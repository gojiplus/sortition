"""Policies as versioned files, and swapping them without a restart.

Two properties make this worth having beyond "config in a file".

**The version is a content hash.** Two deployments cannot disagree about what
"rules-v4" means, and a policy that was edited but not renamed cannot
impersonate the one it replaced. Since every log row carries ``policy_version``,
this is what stops an estimate from silently pooling two different policies.

**Reload cannot fail open.** A corrupt or unreadable artifact leaves the running
policy in place and increments an error count. The alternative -- falling back to
some default -- would route real traffic under a policy nobody chose, and would
do it quietly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sortition import metrics
from sortition.decide.engine import DecisionEngine, ExplorationConfig
from sortition.decide.policy import (
    ConstantPolicy,
    CostAwarePolicy,
    Policy,
    WeightedPolicy,
)
from sortition.decide.rules import RulesPolicy
from sortition.decide.tree import TreePolicy
from sortition.schema import Decision, PolicyArtifact

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 5.0


def _payload_of(policy: Policy) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    """Serialize a policy to its kind, body and arm list.

    Args:
        policy: The policy to serialize.

    Returns:
        A ``(kind, payload, arms)`` triple.

    Raises:
        TypeError: If the policy type has no artifact representation.
    """
    if isinstance(policy, RulesPolicy):
        return (
            "rules",
            {
                "arms": list(policy.arms),
                "default": list(policy.default),
                "name": policy.name,
                "rules": [
                    {
                        "when": rule.when,
                        "prefer": list(rule.prefer),
                        "exclude": list(rule.exclude),
                        "label": rule.label,
                    }
                    for rule in policy.rules
                ],
            },
            policy.arms,
        )
    if isinstance(policy, TreePolicy):
        return (
            "tree",
            {
                "booster_text": policy.booster_text,
                "feature_spec": list(policy.feature_spec),
                "arms": list(policy.arms),
                "cost_booster_text": policy.cost_booster_text,
                "cost_scale": policy.cost_scale,
                "cost_weight": policy.cost_weight,
                "name": policy.name,
            },
            policy.arms,
        )
    if isinstance(policy, ConstantPolicy):
        return "constant", {"arm": policy.arm}, (policy.arm,)
    if isinstance(policy, WeightedPolicy):
        return (
            "weighted",
            {"weights": dict(policy.weights), "name": policy.name},
            tuple(sorted(policy.weights)),
        )
    if isinstance(policy, CostAwarePolicy):
        return (
            "cost_aware",
            {
                "quality": dict(policy.quality),
                "cost_usd": dict(policy.cost_usd),
                "cost_weight": policy.cost_weight,
                "name": policy.name,
            },
            tuple(sorted(policy.quality)),
        )
    raise TypeError(
        f"{type(policy).__name__} has no artifact representation; add one to "
        "_payload_of and _policy_from so it can be versioned and reloaded"
    )


def _policy_from(kind: str, payload: dict[str, Any]) -> Policy:
    """Rebuild a policy from its serialized form.

    Args:
        kind: The policy kind recorded in the artifact.
        payload: The serialized body.

    Returns:
        The reconstructed policy.

    Raises:
        ValueError: If the kind is unknown.
    """
    if kind == "rules":
        return RulesPolicy.from_dict(payload)
    if kind == "tree":
        return TreePolicy(
            booster_text=payload["booster_text"],
            feature_spec=tuple(payload["feature_spec"]),
            arms=tuple(payload["arms"]),
            cost_booster_text=payload.get("cost_booster_text"),
            cost_scale=float(payload.get("cost_scale", 0.0)),
            cost_weight=float(payload.get("cost_weight", 0.0)),
            name=payload.get("name", "tree"),
        )
    if kind == "constant":
        return ConstantPolicy(payload["arm"])
    if kind == "weighted":
        return WeightedPolicy(
            weights=dict(payload["weights"]), name=payload.get("name", "weighted")
        )
    if kind == "cost_aware":
        return CostAwarePolicy(
            quality=dict(payload["quality"]),
            cost_usd=dict(payload["cost_usd"]),
            cost_weight=float(payload.get("cost_weight", 0.0)),
            name=payload.get("name", "cost-aware"),
        )
    raise ValueError(f"unknown policy kind {kind!r}")


def content_hash(
    kind: str, payload: dict[str, Any], exploration: dict[str, Any]
) -> str:
    """Hash everything that changes what the policy does.

    Exploration is included because a policy at epsilon 0.05 and the same policy
    at 0.2 are different sampling distributions, and pooling their logs into one
    estimate would be wrong.

    Args:
        kind: The policy kind.
        payload: The serialized policy body.
        exploration: The exploration configuration.

    Returns:
        A short hex digest.
    """
    canonical = json.dumps(
        {"kind": kind, "payload": payload, "exploration": exploration},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def build(
    policy: Policy,
    exploration: ExplorationConfig | None = None,
    *,
    name: str | None = None,
    git_sha: str | None = None,
) -> PolicyArtifact:
    """Package a policy and its exploration setting into a versioned artifact.

    Args:
        policy: The policy to package.
        exploration: How it explores. Defaults to the standard epsilon floor.
        name: Optional label prefixed to the content hash.
        git_sha: Optional commit the policy was built from.

    Returns:
        The artifact, with ``policy_version`` derived from its content.
    """
    exploration = exploration or ExplorationConfig()
    kind, payload, arms = _payload_of(policy)
    settings = exploration.as_dict()
    digest = content_hash(kind, payload, settings)
    label = name or getattr(policy, "name", kind)
    return PolicyArtifact(
        policy_version=f"{label}-{digest}",
        kind=kind,  # type: ignore[arg-type]
        created_at=datetime.now(UTC),
        arms=arms,
        exploration=settings,
        git_sha=git_sha,
        payload=payload,
    )


def save(artifact: PolicyArtifact, path: str | Path) -> Path:
    """Write an artifact to disk atomically.

    Written to a temporary file and renamed, so a reloading process can never
    read a half-written policy.

    Args:
        artifact: The artifact to write.
        path: Destination file.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    tmp.rename(target)
    logger.info("wrote policy %s to %s", artifact.policy_version, target)
    return target


def load(path: str | Path) -> tuple[Policy, ExplorationConfig, PolicyArtifact]:
    """Read an artifact and rebuild the policy it describes.

    Args:
        path: The artifact file.

    Returns:
        The policy, its exploration configuration, and the artifact itself.

    Raises:
        ValueError: If the file's contents do not match its recorded version,
            which means it was edited without being rebuilt.
    """
    artifact = PolicyArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    policy = _policy_from(artifact.kind, artifact.payload)

    digest = content_hash(artifact.kind, artifact.payload, artifact.exploration)
    if not artifact.policy_version.endswith(digest):
        raise ValueError(
            f"{path} claims version {artifact.policy_version!r} but its contents "
            f"hash to {digest!r}. It was edited without being rebuilt, so its "
            "logs would be attributed to a policy that never ran."
        )

    settings = dict(artifact.exploration)
    exploration = ExplorationConfig(
        strategy=settings.get("strategy", "epsilon_greedy"),
        epsilon=float(settings.get("epsilon", 0.05)),
    )
    return policy, exploration, artifact


class ReloadingEngine:
    """A decision engine that picks up policy changes without a restart.

    The artifact is checked at most once per ``poll_interval``, on the decision
    path, which costs one clock comparison per request and avoids a background
    thread. Swapping is a single attribute assignment, so a request sees the old
    policy or the new one and never a mixture.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        engine_factory: Any = None,
    ) -> None:
        """Load a policy artifact and watch it for changes.

        Args:
            path: The artifact to load and watch.
            poll_interval: Seconds between staleness checks.
            engine_factory: Builds a :class:`DecisionEngine` from
                ``(policy, exploration, policy_version)``. Injectable for tests.

        An unreadable or invalid artifact propagates out of here rather than
        being caught. That is deliberate: reloads fall back to the running
        policy, but starting with no policy at all is worse than not starting.
        """
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._factory = engine_factory or self._default_factory
        self._lock = threading.Lock()
        self._last_check = 0.0
        self.reloads = 0
        self.reload_errors = 0

        self._engine = self._build()
        self._fingerprint = self._read_fingerprint()
        # Start the clock now: the artifact was just read, so the next check is
        # a full interval away rather than immediately due.
        self._last_check = time.monotonic()

    @staticmethod
    def _default_factory(
        policy: Policy, exploration: ExplorationConfig, policy_version: str
    ) -> DecisionEngine:
        return DecisionEngine(
            policy=policy, exploration=exploration, policy_version=policy_version
        )

    def _build(self) -> DecisionEngine:
        policy, exploration, artifact = load(self.path)
        return self._factory(policy, exploration, artifact.policy_version)

    def _read_fingerprint(self) -> tuple[float, int] | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        # mtime and size are enough to notice an edit and cost no read.
        return (stat.st_mtime, stat.st_size)

    @property
    def policy_version(self) -> str:
        """The version currently serving traffic."""
        return self._engine.policy_version or "unknown"

    def decide(self, **kwargs: Any) -> Decision:
        """Route one request, reloading the policy first if it has changed.

        Args:
            **kwargs: Passed through to :meth:`DecisionEngine.decide`.

        Returns:
            The decision, tagged with whichever policy version served it.
        """
        self.maybe_reload()
        return self._engine.decide(**kwargs)

    def maybe_reload(self) -> bool:
        """Reload if the artifact changed and enough time has passed.

        Returns:
            True if a new policy is now serving.
        """
        now = time.monotonic()
        if now - self._last_check < self.poll_interval:
            return False

        with self._lock:
            if now - self._last_check < self.poll_interval:
                return False
            self._last_check = now
            fingerprint = self._read_fingerprint()
            if fingerprint is None or fingerprint == self._fingerprint:
                return False

            try:
                engine = self._build()
            except Exception:
                self.reload_errors += 1
                metrics.policy_reload_errors.inc()
                # Deliberately keeps serving the old policy. Falling back to a
                # default would route real traffic under a policy nobody chose.
                logger.exception(
                    "failed to reload %s; continuing on %s",
                    self.path,
                    self.policy_version,
                )
                # Remember the bad fingerprint so a broken file is not retried
                # on every poll.
                self._fingerprint = fingerprint
                return False

            previous = self.policy_version
            # One assignment: a concurrent decide() sees the old engine or the
            # new one, never a partially constructed policy.
            self._engine = engine
            self._fingerprint = fingerprint
            self.reloads += 1
            metrics.policy_reloads.inc()
            logger.info(
                "reloaded policy %s -> %s from %s",
                previous,
                self.policy_version,
                self.path,
            )
            return True
