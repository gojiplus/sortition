# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Log schema v1 (`sortition.schema`): `Decision`, `DecisionRow`, `ExecutionRow`,
  `OutcomeRow`, `PolicyArtifact`. Versioned and additive-only.
- `sortition.sim`: synthetic contextual bandits with exactly known policy values,
  used as the oracle for every estimator test.
- `sortition.eval`: IPS, SNIPS, direct method, doubly robust, switch-DR and DR
  with optimistic shrinkage; cross-fitted outcome models; Waudby-Smith–Ramdas
  hedged betting intervals for bounded outcomes and bootstrap intervals for cost
  and latency; overlap, effective-sample-size, support-violation and leakage
  diagnostics that mark an estimate untrustworthy rather than reporting a number
  the data cannot support.
- `sortition.targets`: counterfactual policy specifications (`always:<arm>`,
  `uniform`, mixtures, exploration floors).
- `sortition.frame`: the flat log table and its reduction to estimator inputs,
  including exclusion and reporting of gateway-fallback rows.
- `evaluate`, `compare` and `doctor`, plus a `sortition` CLI over parquet, CSV
  and ndjson logs.

### Changed

- Adopted the py-canon fleet standard via `preen adopt`: pyright in place of
  mypy, google-style docstrings, 88-column formatting, and tag-derived
  versioning.
