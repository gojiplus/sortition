"""Every test module that imports an optional extra is declared as needing it.

CI installs the built wheel with base dependencies only and runs this suite
against it, which is how the minimal install gets proven to work. A module with
``import polars`` at the top that is not registered in ``conftest``'s ``_NEEDS``
does not skip there -- it raises at collection, which fails the whole job rather
than the one module. That is a one-line omission with no local symptom, so it is
checked rather than remembered.

Only top-level imports of the test file itself are checked, because those are
what collection evaluates. An extra needed further in -- LightGBM, which
``sortition.train`` imports inside a function -- makes the test error rather than
the job collapse, so it belongs in ``_NEEDS`` too but cannot be found this way.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from conftest import _NEEDS

OPTIONAL = {"scipy", "sklearn", "polars", "duckdb", "rich", "lightgbm", "typer"}

TESTS = Path(__file__).parent


def _top_level_imports(path: Path) -> set[str]:
    """Distribution names the module imports at import time.

    Imports nested inside a function or a ``TYPE_CHECKING`` block do not run at
    collection, so only module-scope statements count.

    Args:
        path: The test file.

    Returns:
        Top-level names of everything imported at module scope.
    """
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "module", sorted(p.name for p in TESTS.glob("test_*.py")), ids=lambda name: name
)
def test_optional_imports_are_declared(module: str) -> None:
    missing = (_top_level_imports(TESTS / module) & OPTIONAL) - set(
        _NEEDS.get(module, ())
    )
    assert not missing, (
        f"{module} imports {sorted(missing)} at module scope but "
        "conftest._NEEDS does not say so, so on the minimal install it fails "
        "collection -- taking the whole job with it -- instead of skipping"
    )


def test_every_declared_module_exists() -> None:
    # A renamed test file leaves a stale entry that silently guards nothing.
    stale = [name for name in _NEEDS if not (TESTS / name).exists()]
    assert not stale, f"conftest._NEEDS names files that are gone: {stale}"
