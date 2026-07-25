"""Guard: the API reference stays in sync with the code.

The docs/api pages render whole modules via ``::: pybvh_ml.<module>``
mkdocstrings blocks, so individual members can't silently drop out of the
docs the way curated member lists can (the failure mode pybvh's
test_docs_api_coverage.py guards against). What can still rot silently:

- a new public module never gets an api page,
- a stale ``:::`` block points at a renamed or removed module,
- an api page falls out of the mkdocs nav,
- an ``__all__`` name stops resolving.

These tests turn each of those into a test failure that names the offender.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS_API = REPO / "docs" / "api"
PACKAGE = REPO / "pybvh_ml"

_BLOCK_RE = re.compile(r"^::: ([\w.]+)\s*$", re.MULTILINE)


def _public_modules() -> set[str]:
    """Dotted paths of every public module in the package (recursively)."""
    modules = set()
    for path in PACKAGE.rglob("*.py"):
        rel = path.relative_to(PACKAGE)
        if any(part.startswith("_") for part in rel.parts):
            continue
        modules.add("pybvh_ml." + ".".join(rel.with_suffix("").parts))
    return modules


def _documented_targets() -> set[str]:
    """Every ``::: target`` across the api pages."""
    targets = set()
    for page in DOCS_API.glob("*.md"):
        targets.update(_BLOCK_RE.findall(page.read_text()))
    return targets


def test_api_pages_cover_every_public_module_two_way():
    actual = _public_modules()
    documented = _documented_targets()
    undocumented = sorted(actual - documented)
    stale = sorted(documented - actual)
    problems = []
    if undocumented:
        problems.append(
            f"public modules missing from docs/api/ pages: {undocumented}")
    if stale:
        problems.append(
            f"docs/api/ pages document modules that no longer exist: {stale}")
    assert not problems, "; ".join(problems)


def test_every_api_page_is_in_the_nav():
    mkdocs_yml = (REPO / "mkdocs.yml").read_text()
    missing = sorted(
        p.name for p in DOCS_API.glob("*.md") if f"api/{p.name}" not in mkdocs_yml
    )
    assert not missing, f"api pages not referenced in mkdocs.yml nav: {missing}"


def test_top_level_all_resolves():
    import pybvh_ml

    broken = [n for n in pybvh_ml.__all__ if not hasattr(pybvh_ml, n)]
    assert not broken, f"pybvh_ml.__all__ names that don't resolve: {broken}"


def test_torch_all_resolves():
    pytest.importorskip("torch")
    import pybvh_ml.torch

    broken = [n for n in pybvh_ml.torch.__all__ if not hasattr(pybvh_ml.torch, n)]
    assert not broken, f"pybvh_ml.torch.__all__ names that don't resolve: {broken}"
