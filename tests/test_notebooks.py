"""Freshness and completeness guards for the committed notebooks.

Every notebook in the repo ships its outputs: the docs gallery page is
generated from ``feature_gallery.ipynb``'s *committed* outputs (no
execution at docs-build time), and the tutorials are read on GitHub as
rendered. Stale or hollow outputs therefore publish directly. CI executes
the notebooks in ``test.yml`` (nbmake), which catches cells that no longer
*run*; everything here catches cells that were never *re-run*, and figures
that were never captured.

Four checks apply to every notebook — jupytext pair in sync, execution
counts sequential, no error outputs, and (for plotting notebooks) figures
actually present. One check is deliberately narrower:

**stderr is required clean in the gallery only.** The gallery is published
as a docs page, where a stray warning reads as a defect. The tutorials do
the opposite on purpose: ``01`` demonstrates the seeded-without-``set_epoch``
warning and ``03`` the non-uniform frame-rate and rest-pose-axis warnings,
so their stderr *is* the lesson. Enforcing repo-wide silence would mean
deleting the teaching; enforcing it nowhere would let a real warning onto
the gallery page. Hence ``STDERR_MUST_BE_CLEAN``, listing the notebooks
whose stderr is a bug rather than content.

The figure checks exist because a lost plot is invisible. Jupyter captures
matplotlib output through the inline backend, which is the kernel's default
— but only a default: ``MPLBACKEND`` set in the environment (``Agg``, the
reflex for headless runs) overrides it, and a notebook re-executed in such
a shell is written back with its code, its text output, sequential
execution counts, and not a single picture. It looks healthy. A cell
calling ``plt.show()`` at least emits a ``FigureCanvasAgg is
non-interactive`` warning; a cell relying on the inline backend's
end-of-cell flush drops its figure in total silence. So the guard asserts
the outcome rather than the symptom: the magic is pinned (it beats
``MPLBACKEND``, which is why the notebooks are made portable instead of CI
dropping ``MPLBACKEND=Agg`` — headless runs want that), every ``plt.show()``
cell carries an image, and each notebook clears an explicit figure floor.
The floor is the only check that survives a notebook with no ``plt.show()``
in it at all.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".ipynb_checkpoints", "site", "dist", ".git"}

INLINE_MAGIC = "%matplotlib inline"

# Notebooks published as docs pages, where any warning on stderr is a
# defect. Tutorials are absent by design — see the module docstring.
STDERR_MUST_BE_CLEAN = {"gallery/feature_gallery.ipynb"}

# Lower bound on figures each plotting notebook must still publish. A
# floor, not an exact count — adding figures is free, losing them all is
# the bug this module exists for. A new plotting notebook must be listed
# here; the test fails until it is, which is the point.
# The gallery's two animated clips are NOT in its count: they are committed
# .gif files displayed from markdown cells (see the clip-visibility tests),
# not image outputs.
MIN_FIGURES = {
    "gallery/feature_gallery.ipynb": 13,
    "tutorials/02_augmentation_visualized.ipynb": 3,
}

RE_EXECUTE = ("re-execute it: `jupyter nbconvert --to notebook --execute "
              "--inplace {}`")


def _notebooks():
    return sorted(p for p in REPO.rglob("*.ipynb")
                  if not SKIP_DIRS & set(p.relative_to(REPO).parts))


def _source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _normalized_cells(cells):
    """(cell_type, normalized source) pairs — whitespace-insensitive."""
    return [(cell["cell_type"],
             "\n".join(line.rstrip() for line in _source(cell).splitlines()).strip())
            for cell in cells]


def _code_cells(path):
    nb = json.loads(path.read_text())
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _figure_count(cell):
    return sum(1 for out in cell.get("outputs", [])
               for mime in out.get("data", {}) if mime.startswith("image/"))


def _is_plotting(path):
    """Does this notebook draw? It imports matplotlib or ships figures."""
    cells = _code_cells(path)
    return ("matplotlib" in "\n".join(_source(c) for c in cells)
            or any(_figure_count(c) for c in cells))


ALL = _notebooks()
ALL_IDS = [str(p.relative_to(REPO)) for p in ALL]
PLOTTING = [p for p in ALL if _is_plotting(p)]
PLOTTING_IDS = [str(p.relative_to(REPO)) for p in PLOTTING]


# --------------------------------------------------------------------------
# Freshness — the committed outputs match the committed source
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", ALL, ids=ALL_IDS)
def test_jupytext_pair_in_sync(path):
    jupytext = pytest.importorskip("jupytext")
    py = path.with_suffix(".py")
    assert py.exists(), f"{path.relative_to(REPO)} has no jupytext .py pair"
    rel = path.relative_to(REPO)
    py_cells = _normalized_cells(jupytext.read(py).cells)
    nb_cells = _normalized_cells(json.loads(path.read_text())["cells"])
    assert len(py_cells) == len(nb_cells), (
        f"{rel}: cell count differs — {len(py_cells)} in .py vs "
        f"{len(nb_cells)} in .ipynb; run `jupytext --sync {rel}`")
    for i, (pc, nc) in enumerate(zip(py_cells, nb_cells)):
        assert pc == nc, (
            f"{rel}: cell {i} differs between the .py and the .ipynb — "
            f"run `jupytext --sync {rel}`")


@pytest.mark.parametrize("path", ALL, ids=ALL_IDS)
def test_notebook_was_fully_executed_in_order(path):
    counts = [c.get("execution_count") for c in _code_cells(path)]
    rel = path.relative_to(REPO)
    assert counts == list(range(1, len(counts) + 1)), (
        f"{rel}: execution counts are not sequential 1..N — the committed "
        "outputs are stale (a cell was edited without a full re-run); "
        + RE_EXECUTE.format(rel))


@pytest.mark.parametrize("path", ALL, ids=ALL_IDS)
def test_notebook_has_no_error_outputs(path):
    nb = json.loads(path.read_text())
    errors = [f"cell {i}: {out.get('ename')}"
              for i, cell in enumerate(nb["cells"])
              for out in cell.get("outputs", [])
              if out.get("output_type") == "error"]
    assert not errors, (
        f"{path.relative_to(REPO)} committed error outputs: "
        + "; ".join(errors))


@pytest.mark.parametrize("rel", sorted(STDERR_MUST_BE_CLEAN))
def test_published_notebook_has_no_stderr(rel):
    nb = json.loads((REPO / rel).read_text())
    noisy = []
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                noisy.append(f"cell {i}: {''.join(out.get('text', []))[:80]!r}")
    assert not noisy, (
        f"{rel} would publish warnings on its docs page: " + "; ".join(noisy))


# --------------------------------------------------------------------------
# Completeness — plotting notebooks actually contain their pictures
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", PLOTTING, ids=PLOTTING_IDS)
def test_inline_backend_is_pinned_before_the_first_figure(path):
    cells = _code_cells(path)
    magic_at = next((i for i, c in enumerate(cells)
                     if INLINE_MAGIC in _source(c)), None)
    assert magic_at is not None, (
        f"{path.relative_to(REPO)} plots but never runs `{INLINE_MAGIC}` — "
        "re-executing it in a shell with MPLBACKEND set would commit a "
        "notebook with no figures. Add the magic to the setup cell (write "
        "it as `# %matplotlib inline` in the paired .py; jupytext uncomments "
        "it, and running the .py as a script just sees a comment)")
    first_figure_at = next((i for i, c in enumerate(cells)
                            if _figure_count(c)), None)
    if first_figure_at is not None:
        assert magic_at <= first_figure_at, (
            f"{path.relative_to(REPO)} runs `{INLINE_MAGIC}` in cell "
            f"{magic_at}, after the first figure in cell {first_figure_at} — "
            "the magic must precede every plot to have any effect")


@pytest.mark.parametrize("path", PLOTTING, ids=PLOTTING_IDS)
def test_paired_py_carries_the_commented_magic(path):
    """The .py is what people edit, so the magic has to survive there too."""
    py = path.with_suffix(".py")
    assert f"# {INLINE_MAGIC}" in py.read_text(), (
        f"{py.relative_to(REPO)} lost `# {INLINE_MAGIC}` — a re-sync would "
        "strip it from the notebook and figures would stop being captured")


@pytest.mark.parametrize("path", PLOTTING, ids=PLOTTING_IDS)
def test_every_plt_show_cell_captured_a_figure(path):
    bare = [i for i, cell in enumerate(_code_cells(path))
            if "plt.show()" in _source(cell) and not _figure_count(cell)]
    rel = path.relative_to(REPO)
    assert not bare, (
        f"{rel}: cells {bare} call plt.show() but committed no image output "
        "— the notebook was executed with a non-inline backend; "
        + RE_EXECUTE.format(rel))


@pytest.mark.parametrize("path", PLOTTING, ids=PLOTTING_IDS)
def test_notebook_meets_its_figure_floor(path):
    rel = str(path.relative_to(REPO))
    assert rel in MIN_FIGURES, (
        f"{rel} plots but has no entry in MIN_FIGURES — add its figure count "
        "as a floor, so a run that silently captures nothing gets caught")
    total = sum(_figure_count(c) for c in _code_cells(path))
    assert total >= MIN_FIGURES[rel], (
        f"{rel} publishes {total} figures, below its floor of "
        f"{MIN_FIGURES[rel]} — either figures were lost to a non-inline "
        "backend, or they were intentionally removed and the floor needs "
        "lowering")


# --------------------------------------------------------------------------
# Clip visibility — animated GIFs must be linked files, not cell outputs
# --------------------------------------------------------------------------

RAW_PREFIX = "https://raw.githubusercontent.com/VictorS-67/pybvh-ml/main/"


def _servable_from_github(repo_rel):
    """The file exists and is not gitignored — ``exists()`` alone would accept a local byproduct that raw.githubusercontent.com will 404 on."""
    if not (REPO / repo_rel).exists():
        return False
    if not (REPO / ".git").exists():
        return True                     # sdist/tarball: existence is all we have
    ignored = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", repo_rel],
        capture_output=True)
    return ignored.returncode != 0


@pytest.mark.parametrize("path", ALL, ids=ALL_IDS)
def test_no_gif_cell_outputs(path):
    """GitHub's notebook renderer displays ``image/png`` outputs but silently drops ``image/gif`` ones — the reader sees ``<IPython.core.display.Image object>`` where the clip should play, and the figure floor never notices (GIF outputs are not PNG outputs). A clip belongs in a committed ``.gif`` displayed from a markdown cell; see ``test_markdown_images_are_absolute_and_resolve`` for the form that cell must take."""
    offenders = [i for i, cell in enumerate(_code_cells(path))
                 for out in cell.get("outputs", [])
                 if "image/gif" in out.get("data", {})]
    rel = path.relative_to(REPO)
    assert not offenders, (
        f"{rel}: code cells {offenders} embed image/gif outputs, invisible "
        f"on github.com. Return the path from the gallery_plots helper, "
        f"commit the GIF, and display it from a markdown cell with an "
        f"absolute raw.githubusercontent.com URL.")


@pytest.mark.parametrize("path", ALL, ids=ALL_IDS)
def test_markdown_images_are_absolute_and_resolve(path):
    """GitHub's notebook renderer does not resolve *relative* image paths in markdown cells (the reader sees only the alt text), so every markdown image must use an absolute ``raw.githubusercontent.com`` URL — possible because this repo is public. Resolving each URL against the working tree (and against .gitignore, since raw.githubusercontent.com serves only committed files) catches a renamed, ignored, or never-committed file locally instead of as a broken-image icon on github.com."""
    nb = json.loads(path.read_text())
    rel = path.relative_to(REPO)
    problems = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        for src in re.findall(r"!\[[^\]]*\]\((\S+?)\)", _source(cell)):
            if not src.startswith(RAW_PREFIX):
                problems.append(
                    f"cell {i}: {src!r} is not an absolute {RAW_PREFIX} URL "
                    f"(GitHub shows only the alt text for relative paths)")
            elif not _servable_from_github(src[len(RAW_PREFIX):]):
                problems.append(
                    f"cell {i}: {src!r} does not resolve to a committed, "
                    f"non-gitignored file (GitHub would show a broken-image "
                    f"icon)")
    assert not problems, f"{rel}: " + "; ".join(problems)
