# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 3. Heterogeneous datasets: `harmonize`, `skip_errors`, topology checks
#
# Real motion-capture datasets are messy:
#
# - Clips from different motion capture systems with different skeleton topologies.
# - Frame rates ranging from 24 fps to 120 fps in the same corpus.
# - Different up-axis conventions (Y-up vs. Z-up, Mixamo-style +z-facing vs. 3ds Max +y-facing).
# - The occasional corrupt file that breaks the whole batch.
#
# This notebook shows how to handle each of these before `preprocess_directory`.

# %% [markdown]
# ## Setup

# %%
import shutil
import tempfile
from pathlib import Path

from pybvh import read_bvh_file, read_bvh_directory, harmonize, write_bvh_file
from pybvh_ml import preprocess_directory

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
BVH_DIR = REPO_ROOT / "bvh_data"

# Peek at the fixtures' skeletons.
for name in ("bvh_test1.bvh", "bvh_test2.bvh", "bvh_test3.bvh"):
    b = read_bvh_file(BVH_DIR / name)
    print(f"{name:22s}  joints={b.joint_count:2d}  fps={1/b.frame_time:6.2f}  up={b.world_up}")

# %% [markdown]
# The fixtures have different skeletons (different joint counts and/or axes).
# A preprocessing pipeline that blindly loads all three into one `.npz` would
# produce garbage.

# %% [markdown]
# ## The safety net: `require_matching_topology=True`
#
# Since pybvh-ml 0.3, `preprocess_directory` validates skeleton compatibility by default.
# Mixing incompatible clips raises `ValueError`:

# %%
work_dir = Path(tempfile.mkdtemp(prefix="pybvh_ml_hetero_"))

# Copy the two incompatible fixtures.
shutil.copy(BVH_DIR / "bvh_test1.bvh", work_dir / "a.bvh")
shutil.copy(BVH_DIR / "bvh_test3.bvh",  work_dir / "b.bvh")

try:
    preprocess_directory(work_dir, work_dir / "bad.npz")
except ValueError as e:
    print("As expected:")
    print(" ", e)

# %% [markdown]
# ## Fix #0 — axis-only mismatch with `target_world_up` / `target_rest_forward` / `target_rest_up`
#
# The cheaper case: every clip shares topology but axes disagree.
# Three symptoms, three kwargs on `preprocess_directory`:
#
# - `target_world_up` — animation up differs across files.
# - `target_rest_forward` — rest-pose facing direction differs across files.
# - `target_rest_up` — rest-pose up disagrees with the file's own animation
#   up (a per-file anomaly; see subsection below).
#
# `preprocess_directory` audits all three after loading and emits one
# aggregated warning per category with the kwarg that fixes it.  Each
# `target_*` reorients every clip before rotation data is extracted —
# FK positions are preserved; joint-angle numbers change.
#
# Compared with `pybvh.harmonize` (next section), `target_*` is the
# minimal tool when only axes disagree — no topology retargeting, no
# FPS resampling.

# %%
# Synthesize a cross-convention mix: keep bvh_test1 as-is and add a
# +y-up copy.  Both share topology.
axis_dir = Path(tempfile.mkdtemp(prefix="pybvh_ml_axis_"))
shutil.copy(BVH_DIR / "bvh_test1.bvh", axis_dir / "captureA.bvh")
orig = read_bvh_file(BVH_DIR / "bvh_test1.bvh")
write_bvh_file(orig.reorient_world_up("+y"), axis_dir / "captureB.bvh")

# Topologies match so preprocess_directory doesn't raise, but it emits
# one aggregated UserWarning per heterogeneous axis and records the
# distribution in summary["uniformity"] — a machine-readable snapshot
# you can grep from CI to fail builds on surprise heterogeneity.
summary = preprocess_directory(
    axis_dir, axis_dir / "axis_mismatch.npz", representation="6d")

for key, dist in summary["uniformity"].items():
    counts = {k: len(v) for k, v in dist.items()} if isinstance(dist, dict) else dist
    print(f"{key}: {counts}")

# %% [markdown]
# Now pass the `target_*` kwargs to reorient every clip to a shared
# axis.  Warnings go away; outputs are directly comparable across the
# two captures:

# %%
summary = preprocess_directory(
    axis_dir, axis_dir / "axis_harmonized.npz",
    representation="6d",
    target_world_up="+z",        # reorient every clip to +z up
    target_rest_forward="+y",    # and +y rest-pose forward
)
print("num_clips:", summary["num_clips"])

# %% [markdown]
# ### `target_rest_up` — rest-pose up disagrees with animation up
#
# Some exporters author the rest (T-pose) in one convention while the
# animation plays in another.  `preprocess_directory` lists affected
# files under `uniformity["rest_anim_mismatch"]` (printed above) and
# emits one aggregated warning; the per-file pybvh warning is
# suppressed in favor of this batch-level message.
#
# It matters because **every rotation representation is affected**: for the same world pose, quaternions / 6D / axis-angle
# tensors all differ between a `+z`-rest file and a `+y`-rest file.
# Training tensors silently diverge.  Fix with `target_rest_up`:

# %%
summary = preprocess_directory(
    axis_dir, axis_dir / "axis_harmonized.npz",
    representation="6d",
    target_world_up="+z",        # reorient every clip to +z up
    target_rest_forward="+y",    # and +y rest-pose forward
    target_rest_up="+z"          # and +z rest-pose up
)
print("num_clips:", summary["num_clips"])


# Cleanup
shutil.rmtree(axis_dir)

# %% [markdown]
# ## Fix #1 — topology / FPS mismatch with `harmonize`
#
# `pybvh.harmonize` is the heavier dataset-level preprocessor.  Give it a
# reference skeleton plus optional `target_fps`, `target_world_up`,
# `target_rest_forward`, `target_rest_up` — and it drops clips that can't
# be made compatible (`on_incompatible="drop"`, the default) and retargets
# / resamples / reorients the rest.  The three `target_*` reorient kwargs
# mirror `preprocess_directory` exactly, so once you're calling
# `harmonize` it handles every axis concern too; reach for it when
# topology or frame rate differs.

# %%
clips = read_bvh_directory(work_dir, parallel=False)
print(f"Loaded {len(clips)} clips: {[b.joint_count for b in clips]} joints each")

reference = clips[0]
harmonized = harmonize(
    clips,
    reference=reference,         # topology must match this
    target_fps=30,               # SLERP-resample any fps mismatch
    target_world_up="+z",        # rotate the world if up axes disagree
    on_incompatible="drop",      # drop mismatches (alternative: "raise")
    verbose=True,                # UserWarning per dropped clip
)

print(f"\n{len(harmonized)} clips survived; joints: {[b.joint_count for b in harmonized]}")

# %% [markdown]
# Now write the harmonized clips to a clean directory and preprocess normally:

# %%
harmonized_dir = Path(tempfile.mkdtemp(prefix="pybvh_ml_hetero_ok_"))
for i, b in enumerate(harmonized):
    write_bvh_file(b, harmonized_dir / f"clip_{i:03d}.bvh")

summary = preprocess_directory(
    harmonized_dir,
    harmonized_dir / "train.npz",
    representation="6d",
    require_matching_topology=True,    # now safe because we pre-harmonized
)
print("preprocessed:", summary["num_clips"], "clips,", summary["representation"])

# %% [markdown]
# ## Fix #2 — tolerate corrupt files with `skip_errors=True`
#
# Real datasets have the occasional truncated / malformed file. Passing `skip_errors=True`
# makes `preprocess_directory` warn and skip each failure instead of bailing out on the
# first bad file.

# %%
# Drop a deliberately broken file into our clean directory.  Jupyter will
# render preprocess_directory's "skipping …" UserWarning inline.
(harmonized_dir / "broken.bvh").write_text("not a valid bvh file\n")

summary = preprocess_directory(
    harmonized_dir, harmonized_dir / "robust.npz",
    representation="6d", skip_errors=True,
)
print(f"num_clips preprocessed: {summary['num_clips']}")

# %% [markdown]
# ## Optional — apply an explicit `lr_mapping` uniformly
#
# pybvh's auto-detector recognizes most conventions out of the box —
# `.L`/`.R` and `_L`/`_R` suffixes, `Left`/`Right` substrings, and the
# `L*`/`R*` prefix form, with `mixamorig:`-style namespace prefixes and
# Blender's `.001` duplicate suffixes stripped before matching.  It fails
# only on names that fit none of those (e.g. non-English naming or pairs
# that differ solely by a numeric ID).  In that case, pass an explicit
# `lr_mapping` to `preprocess_directory` and it propagates to every
# `read_bvh_file` call:
#
# ```python
# preprocess_directory(
#     work_dir, out,
#     lr_mapping={"LeftArm": "RightArm", "LeftLeg": "RightLeg", ...},
#     world_up="+y",    # same idea for up axis
# )
# ```
#
# Both kwargs are pure pass-throughs to `pybvh.read_bvh_file`.

# %% [markdown]
# ## Checklist: a robust preprocessing recipe
#
# 1. **Inspect first.** `read_bvh_directory(..., skip_errors=True)` + check
#    `b.joint_count`, `1/b.frame_time`, `b.world_up` — print them and look for outliers.
# 2. **Pick the right harmonization tool.**
#    - Axes disagree but topology / FPS match → `target_world_up`,
#      `target_rest_forward`, `target_rest_up` on `preprocess_directory`
#      (fast, in-process; no intermediate files).
#    - Topology / FPS / missing joints differ → `pybvh.harmonize(..., on_incompatible="drop")`
#      for a forgiving pass, `"raise"` for strict CI.  Its `target_*`
#      reorient kwargs are a strict superset of `preprocess_directory`'s,
#      so if you're running `harmonize` anyway, pass the axes to it and
#      skip the `preprocess_directory` kwargs.
# 3. **Write out.** If you used `harmonize`, `write_bvh_file` the harmonized clips to a clean
#    directory so `preprocess_directory` reads from a known-good input.
# 4. **Preprocess.** `preprocess_directory(..., skip_errors=True, parallel=True,
#    require_matching_topology=True)` gives you throughput + a last-line-of-defence check.
#    Inspect `summary["uniformity"]` in CI to guard against silent axis drift.
# 5. **Normalize.** `load_preprocessed(out)["mean"] / ["std"]` are ready for
#    `pybvh.normalize_array(...)` at training time; `["constant_channels"]` tells you
#    which columns had zero variance.

# %%
# Clean up
shutil.rmtree(work_dir)
shutil.rmtree(harmonized_dir)
