# Preprocessing Pipelines

The "run once" step: batch convert a directory of BVH files into a single on-disk dataset that training jobs load in milliseconds. Preprocessing and runtime are deliberately separate — this page is the disk side; [Runtime Augmentation](augmentation.md) is the every-epoch side.

## One call

```python
from pybvh_ml import preprocess_directory, load_preprocessed

summary = preprocess_directory(
    "dataset/", "train.npz",
    representation="6d",
    parallel=True,         # threaded loading for large directories
    skip_errors=True,      # skip + warn on malformed files
)

data = load_preprocessed("train.npz")
clips = data["clips"]                # list of per-clip dicts
mean, std = data["mean"], data["std"]
skel = data["skeleton_info"]         # edges, lr_pairs, joint_names, world_up, ...
```

Supported representations: `"euler"`, `"quat"`, `"6d"`, `"axisangle"` (validated up front, before any file is parsed). Output formats: `.npz` (always available) and `.hdf5` / `.h5` (requires the [`hdf5` extra](../getting-started/installation.md)); the suffix picks the format and anything else raises immediately.

## What the file stores

The dataset file is self-sufficient — everything needed at training time, no reopening source BVHs:

- **Per-clip arrays**: `root_pos`, `joint_data`, plus optional velocities, foot contacts, quaternions, and labels (below).
- **Skeleton metadata** (`skeleton_info`): joint names, edges, L/R pairs, Euler orders — and the `world_up` / `rest_forward` / `rest_up` axis strings, so runtime augmentation can be configured straight from the loaded dict.
- **Normalization statistics**: per-channel `mean` / `std` over all frames, plus a `constant_channels` bool mask (columns whose raw std was below `1e-8`, guarded to `1.0`).
- **The `center_root` flag**: whether the stored `root_pos` arrays were centered at preprocessing time (default `True`). Files from older versions load with `None` (unknown).
- **The uniformity audit** (`uniformity`): per-axis value counts across the corpus, and — when harmonizing — `harmonized_to` with the resolved targets, the `retarget` choice, and per-stage modification counts. The transformation trail is auditable from the file itself.

## Richer outputs

```python
preprocess_directory(
    "dataset/", "train.npz", representation="6d",
    include_velocities=True,      # per-joint linear velocities (F, J, 3)
    include_foot_contacts=True,   # binary contact labels + foot joint names
    foot_joints=["LeftFoot", "RightFoot"],   # explicit; auto-detected when None
    include_quaternions=True,     # pre-computed quats for runtime augmentation
    label_fn=lambda stem: 0,      # filename stem → integer class
    filter_fn=lambda stem: True,  # skipped files are never parsed
    world_up="auto",              # forwarded to every pybvh.read_bvh_file call
    lr_mapping=None,              # ditto
)
```

Two notes:

- Pass `foot_joints=` explicitly for footless or nonstandard rigs where auto-detection finds nothing.
- For `representation="quat"`, `include_quaternions=True` stores nothing extra — the main `joint_data` already *is* the quaternion array, and the loader aliases `clip["joint_quats"]` to it instead of duplicating storage.

## Harmonizing heterogeneous datasets

When clips come from different skeletons, frame rates, up-axis conventions, or — for order-sensitive representations like `"euler"` / `"axisangle"` — different per-joint Euler orders, pass `harmonize=True`:

```python
preprocess_directory(
    "raw/", "train.npz",
    representation="euler",
    harmonize=True,                  # runs pybvh.harmonize after loading
    target_world_up="+y",            # (optional) explicit target; majority otherwise
    skip_errors=True,
)
```

`harmonize=True` runs [`pybvh.harmonize`](https://victors-67.github.io/pybvh/api/batch/), using majority values from the uniformity audit for any `target_*` you didn't set explicitly. For order-sensitive representations it also auto-picks a `target_euler_order` (the most common per-joint order across the dataset).

**Harmonization is pure reorientation/resampling — each actor's bone lengths are preserved.** Pass `retarget=True` to additionally retarget every clip's bone offsets to the first clip, when the whole dataset should share one skeleton geometry (e.g. for fixed-topology GCNs). Hierarchy mismatches raise loudly either way — no silent drops.

!!! warning "Heterogeneity warnings are advice, not errors"
    Without `harmonize=True`, mixed axis conventions produce warnings telling you which values disagree and that `harmonize=True` will unify them to the majority. Heeding them matters: training on clips with mixed up-axes teaches your model that "up" is two different directions.

For workflows that need to inspect or persist the harmonized intermediates, call `pybvh.harmonize` directly, write the results, and preprocess the written directory:

```python
from pybvh import read_bvh_directory, harmonize, write_bvh_file
from pybvh_ml import preprocess_directory
from pathlib import Path

clips = read_bvh_directory("raw/", parallel=True, skip_errors=True)
harmonized = harmonize(clips, reference=clips[0], target_fps=30, target_world_up="+y")

out_dir = Path("harmonized/")
out_dir.mkdir(exist_ok=True)
for b, src in zip(harmonized, clips):
    write_bvh_file(b, out_dir / Path(src.source_path).name)
preprocess_directory(out_dir, "train.npz", representation="6d")
```

## Normalization

Per-channel z-score normalization following the `Mean.npy` / `Std.npy` convention used by HumanML3D and MDM:

```python
from pybvh import read_bvh_directory
from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array

bvhs = read_bvh_directory("dataset/")
stats = compute_normalization_stats(bvhs, representation="6d")
x_norm = normalize_array(x, stats)     # (x - mean) / std
x = denormalize_array(x_norm, stats)   # back to BVH units
```

`compute_normalization_stats` takes a list of `Bvh` objects and computes stats over all frames in the flat `[root_pos, joint_data]` channel layout (`include_root_pos=False` drops the first 3 columns). Zero-variance channels get their std guarded to `1.0` and flagged in the `constant_channels` mask.

`preprocess_directory` stores the same stats in its output file, so after `load_preprocessed` you can pass the loaded dict straight to `normalize_array` — the direct entry point is for workflows that skip the on-disk artifact. Pass `center_root=True` to reproduce exactly the stats a `preprocess_directory` run stores under its default first-frame root centering.

## See also

- [Preprocessing & Normalization API](../api/preprocessing.md) — full signatures
- [CHANGELOG](https://github.com/VictorS-67/pybvh-ml/blob/main/CHANGELOG.md) — migration notes for the 0.5 `retarget` default change and the normalization-trio move from pybvh
- [Tutorial 3: Heterogeneous preprocessing](../tutorials.md) — mixed skeletons, frame rates, and up-axes as a runnable recipe
