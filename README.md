# pybvh-ml

[![PyPI version](https://img.shields.io/pypi/v/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![Python](https://img.shields.io/pypi/pyversions/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![Tests](https://github.com/VictorS-67/pybvh-ml/actions/workflows/test.yml/badge.svg)](https://github.com/VictorS-67/pybvh-ml/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ML bridge layer for [pybvh](https://github.com/VictorS-67/pybvh) — turn motion capture data into training-ready inputs for skeleton-based ML models.

> **Status:** pre-1.0. Minor versions can include breaking API changes; see [CHANGELOG.md](CHANGELOG.md) for migration notes.

## Features

- **Tensor packing** to `(C, T, V)`, `(T, V, C)`, and flat `(T, D)` layouts with round-trip unpacking.
- **Array-level augmentation** in quaternion, 6D, axis-angle, rotmat, and Euler — keyword-only, no `Bvh` round-trip, with composable pipelines and reproducible per-epoch seeding.
- **Preprocessing pipelines** — BVH directory → on-disk dataset (`.npz` / `.hdf5`) with skeleton-aware harmonization for heterogeneous corpora and dataset-wide z-score normalization (`compute_normalization_stats` / `normalize_array` / `denormalize_array`).
- **Skeleton-graph metadata** — edge lists, body-part partitions, L/R joint pairs for GCN and Transformer models.
- **Optional PyTorch integration** — `MotionDataset` / `OnTheFlyDataset` / `collate_motion_batch` with variable-length padding.

## Philosophy

pybvh-ml is the layer between [pybvh](https://github.com/VictorS-67/pybvh) (which parses BVH files and does rotation math) and your model (which consumes tensors). It handles the data plumbing — tensor layout, augmentation, preprocessing, dataset construction — without making assumptions about your model or task. All core functions use NumPy; PyTorch is optional.

It replaces the ~150 lines of preprocessing, augmentation, and dataset-class boilerplate that most BVH-based ML pipelines reinvent. Composable enough to use one piece at a time (just the packer, just the augmentor); opinionated enough to give you a working data loader in a dozen lines.

## Stability and versioning

**pybvh-ml is in 0.x — expect breaking changes between minor versions.**

We treat 0.x as design space: when a past choice turns out to be wrong, we fix it at the root rather than carry scar tissue forward. No deprecation cycles, no compatibility shims; each release ships a single clean migration path, documented in the [CHANGELOG](CHANGELOG.md). If you depend on pybvh-ml from production code, **pin to an exact version** (`pybvh-ml==0.5.0`) and read the upgrade notes before bumping.

This will change at **1.0**: from then on, pybvh-ml will commit to strict semver — no breaking changes within a major version, deprecation warnings (at least one minor release) before any future removal. Until 1.0, "make the library better" wins over "preserve the old behavior."

## Installation

```bash
pip install pybvh-ml
```

This pulls in [pybvh](https://github.com/VictorS-67/pybvh) `>= 0.8, < 0.9` automatically — pybvh-ml 0.5 tracks pybvh 0.8's API (short representation tokens, radians-first parameters).

With optional dependencies:

```bash
pip install "pybvh-ml[torch]"    # PyTorch Dataset classes
pip install "pybvh-ml[hdf5]"     # HDF5 output support
```

## Quick Start

```python
import pybvh_ml

# Preprocess a directory of BVH files into a training-ready .npz.
summary = pybvh_ml.preprocess_directory(
    "walks/", "train.npz", representation="6d",
)
print(f"{summary['num_clips']} clips, "
      f"{summary['skeleton_info']['num_joints']} joints")

# Load back: per-clip arrays, normalization stats, skeleton metadata.
data = pybvh_ml.load_preprocessed("train.npz")
root_pos = data["clips"][0]["root_pos"]      # (F, 3)
joint_data = data["clips"][0]["joint_data"]  # (F, J, 6) for 6D
mean, std = data["mean"], data["std"]        # for input normalization
```

That's the headline workflow. The sections below cover the pieces in more depth, from the preprocessing entry point through augmentation and PyTorch integration.

## Tutorials

Runnable end-to-end notebooks in [`tutorials/`](tutorials/):

1. **[End-to-end pipeline](tutorials/01_end_to_end_pipeline.ipynb)** — BVH directory → `preprocess_directory` → `MotionDataset` with augmentation → tiny MLP classifier, training loop included.
2. **[Augmentation visualized](tutorials/02_augmentation_visualized.ipynb)** — every array-level augmentation (`rotate_vertical`, `mirror`, `speed_perturbation_arrays`, `dropout_arrays`, `add_joint_noise`) shown before/after on a real skeleton, plus pipeline composition and `set_epoch` reproducibility.
3. **[Heterogeneous preprocessing](tutorials/03_heterogeneous_preprocessing.ipynb)** — mixing skeletons, frame rates, and up-axes: `harmonize=True` + `skip_errors` + the rep-aware compatibility check as a robust ingest recipe.

Notebooks execute in CI via `pytest --nbmake tutorials/`, so they can't silently rot.

## Preprocessing

Batch convert a BVH directory to an on-disk dataset in one call:

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
skel = data["skeleton_info"]         # edges, lr_pairs, lr_mapping, joint_names
```

Output formats: `.npz` (always available) and `.hdf5` (requires `h5py`). The file stores per-clip arrays, skeleton metadata, normalization statistics, and a `constant_channels` bool mask (columns whose raw std was below `1e-8`, guarded to 1.0 for normalization).

Power-user kwargs for richer outputs:

- `include_velocities=True` — store per-joint linear velocities `(F, J, 3)` alongside joint data.
- `include_foot_contacts=True` — store binary foot-contact labels and the detected foot joint names.
- `include_quaternions=True` — store pre-computed quaternion arrays for runtime augmentation that needs them.
- `label_fn(stem) -> int` — attach per-clip integer labels.
- `filter_fn(stem) -> bool` — filter files before loading (skipped files are never parsed).

### Normalization

Per-channel z-score normalization following the `Mean.npy` / `Std.npy` convention used by HumanML3D and MDM:

```python
from pybvh import read_bvh_directory
from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array

bvhs = read_bvh_directory("dataset/")
stats = compute_normalization_stats(bvhs, representation="6d")  # {"mean", "std", "constant_channels"}
x_norm = normalize_array(x, stats)     # (x - mean) / std
x = denormalize_array(x_norm, stats)   # back to BVH units
```

`compute_normalization_stats` takes a list of `Bvh` objects and computes stats over all frames in the flat `[root_pos, joint_data]` channel layout (`include_root_pos=False` drops the first 3 columns); zero-variance channels get their std guarded to `1.0` and flagged in the `constant_channels` mask. `preprocess_directory` stores the same stats in its output file, so after `load_preprocessed` you can pass the loaded dict straight to `normalize_array` — the direct entry point is for workflows that skip the on-disk artifact.

### Harmonizing heterogeneous datasets

When clips come from different skeletons, frame rates, up-axis conventions, or — for order-sensitive representations like `"euler"` / `"axisangle"` — different per-joint Euler orders, pass `harmonize=True`:

```python
from pybvh_ml import preprocess_directory

preprocess_directory(
    "raw/", "train.npz",
    representation="euler",
    harmonize=True,                  # runs pybvh.harmonize after loading
    target_world_up="+y",            # (optional) explicit target; majority otherwise
    skip_errors=True,
)
```

`harmonize=True` runs `pybvh.harmonize`, using majority values from the uniformity audit for any `target_*` you didn't set explicitly. For order-sensitive representations it also auto-picks a `target_euler_order` (most common per-joint order across the dataset). Harmonization is pure reorientation/resampling — each actor's bone lengths are preserved; pass `retarget=True` to additionally retarget every clip's bone offsets to the first clip when the whole dataset should share one skeleton geometry (e.g. for fixed-topology GCNs). Hierarchy mismatches raise loudly either way — no silent drops. The `uniformity` audit (including `harmonized_to` with the resolved targets, the `retarget` choice, per-stage modification counts, and a JSON-serializable `HarmonizeReport`) is returned *and* persisted in the saved dataset, so the transformation trail is auditable via `load_preprocessed(...)["uniformity"]`.

For workflows that need to inspect or persist intermediates, call `pybvh.harmonize` directly:

```python
from pybvh import read_bvh_directory, harmonize, write_bvh_file
from pybvh_ml import preprocess_directory
from pathlib import Path

clips = read_bvh_directory("raw/", parallel=True, skip_errors=True)
harmonized = harmonize(
    clips,
    reference=clips[0],
    target_fps=30,
    target_world_up="+y",
)

out_dir = Path("harmonized/")
out_dir.mkdir(exist_ok=True)
for b, src in zip(harmonized, clips):
    write_bvh_file(b, out_dir / Path(src.source_path).name)
preprocess_directory(out_dir, "train.npz", representation="6d")
```

## Augmentation

Array-level augmentation operates directly on NumPy arrays — no Bvh object reconstruction needed. All augmentation functions take keyword-only arguments, angles are in radians (matching pybvh), and every representation (`"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"`) is handled by the same unified functions:

```python
import numpy as np
from pybvh_ml import (
    rotate_vertical, mirror,
    speed_perturbation_arrays, dropout_arrays, add_joint_noise,
    get_lr_pairs,
)

rng = np.random.default_rng(42)

# Vertical rotation — up_axis is a signed axis string matching bvh.world_up.
# The sign flips the rotation direction, so '+y' and '-y' yaw oppositely.
root_pos, quats = rotate_vertical(
    root_pos=root_pos, joint_data=quats,
    angle=np.pi / 2, up_axis=bvh.world_up,
    representation="quat")

# Left-right mirroring — lateral_axis uses the same signed-string form
# but is sign-invariant ('+x' and '-x' are equivalent).
lr_pairs = get_lr_pairs(bvh)
root_pos, quats = mirror(
    root_pos=root_pos, joint_data=quats,
    lr_joint_pairs=lr_pairs, lateral_axis="+x",
    representation="quat")

# Speed perturbation (SLERP interpolation), frame dropout, joint noise.
root_pos, quats = speed_perturbation_arrays(
    root_pos=root_pos, joint_data=quats,
    factor=1.2, representation="quat")
root_pos, quats = dropout_arrays(
    root_pos=root_pos, joint_data=quats,
    drop_rate=0.1, representation="quat", rng=rng)
root_pos, quats = add_joint_noise(
    root_pos=root_pos, joint_data=quats,
    sigma=np.radians(1.0), representation="quat", rng=rng)
```

Euler arrays additionally require `euler_orders=bvh.euler_orders`.

### Augmentation Pipeline

Compose augmentations with per-step probabilities for use in data loaders. Kwargs can be callables for per-sample random parameters:

```python
import numpy as np
from pybvh_ml import AugmentationPipeline
from pybvh_ml.augmentation import rotate_vertical, mirror, add_joint_noise

pipeline = AugmentationPipeline([
    (rotate_vertical, 1.0, {
        "angle": lambda rng: rng.uniform(-np.pi, np.pi),  # random each sample
        "up_axis": bvh.world_up,
        "representation": "quat",
    }),
    (mirror, 0.5, {
        "lr_joint_pairs": lr_pairs,
        "lateral_axis": "+x",
        "representation": "quat",
    }),
    (add_joint_noise, 1.0, {
        "sigma": np.radians(1.0),
        "representation": "quat",
    }),
])

rng = np.random.default_rng(42)
root_pos, quats = pipeline(root_pos=root_pos, joint_data=quats, rng=rng)
```

For the common case, skip the boilerplate and use the `standard` factory — it wires rotate + mirror + noise + speed from a `skeleton_info` dict:

```python
from pybvh_ml import AugmentationPipeline, get_skeleton_info

pipeline = AugmentationPipeline.standard(
    get_skeleton_info(bvh),
    representation="quat",
    up_axis=bvh.world_up,
    # rotate_angle_range=(-np.pi, np.pi), mirror_prob=0.5, noise_sigma=np.radians(1.0),
    # speed_factor_range=(0.8, 1.2)  — defaults shown; pass None to disable a step
)
```

## PyTorch Integration

Optional — install with `pip install "pybvh-ml[torch]"`. End-to-end: preprocess → `MotionDataset` → `DataLoader` → training batch.

```python
from pybvh_ml import preprocess_directory, load_preprocessed, AugmentationPipeline, get_skeleton_info
from pybvh_ml.torch import MotionDataset, collate_motion_batch
from torch.utils.data import DataLoader
import pybvh

# 1. One-time preprocess (label_fn: map filename stem → integer class).
preprocess_directory("walks/", "train.npz", representation="6d",
                     label_fn=lambda stem: 0)

# 2. Build dataset + augmentation pipeline.
data = load_preprocessed("train.npz")
skel = data["skeleton_info"]
pipeline = AugmentationPipeline.standard(
    skel, representation="6d", up_axis="+y")

dataset = MotionDataset(
    data["clips"], labels=data["labels"],
    target_length=128, augmentation=pipeline,
    seed=42,  # reproducible — see set_epoch note below
)

# 3. Variable-length batching with padding and masks.
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_motion_batch)

# 4. Training loop.
for epoch in range(num_epochs):
    dataset.set_epoch(epoch)    # fresh aug per epoch, reproducible across runs
    for batch in loader:
        data_tensor = batch["data"]       # (B, T_max, D)
        mask = batch["mask"]              # (B, T_max) bool
        lengths = batch["lengths"]        # (B,)
        labels = batch["labels"]
        # model(data_tensor, mask) ...
```

**Reproducible per-epoch augmentation.** When `seed` is set on `MotionDataset` / `OnTheFlyDataset`, the tuple `(seed, epoch, idx)` feeds a `numpy.random.SeedSequence`, so two runs with the same seed produce the same augmentation trajectory while each epoch still sees a different draw. Call `dataset.set_epoch(epoch)` at the top of each epoch — same contract as `torch.utils.data.distributed.DistributedSampler`. With `seed=None`, every call uses fresh OS entropy (simplest; no reproducibility).

`OnTheFlyDataset` skips the preprocessed file and loads BVH paths directly, converting on the fly — useful when you want augmentation but don't want the preprocessing artifact on disk.

## Skeleton Graph Metadata

Extract the topology data that GCN and Transformer models need:

```python
import pybvh_ml

edges = pybvh_ml.get_edge_list(bvh)           # [(child, parent), ...]
lr_pairs = pybvh_ml.get_lr_pairs(bvh)         # [(left, right), ...]
partitions = pybvh_ml.get_body_partitions(bvh) # {"torso": [...], "left_arm": [...], ...}

# All-in-one (include_partitions=True adds "body_partitions")
info = pybvh_ml.get_skeleton_info(bvh)
# {"edges", "lr_pairs", "lr_mapping",
#  "joint_names", "euler_orders", "num_joints"}
```

## Additional utilities

### Representation conversion

Convert between any pair of rotation representations on `(F, J, C)` arrays:

```python
from pybvh_ml import convert_arrays

# Euler to 6D (respects per-joint Euler orders)
rot6d = convert_arrays(euler_data, from_repr="euler", to_repr="6d",
                       euler_orders=bvh.euler_orders)

# Quaternion to rotation matrix
rotmat = convert_arrays(quats, from_repr="quat", to_repr="rotmat")
```

Supported: `"euler"`, `"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`.

### Sequence utilities

```python
from pybvh_ml import sliding_window, standardize_length

windows = sliding_window(data, window_size=64, stride=32)  # (num_windows, 64, ...)
padded = standardize_length(data, target_length=128, method="pad")
cropped = standardize_length(data, target_length=64, method="crop")
```

### Temporal sampling

PySKL-style uniform segment sampling for skeleton-based recognition:

```python
from pybvh_ml import uniform_temporal_sample, sample_temporal

indices = uniform_temporal_sample(num_frames=200, clip_length=64, mode="train", rng=rng)
clip = data[indices]                                              # (64, ...)
clip = sample_temporal(data, clip_length=64, mode="train", rng=rng)  # convenience wrapper
clips = sample_temporal(data, clip_length=64, num_samples=5, mode="train", rng=rng)
```

### Feature metadata

Know what each column in a packed array represents:

```python
from pybvh_ml import describe_features

desc = describe_features(num_joints=24, representation="6d", include_root_pos=True)
desc["root_pos"]                  # (0, 3)
desc["joint_rotations"]           # (3, 147)
desc.slice("joint_rotations")     # slice(3, 147)
```

For the richer layout that covers velocities and foot contacts, use pybvh's `Bvh.feature_array_layout(...)` alongside `Bvh.to_feature_array(...)`.

## Requirements

- Python >= 3.9
- [pybvh](https://github.com/VictorS-67/pybvh) >= 0.8, < 0.9
- NumPy >= 1.21

Optional: PyTorch >= 2.0 (`pip install "pybvh-ml[torch]"`), h5py >= 3.0 (`pip install "pybvh-ml[hdf5]"`).

## Development

```bash
pip install "pybvh-ml[dev]"
pytest tests/test_pybvh_ml.py             # unit tests
pytest --nbmake tutorials/                # tutorial notebook execution
```

## License

MIT
