# pybvh-ml

[![PyPI version](https://img.shields.io/pypi/v/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![Python](https://img.shields.io/pypi/pyversions/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ML bridge layer for [pybvh](https://github.com/VictorS-67/pybvh) — turn motion capture data into training-ready inputs for skeleton-based ML models.

## Features

- **Tensor packing** to `(C,T,V)`, `(T,V,C)`, and flat `(T,D)` layouts with round-trip unpacking
- **Array-level augmentation** in quaternion and 6D space — rotation, mirroring, speed perturbation, dropout, joint noise — all on pre-extracted NumPy arrays, no Bvh objects needed
- **Representation conversion** between euler, quaternion, 6D, axis-angle, and rotation matrices
- **Composable augmentation pipelines** with per-step probabilities, callable kwargs for random parameters, and seeded randomization
- **Preprocessing pipelines** — batch convert BVH directories to on-disk datasets (npz, hdf5) with normalization stats
- **Skeleton graph metadata** — edge lists, body-part partitions, L/R joint pairs for GCN and Transformer models
- **Sequence utilities** — sliding windows, length standardization (pad, crop, resample), and PySKL-style uniform temporal sampling
- **Feature metadata** — column descriptors that map packed array channels to their meaning
- **PyTorch integration** (optional) — `MotionDataset`, `OnTheFlyDataset`, and `collate_motion_batch` for variable-length sequences

## Philosophy

pybvh-ml is the layer between [pybvh](https://github.com/VictorS-67/pybvh) (which parses BVH files and does rotation math) and your model (which consumes tensors). It handles the data plumbing — tensor layout, augmentation, preprocessing, dataset construction — without making assumptions about your model or task. All core functions use NumPy; PyTorch is optional.

## Tutorials

Runnable end-to-end notebooks in [`tutorials/`](tutorials/):

1. **[End-to-end pipeline](tutorials/01_end_to_end_pipeline.ipynb)** — BVH directory →
   `preprocess_directory` → `MotionDataset` with augmentation → tiny MLP classifier, training loop included.
2. **[Augmentation visualized](tutorials/02_augmentation_visualized.ipynb)** — every
   array-level augmentation (`rotate_vertical`, `mirror`, `speed_perturbation_arrays`,
   `dropout_arrays`, `add_joint_noise`) shown before/after on a real skeleton, plus
   pipeline composition and `set_epoch` reproducibility.
3. **[Heterogeneous preprocessing](tutorials/03_heterogeneous_preprocessing.ipynb)** —
   mixing skeletons, frame rates, and up-axes: `pybvh.harmonize` + `skip_errors` +
   `require_matching_topology` as a robust ingest recipe.

Notebooks execute in CI via `pytest --nbmake tutorials/`, so they can't silently rot.

## Installation

```bash
pip install pybvh-ml
```

With optional dependencies:

```bash
pip install "pybvh-ml[torch]"    # PyTorch Dataset classes
pip install "pybvh-ml[hdf5]"     # HDF5 output support
```

## Quick Start

```python
import pybvh
import pybvh_ml

# Load a BVH file and extract rotation data
bvh = pybvh.read_bvh_file("walk.bvh")
root_pos, quats = bvh.to_quaternions()

# Pack into (C, T, V) layout for ST-GCN style models
data = pybvh_ml.pack_to_ctv(root_pos, quats)  # (4, F, J+1)

# Or flat (T, D) for MLP / Transformer
data = pybvh_ml.pack_to_flat(root_pos, quats)  # (F, 3 + J*4)
```

## Augmentation

Array-level augmentation operates directly on NumPy arrays — no Bvh object reconstruction needed.
All augmentation functions take keyword-only arguments, and every representation (`"quaternion"`,
`"6d"`, `"axisangle"`, `"rotmat"`, `"euler"`) is handled by the same unified functions:

```python
import numpy as np
from pybvh_ml import (
    rotate_vertical,
    mirror,
    speed_perturbation_arrays,
    dropout_arrays,
    add_joint_noise,
    get_lr_pairs,
)

rng = np.random.default_rng(42)

# Vertical rotation — up_axis is a signed axis string matching bvh.world_up.
# The sign flips the rotation direction, so '+y' and '-y' yaw oppositely.
root_pos, quats = rotate_vertical(
    root_pos=root_pos, joint_data=quats,
    angle_deg=90, up_axis=bvh.world_up,
    representation="quaternion")

# Left-right mirroring — lateral_axis uses the same signed-string form,
# but mirror is sign-invariant so '+x' and '-x' are equivalent.
lr_pairs = get_lr_pairs(bvh)
root_pos, quats = mirror(
    root_pos=root_pos, joint_data=quats,
    lr_joint_pairs=lr_pairs, lateral_axis="+x",
    representation="quaternion")

# Speed perturbation (SLERP-based interpolation)
root_pos, quats = speed_perturbation_arrays(
    root_pos=root_pos, joint_data=quats,
    factor=1.2, representation="quaternion")

# Frame dropout with SLERP fill
root_pos, quats = dropout_arrays(
    root_pos=root_pos, joint_data=quats,
    drop_rate=0.1, representation="quaternion", rng=rng)

# Joint noise (Gaussian rotation perturbations)
root_pos, quats = add_joint_noise(
    root_pos=root_pos, joint_data=quats,
    sigma_deg=1.0, representation="quaternion", rng=rng)
```

For 6D, pass `representation="6d"`; `rotate_vertical` and `mirror` take fast paths that skip the
quaternion round-trip entirely. Euler arrays also require `euler_orders=bvh.euler_orders`.

## Augmentation Pipeline

Compose augmentations with per-step probabilities for use in data loaders. Kwargs can be callables
for per-sample random parameters:

```python
import numpy as np
from pybvh_ml import AugmentationPipeline
from pybvh_ml.augmentation import rotate_vertical, mirror, add_joint_noise

pipeline = AugmentationPipeline([
    (rotate_vertical, 1.0, {
        "angle_deg": lambda rng: rng.uniform(-180, 180),  # random each sample
        "up_axis": bvh.world_up,
        "representation": "quaternion",
    }),
    (mirror, 0.5, {
        "lr_joint_pairs": lr_pairs,
        "lateral_axis": "+x",
        "representation": "quaternion",
    }),
    (add_joint_noise, 1.0, {
        "sigma_deg": 1.0,
        "representation": "quaternion",
    }),
])

rng = np.random.default_rng(42)
root_pos, quats = pipeline(root_pos=root_pos, joint_data=quats, rng=rng)
```

For the common case, skip the boilerplate and use the `standard` factory — it wires rotate +
mirror + noise + speed from a `skeleton_info` dict:

```python
from pybvh_ml import AugmentationPipeline, get_skeleton_info

pipeline = AugmentationPipeline.standard(
    get_skeleton_info(bvh),
    representation="quaternion",
    up_axis=bvh.world_up,
    # rotate_angle_range=(-180, 180), mirror_prob=0.5, noise_sigma_deg=1.0,
    # speed_factor_range=(0.8, 1.2)  — defaults shown; pass None to disable a step
)
```

## Representation Conversion

Convert between any pair of rotation representations on `(F, J, C)` arrays:

```python
from pybvh_ml import convert_arrays

# Euler to 6D (respects per-joint Euler orders)
rot6d = convert_arrays(euler_data, from_repr="euler", to_repr="6d",
                       euler_orders=bvh.euler_orders)

# Quaternion to rotation matrix
rotmat = convert_arrays(quats, from_repr="quaternion", to_repr="rotmat")
```

Supported: `"euler"`, `"quaternion"`, `"6d"`, `"axisangle"`, `"rotmat"`.

## Preprocessing

Batch convert a BVH directory to an on-disk dataset in one call:

```python
from pybvh_ml import preprocess_directory, load_preprocessed

# Convert to npz with 6D representation
summary = preprocess_directory(
    "dataset/",
    "train.npz",
    representation="6d",
    parallel=True,                   # threaded loading for large directories
    skip_errors=True,                # skip + warn on malformed files
    include_velocities=True,         # static per-joint velocities (F, N, 3)
    include_foot_contacts=True,      # static binary foot contacts (F, num_feet)
)

# Or HDF5 (requires h5py)
preprocess_directory("dataset/", "train.hdf5", representation="quaternion")

# Load back — returns a dict
data = load_preprocessed("train.npz")
clips = data["clips"]                # list of per-clip dicts
mean, std = data["mean"], data["std"]
skel = data["skeleton_info"]         # includes edges, lr_pairs, lr_mapping
constant_channels = data.get("constant_channels")  # bool mask (0.3+)
```

The output file stores arrays, skeleton metadata, and normalization statistics together.
`constant_channels` marks columns whose raw std was below `1e-8` (guarded to 1.0 for
normalization); exclude them from per-channel diagnostics.

### Harmonizing heterogeneous datasets

When clips come from different skeletons, frame rates, or up-axis conventions, preprocess
with `require_matching_topology=True` (the default) will reject the batch. Pre-harmonize
with `pybvh.harmonize`:

```python
from pybvh import read_bvh_directory, harmonize, write_bvh_file
from pathlib import Path

clips = read_bvh_directory("raw/", parallel=True, skip_errors=True)
reference = clips[0]

harmonized = harmonize(
    clips,
    reference=reference,             # retarget to this skeleton
    target_fps=30,                   # SLERP resample
    target_world_up="+y",            # reorient animation up
    target_rest_forward="+z",        # (optional) unify rest-pose facing
    target_rest_up="+y",             # (optional) unify rest-pose up
    on_incompatible="drop",          # skip clips whose topology doesn't match
)

# Write the harmonized clips to disk and preprocess normally
out_dir = Path("harmonized/")
out_dir.mkdir(exist_ok=True)
for b, src in zip(harmonized, clips):
    write_bvh_file(b, out_dir / Path(src.filepath).name)  # or your own naming
preprocess_directory(out_dir, "train.npz", representation="6d")
```

## Skeleton Graph Metadata

Extract the topology data that GCN and Transformer models need:

```python
import pybvh_ml

edges = pybvh_ml.get_edge_list(bvh)           # [(child, parent), ...]
lr_pairs = pybvh_ml.get_lr_pairs(bvh)         # [(left, right), ...]
partitions = pybvh_ml.get_body_partitions(bvh) # {"torso": [0,1,...], "left_arm": [...], ...}

# All-in-one
info = pybvh_ml.get_skeleton_info(bvh)
# {"edges", "lr_pairs", "body_partitions", "joint_names", "euler_orders"}
```

## Sequence Utilities

```python
from pybvh_ml import sliding_window, standardize_length

# Fixed-length windows for training
windows = sliding_window(data, window_size=64, stride=32)  # (num_windows, 64, ...)

# Standardize to target length
padded = standardize_length(data, target_length=128, method="pad")
cropped = standardize_length(data, target_length=64, method="crop")
```

## Temporal Sampling

PySKL-style uniform segment sampling for skeleton-based recognition:

```python
from pybvh_ml import uniform_temporal_sample, sample_temporal

# Get frame indices (handles short, dense, and long sequences)
indices = uniform_temporal_sample(num_frames=200, clip_length=64, mode="train", rng=rng)
clip = data[indices]  # (64, ...)

# Or apply directly to an array with wraparound for short sequences
clip = sample_temporal(data, clip_length=64, mode="train", rng=rng)  # (64, ...)

# Multiple independent samples
clips = sample_temporal(data, clip_length=64, num_samples=5, mode="train", rng=rng)
# (5, 64, ...)
```

## PyTorch Integration

Optional — install with `pip install "pybvh-ml[torch]"`:

```python
from pybvh_ml.torch import MotionDataset, OnTheFlyDataset, collate_motion_batch
from torch.utils.data import DataLoader

# From preprocessed data
data = load_preprocessed("train.npz")
dataset = MotionDataset(
    data["clips"], labels=data["labels"],
    target_length=128, augmentation=pipeline,
    seed=42,  # reproducible; see set_epoch note below
)

# From raw BVH files (converts on-the-fly)
dataset = OnTheFlyDataset(bvh_paths, representation="6d", augmentation=pipeline, seed=42)

# Variable-length batching with padding and masks
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_motion_batch)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)    # fresh aug per epoch, reproducible across runs
    for batch in loader:
        data = batch["data"]       # (B, T_max, D)
        mask = batch["mask"]       # (B, T_max) bool
        lengths = batch["lengths"] # (B,)
```

**Reproducible per-epoch augmentation.** When `seed` is set, `(seed, epoch, idx)` feeds
a `numpy.random.SeedSequence`, so two runs with the same seed produce the same
augmentation trajectory while each epoch still sees a different draw. Call
`dataset.set_epoch(epoch)` at the top of each epoch — same contract as
`torch.utils.data.distributed.DistributedSampler`. With `seed=None`, every call uses
fresh OS entropy (simplest; no reproducibility).

## Feature Metadata

Know what each column in a packed array represents:

```python
from pybvh_ml import describe_features

desc = describe_features(num_joints=24, representation="6d", include_root_pos=True)
desc["root_pos"]           # (0, 3)
desc["joint_rotations"]    # (3, 147)
desc.slice("joint_rotations")  # slice(3, 147)
```

For the richer layout that covers velocities and foot contacts, use pybvh's
`Bvh.feature_array_layout(...)` alongside `Bvh.to_feature_array(...)`.

## Running tests

Tests run against the small fixtures under [bvh_data/](bvh_data/) and need no extra setup:

```bash
pytest tests/test_pybvh_ml.py
```

## Requirements

- Python >= 3.9
- [pybvh](https://github.com/VictorS-67/pybvh) >= 0.6.0
- NumPy >= 1.21

Optional: PyTorch >= 2.0 (`pip install "pybvh-ml[torch]"`), h5py >= 3.0 (`pip install "pybvh-ml[hdf5]"`).

## License

MIT
