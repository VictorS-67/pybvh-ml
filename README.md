# pybvh-ml

[![PyPI version](https://img.shields.io/pypi/v/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![Python](https://img.shields.io/pypi/pyversions/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ML bridge layer for [pybvh](https://github.com/VictorS-67/pybvh) — turn motion capture data into training-ready inputs for skeleton-based ML models.

## Features

- **Tensor packing** to `(C,T,V)`, `(T,V,C)`, and flat `(T,D)` layouts with round-trip unpacking
- **Array-level augmentation** in quaternion and 6D space — rotation, mirroring, speed perturbation, dropout — all on pre-extracted NumPy arrays, no Bvh objects needed
- **Representation conversion** between euler, quaternion, 6D, axis-angle, and rotation matrices
- **Composable augmentation pipelines** with per-step probabilities and seeded randomization
- **Preprocessing pipelines** — batch convert BVH directories to on-disk datasets (npz, hdf5) with normalization stats
- **Skeleton graph metadata** — edge lists, body-part partitions, L/R joint pairs for GCN and Transformer models
- **Sequence utilities** — sliding windows and length standardization (pad, crop, resample)
- **Feature metadata** — column descriptors that map packed array channels to their meaning
- **PyTorch integration** (optional) — `MotionDataset`, `OnTheFlyDataset`, and `collate_motion_batch` for variable-length sequences

## Philosophy

pybvh-ml is the layer between [pybvh](https://github.com/VictorS-67/pybvh) (which parses BVH files and does rotation math) and your model (which consumes tensors). It handles the data plumbing — tensor layout, augmentation, preprocessing, dataset construction — without making assumptions about your model or task. All core functions use NumPy; PyTorch is optional.

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
root_pos, quats, joints = bvh.get_frames_as_quaternion()

# Pack into (C, T, V) layout for ST-GCN style models
data = pybvh_ml.pack_to_ctv(root_pos, quats)  # (4, F, J+1)

# Or flat (T, D) for MLP / Transformer
data = pybvh_ml.pack_to_flat(root_pos, quats)  # (F, 3 + J*4)
```

## Augmentation

Array-level augmentation operates directly on NumPy arrays — no Bvh object reconstruction needed:

```python
from pybvh_ml import (
    rotate_quaternions_vertical,
    mirror_quaternions,
    speed_perturbation_arrays,
    dropout_arrays,
)

# Vertical rotation (e.g., Y-up skeleton)
quats, root_pos = rotate_quaternions_vertical(quats, root_pos, angle_deg=90, up_idx=1)

# Left-right mirroring
lr_pairs = pybvh_ml.get_lr_pairs(bvh)
quats, root_pos = mirror_quaternions(quats, root_pos, lr_joint_pairs=lr_pairs, lateral_idx=0)

# Speed perturbation (SLERP-based interpolation)
quats, root_pos = speed_perturbation_arrays(quats, root_pos, factor=1.2)

# Frame dropout with SLERP fill
quats, root_pos = dropout_arrays(quats, root_pos, drop_rate=0.1, rng=rng)
```

6D augmentation avoids the quaternion round-trip in hot data loader paths:

```python
from pybvh_ml import rotate_rot6d_vertical, mirror_rot6d

root_pos, rot6d, joints = bvh.get_frames_as_6d()
rot6d, root_pos = rotate_rot6d_vertical(rot6d, root_pos, angle_deg=45, up_idx=1)
rot6d, root_pos = mirror_rot6d(rot6d, root_pos, lr_joint_pairs=lr_pairs, lateral_idx=0)
```

## Augmentation Pipeline

Compose augmentations with per-step probabilities for use in data loaders:

```python
import numpy as np
from pybvh_ml import AugmentationPipeline
from pybvh_ml.augmentation import rotate_quaternions_vertical, mirror_quaternions

pipeline = AugmentationPipeline([
    (rotate_quaternions_vertical, 0.5, {"angle_deg": 90, "up_idx": 1}),
    (mirror_quaternions, 0.5, {"lr_joint_pairs": lr_pairs, "lateral_idx": 0}),
])

rng = np.random.default_rng(42)
quats, root_pos = pipeline(quats, root_pos, rng=rng)
```

## Representation Conversion

Convert between any pair of rotation representations on `(F, J, C)` arrays:

```python
from pybvh_ml import convert_arrays

# Euler to 6D (respects per-joint Euler orders)
rot6d = convert_arrays(euler_data, "euler", "6d", euler_orders=bvh.euler_orders)

# Quaternion to rotation matrix
rotmat = convert_arrays(quats, "quaternion", "rotmat")
```

Supported: `"euler"`, `"quaternion"`, `"6d"`, `"axisangle"`, `"rotmat"`.

## Preprocessing

Batch convert a BVH directory to an on-disk dataset in one call:

```python
from pybvh_ml import preprocess_directory, load_preprocessed

# Convert to npz with 6D representation
stats = preprocess_directory(
    "dataset/",
    "train.npz",
    representation="6d",
)

# Or HDF5 (requires h5py)
stats = preprocess_directory("dataset/", "train.hdf5", representation="quaternion")

# Load back
clips, metadata = load_preprocessed("train.npz")
```

The output file stores arrays, skeleton metadata, and normalization statistics together.

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

## PyTorch Integration

Optional — install with `pip install "pybvh-ml[torch]"`:

```python
from pybvh_ml.torch import MotionDataset, OnTheFlyDataset, collate_motion_batch
from torch.utils.data import DataLoader

# From preprocessed data
clips, metadata = load_preprocessed("train.npz")
dataset = MotionDataset(clips, target_length=128, augmentation=pipeline)

# From raw BVH files (converts on-the-fly)
dataset = OnTheFlyDataset(bvh_paths, representation="6d", augmentation=pipeline)

# Variable-length batching with padding and masks
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_motion_batch)
for batch in loader:
    data = batch["data"]       # (B, T_max, D)
    mask = batch["mask"]       # (B, T_max) bool
    lengths = batch["lengths"] # (B,)
```

## Feature Metadata

Know what each column in a packed array represents:

```python
from pybvh_ml import describe_features

desc = describe_features("6d", include_root_pos=True)
# desc.root_pos_slice, desc.joint_data_slice, desc.channels_per_joint, ...
```

## Requirements

- Python >= 3.9
- [pybvh](https://github.com/VictorS-67/pybvh) >= 0.4.0
- NumPy >= 1.21

Optional: PyTorch >= 2.0 (`pip install "pybvh-ml[torch]"`), h5py >= 3.0 (`pip install "pybvh-ml[hdf5]"`).

## License

MIT
