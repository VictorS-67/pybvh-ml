# pybvh-ml

[![PyPI version](https://img.shields.io/pypi/v/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![Python](https://img.shields.io/pypi/pyversions/pybvh-ml)](https://pypi.org/project/pybvh-ml/)
[![Tests](https://github.com/VictorS-67/pybvh-ml/actions/workflows/test.yml/badge.svg)](https://github.com/VictorS-67/pybvh-ml/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ML bridge layer for [pybvh](https://github.com/VictorS-67/pybvh) — turn motion capture data into training-ready inputs for skeleton-based ML models.

> **Status:** pre-1.0. Minor versions can include breaking API changes; see [CHANGELOG.md](CHANGELOG.md) for migration notes.

**[Documentation](https://victors-67.github.io/pybvh-ml/)** · [Quick Start](https://victors-67.github.io/pybvh-ml/getting-started/quickstart/) · [Find a function](https://victors-67.github.io/pybvh-ml/api/) · [User Guide](https://victors-67.github.io/pybvh-ml/guide/tensor-layouts/) · [Tutorials](https://victors-67.github.io/pybvh-ml/tutorials/)

## Features

- **Tensor packing** to `(C, T, V)`, `(T, V, C)`, and flat `(T, D)` layouts with round-trip unpacking.
- **Array-level augmentation** in quaternion, 6D, axis-angle, rotmat, and Euler — keyword-only, no `Bvh` round-trip, with composable pipelines and reproducible per-epoch seeding.
- **Preprocessing pipelines** — BVH directory → on-disk dataset (`.npz` / `.hdf5`) with skeleton-aware harmonization for heterogeneous corpora and dataset-wide z-score normalization.
- **Skeleton-graph metadata** — edge lists, body-part partitions, L/R joint pairs for GCN and Transformer models.
- **Optional PyTorch integration** — `MotionDataset` / `OnTheFlyDataset` / `collate_motion_batch` with variable-length padding.

## Philosophy

pybvh-ml is the layer between [pybvh](https://github.com/VictorS-67/pybvh) (which parses BVH files and does rotation math) and your model (which consumes tensors). It handles the data plumbing — tensor layout, augmentation, preprocessing, dataset construction — without making assumptions about your model or task. All core functions use NumPy; PyTorch is optional.

It replaces the ~150 lines of preprocessing, augmentation, and dataset-class boilerplate that most BVH-based ML pipelines reinvent. Composable enough to use one piece at a time (just the packer, just the augmentor); opinionated enough to give you a working data loader in a dozen lines.

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

From there: runtime [augmentation](https://victors-67.github.io/pybvh-ml/guide/augmentation/) in your data loader, [PyTorch Datasets](https://victors-67.github.io/pybvh-ml/guide/pytorch/) with worker-safe per-epoch seeding, and [skeleton graph metadata](https://victors-67.github.io/pybvh-ml/guide/skeleton-metadata/) for GCNs — the [documentation](https://victors-67.github.io/pybvh-ml/) covers each piece.

## Tutorials

Runnable end-to-end notebooks in [`tutorials/`](tutorials/):

1. **[End-to-end pipeline](tutorials/01_end_to_end_pipeline.ipynb)** — BVH directory → preprocess → `MotionDataset` with augmentation → tiny MLP classifier.
2. **[Augmentation visualized](tutorials/02_augmentation_visualized.ipynb)** — every augmentation before/after on a real skeleton, plus `set_epoch` reproducibility.
3. **[Heterogeneous preprocessing](tutorials/03_heterogeneous_preprocessing.ipynb)** — mixing skeletons, frame rates, and up-axes robustly.

Notebooks execute in CI via `pytest --nbmake tutorials/`, so they can't silently rot.

## Stability and versioning

**pybvh-ml is in 0.x — expect breaking changes between minor versions.**

We treat 0.x as design space: when a past choice turns out to be wrong, we fix it at the root rather than carry scar tissue forward. No deprecation cycles, no compatibility shims; each release ships a single clean migration path, documented in the [CHANGELOG](CHANGELOG.md). If you depend on pybvh-ml from production code, **pin to an exact version** (`pybvh-ml==0.5.0`) and read the upgrade notes before bumping.

This will change at **1.0**: from then on, pybvh-ml will commit to strict semver — no breaking changes within a major version, deprecation warnings (at least one minor release) before any future removal. Until 1.0, "make the library better" wins over "preserve the old behavior."

## Requirements

- Python >= 3.9
- [pybvh](https://github.com/VictorS-67/pybvh) >= 0.8, < 0.9
- NumPy >= 1.21

Optional: PyTorch >= 2.0 (`pip install "pybvh-ml[torch]"`), h5py >= 3.0 (`pip install "pybvh-ml[hdf5]"`).

## Development

```bash
pip install "pybvh-ml[dev]"
pytest tests/                             # unit tests
pytest --nbmake tutorials/                # tutorial notebook execution
mkdocs serve                              # docs preview (pip install "pybvh-ml[docs]")
```

## License

MIT
