# pybvh-ml

ML bridge layer for [pybvh](https://victors-67.github.io/pybvh/) — turn motion capture data into training-ready inputs for skeleton-based ML models.

pybvh-ml is the layer between pybvh (which parses BVH files and does rotation math) and your model (which consumes tensors). It handles the data plumbing — tensor layout, augmentation, preprocessing, dataset construction — without making assumptions about your model or task. All core functions use NumPy; PyTorch is optional.

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quick Start](getting-started/quickstart.md)** — BVH directory to training batch in five minutes
- :material-magnify: **[Find a function](api/index.md)** — "I want to…" → the exact call → its reference page
- :material-book-open-variant: **[User Guide](guide/tensor-layouts.md)** — layouts, preprocessing, augmentation, PyTorch, skeleton graphs
- :material-swap-horizontal: **[CHANGELOG](https://github.com/VictorS-67/pybvh-ml/blob/main/CHANGELOG.md)** — the migration record: breaking changes and upgrade notes for every release
- :material-school: **[Tutorials](tutorials.md)** — three notebooks from first preprocess to a trained classifier
- :material-download: **[Install](getting-started/installation.md)** — `pip install pybvh-ml`; optional torch and hdf5 extras

</div>

## What can pybvh-ml do?

- **Tensor packing** to `(C, T, V)`, `(T, V, C)`, and flat `(T, D)` layouts with round-trip unpacking — [Tensor Layouts guide](guide/tensor-layouts.md)
- **Array-level augmentation** in quaternion, 6D, axis-angle, rotmat, and Euler — keyword-only, no `Bvh` round-trip, with [composable pipelines](guide/augmentation.md) and reproducible per-epoch seeding
- **Preprocessing pipelines** — BVH directory → on-disk dataset (`.npz` / `.hdf5`) with [skeleton-aware harmonization](guide/preprocessing.md#harmonizing-heterogeneous-datasets) for heterogeneous corpora and dataset-wide [z-score normalization](guide/preprocessing.md#normalization)
- **Skeleton-graph metadata** — [edge lists, body-part partitions, L/R joint pairs](guide/skeleton-metadata.md) for GCN and Transformer models
- **Optional PyTorch integration** — [`MotionDataset` / `OnTheFlyDataset` / `collate_motion_batch`](guide/pytorch.md) with variable-length padding and worker-safe per-epoch augmentation

It replaces the ~150 lines of preprocessing, augmentation, and dataset-class boilerplate that most BVH-based ML pipelines reinvent. Composable enough to use one piece at a time (just the packer, just the augmentor); opinionated enough to give you a working data loader in a dozen lines.

## Quick example

```python
import pybvh_ml

# Preprocess a directory of BVH files into a training-ready .npz.
summary = pybvh_ml.preprocess_directory(
    "walks/", "train.npz", representation="6d",
)

# Load back: per-clip arrays, normalization stats, skeleton metadata.
data = pybvh_ml.load_preprocessed("train.npz")
root_pos = data["clips"][0]["root_pos"]      # (F, 3)
joint_data = data["clips"][0]["joint_data"]  # (F, J, 6) for 6D
mean, std = data["mean"], data["std"]        # for input normalization
```

## Foundation library

pybvh-ml depends on [pybvh](https://victors-67.github.io/pybvh/) for all BVH parsing, rotation math, and spatial transforms — it never reimplements what pybvh provides. Concepts like world-up detection, rotation representations, and forward kinematics are documented there.

## Stability and versioning

**pybvh-ml is in 0.x — expect breaking changes between minor versions.** We treat 0.x as design space: when a past choice turns out to be wrong, we fix it at the root rather than carry scar tissue forward. Each release has a clear migration path in the [CHANGELOG](https://github.com/VictorS-67/pybvh-ml/blob/main/CHANGELOG.md), no deprecation cycles. If you depend on pybvh-ml from production code, **pin to an exact version** (`pybvh-ml==0.5.0`) and read the upgrade notes before bumping.

pybvh-ml will commit to strict semver at **1.0**: no breaking changes within a major version, deprecation warnings (at least one minor release) before any removal. Until then, "make the library better" wins over "preserve the old behavior."
