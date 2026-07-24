# Quick Start

The headline workflow: a directory of BVH files in, a training-ready PyTorch data loader out. Every piece also works standalone — use just the packer, just the augmentor, or just the preprocessor.

## Preprocess once

```python
import pybvh_ml

summary = pybvh_ml.preprocess_directory(
    "walks/", "train.npz",
    representation="6d",
    label_fn=lambda stem: 0,   # map filename stem → integer class
)
print(f"{summary['num_clips']} clips, "
      f"{summary['skeleton_info']['num_joints']} joints")
```

One call loads every `.bvh` file, converts rotations to your representation of choice (`"euler"`, `"quat"`, `"6d"`, `"axisangle"`), computes dataset-wide normalization statistics, and writes everything to one `.npz` (or `.hdf5`) file. Mixed skeletons, frame rates, or up-axes? See [harmonization](../guide/preprocessing.md#harmonizing-heterogeneous-datasets).

## Load back

```python
data = pybvh_ml.load_preprocessed("train.npz")

clip = data["clips"][0]
clip["root_pos"]        # (F, 3) root translation
clip["joint_data"]      # (F, J, 6) for 6D
mean, std = data["mean"], data["std"]   # per-channel z-score stats
skel = data["skeleton_info"]            # edges, lr_pairs, world_up, ...
```

The file is self-sufficient: it carries the skeleton metadata (including the axis strings augmentation needs), so you never reopen the source BVHs at training time.

## Augment at runtime

```python
import numpy as np
from pybvh_ml import AugmentationPipeline

pipeline = AugmentationPipeline.standard(
    skel, representation="6d", up_axis=skel["world_up"],
)
rng = np.random.default_rng(42)
root_pos, joint_data = pipeline(
    root_pos=clip["root_pos"], joint_data=clip["joint_data"], rng=rng,
)
```

The `standard` factory wires vertical rotation + mirroring + joint noise + speed perturbation from the skeleton metadata. Each step is also a standalone function — the [Augmentation guide](../guide/augmentation.md) covers all five.

## Train (optional PyTorch layer)

```python
from pybvh_ml.torch import MotionDataset, collate_motion_batch
from torch.utils.data import DataLoader

dataset = MotionDataset(
    data["clips"], labels=data["labels"],
    target_length=128, augmentation=pipeline, seed=42,
)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_motion_batch)

for epoch in range(10):
    dataset.set_epoch(epoch)   # fresh augmentation per epoch, reproducibly
    for batch in loader:
        batch["data"]     # (B, T_max, D) padded
        batch["mask"]     # (B, T_max) bool validity mask
        batch["lengths"]  # (B,) valid frames per item
        batch["labels"]   # (B,)
```

With a `seed`, augmentation is bit-identical across runs regardless of `num_workers` or shuffle order — the [PyTorch guide](../guide/pytorch.md) explains the `set_epoch` contract.

## Or pack arrays yourself

If you'd rather own the loop, the packers convert pybvh's structured arrays into model layouts directly:

```python
from pybvh_ml import pack_to_ctv, pack_to_flat
import pybvh

bvh = pybvh.read_bvh_file("walk.bvh")
root_pos, rot6d = bvh.to_6d()

ctv = pack_to_ctv(root_pos, rot6d)     # (C, T, V) for GCNs
flat = pack_to_flat(root_pos, rot6d)   # (T, D) for MLPs / Transformers
```

## Where to next

- **[Find a function](../api/index.md)** — the "I want to… → call" capability map.
- **[Tensor Layouts & Packing](../guide/tensor-layouts.md)** — the `(C, T, V)` conventions and `center_root` semantics; ten minutes that prevent most bugs.
- **[Preprocessing Pipelines](../guide/preprocessing.md)** — harmonization, normalization, and everything the dataset file stores.
- **[Tutorials](../tutorials.md)** — three notebooks with detailed walkthroughs, ending in a trained classifier.
