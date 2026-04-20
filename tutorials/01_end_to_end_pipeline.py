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
# # 1. End-to-end pipeline: BVH directory → trained classifier
#
# This notebook walks through the canonical pybvh-ml workflow:
# **BVH files on disk → preprocessed dataset → augmented `MotionDataset` → `DataLoader` → trained model.**
#
# It's intentionally minimal — a 2-class dataset built from the repo's two compatible
# fixture (`bvh_test1.bvh`, copied twice to simulate two classes) and a tiny MLP — so the shape of the pipeline
# is visible end to end. In practice you'd have hundreds of clips and a real model architecture;
# the glue code is identical.

# %% [markdown]
# ## Setup

# %%
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import pybvh_ml
print("pybvh-ml version:", pybvh_ml.__version__)

import pybvh
print("pybvh version:", pybvh.__version__)

# %% [markdown]
# ## Step 1 — Preprocess a BVH directory
#
# `preprocess_directory` walks a directory, loads each file, extracts the chosen rotation
# representation, computes dataset-wide normalization stats, and writes everything to a single
# `.npz` or `.hdf5` file.
#
# For this demo we use two fixtures that share a topology:

# %%
# The repo's bvh_data/ directory contains 5 fixture BVHs, but only two share a skeleton.
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
BVH_DIR = REPO_ROOT / "bvh_data"

# Copy bvh_test1.bvh twice under different names so we have a 2-class dataset
# with matching topology — real pipelines would of course use distinct clips.
work_dir = Path(tempfile.mkdtemp(prefix="pybvh_ml_demo_"))
import shutil
for name in ("clip_a.bvh", "clip_b.bvh"):
    shutil.copy(BVH_DIR / "bvh_test1.bvh", work_dir / name)

print("Source directory:", work_dir)
print("Files:", sorted(p.name for p in work_dir.iterdir()))

# %%
from pybvh_ml import preprocess_directory

# Label files by their stem. Real pipelines usually parse the label from the filename
# (e.g. emotion tag, action class) — here we just map the two fixtures to two classes.
def label_fn(stem: str) -> int:
    return 0 if stem == "clip_a" else 1

out_path = work_dir / "train.npz"
summary = preprocess_directory(
    work_dir,
    out_path,
    representation="6d",                 # continuous, recommended for ML models
    center_root=True,
    label_fn=label_fn,
    require_matching_topology=True,      # safety — raises if the skeletons disagree
)

print("num_clips:", summary["num_clips"])
print("representation:", summary["representation"])
print("skeleton joints:", summary["skeleton_info"]["num_joints"])
print("saved to:", out_path)

# %% [markdown]
# ## Step 2 — Load the preprocessed file
#
# `load_preprocessed` returns a dict: clips, labels, mean/std, skeleton metadata, and
# (since 0.3) a `constant_channels` mask. Each clip is itself a dict with `root_pos` +
# `joint_data` (6D rotations, shape `(F, J, 6)`).

# %%
from pybvh_ml import load_preprocessed

data = load_preprocessed(out_path)
print("keys:", sorted(data.keys()))
print("clip[0] keys:", sorted(data["clips"][0].keys()))
print("clip[0] root_pos shape:", data["clips"][0]["root_pos"].shape)
print("clip[0] joint_data shape:", data["clips"][0]["joint_data"].shape)

# %% [markdown]
# ## Step 3 — Add an on-the-fly augmentation pipeline
#
# Augmentation runs inside `__getitem__` so each epoch sees freshly augmented clips.
# We'll compose two array-level augmentations: vertical rotation and L/R mirror.
#
# pybvh-ml provides **6D-native** versions of both — they operate directly on `(F, J, 6)` arrays
# without converting to quaternions first, which keeps the data-loader path fast. The 6D data goes
# straight to the model without any intermediate conversion.

# %%
from pybvh_ml import AugmentationPipeline
from pybvh_ml.augmentation import rotate_vertical, mirror
from pybvh_ml.skeleton import get_lr_pairs

# The L/R joint pairs we need for mirroring come from the skeleton metadata.
# We reconstruct them from one of the loaded BVH files.
from pybvh import read_bvh_file
reference_bvh = read_bvh_file(work_dir / "clip_a.bvh")
lr_pairs = get_lr_pairs(reference_bvh)

# Signed-axis strings like '+y' or '-z' match pybvh's world_up / forward_at
# / left_at convention. The sign matters for rotation direction; mirror is
# sign-invariant.
up_axis = reference_bvh.world_up
lateral_axis = reference_bvh.left_at(0)

# 6D-native augmentations operate directly on (F, J, 6) arrays —
# no quaternion round-trip, so this is safe inside a DataLoader worker.
pipeline = AugmentationPipeline([
    # 100% of the time: random vertical rotation in [-180, 180] degrees
    (rotate_vertical, 1.0, {
        "angle_deg": lambda rng: rng.uniform(-180, 180),
        "up_axis": up_axis,
        "representation": "6d",
    }),
    # 50% of the time: L/R mirror
    (mirror, 0.5, {
        "lr_joint_pairs": lr_pairs,
        "lateral_axis": lateral_axis,
        "representation": "6d",
    }),
])

print(pipeline)

# %%
from pybvh_ml.torch import MotionDataset

dataset = MotionDataset(
    data["clips"],
    labels=data["labels"],
    target_length=32,                # crop/pad to fixed length for batching
    augmentation=pipeline,
    seed=42,                         # reproducible across runs
)
print("dataset size:", len(dataset))
print("sample[0] keys:", sorted(dataset[0].keys()))
print("sample[0] data shape:", dataset[0]["data"].shape)
print("sample[0] label:", dataset[0]["label"])

# %% [markdown]
# ## Step 4 — `DataLoader` with variable-length collation
#
# `collate_motion_batch` zero-pads clips to the longest in the batch and returns a
# validity `mask`. With `target_length` already set above, the lengths are uniform
# in this demo; in variable-length setups the collate does real work.

# %%
from pybvh_ml.torch import collate_motion_batch

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_motion_batch,
)

batch = next(iter(loader))
print("data:", tuple(batch["data"].shape))
print("mask:", tuple(batch["mask"].shape))
print("lengths:", batch["lengths"].tolist())
print("labels:", batch["labels"].tolist())


# %% [markdown]
# ## Step 5 — A tiny MLP classifier
#
# Nothing pybvh-ml-specific here; this is the simplest possible model that consumes
# the `(B, T, D)` tensor produced by our collate function.

# %%
class TinyClassifier(torch.nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_classes),
        )

    def forward(self, x, mask):
        # Mean-pool over valid frames, then classify.
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.net(pooled)

feat_dim = batch["data"].shape[-1]
model = TinyClassifier(feat_dim=feat_dim, num_classes=2)
print(model)

# %% [markdown]
# ## Step 6 — Training loop with `set_epoch`
#
# The **only pybvh-ml-specific bit** in the training loop is `dataset.set_epoch(epoch)` at the
# top of each epoch. It feeds `(seed, epoch, idx)` into a `SeedSequence` so every epoch draws
# a fresh augmentation *and* two runs with the same seed produce the same trajectory.
#
# Same contract as `torch.utils.data.distributed.DistributedSampler.set_epoch`.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
    dataset.set_epoch(epoch)   # <-- the reproducibility + variety contract
    total_loss = 0.0
    for batch in loader:
        logits = model(batch["data"], batch["mask"])
        loss = criterion(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch}: loss = {total_loss / len(loader):.4f}")

# %% [markdown]
# ## What's next
#
# - **Swap in your own model.** Anything that eats `(B, T, D)` works: Transformer, LSTM, ST-GCN (see the skeleton graph notebook for the adjacency).
# - **Scale up the dataset.** `preprocess_directory(..., parallel=True, skip_errors=True)` handles thousands of files robustly.
# - **Add more augmentations.** The [augmentation notebook](02_augmentation_visualized.ipynb) shows every function visually.
# - **Heterogeneous clips?** Use `pybvh.harmonize` first. See the [harmonization notebook](03_heterogeneous_preprocessing.ipynb).

# %%
# Clean up the scratch dir.
shutil.rmtree(work_dir)
