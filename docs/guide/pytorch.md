# PyTorch Integration

Optional — install with `pip install "pybvh-ml[torch]"`. The `pybvh_ml.torch` submodule provides two Dataset classes and a collate function; everything else in the library stays NumPy-only, so a broken or absent torch never blocks the core.

## End to end

Preprocess → `MotionDataset` → `DataLoader` → training batch:

```python
from pybvh_ml import preprocess_directory, load_preprocessed, AugmentationPipeline
from pybvh_ml.torch import MotionDataset, collate_motion_batch
from torch.utils.data import DataLoader

# 1. One-time preprocess (label_fn: map filename stem → integer class).
preprocess_directory("walks/", "train.npz", representation="6d",
                     label_fn=lambda stem: 0)

# 2. Build dataset + augmentation pipeline.
data = load_preprocessed("train.npz")
skel = data["skeleton_info"]
pipeline = AugmentationPipeline.standard(
    skel, representation="6d", up_axis=skel["world_up"])

dataset = MotionDataset(
    data["clips"], labels=data["labels"],
    target_length=128, augmentation=pipeline,
    seed=42,  # reproducible — see the set_epoch contract below
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

## The two Dataset classes

**`MotionDataset`** wraps in-memory clip dicts — typically `load_preprocessed(...)["clips"]`, but any list of `{"root_pos", "joint_data", ...}` dicts works (pass `center_root=True` for raw, uncentered hand-built clips).

**`OnTheFlyDataset`** skips the preprocessed file entirely: give it a list of BVH paths (`str` or `Path`) and it loads, converts, and augments per item — useful when you don't want the preprocessing artifact on disk. `world_up=` and `lr_mapping=` are forwarded to every `pybvh.read_bvh_file` call, matching `preprocess_directory`.

Both support Python negative indexing (`ds[-1]` matches `ds[len(ds)-1]` on the same rng stream) and raise `IndexError` cleanly out of range.

## What `__getitem__` returns, and what collate does

Each item is a dict with `"data"` — the flat `(T, D)` pack of the (augmented, length-standardized) clip — plus `"length"` and optionally `"label"`.

**`length` means valid frames in the returned tensor.** When `target_length` pads a shorter clip, `length` is the original frame count (the padded tail is not valid); when it *crops* a longer clip, `length` is `target_length`. If you need the original pre-crop length, recover it from your clip arrays before the dataset.

`collate_motion_batch` stacks variable-length items into `{"data", "mask", "lengths", "labels"}` with zero-padding to the batch maximum and a bool validity mask. Either every item in a batch carries a label or none does — mixed presence raises a `ValueError` naming the offending index rather than silently dropping labels.

## Reproducible per-epoch augmentation

When `seed` is set, the tuple `(seed, epoch, idx)` feeds a `numpy.random.SeedSequence`, so:

- two runs with the same seed produce the same augmentation trajectory,
- each epoch still sees a different draw,
- and the result is **bit-identical regardless of `num_workers` or shuffle order** — the rng depends on the sample index, not on which worker processes it.

Call `dataset.set_epoch(epoch)` at the top of each epoch — the same contract as `torch.utils.data.distributed.DistributedSampler`. Forgetting it doesn't break anything: the dataset warns once and uses epoch 0 (every epoch replays the same augmentation draw).

With `seed=None`, every call uses fresh OS entropy — simplest, no reproducibility.

!!! note "Why `set_epoch` works with worker processes (and what it costs)"
    DataLoader workers hold a *copy* of the dataset, created once — with `persistent_workers=True`, that copy lives for the whole training run, so a plain-attribute epoch would silently freeze at its startup value in every worker. The epoch therefore lives in shared memory (a `multiprocessing.Value`), which worker processes inherit at creation — fork and spawn both. One consequence, stated on both classes: dataset instances can't be `deepcopy`-ed or `torch.save`-ed directly, because shared state only travels to other processes via DataLoader worker inheritance. Save your model, not your dataset.

## See also

- [PyTorch API](../api/torch.md) — full signatures for both classes and the collate function
- [Runtime Augmentation](augmentation.md) — building the pipeline the datasets call
- [Tutorial 1: End-to-end pipeline](../tutorials.md) — this page as a runnable notebook, ending in a trained classifier
