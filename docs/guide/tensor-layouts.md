# Tensor Layouts & Packing

pybvh gives you structured arrays: `root_pos` of shape `(F, 3)` and `joint_data` of shape `(F, J, C)`. Models want tensors in a specific layout. The packers convert between the two, in both directions.

## The three layouts

| Layout | Shape | Typical consumer |
|---|---|---|
| CTV | `(C, T, V)` | GCNs (ST-GCN family) |
| TVC | `(T, V, C)` | Transformers with per-joint tokens |
| Flat | `(T, D)` | MLPs, sequence models over pose vectors |

Conventions, shared by every packer:

- **C** = channels — `max(3, joint channels)`: 3 for Euler/axis-angle, 4 for quat, 6 for 6D, 9 for rotmat
- **T** = time / frames
- **V** = vertices / joints — the **root is vertex 0**, joints are `1..J`
- **D** = flat feature dimension — `3 + J * C_joint`

The root vertex carries 3 position channels; when `C > 3` (quat, 6D, or rotmat joint data), the root vertex's remaining `C - 3` channels are zero padding — the position values themselves are unchanged.

## Packing and unpacking

```python
from pybvh_ml import (
    pack_to_ctv, pack_to_tvc, pack_to_flat,
    unpack_from_ctv, unpack_from_tvc, unpack_from_flat,
)
import pybvh

bvh = pybvh.read_bvh_file("walk.bvh")
root_pos, rot6d = bvh.to_6d()          # (F, 3), (F, J, 6)

ctv = pack_to_ctv(root_pos, rot6d)     # (6, F, J+1)
tvc = pack_to_tvc(root_pos, rot6d)     # (F, J+1, 6)
flat = pack_to_flat(root_pos, rot6d)   # (F, 3 + J*6)

root_back, rot_back = unpack_from_ctv(ctv)
```

Every `pack_to_*` has a matching `unpack_from_*` that inverts it (up to the `center_root` shift — see below). `unpack_from_flat` needs to know the channel split: pass `joint_channels=` matching the packed representation (e.g. `joint_channels=6` for 6D), or it raises a `ValueError` when `D - root_channels` isn't divisible by it.

## `center_root` — read this once, save a debugging session

All three packers accept `center_root` and it **defaults to `True`**: the first frame's root position is subtracted from every frame, so trajectories start at the origin.

Two things to know:

1. **This is pybvh-ml's convention, not pybvh's.** pybvh-ml's `center_root` subtracts the full 3D first-frame root position. pybvh's `centered="first"` is ground-plane-only (the up coordinate is untouched there).
2. **Don't center twice — but know when twice is harmless.** `preprocess_directory` also centers by default and records the choice in the dataset metadata (`load_preprocessed(...)["center_root"]`). Re-centering a whole already-centered clip is idempotent (its first frame is already at the origin). The real hazard is **windowed sub-clips**: center the whole clip, then cut windows — if you instead pack each window with `center_root=True`, every window is re-based to its own first frame and the global trajectory is destroyed.

When packing arrays that came out of `load_preprocessed`, pass `center_root=False` and let the stored metadata tell you whether the data was centered at preprocessing time.

## Knowing what the columns mean

For the flat layout, `describe_features` maps block names to column ranges:

```python
from pybvh_ml import describe_features

desc = describe_features(num_joints=24, representation="6d", include_root_pos=True)
desc["root_pos"]                  # (0, 3)
desc["joint_rotations"]           # (3, 147)
desc.slice("joint_rotations")     # slice(3, 147)
```

This describes the simple `root_pos + joint_rotations` layout produced by `pack_to_flat`. For the richer layout that also covers velocities and foot contacts (as written by `pybvh.Bvh.to_feature_array`), use `pybvh.Bvh.feature_array_layout` — it returns a `{block_name: slice}` dict for the full feature array.

## Sequence length utilities

Models want fixed-length inputs; clips aren't. Three tools, from dumb to smart:

```python
from pybvh_ml import sliding_window, standardize_length, sample_temporal

# Cut overlapping windows: (num_windows, 64, ...)
windows = sliding_window(data, window_size=64, stride=32)

# Pad or crop to an exact length along axis 0
padded = standardize_length(data, target_length=128, method="pad")
cropped = standardize_length(data, target_length=64, method="crop")

# PySKL-style uniform segment sampling (skeleton-based recognition)
clip = sample_temporal(data, clip_length=64, mode="train", rng=rng)
clips = sample_temporal(data, clip_length=64, num_samples=5, mode="train", rng=rng)
```

`sample_temporal` divides the sequence into `clip_length` equal segments and picks one frame per segment — random offsets in `"train"` mode (temporal augmentation), deterministic offsets in `"test"` mode (reproducible evaluation, and multiple `num_samples` are still distinct draws). The index-level primitive is `uniform_temporal_sample` if you want the indices without applying them.

## See also

- [Packing API](../api/packing.md) — full signatures
- [Sequences API](../api/sequences.md) — windowing and sampling reference
- [Feature Metadata API](../api/metadata.md) — `FeatureDescriptor`
- [PyTorch Integration](pytorch.md) — where `target_length` and collate-time padding take over
