# Tensor Layouts & Packing

pybvh gives you structured arrays: `root_pos` of shape `(F, 3)` and per-joint rotations of shape `(F, J, C)`. pybvh-ml carries the pair as a [`MotionArrays`](../api/arrays.md), and models want tensors in a specific layout. The packers convert between the two, in both directions.

## The three layouts

| Layout | Shape | Typical consumer |
|---|---|---|
| CTV | `(C, T, V)` | GCNs (ST-GCN family) |
| TVC | `(T, V, C)` | Transformers with per-joint tokens |
| Flat | `(T, D)` | MLPs, sequence models over pose vectors |

Conventions, shared by every packer, for the default `streams=("root_pos", "joint_rot")`:

- **C** = channels — `max(3, joint channels)`: 3 for Euler/axis-angle, 4 for quat, 6 for 6D, 9 for rotmat
- **T** = time / frames
- **V** = vertices / joints — the **root is vertex 0**, joints are `1..J`
- **D** = flat feature dimension — `3 + J * C_joint`

The root vertex carries 3 position channels; when `C > 3` (quat, 6D, or rotmat joint data), the root vertex's remaining `C - 3` channels are zero padding — the position values themselves are unchanged.

![One clip in all three layouts: CTV and TVC single-frame slices with the root vertex's zero-padded channels marked, and the full flat clip with its columns mapped by describe_features](../gallery/img/layouts.png)

*One real clip, all three layouts — the root zero-padding and the flat column map, drawn. ([Gallery](../gallery/index.md) for every figure.)*

## Packing and unpacking

```python
from pybvh_ml import (
    MotionArrays,
    pack_to_ctv, pack_to_tvc, pack_to_flat,
    unpack_from_ctv, unpack_from_tvc, unpack_from_flat,
)
import pybvh

bvh = pybvh.read_bvh_file("walk.bvh")
root_pos, rot6d = bvh.to_6d()          # (F, 3), (F, J, 6)

arrays = MotionArrays(root_pos=root_pos, joint_rot=rot6d)

ctv = pack_to_ctv(arrays)     # (6, F, J+1)
tvc = pack_to_tvc(arrays)     # (F, J+1, 6)
flat = pack_to_flat(arrays)   # (F, 3 + J*6)

root_back, rot_back = unpack_from_ctv(ctv)
```

Every `pack_to_*` has a matching `unpack_from_*` that inverts it, returning a `MotionArrays` (up to the `center_root` shift — see below). `unpack_from_flat` needs to know the channel split: pass `joint_channels=` matching the packed representation (e.g. `joint_channels=6` for 6D), or it raises a `ValueError` when `D - root_channels` isn't divisible by it.

## Choosing what gets packed: `streams=`

`streams=` names what is packed and in what order — channel order in the graph layouts, column order in flat. It defaults to `("root_pos", "joint_rot")`, which is byte-identical to what every earlier version produced.

```python
arrays = MotionArrays.from_bvh(bvh, "6d", include_positions=True)

pack_to_ctv(arrays, streams=("joint_pos",))              # (3, T, J) — ST-GCN
pack_to_ctv(arrays, streams=("joint_pos", "joint_rot"))  # (3+6, T, J)
```

| `streams` | CTV shape | note |
|---|---|---|
| `("root_pos", "joint_rot")` | `(max(3, C_rot), T, 1+J)` | the default |
| `("joint_pos",)` | `(3, T, J)` | canonical ST-GCN / CTR-GCN input |
| `("node_pos",)` | `(3, T, N)` | full visual skeleton, end effectors included |
| `("joint_pos", "joint_rot")` | `(3 + C_rot, T, J)` | multi-stream on `C` |
| `("root_pos", "joint_pos")` | `(3, T, 1+J)` | vertex 0 duplicates joint 0 under `"world"` centering |

Two rules:

- **`"root_pos"` in the list adds the root as vertex 0** (`V = 1 + J`); omit it and `V = J`. This is what removes the off-by-one between packed vertices and `skeleton_info["edges"]` — see [Skeleton Graph Metadata](skeleton-metadata.md#which-key-indexes-which-packing).
- **`node_pos` cannot share a vertex axis with a joint-space stream.** Node space includes end sites, so it has `N` vertices where joint space has `J`; the packer raises naming the mismatch rather than broadcasting anything.

The **unpackers keep their pre-0.6.0 signature** and invert only the default streams — everything past the root's channels comes back as `joint_rot`. A streams-aware `unpack_from_*` is purely additive whenever it lands; the asymmetry is a deliberate deferral. Until then, slice the channel axis yourself using the widths above.

## `center_root` — read this once, save a debugging session

All three packers accept `center_root` and it **defaults to `True`**: the first frame's root position is subtracted from every frame, so trajectories start at the origin.

Two things to know:

1. **This is pybvh-ml's convention, not pybvh's.** pybvh-ml's `center_root` subtracts the full 3D first-frame root position. pybvh's `centered="first"` is ground-plane-only (the up coordinate is untouched there).
2. **Don't center twice — but know when twice is harmless.** `preprocess_directory` also centers by default and records the choice in the dataset metadata (`load_preprocessed(...)["center_root"]`). Re-centering a whole already-centered clip is idempotent (its first frame is already at the origin). The real hazard is **windowed sub-clips**: center the whole clip, then cut windows — if you instead pack each window with `center_root=True`, every window is re-based to its own first frame and the global trajectory is destroyed.

![Top view of a root trajectory: centering the clip once keeps windows on the global path; packing each window with center_root=True re-bases every window to its own origin](../gallery/img/center-root-hazard.png)

*The hazard, drawn: the same windows on the preserved global path (left) vs each re-based to its own origin (right).*

When packing arrays that came out of `load_preprocessed`, pass `center_root=False` and let the stored metadata tell you whether the data was centered at preprocessing time.

**With positions in the container, `center_root` reaches them too.** Under `position_centering="world"` or `"first"` the identical first-frame shift is applied to every position vertex; under `"skeleton"` they are already root-relative and are left alone; with `position_centering=None` the packer raises rather than guess. Centering only the root would move vertex 0 away from a body that stayed put — the same inconsistency `add_root_position_noise` guards against, produced at pack time instead.

## Knowing what the columns mean

For the flat layout, `describe_features` maps block names to column ranges:

```python
from pybvh_ml import describe_features

desc = describe_features(num_joints=24, representation="6d", include_root_pos=True)
desc["root_pos"]                  # (0, 3)
desc["joint_rotations"]           # (3, 147)
desc.slice("joint_rotations")     # slice(3, 147)
```

It takes the same `streams=` as the packer, and blocks are named `root_pos`, `joint_rotations`, `joint_positions`, `node_positions`:

```python
desc = describe_features(24, streams=("joint_pos", "joint_rot"))
desc.slice("joint_positions")     # slice(0, 72)
desc.slice("joint_rotations")     # slice(72, 216)
```

`num_nodes=` is required for a `node_pos` block — nodes are joints plus end sites, so `N` cannot be derived from `num_joints`.

For the richer layout that also covers velocities and foot contacts (as written by `pybvh.Bvh.to_feature_array`), use `pybvh.Bvh.feature_array_layout` — it returns a `{block_name: slice}` dict for the full feature array.

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

![Segment-shaded timeline with the frame indices picked by three train-mode draws and the deterministic test-mode draw](../gallery/img/temporal-sample.png)

*One frame per segment: three train draws jitter within their segments; test mode is fixed.*

## See also

- [Packing API](../api/packing.md) — full signatures
- [Sequences API](../api/sequences.md) — windowing and sampling reference
- [Feature Metadata API](../api/metadata.md) — `FeatureDescriptor`
- [PyTorch Integration](pytorch.md) — where `target_length` and collate-time padding take over
