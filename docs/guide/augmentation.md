# Runtime Augmentation

The "every epoch" step: fast, array-level augmentation designed for on-the-fly use inside data loaders. All functions operate directly on pre-extracted NumPy arrays — no `Bvh` object reconstruction — and every rotation representation (`"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"`) is handled by the same unified functions.

Three conventions, everywhere:

- **Keyword-only arguments** — call sites stay readable inside pipeline configs.
- **Angles are in radians**, matching pybvh 0.8's radians-first API. Coming from degrees? `np.radians(90)`.
- **Invalid inputs raise `ValueError`** — mismatched `root_pos` / `joint_rot` frame counts (checked once, when the `MotionArrays` is built), negative sigmas, out-of-range drop rates, empty joint arrays, zero-norm quaternions. No silent no-ops.

## `MotionArrays`, the thing every function takes

One clip's motion travels through the library as a single
[`MotionArrays`](../api/arrays.md): `root_pos` plus `joint_rot`.  Every
augmentation takes one and returns a new one, so a pipeline is a chain of
`MotionArrays -> MotionArrays` and adding a stream later — per-joint
positions are next — costs no call-site changes.

```python
from pybvh_ml import MotionArrays

arrays = MotionArrays.from_bvh(bvh, "quat")     # or build it yourself:
arrays = MotionArrays(root_pos=root_pos, joint_rot=quats)
```

It is frozen: derive a new one with `arrays.replace(joint_rot=...)`
rather than assigning to a field.  That is what lets every function in
the package rely on the frame counts having been checked once.

## The six functions

```python
import numpy as np
from pybvh_ml import (
    MotionArrays, rotate_vertical, mirror,
    speed_perturbation_arrays, dropout_arrays,
    add_joint_rotation_noise, add_root_position_noise,
    get_lr_pairs,
)

rng = np.random.default_rng(42)
arrays = MotionArrays(root_pos=root_pos, joint_rot=quats)

# Vertical rotation — up_axis is a signed axis string matching bvh.world_up.
# The sign flips the rotation direction, so '+y' and '-y' yaw oppositely.
arrays = rotate_vertical(
    arrays, angle=np.pi / 2, up_axis="+y", representation="quat")

# Left-right mirroring — lateral_axis uses the same signed-string form
# but is sign-invariant ('+x' and '-x' are equivalent).
lr_pairs = get_lr_pairs(bvh)
arrays = mirror(
    arrays, lr_joint_pairs=lr_pairs, lateral_axis="+x",
    representation="quat")

# Speed perturbation (SLERP interpolation between frames).
arrays = speed_perturbation_arrays(
    arrays, factor=1.2, representation="quat")

# Frame dropout (drop and re-interpolate random frames).
arrays = dropout_arrays(
    arrays, drop_rate=0.1, representation="quat", rng=rng)

# Joint rotation noise (applied in quaternion space internally).
# `degrees=True` would let you write sigma=1.0 instead.
arrays = add_joint_rotation_noise(
    arrays, sigma=np.radians(1.0), representation="quat", rng=rng)

# Root translation noise — a separate function because its sigma is a
# length, not an angle, so no single `degrees=` flag could serve both.
arrays = add_root_position_noise(arrays, sigma=0.5, rng=rng)

root_pos, quats = arrays.root_pos, arrays.joint_rot
```

Representation-specific notes:

- **`"euler"`** additionally requires `euler_orders=bvh.euler_orders` — per-joint orders are respected, including mixed-order L/R pairs under `mirror`.
- **`"rotmat"`** is carried flat as `(F, J, 9)` — the layout [`convert_arrays`](../api/convert.md) documents and produces; the 3×3 reshape happens internally.
- **`"quat"` is the fast path**: every function works in quaternion space internally, so non-quat representations pay one conversion in and one out per call. The [pipeline](#composing-a-pipeline) eliminates the intermediate round trips.

## Composing a pipeline

`AugmentationPipeline` chains steps with per-step probabilities. Kwargs can be callables taking `rng`, drawn fresh per sample:

```python
import numpy as np
from pybvh_ml import AugmentationPipeline
from pybvh_ml.augmentation import (
    rotate_vertical, mirror, add_joint_rotation_noise)

pipeline = AugmentationPipeline([
    (rotate_vertical, 1.0, {
        "angle": lambda rng: rng.uniform(-np.pi, np.pi),  # random each sample
        "up_axis": "+y",
    }),
    (mirror, 0.5, {
        "lr_joint_pairs": lr_pairs,
        "lateral_axis": "+x",
    }),
    (add_joint_rotation_noise, 1.0, {"sigma": np.radians(1.0)}),
], representation="quat")   # pipeline-level default; a step may still override

rng = np.random.default_rng(42)
arrays = pipeline(MotionArrays(root_pos=root_pos, joint_rot=quats), rng=rng)
```

A pipeline is homogeneous in practice, so declare `representation` once at the pipeline level rather than repeating it on every step — repeating it is the copy-paste surface where one step in five ends up disagreeing with the rest. A step that declares its own keeps it, and the default only reaches functions that *name* the parameter (a `**kwargs` catch-all doesn't count), so a custom step taking neither is called with exactly its own kwargs. `euler_orders` takes a pipeline-level default the same way.

Built-in steps have to agree on the result: each one's output is the next one's input, so two of them declaring different representation tokens with nothing in between to convert raises at construction rather than quietly producing different arrays under `cache_quats=True` and `False`. A custom step between them lifts the restriction — it may legitimately be doing the conversion.

Steps are `AugmentationStep` named tuples, so introspecting a configured pipeline reads:

```python
pipeline.augmentations[1].prob            # 0.5
pipeline.augmentations[1].kwargs["lateral_axis"]   # "+x"
```

A `NamedTuple` is a `tuple`, so positional access (`pipeline.augmentations[1][2]`) and unpacking still work.

For the common case, skip the boilerplate: the `standard` factory wires rotate + mirror + noise + speed from a `skeleton_info` dict — which a [preprocessed dataset](preprocessing.md#what-the-file-stores) already carries:

```python
from pybvh_ml import AugmentationPipeline, get_skeleton_info

pipeline = AugmentationPipeline.standard(
    get_skeleton_info(bvh),
    representation="quat",
    up_axis="+y",
    # rotate_angle_range=(-np.pi, np.pi), mirror_prob=0.5, noise_sigma=np.radians(1.0),
    # speed_factor_range=(0.8, 1.2)  — defaults shown; pass None to disable a step
)
```

### How the pipeline avoids conversion churn

With `cache_quats=True` (the default), the pipeline converts your `joint_rot` to quaternions once, runs quaternion-internal steps back to back without leaving quat space, and converts back to your declared representation at the end. This is why the pipeline needs to *know* the representation: **something must declare it** — `AugmentationPipeline(..., representation=...)` or at least one step's kwargs (or pass `cache_quats=False` to run every step exactly as written). It raises otherwise, rather than guessing and corrupting non-quat inputs.

Two guarantees, identical on both paths:

- **Custom steps see your declared representation.** A step function the pipeline doesn't recognize receives `arrays.joint_rot` in the pipeline's current declared representation — never in whatever internal state a previous built-in step left behind. `cache_quats=True` and `cache_quats=False` are bit-identical.
- **Outputs never alias inputs.** Even when no step fires, `pipeline(...)` returns freshly allocated arrays — safe to mutate without corrupting a Dataset's cached clips.

### Seeing what a call actually drew

A pipeline with probabilities and callable kwargs makes a different decision for every sample. `return_params=True` reports those decisions alongside the arrays — it is the only thing that changes the return arity, turning the `MotionArrays` into an `(arrays, steps)` pair:

```python
arrays, steps = pipeline(
    MotionArrays(root_pos=root_pos, joint_rot=joint_rot),
    rng=rng, return_params=True)

[(s["name"], s["applied"], s["params"]) for s in steps]
# [('rotate_vertical',           True,  {'angle': -1.4464727375963786}),
#  ('mirror',                    True,  {}),
#  ('add_joint_rotation_noise',  True,  {}),
#  ('speed_perturbation_arrays', True,  {'factor': 1.1399704034814648})]
```

One record per configured step, in pipeline order, so `steps[i]` describes `pipeline.augmentations[i]`. `applied` is the probability draw's outcome — `mirror` at `p=0.5` reports `False` on the samples it skipped. `params` holds what this call *sampled*: the kwargs whose spec is a callable, resolved to the values the augmentation received. Static kwargs (`sigma`, `up_axis`, `lr_joint_pairs`) are pipeline configuration you already have — read them from `pipeline.augmentations` — and `rng` is machinery, so neither clutters the record.

The records are plain dicts, JSON-native for every built-in step: log them next to your training metrics to answer "which augmentation produced this loss spike", or replay a specific sample's draw. (A custom step whose callable returns something exotic — an array, say — is reported verbatim, so serialize those with care.) Asking for them never changes the random stream — the same `rng` yields identical arrays with or without the flag — so it is safe to switch on in a debugging run and off again.

## Reproducibility

Every random step takes the pipeline's `rng`. Pass a seeded `np.random.default_rng(seed)` and the whole pipeline is deterministic; pass nothing and it draws OS entropy. Inside the PyTorch datasets, the rng is derived from `(seed, epoch, idx)` so results are bit-identical regardless of worker count or shuffle order — see [PyTorch Integration](pytorch.md#reproducible-per-epoch-augmentation).

## Converting between representations

The same conversion core is exposed standalone, for `(F, J, C)` arrays:

```python
from pybvh_ml import convert_arrays

rot6d = convert_arrays(euler_data, from_repr="euler", to_repr="6d",
                       euler_orders=bvh.euler_orders)
rotmat = convert_arrays(quats, from_repr="quat", to_repr="rotmat")
```

Supported: `"euler"`, `"quat"`, `"6d"`, `"axisangle"`, `"rotmat"` — any pair. Euler angles are radians, per-joint orders respected.

## See also

- [Augmentation API](../api/augmentation.md) / [Pipeline API](../api/pipeline.md) — full signatures
- [Tutorial 2: Augmentation visualized](../tutorials.md) — every function shown before/after on a real skeleton
- [pybvh's transforms](https://victors-67.github.io/pybvh/guide/augmentation/) — the `Bvh`-level equivalents, when you're not in a data loader
