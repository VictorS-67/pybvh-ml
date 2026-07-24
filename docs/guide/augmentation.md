# Runtime Augmentation

The "every epoch" step: fast, array-level augmentation designed for on-the-fly use inside data loaders. All functions operate directly on pre-extracted NumPy arrays — no `Bvh` object reconstruction — and every rotation representation (`"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"`) is handled by the same unified functions.

Three conventions, everywhere:

- **Keyword-only arguments** — call sites stay readable inside pipeline configs.
- **Angles are in radians**, matching pybvh 0.8's radians-first API. Coming from degrees? `np.radians(90)`.
- **Invalid inputs raise `ValueError`** — mismatched `root_pos` / `joint_data` frame counts, negative sigmas, out-of-range drop rates, empty joint arrays, zero-norm quaternions. No silent no-ops.

## The five functions

```python
import numpy as np
from pybvh_ml import (
    rotate_vertical, mirror,
    speed_perturbation_arrays, dropout_arrays, add_joint_noise,
    get_lr_pairs,
)

rng = np.random.default_rng(42)

# Vertical rotation — up_axis is a signed axis string matching bvh.world_up.
# The sign flips the rotation direction, so '+y' and '-y' yaw oppositely.
root_pos, quats = rotate_vertical(
    root_pos=root_pos, joint_data=quats,
    angle=np.pi / 2, up_axis="+y",
    representation="quat")

# Left-right mirroring — lateral_axis uses the same signed-string form
# but is sign-invariant ('+x' and '-x' are equivalent).
lr_pairs = get_lr_pairs(bvh)
root_pos, quats = mirror(
    root_pos=root_pos, joint_data=quats,
    lr_joint_pairs=lr_pairs, lateral_axis="+x",
    representation="quat")

# Speed perturbation (SLERP interpolation between frames).
root_pos, quats = speed_perturbation_arrays(
    root_pos=root_pos, joint_data=quats,
    factor=1.2, representation="quat")

# Frame dropout (drop and re-interpolate random frames).
root_pos, quats = dropout_arrays(
    root_pos=root_pos, joint_data=quats,
    drop_rate=0.1, representation="quat", rng=rng)

# Joint rotation noise (applied in quaternion space internally).
root_pos, quats = add_joint_noise(
    root_pos=root_pos, joint_data=quats,
    sigma=np.radians(1.0), representation="quat", rng=rng)
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
from pybvh_ml.augmentation import rotate_vertical, mirror, add_joint_noise

pipeline = AugmentationPipeline([
    (rotate_vertical, 1.0, {
        "angle": lambda rng: rng.uniform(-np.pi, np.pi),  # random each sample
        "up_axis": "+y",
        "representation": "quat",
    }),
    (mirror, 0.5, {
        "lr_joint_pairs": lr_pairs,
        "lateral_axis": "+x",
        "representation": "quat",
    }),
    (add_joint_noise, 1.0, {
        "sigma": np.radians(1.0),
        "representation": "quat",
    }),
])

rng = np.random.default_rng(42)
root_pos, quats = pipeline(root_pos=root_pos, joint_data=quats, rng=rng)
```

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

With `cache_quats=True` (the default), the pipeline converts your `joint_data` to quaternions once, runs quaternion-internal steps back to back without leaving quat space, and converts back to your declared representation at the end. This is why the pipeline needs to *know* the representation: **at least one step must declare a `representation` kwarg** (or pass `cache_quats=False` to run every step exactly as written). It raises at construction time otherwise — guessing would corrupt non-quat inputs.

Two guarantees, identical on both paths:

- **Custom steps see your declared representation.** A step function the pipeline doesn't recognize receives `joint_data` in the pipeline's current declared representation — never in whatever internal state a previous built-in step left behind. `cache_quats=True` and `cache_quats=False` are bit-identical.
- **Outputs never alias inputs.** Even when no step fires, `pipeline(...)` returns freshly allocated arrays — safe to mutate without corrupting a Dataset's cached clips.

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
