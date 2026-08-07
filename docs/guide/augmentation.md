# Runtime Augmentation

The "every epoch" step: fast, array-level augmentation designed for on-the-fly use inside data loaders. All functions operate directly on pre-extracted NumPy arrays — no `Bvh` object reconstruction — and every rotation representation (`"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"`) is handled by the same unified functions.

Three conventions, everywhere:

- **Keyword-only arguments** — call sites stay readable inside pipeline configs.
- **Angles are in radians**, matching pybvh 0.8's radians-first API. Coming from degrees? `np.radians(90)`.
- **Invalid inputs raise `ValueError`** — mismatched `root_pos` / `joint_rot` frame counts (checked once, when the `MotionArrays` is built), negative sigmas, out-of-range drop rates, empty joint arrays, zero-norm quaternions. No silent no-ops.

## `MotionArrays`, the thing every function takes

One clip's motion travels through the library as a single
[`MotionArrays`](../api/arrays.md): `root_pos`, plus any of `joint_rot`,
`joint_pos` and `node_pos`.  Every augmentation takes one and returns a
new one, so a pipeline is a chain of `MotionArrays -> MotionArrays` and
the position streams 0.6.0 added cost no call-site changes.

```python
from pybvh_ml import MotionArrays

arrays = MotionArrays.from_bvh(bvh, "quat")     # or build it yourself:
arrays = MotionArrays(root_pos=root_pos, joint_rot=quats)

# Rotations and positions together, or positions alone:
arrays = MotionArrays.from_bvh(bvh, "quat", include_positions=True)
keypoints = MotionArrays.from_bvh(
    bvh, representation=None, include_positions=True,
    position_centering="skeleton")
```

It is frozen: derive a new one with `arrays.replace(joint_rot=...)` rather than assigning to a field. That is what lets every function in the package rely on the frame counts having been checked once. The fields are read-only *views* of the arrays you passed in — writing through them raises, so a container built over a Dataset's cached clips cannot rewrite the cache, but it does not copy either, so mutating your own array still changes what the container reads. Take `np.array(arrays.joint_rot)` when you need a writable working array.

dtype is preserved rather than promoted: `float32` in, `float32` out, so holding a cached clip in a container doesn't double its memory. Non-floating input (an integer array) is promoted to `float64`. Each stream follows its own input, so `float32` positions next to `float64` rotations is legal.

**Augmentation preserves the dtype without computing in it.** Every function and the pipeline run the math in `float64` — pybvh's dtype, and the only one its conversions are exact in — then return each stream as it arrived. Widening is lossless, so the `float32` result is exactly the `float64` result narrowed. Two things depend on doing it this way rather than simply letting the input dtype flow through: a probabilistic pipeline's output dtype must not depend on which steps happened to fire for that sample, and `cache_quats=True` / `False` have to stay bit-identical (the staged `6d` fast path writes into a copy of its input, so a `float32` clip would otherwise have that step computed in single precision on one path and double on the other).

Where preservation stops: the packers and `standardize_length(method="resample_linear")` produce `float64` regardless, so the array a model receives is `float64` either way — the PyTorch datasets then emit `torch.float32`. This is about what a clip costs to hold and pass through augmentation, not an end-to-end single-precision path.

## Position streams

`joint_pos` is `(F, J, 3)`, index-aligned with `joint_rot`; `node_pos` is `(F, N, 3)` and includes end sites — fingertips, toe tips, the head top — so it pairs with the node-space edge list. **Pick one**: `node_pos` already contains `joint_pos` (`node_pos[:, joint_idx >= 0]` is exactly it), and only `joint_pos` can share a vertex axis with `joint_rot`.

Positions carry a **frame convention**, `position_centering`, and it travels with the arrays rather than only with the dataset because at least one step's correctness depends on it:

| value | meaning |
|---|---|
| `"world"` | same frame as `root_pos` — a joint position already contains the root trajectory |
| `"skeleton"` | root at the origin every frame; the trajectory is carried by `root_pos` alone. What most NTU-style pipelines feed a model |
| `"first"` | pybvh's ground-plane centering — the first frame's root subtracted in the two axes perpendicular to world up |
| `None` | unknown |

The three coincide only for a clip whose root never moves. `None` is legal and **fails at use, not at construction**: most of the surface — mirror, speed perturbation, dropout, keypoint jitter, `rotate_vertical`, `pack_to_*(center_root=False)` — is a rigid or temporal operation applied identically to both streams and does not care. The three that do care — `add_root_position_noise`, the FK refresh inside `add_joint_rotation_noise`, and `pack_to_*(center_root=True)` — raise naming the field. A guessed `"world"` from someone who does not actually know is worse than an honest `None`; anything *this library* writes records it.

Why it has to be on the container and not just in dataset metadata: under `"world"` centering, jittering `root_pos` while leaving the positions alone leaves the two streams mutually inconsistent — and under the canonical ST-GCN pack `streams=("joint_pos",)`, where `root_pos` is not packed at all, it degrades into an augmentation the model never sees and nothing raises.

## Streams: a step handles every one, or refuses it

> **A step must handle every stream the sample carries, or refuse it.** A pipeline never carries a stream a step left behind.

Every function declares what it handles, and the declaration is checked **once at `AugmentationPipeline.__call__` entry, before any step runs** — a `p=0.1` step with the wrong stream support would otherwise raise on one sample in ten.

```python
from pybvh_ml import stream_support, handles_streams

stream_support(mirror)                    # every stream
stream_support(add_joint_position_noise)  # {"root_pos", "joint_pos"}
```

All four geometric steps and both noise functions handle everything. Only the two keypoint-jitter functions decline anything, and they decline `joint_rot` for the governing reason: **positions are derived from rotations**, so rotation → position is computable (forward kinematics) while position → rotation is not (inverse kinematics). A jittered position stream cannot be pushed back into the rotations beside it.

**"Handles" means the stream is left correct**, by one of two routes — *transformed* from its own input (all four geometric steps), or *re-derived* from another stream (`add_joint_rotation_noise`, which replaces the positions with FK of the noised rotations). It does **not** mean positions stay the exact FK of the rotations beside them, except immediately after a re-derivation. Two divergences are intrinsic and every pipeline in the field has them:

- **Mirror.** Positions reflect exactly in world space; rotations reflect in parent-local space. They agree on a laterally symmetric rest pose and diverge on asymmetric rigs, with the error accumulating down the chain. Each stream stays individually correct; the pair stops being FK partners.
- **Speed perturbation and dropout.** Positions are linearly interpolated, rotations slerped — chord versus arc. They agree at the knots and drift between them.

A custom step that declares nothing is assumed to handle `{"root_pos", "joint_rot"}` — exactly the capability of every step written against 0.5, so no existing pipeline changes behaviour, and the first positions-carrying sample raises a message naming the decorator:

```python
@handles_streams("root_pos", "joint_rot", "joint_pos")
def scale_positions(arrays, *, factor):
    return arrays.replace(joint_pos=arrays.joint_pos * factor)
```

## The eight functions

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

Plus the two keypoint-jitter functions, for samples that carry positions and no rotations:

```python
from pybvh_ml import add_joint_position_noise, add_node_position_noise

keypoints = add_joint_position_noise(keypoints, sigma=0.02, rng=rng)
```

They are two names rather than one `streams=` kwarg so the index space is visible at the call site — the whole point of pybvh's `joint_` / `node_` vocabulary. Note the difference from `add_root_position_noise`: keypoint jitter moves every vertex by its own draw (a pose estimator's per-joint error), while root noise moves the whole body rigidly by one offset per frame.

Representation-specific notes:

- **`"euler"`** additionally requires `euler_orders=bvh.euler_orders` — per-joint orders are respected, including mixed-order L/R pairs under `mirror`.
- **`"rotmat"`** is carried flat as `(F, J, 9)` — the layout [`convert_rotations`](../api/convert.md) documents and produces; the 3×3 reshape happens internally.
- **`"quat"` is the fast path**: every function works in quaternion space internally, so non-quat representations pay one conversion in and one out per call. The [pipeline](#composing-a-pipeline) eliminates the intermediate round trips.
- **`representation=` is optional** as of 0.6.0, and required only when the sample carries `joint_rot`. A positions-only clip has no rotation array for the token to describe.

### Positions and rotations together: the FK refresh

`add_joint_rotation_noise` is the one step that handles a stream by *re-deriving* it. When the sample carries positions it recomputes them from the noised rotations, so the two streams come out of it as genuine FK partners:

```python
from pybvh_ml import build_fk_topology, get_skeleton_info

info = get_skeleton_info(bvh)
topology = build_fk_topology(info)          # once per dataset

arrays = add_joint_rotation_noise(
    arrays, sigma=np.radians(1.0), representation="6d",
    fk_topology=topology, rng=rng)
```

- **`fk_topology=` is required** when the sample carries positions, and ignored otherwise. `AugmentationPipeline.standard` wires it from `skeleton_info` for you.
- **`world_up=` is required under `position_centering="first"`** — ground-plane centering has to know which coordinate it leaves alone, and an `FkTopology` carries no gravity direction. Pass `skeleton_info["world_up"]`.
- **It costs about 0.9 ms per sample at `F=64`** on a 31-joint rig (0.64 ms FK plus 0.25 ms 6d→euler), roughly tripling per-sample augmentation cost when it fires. Dataloader workers absorb it, and it only fires when a rotation-space step actually meets a position stream.

The consequence worth knowing is an ordering one: **a re-derivation discards whatever history the position stream carried.** On a rig with asymmetric rest offsets, `[mirror, add_joint_rotation_noise]` ends with FK of locally-mirrored rotations — throwing away the world-exact reflection the positions held — while `[add_joint_rotation_noise, mirror]` keeps it. Both are defensible and neither is a bug, but they do not produce the same positions.

The one composition that *would* be destructive is impossible by construction: keypoint jitter can never be silently wiped by a later FK refresh, because `add_joint_position_noise` declines rotation-carrying samples and `add_joint_rotation_noise` requires them, so the two can never share a pipeline.

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

For a **positions-only** dataset, pass `representation=None` and a keypoint sigma:

```python
pipeline = AugmentationPipeline.standard(
    data["skeleton_info"],
    representation=None,               # skips the rotation-noise step
    position_noise_sigma=0.02,
    # position_space="joint",          # else read from skeleton_info
)
```

Two resolutions happen at construction, because the pipeline is built before any sample is seen:

- **`representation=None` skips the rotation-noise step.** `standard()` wires `add_joint_rotation_noise` by default, and that step is meaningless on a clip with no rotations, so a positions-only pipeline would otherwise refuse every sample. Skipping is the established pattern here — mirror is already silently skipped when no L/R pairs were detected. (A *direct* `add_joint_rotation_noise` call on such a sample still raises: a factory declining to configure a meaningless step and a function refusing a meaningless call are different questions.)
- **The keypoint-jitter index space is resolved here too.** Joint-space and node-space jitter are different functions with different stream declarations, so `position_space=` (or `skeleton_info["position_space"]`, which preprocessing records) decides which one is wired. Wiring one unconditionally would make the pipeline refuse every sample of a dataset stored in the other space.

### How the pipeline avoids conversion churn

With `cache_quats=True` (the default), the pipeline converts your `joint_rot` to quaternions once, runs quaternion-internal steps back to back without leaving quat space, and converts back to your declared representation at the end. This is why the pipeline needs to *know* the representation: **something must declare it** — `AugmentationPipeline(..., representation=...)` or at least one step's kwargs (or pass `cache_quats=False` to run every step exactly as written). It raises otherwise, rather than guessing and corrupting non-quat inputs.

Two guarantees, identical on both paths:

- **Custom steps see your declared representation.** A step function the pipeline doesn't recognize receives `arrays.joint_rot` in the pipeline's current declared representation — never in whatever internal state a previous built-in step left behind. `cache_quats=True` and `cache_quats=False` are bit-identical.
- **Outputs never alias inputs.** Even when no step fires, `pipeline(...)` returns freshly allocated arrays, so nothing it hands back shares storage with a Dataset's cached clips. (The returned container's fields are read-only like any other, so a cache is safe from both directions.) This is the *pipeline's* guarantee: a single augmentation function may return a stream it never touched by reference — `add_root_position_noise` passes `joint_rot` through, and the keypoint-jitter functions pass through the stream they were not pointed at — which read-only fields make safe, since nothing can write through the shared view.

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

Conversion sits at the same level as augmentation and packing, so it takes and returns a `MotionArrays` too — `root_pos` is carried through unchanged, since no rotation representation applies to a translation:

```python
from pybvh_ml import convert_arrays

arrays = MotionArrays.from_bvh(bvh, "euler")
arrays = convert_arrays(arrays, "euler", "6d", euler_orders=bvh.euler_orders)
```

When there is no root stream to carry — a model's rotation output, a cached quaternion array — use the rotation-level form:

```python
from pybvh_ml import convert_rotations

rot6d = convert_rotations(euler_data, from_repr="euler", to_repr="6d",
                          euler_orders=bvh.euler_orders)
rotmat = convert_rotations(quats, from_repr="quat", to_repr="rotmat")
```

Supported by both: `"euler"`, `"quat"`, `"6d"`, `"axisangle"`, `"rotmat"` — any pair. Euler angles are radians, per-joint orders respected.

## A HumanML3D-style frame vector, and what 0.6.0 does not ship

The rotation and position halves now travel together, from one preprocessing run to one flat vector, augmented coherently:

```python
preprocess_directory("dataset/", "train.npz", representation="6d", target_fps=20,
                     include_positions=True, position_centering="skeleton")

data = load_preprocessed("train.npz")
ds = MotionDataset.from_preprocessed(
    data, layout="flat",
    streams=("root_pos", "joint_rot", "joint_pos", "joint_vel"),
    augmentation=AugmentationPipeline.standard(data["skeleton_info"]),
    temporal="crop", target_length=64)
# D = 3 + J*6 + J*3 + J*3, with describe_features(J, "6d", streams=...) naming the blocks
```

`position_centering="skeleton"` is what makes the position block root-relative, HumanML3D's `ric_data` convention; the trajectory then lives only in the `root_pos` columns.

HumanML3D's full 263-dimensional frame vector (4 root + 63 joint positions + 126 rotations as 6D + 66 velocities + 4 foot contacts) is what motivates carrying rotations and positions together, but it is not fully served by this release: **foot contacts and the 4 root channels stay deferred.** `preprocess_directory` can store foot contacts, but as a static feature that augmentation does not refresh. The root channels differ in kind — HumanML3D stores root *velocities* (angular about the up axis, linear in the ground plane) plus root height, in a per-frame body-facing frame; `root_pos` is the raw trajectory, and the gap is a canonicalization rather than a derivative.

**Velocities are served**, by [`joint_vel` / `joint_acc`](tensor-layouts.md#velocity-and-acceleration-joint_vel-joint_acc) — but as *derived* streams, not carried ones. A velocity array looks like another `(F, J, 3)` stream and is not one: under `speed_perturbation_arrays` a carried velocity would have to be *rescaled by the factor*, not merely resampled, and every geometric step would need its own rule. Differencing at packing time instead — after all augmentation, before temporal standardization — makes the rescale fall out of the arithmetic, and is what lets the library get the padding and resampling cases right that a hand-rolled `torch.diff` in a `collate_fn` cannot.

## See also

- [Augmentation API](../api/augmentation.md) / [Pipeline API](../api/pipeline.md) — full signatures
- [Tutorial 2: Augmentation visualized](../tutorials.md) — every function shown before/after on a real skeleton
- [pybvh's transforms](https://victors-67.github.io/pybvh/guide/augmentation/) — the `Bvh`-level equivalents, when you're not in a data loader
