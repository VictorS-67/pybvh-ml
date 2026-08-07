# Preprocessing Pipelines

The "run once" step: batch convert a directory of BVH files into a single on-disk dataset that training jobs load in milliseconds. Preprocessing and runtime are deliberately separate — this page is the disk side; [Runtime Augmentation](augmentation.md) is the every-epoch side.

## One call

```python
from pybvh_ml import preprocess_directory, load_preprocessed

summary = preprocess_directory(
    "dataset/", "train.npz",
    representation="6d",
    parallel=True,         # threaded loading for large directories
    skip_errors=True,      # skip + warn on malformed files
)

data = load_preprocessed("train.npz")
clips = data["clips"]                # list of per-clip dicts
mean, std = data["mean"], data["std"]
skel = data["skeleton_info"]         # edges, lr_pairs, joint_names, world_up, ...
```

Supported representations: `"euler"`, `"quat"`, `"6d"`, `"axisangle"` (validated up front, before any file is parsed). Output formats: `.npz` (always available) and `.hdf5` / `.h5` (requires the [`hdf5` extra](../getting-started/installation.md)); the suffix picks the format and anything else raises immediately.

## What the file stores

The dataset file is self-sufficient — everything needed at training time, no reopening source BVHs:

- **Per-clip arrays**: `root_pos`, `joint_rot`, plus optional positions (below), velocities, foot contacts, quaternions, and labels. Datasets written before 0.5.0 name the rotations `joint_data` on disk; both keys load, and `load_preprocessed` always hands back `joint_rot`.

  Velocities, foot contacts and quaternions are **static features**: unlike `joint_rot` they are not refreshed by augmentation, so use them for evaluation and targets rather than as augmentation-invariant training inputs. `joint_pos` / `node_pos` differ in kind — they are augmentable streams that every geometric step transforms and `add_joint_rotation_noise` re-derives.
- **Skeleton metadata** (`skeleton_info`): joint names, edges, L/R pairs, Euler orders, the node-space equivalents, the FK topology — and the `world_up` / `rest_forward` / `rest_up` axis strings, so runtime augmentation can be configured straight from the loaded dict. Every key is always present: a dataset written before a key existed loads it as `None` rather than omitting it, so there's no `.get()` dance.
- **Normalization statistics**: per-channel `mean` / `std` over all frames, plus a `constant_channels` bool mask (columns whose raw std was below `1e-8`, guarded to `1.0`). Positions get their **own** `position_stats` block rather than widening these — see below.
- **The `center_root` flag**: whether the stored `root_pos` arrays were centered at preprocessing time (default `True`). Files from older versions load with `None` (unknown).
- **The `position_centering` value**: which frame the stored positions are in, or `None` when the dataset carries none.
- **The uniformity audit** (`uniformity`): per-axis and per-frame-rate value counts across the corpus — the *pre-transform* snapshot — plus a record of what was then applied to it: `harmonized_to` (resolved targets, the `retarget` choice, per-stage modification counts) when harmonizing, or `applied_targets` (the `target_*` kwargs this call applied directly) when not. The transformation trail is auditable from the file itself: a corpus resampled to 30 Hz records both the rates it came from and the rate it is now at. Rigs whose rest pose is too degenerate to measure appear under the `rest_up` key `"unknown"`.

## Richer outputs

```python
preprocess_directory(
    "dataset/", "train.npz", representation="6d",
    include_velocities=True,      # per-joint linear velocities (F, J, 3)
    include_foot_contacts=True,   # binary contact labels + foot joint names
    foot_joints=["LeftFoot", "RightFoot"],   # explicit; auto-detected when None
    include_quaternions=True,     # pre-computed quats for runtime augmentation
    label_fn=lambda stem: 0,      # filename stem → integer class
    filter_fn=lambda stem: True,  # skipped files are never parsed
    world_up="auto",              # forwarded to every pybvh.read_bvh_file call
    lr_mapping=None,              # ditto
)
```

Two notes:

- Pass `foot_joints=` explicitly for footless or nonstandard rigs where auto-detection finds nothing.
- For `representation="quat"`, `include_quaternions=True` stores nothing extra — the main `joint_rot` already *is* the quaternion array, and the loader aliases `clip["joint_quats"]` to it instead of duplicating storage.

## Storing positions

Skeleton action recognition consumes joint *positions* almost exclusively. `include_positions=True` stores them alongside (or instead of) the rotations:

```python
preprocess_directory(
    "dataset/", "train.npz", representation="6d",
    include_positions=True,
    position_space="joint",          # or "node" — end sites included
    position_centering="skeleton",   # or "world" / "first"
    center_root=False,
)
```

They come from `bvh.joint_positions()` / `bvh.node_positions()`, both backed by pybvh's cached world-frame FK, so requesting positions alongside a rotation representation costs one array derivation rather than a second kinematics pass.

**The two settings live in different places, deliberately.** `position_space` goes into **`skeleton_info`**: it is a topology fact — which index space, and therefore which `V`, which edge list, which L/R pair list — sitting next to `num_joints` / `num_nodes` / `edges`, exactly as `foot_joints` does. `position_centering` goes into **dataset-level metadata**, next to `center_root`, which is its exact analogue: a statement about the values, not the topology.

Pick the centering deliberately — the three coincide only for a clip whose root never moves:

- **`"world"`** keeps positions in the same frame as `root_pos`, so `rotate_vertical` acts identically on both and a joint position already contains the root trajectory.
- **`"skeleton"`** puts the root at the origin every frame — the form most NTU-style pipelines feed a model, with the trajectory then carried only by `root_pos`.
- **`"first"`** is pybvh's ground-plane centering.

`center_root=True` (the default) combines with `"world"` by applying the identical all-three-component shift to every position vertex, and with `"skeleton"` by leaving the positions alone (they are already root-relative). **`center_root=True` with `"first"` is rejected.** Ground-plane centering subtracts only the two non-up components, so those positions are in a frame offset from `root_pos` and stay offset however the root is centered. A transient container tolerates that — the packers shift both and preserve the relationship — but a *written* dataset must not, because the recorded `center_root=True` would suggest a coherence between the two streams that ground-plane centering never established.

Recording the centering is mandatory for anything this library writes — a stored position array whose frame convention we failed to record is unrecoverable — and `load_preprocessed` surfaces it so it can be threaded onto every `MotionArrays` the Dataset classes mint.

### Position normalization statistics

`position_stats` is a separate `{"mean", "std", "constant_channels"}` block over the `(F, V*3)` flattening, **not** a widened `mean` / `std`. The existing vector's `D = 3 + J*C` layout is a documented public contract matched by `pack_to_flat`, `describe_features` and the HumanML3D `Mean.npy` / `Std.npy` convention; silently changing `D` based on a preprocessing flag would make one file format mean two things.

Ignoring these stats entirely is a legitimate choice: ST-GCN pipelines more commonly root-center or normalize by bone length than z-score raw coordinates.

## Frame rate

Capture rates and training rates rarely match — 120 Hz mocap feeding a model that trains at 30 Hz. `target_fps=` resamples every clip via SLERP before anything is extracted:

```python
preprocess_directory("dataset/", "train.npz", target_fps=30)
```

**Before extraction is the point.** `joint_rot`, `include_velocities` and `include_foot_contacts` are all derived from the resampled clip, so they describe the motion at the target rate. Decimating the finished `.npz` afterwards can't reproduce this — beyond needing a hand-maintained list of which stored keys are frame-indexed, velocities are finite differences whose stencil baseline is the *original* `frame_time`, so a decimated velocity array is simply the wrong number.

A directory with mixed rates warns, the way a mixed up-axis does, and `uniformity["fps"]` records the distribution — the rates the clips *came from*. The rate they are now at is `uniformity["applied_targets"]["target_fps"]` (or `harmonized_to["targets"]["target_fps"]` under `harmonize=True`), so a loader can tell a genuinely mixed-rate dataset from a unified one without reopening any BVH. Under `harmonize=True`, `target_fps` becomes the explicit target and the dataset majority fills in when you don't set one.

## Harmonizing heterogeneous datasets

When clips come from different skeletons, frame rates, up-axis conventions, or — for order-sensitive representations like `"euler"` / `"axisangle"` — different per-joint Euler orders, pass `harmonize=True`:

```python
preprocess_directory(
    "raw/", "train.npz",
    representation="euler",
    harmonize=True,                  # runs pybvh.harmonize after loading
    target_world_up="+y",            # (optional) explicit target; majority otherwise
    skip_errors=True,
)
```

`harmonize=True` runs [`pybvh.harmonize`](https://victors-67.github.io/pybvh/api/batch/), using majority values from the uniformity audit for any `target_*` you didn't set explicitly. For order-sensitive representations it also auto-picks a `target_euler_order` (the most common per-joint order across the dataset).

**Harmonization is pure reorientation/resampling — each actor's bone lengths are preserved.** Pass `retarget=True` to additionally retarget every clip's bone offsets to the first clip, when the whole dataset should share one skeleton geometry (e.g. for fixed-topology GCNs). Hierarchy mismatches raise loudly either way — no silent drops.

!!! warning "Heterogeneity warnings are advice, not errors"
    Without `harmonize=True`, mixed axis conventions produce warnings telling you which values disagree and that `harmonize=True` will unify them to the majority. Heeding them matters: training on clips with mixed up-axes teaches your model that "up" is two different directions.

For workflows that need to inspect or persist the harmonized intermediates, call `pybvh.harmonize` directly, write the results, and preprocess the written directory:

```python
from pybvh import read_bvh_directory, harmonize, write_bvh_file
from pybvh_ml import preprocess_directory
from pathlib import Path

clips = read_bvh_directory("raw/", parallel=True, skip_errors=True)
harmonized = harmonize(clips, reference=clips[0], target_fps=30, target_world_up="+y")

out_dir = Path("harmonized/")
out_dir.mkdir(exist_ok=True)
for b, src in zip(harmonized, clips):
    write_bvh_file(b, out_dir / Path(src.source_path).name)
preprocess_directory(out_dir, "train.npz", representation="6d")
```

## Normalization

Per-channel z-score normalization following the `Mean.npy` / `Std.npy` convention used by HumanML3D and MDM:

```python
from pybvh import read_bvh_directory
from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array

bvhs = read_bvh_directory("dataset/")
stats = compute_normalization_stats(bvhs, representation="6d")
x_norm = normalize_array(x, stats)     # (x - mean) / std
x = denormalize_array(x_norm, stats)   # back to BVH units
```

`compute_normalization_stats` takes a list of `Bvh` objects and computes stats over all frames in the flat `[root_pos, joint_rot]` channel layout (`include_root_pos=False` drops the first 3 columns). Zero-variance channels get their std guarded to `1.0` and flagged in the `constant_channels` mask.

`preprocess_directory` stores the same stats in its output file, so after `load_preprocessed` you can pass the loaded dict straight to `normalize_array` — the direct entry point is for workflows that skip the on-disk artifact. Pass `center_root=True` to reproduce exactly the stats a `preprocess_directory` run stores under its default first-frame root centering.

## See also

- [Preprocessing & Normalization API](../api/preprocessing.md) — full signatures
- [CHANGELOG](https://github.com/VictorS-67/pybvh-ml/blob/main/CHANGELOG.md) — migration notes for the 0.5 `retarget` default change and the normalization-trio move from pybvh
- [Tutorial 3: Heterogeneous preprocessing](../tutorials.md) — mixed skeletons, frame rates, and up-axes as a runnable recipe
