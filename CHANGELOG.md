# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-20

### Breaking changes

- **Augmentation functions and `AugmentationPipeline.__call__` are now
  keyword-only.**  Every public augmentation (`rotate_vertical`, `mirror`,
  `add_joint_noise`, `speed_perturbation_arrays`, `dropout_arrays`) and the
  pipeline's `__call__` require named arguments; positional calls raise
  `TypeError` at invocation.
  - **Why**: `root_pos` and `joint_data` are both ndarrays with a shared
    leading dim `F`, so a swapped positional call silently corrupts rather
    than raising.  Combined with the root-first / joint-first flip below,
    this was a two-foot gun.  Keyword-only makes the swap literally
    unexpressible.
  - **Migration**: name each argument at every call site.
    `rotate_vertical(pos, quats, 45.0, "+y", representation="6d")` →
    `rotate_vertical(root_pos=pos, joint_data=quats, angle_deg=45.0, up_axis="+y", representation="6d")`.
    Same for the other augmentations and for
    `pipeline(root_pos=..., joint_data=..., rng=...)`.
- **Augmentation functions and `AugmentationPipeline` now take and return
  `(root_pos, joint_data)` instead of `(joint_data, root_pos)`.**  Affects
  `rotate_vertical`, `mirror`, `add_joint_noise`,
  `speed_perturbation_arrays`, `dropout_arrays`, and `AugmentationPipeline.__call__`.
  - **Why**: the previous joint-first order was inconsistent with pybvh's
    `Bvh.from_*` / `Bvh.to_*` and with pybvh-ml's own `pack_to_flat` /
    `extract_repr`, which are root-first.  The mismatch forced users to
    mentally swap arguments at every boundary between augmentation and
    packing — a footgun on shape-compatible inputs.
  - **Migration**: swap argument order (and now name them) at every call
    site.  `new_q, new_p = rotate_vertical(q, p, 45, "+y", representation="6d")`
    → `new_p, new_q = rotate_vertical(root_pos=p, joint_data=q, angle_deg=45, up_axis="+y", representation="6d")`.
    Same for pipeline calls.
- **Per-representation augmentation functions unified into one function per
  operation.**  `rotate_quaternions_vertical` / `rotate_rot6d_vertical` →
  `rotate_vertical(..., representation="quaternion" | "6d" | "axisangle" |
  "rotmat" | "euler")`.  Same for `mirror_quaternions` / `mirror_rot6d` →
  `mirror`, and `add_joint_noise_quaternions` → `add_joint_noise`.
  - **Why**: five functions per operation (one per representation) bloated
    the surface and forced users to refactor every call site when switching
    representations.  A single parameterized function covers every supported
    representation and keeps the 6d fast paths internal.
  - **Migration**: rename calls and add `representation=`.
    `rotate_quaternions_vertical(p, q, 45, "+y")` →
    `rotate_vertical(root_pos=p, joint_data=q, angle_deg=45, up_axis="+y", representation="quaternion")`.
- **Vertical/mirror axis arguments are signed-axis strings** instead of
  integer indices.
  - `up_idx: int` (0/1/2) → `up_axis: str` (one of `'+x'`, `'-x'`, `'+y'`, `'-y'`,
    `'+z'`, `'-z'`).
  - `lateral_idx: int` → `lateral_axis: str` (same accepted values; mirror is
    sign-invariant so `'+x'` and `'-x'` produce identical results).
  - **Why**: the previous unsigned-index API silently rotated in the *opposite*
    direction on `-y` / `-z` up skeletons — same latent bug pybvh fixed in its
    own `rotate_vertical` in 0.6.0.  Typical call site becomes
    `up_axis=bvh.world_up`, which is correct by construction.
  - **Migration**: `up_idx=1` → `up_axis="+y"`, `lateral_idx=0` → `lateral_axis="+x"`.
    For dynamic selection, build the string from the signed axis: `up_axis=bvh.world_up`.
    Invalid strings raise `ValueError` — no silent mis-rotation.
- **`MotionDataset.use_quats_for_augmentation` is removed.** The flag silently
  discarded its own output — augmented quaternions were never re-packed, so
  only the side-effect on `root_pos` survived.  Users who need quat-space
  augmentation on a non-quat primary representation should convert inside the
  `AugmentationPipeline` themselves (via `convert_arrays`), or preprocess with
  `representation="quaternion"`.
- **`standardize_length(method="resample")` is now `method="resample_linear"`.**
  The old name emitted a warning on every call for a decision the caller should
  make at authoring time.  The runtime warning is gone; the new name is
  explicit about the linear-interp semantics.  Still not correct for rotation
  arrays — resample rotations with `pybvh.Bvh.resample()` (SLERP) before extraction.
- **Per-epoch augmentation requires `dataset.set_epoch(epoch)`.** The previous
  `seed + idx` composition produced the same augmentation every epoch,
  defeating the purpose of augmentation.  The new scheme composes
  `(seed, epoch, idx)` through `numpy.random.SeedSequence`.  Call
  `dataset.set_epoch(epoch)` at the top of each epoch — same contract as
  `torch.utils.data.distributed.DistributedSampler`.  With `seed=None`, each
  `__getitem__` uses fresh OS entropy (simplest; no reproducibility).
- **`preprocess_directory(require_matching_topology=True)` is the new default.**
  Mixing skeletons in one output silently produced garbage downstream.  Every
  loaded clip must now match the first clip's `joint_names` and `euler_orders`;
  otherwise `ValueError` points at `pybvh.harmonize()`.  Pass
  `require_matching_topology=False` for the lenient pre-0.3 behaviour.
- **Bumped pybvh floor to `>=0.6.0`.** All deprecated `bvh.get_frames_as_*` /
  `bvh.set_frames_from_*` calls have been migrated to the new `to_*` / `from_*`
  names.  `bvh.to_{quaternions,6d,axisangle,rotmat}` now return a 2-tuple
  (pybvh 0.6.0 change); if your own code still unpacks a 3-tuple
  (`_, data, _ = bvh.to_6d()`), switch to `_, data = bvh.to_6d()`.  If your
  pipeline imports from pybvh directly, migrate at the same time to avoid
  `DeprecationWarning` noise.

### Added

- **`AugmentationPipeline.standard(skeleton_info, ...)` classmethod.**  Opinionated
  factory that builds the canonical `rotate + mirror + noise + speed` pipeline
  from a `skeleton_info` dict (as returned by `get_skeleton_info` or
  `load_preprocessed`), replacing the ~20-line boilerplate every downstream
  project was writing.  Each step is optional: pass `None` (or `0` for
  `mirror_prob`) to skip it.  For anything beyond what these kwargs expose,
  build the pipeline directly with the `(fn, prob, kwargs)` constructor — this
  factory is the blessed common case, not a wrapper around every knob.
- **`MotionDataset` and `OnTheFlyDataset` warn once** when `seed` is set but
  `set_epoch(epoch)` was never called.  Without a seed change per epoch the
  pipeline produces identical augmentation every epoch — a quiet correctness
  bug that surfaces as a flat validation curve.  The warning fires on the
  first `__getitem__` call of a seeded dataset; call `set_epoch(0)` at the
  top of training to acknowledge the contract even when `epoch=0`, or pass
  `seed=None` for fresh OS entropy each call.
- **`AugmentationPipeline` quaternion cache (`cache_quats=True` default).**
  Shares one quaternion conversion across compatible built-in augmentations
  instead of each step independently calling `_to_quats` / `_from_quats`.
  Measured on 20 clips × full-length real clips across three datasets:
  - 6d representation: **~1.5×** speedup
  - axisangle / euler: **~3×** speedup
  - quaternion: neutral (nothing to cache)
  User-defined augmentations not registered in the internal dispatch table
  are supported transparently — the pipeline flushes the cache, converts
  joint data back to the step's declared representation, calls the function
  normally, and resumes staging cold afterward.  Set `cache_quats=False` for
  historical bit-exact behavior.
- **`preprocess_directory` gained `target_world_up`, `target_rest_forward`, and
  `target_rest_up` kwargs** for harmonizing heterogeneous datasets before
  the stats + topology check.  Each defaults to `None` (no reorientation).
  When set, every loaded clip is passed through the corresponding
  `bvh.reorient_world_up` / `reorient_rest_forward` / `reorient_rest_up` so
  downstream extraction sees consistent axes.  FK positions are preserved;
  joint-angle numbers change.
- **Aggregated `UserWarning` for heterogeneous datasets.**  After loading,
  `preprocess_directory` inspects every clip's animation `world_up`, rest-
  pose forward axis, and rest-pose up axis.  If files disagree on any,
  one summary warning per category is emitted with the distribution,
  first 3 example filenames per minority value, and the kwarg that would
  fix it.  An additional aggregated warning fires when any file's rest-
  pose up disagrees with its own animation-derived `world_up` — the
  condition `target_rest_up` repairs, which silently corrupts every
  rotation representation (not just Euler) otherwise.  The per-file pybvh
  `"Rest pose suggests world up…"` warning is suppressed during load (via
  `read_bvh_file(..., warn_on_world_up_disagreement=False)`) in favor of
  this batch-level message.  Suppressed per-category when the corresponding
  `target_*` kwarg is explicitly set.
- **`uniformity` key in the `preprocess_directory` return dict.**  Maps
  `{"world_up": {axis: [stems, …]}, "rest_forward": {…}, "rest_up": {…},
  "rest_anim_mismatch": [stems, …]}` — a machine-readable snapshot of
  the pre-reorient state, useful for CI gates that want to fail on
  cross-file heterogeneity.  `rest_anim_mismatch` lists files whose
  rest-pose up disagrees with their animation-derived `world_up`.
- **`include_velocities` / `include_foot_contacts` kwargs on `preprocess_directory`.**
  Computes per-joint linear velocities (via `bvh.joint_velocities()` with
  central-stencil edge-padded defaults) and binary foot-contact labels (via
  `bvh.foot_contacts()` with `method="combined"`) per clip, stored alongside
  the primary arrays.  Static features — **not** refreshed after augmentation,
  so use for evaluation / targets, not as augmentation-invariant training
  inputs.  `skeleton_info["foot_joints"]` records the detected foot joint names.
- **`parallel=` / `max_workers=` kwargs on `preprocess_directory`.** Threaded
  BVH loading via `ThreadPoolExecutor`; speeds up large-directory
  preprocessing where I/O dominates.
- **`skip_errors=` / `world_up=` / `lr_mapping=` / `filter_fn=` kwargs on
  `preprocess_directory`.** `skip_errors` / `world_up` / `lr_mapping` are
  pass-throughs to `pybvh.read_bvh_file` (`skip_errors` emits `UserWarning`
  per skipped file and continues).  `filter_fn(filename_stem) -> bool`
  excludes files before load, saving I/O and memory when preprocessing a
  subset of a large directory.
- **`set_epoch(epoch)` on `MotionDataset` and `OnTheFlyDataset`.** Required
  for reproducible per-epoch augmentation (see Breaking changes).
- **`extract_repr(bvh, representation)` as a public function** in
  `pybvh_ml.preprocessing`.  Replaces the cross-module private import
  `_extract_repr` used by `OnTheFlyDataset`.
- **`lr_mapping` entry in `get_skeleton_info(bvh)` output.** Mirrors
  `bvh.lr_mapping` (name-keyed dict or `None`).
- **`load_preprocessed` output dict now includes `constant_channels`** when
  the file was written by pybvh-ml ≥ 0.3.  Absent for older files.
- **`py.typed` marker** ships with the package; downstream mypy users now
  see the type annotations.
- **README sections** on harmonizing heterogeneous datasets and per-epoch
  reproducible augmentation.
- **`AugmentationPipeline` docstring note on composition hazards** (mirror ∘
  rotate sign flip, speed perturbation changes `F`, noise order semantics).
- **Runnable tutorial notebooks** under `tutorials/`: end-to-end pipeline,
  augmentation visualized, and heterogeneous preprocessing.  Exercised in CI
  via `pytest --nbmake`.

### Changed

- **Rest-pose axis detection in `preprocess_directory` uses pybvh's
  public `Bvh.rest_up` and `Bvh.rest_forward` properties** instead of
  the private `pybvh.tools._rest_upward` /
  `rest_pose_coords + forward_at(coords=...)` round-trip.  Both
  accessors were added upstream in response to feedback requests from
  this project.
- **`pybvh.harmonize` now uses `target_world_up` (renamed from
  `target_up`) and gains `target_rest_forward` / `target_rest_up`
  kwargs**, aligning its reorient surface with
  `preprocess_directory`'s three `target_*` kwargs.
- **`pybvh_ml.metadata.REPR_CHANNELS` is re-exported from
  `pybvh.rotations.REPRESENTATION_CHANNELS`.** pybvh promoted this constant
  to its public surface; the local copy is dropped.  `pybvh_ml.convert`
  already re-used `metadata.REPR_CHANNELS` and continues to do so.
- **`pybvh_ml.convert.convert_arrays` is a thin wrapper over
  `pybvh.rotations.convert`** added in pybvh 0.6.0.  The internal
  `_euler_to_rotmat_per_joint` / `_rotmat_to_euler_per_joint` helpers are
  gone — per-joint Euler orders are now handled natively by
  `rotations.euler_to_rotmat(angles, ['ZYX', ...])`.
- **`get_edge_list(bvh, include_end_sites=True)` uses `bvh.node_edges`**
  (new pybvh 0.6.0 property) instead of traversing `bvh.nodes` manually.
- **`OnTheFlyDataset.__getitem__` imports `extract_repr` at module scope**
  instead of inside the hot path.

### Fixed

- **`AugmentationPipeline` auto-forwards `rng`** to augmentation functions
  that accept it (via signature inspection).  Previously, functions like
  `add_joint_noise` and `dropout_arrays` received `rng=None` and created
  unseeded generators, breaking reproducibility.  User-provided `rng`
  kwargs still take precedence.
- **`representation="euler"` augmentation path no longer raises.**  The
  internal `_to_quats` / `_from_quats` helpers passed `euler_orders=…` to
  `pybvh.rotations.convert`, but the pybvh API is `order=…`.  Every
  euler-path call through `add_joint_noise`, `speed_perturbation_arrays`,
  `dropout_arrays`, `rotate_vertical` (non-6d), or `mirror` (non-6d) was
  silently broken.  Fix also threads `degrees=True`, matching the existing
  `convert_arrays` behavior.
- **`preprocess_directory` double-call bug.** The per-representation
  extraction was a dict of lambdas, each invoking `bvh.to_*()` twice (once
  for `[0]`, once for `[1]`).  Forward kinematics + rotmat conversion
  therefore ran twice per clip.  Now runs once; measurable throughput
  improvement on non-Euler representations, larger when
  `include_quaternions=True`.
- **`preprocess_directory(include_quaternions=True)` also shares one FK
  pass.** When the primary representation is `"6d"` or `"axisangle"` (both
  of which pivot through rotmat), we now call `bvh.to_rotmat()` once and
  derive both the primary and the quaternion secondary from the shared
  rotation matrix via `rotations.rotmat_to_rot6d` / `rotations.rotmat_to_quat`.
  Halves the FK + rotmat work on that hot path.
- **`get_lr_pairs(bvh)` now just returns `list(bvh.lr_pairs)`.** Uses the
  cached index-space property added in pybvh 0.6.0.

## [0.2.0] - 2026-03-31

### Added

- **`uniform_temporal_sample`** — PySKL-style uniform segment sampling with three regimes (short/wrapping, dense/gap-insertion, uniform/segment-based); train mode with random offsets, test mode deterministic
- **`sample_temporal`** — convenience wrapper that applies sampled indices to an array with wraparound and multi-sample support
- **`add_joint_noise_quaternions`** — Gaussian rotation noise on `(F,J,4)` quaternion arrays via random noise generated as axis-angle perturbations and converted to quaternions; optional root position noise
- **Callable kwargs in `AugmentationPipeline`** — kwargs values can now be callables of the form `lambda rng: value`, resolved per invocation for random parameter sampling (e.g., random rotation angles, random speed factors)

## [0.1.0] - 2026-03-31

### Added

- **Tensor packing** — `pack_to_ctv`, `pack_to_tvc`, `pack_to_flat` and round-trip `unpack_from_*` inverses; root position as vertex 0, zero-padded to match joint channel count
- **Quaternion augmentation** — `rotate_quaternions_vertical`, `mirror_quaternions`, `speed_perturbation_arrays`, `dropout_arrays` operating on `(F,J,4)` arrays with SLERP interpolation
- **6D augmentation** — `rotate_rot6d_vertical`, `mirror_rot6d` operating directly on `(F,J,6)` arrays, avoiding the quaternion round-trip in hot paths
- **`AugmentationPipeline`** — composable augmentation sequence with per-step probabilities and seeded rng
- **`convert_arrays`** — convert between euler, quaternion, 6D, axis-angle, and rotation matrices on `(F,J,C)` arrays, with per-joint Euler order support
- **Preprocessing** — `preprocess_directory` to batch convert BVH directories to npz or hdf5 datasets with normalization stats; `load_preprocessed` to read them back
- **Skeleton graph metadata** — `get_edge_list`, `get_lr_pairs`, `get_body_partitions`, `get_skeleton_info` for GCN and Transformer models
- **Sequence utilities** — `sliding_window` for fixed-length windowing; `standardize_length` with pad, crop, and resample modes
- **Feature metadata** — `FeatureDescriptor` and `describe_features` for programmatic access to packed array columns
- **PyTorch integration** (optional, `pip install "pybvh-ml[torch]"`) — `MotionDataset`, `OnTheFlyDataset`, `collate_motion_batch` for variable-length batching with padding and masks
