# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-07-07

Coordinated release with pybvh 0.8.0 (atomic migration, no shims), in two layers. First, the migration itself: pybvh-ml adopts pybvh's short representation tokens and radians-first API in the same release and absorbs the normalization trio pybvh removed. Second, a full-codebase audit fix-up: euler joint data is now genuinely radians end to end (the conversion sites silently spoke degrees — shipped-0.4.0 corruption), rotmat augmentation works, the two pipeline dispatch paths agree for custom steps, `harmonize=True` stops silently retargeting bone offsets (new `retarget` opt-in), `set_epoch` reaches DataLoader workers via shared memory, and a wide set of fail-loud validation, collate/packing/sequence fixes, and dataset-metadata additions (persisted uniformity audit, skeleton axis strings) land alongside.

### Breaking changes & migration at a glance

| Change | Migration |
|---|---|
| Requires **pybvh >= 0.8, < 0.9** | Upgrade pybvh first; pybvh 0.8.0 ships its own consolidated migration table in its CHANGELOG. |
| Representation token `"quaternion"` → `"quat"` | Pass `representation="quat"` at every call site; stored dataset metadata now reads back as `"quat"`. |
| `bvh.to_quaternions()` / `bvh.from_quaternions()` → `bvh.to_quat()` / `bvh.from_quat()` (via pybvh 0.8.0) | Rename the method calls in your own pipeline code — no dual names. |
| Angles are radians: `rotate_vertical(angle_deg=…)` → `angle=…`, `add_joint_noise(sigma_deg=…)` → `sigma=…`, `AugmentationPipeline.standard(noise_sigma_deg=…)` → `noise_sigma=…` | Rename the kwarg and wrap old degree values in `np.radians(...)`; `rotate_angle_range` defaults to `(-np.pi, np.pi)`. |
| Dataset `length` / batch `lengths` now mean valid frames in the returned tensor | Cropped clips report `target_length`, not the original clip length; recover original lengths from your clip arrays before the dataset if you need them. |
| Invalid augmentation inputs raise `ValueError` (negative `sigma` / `sigma_pos`; `drop_rate` outside `[0, 1)`; mismatched `root_pos` / `joint_data` frame counts; empty joint arrays in `rotate_vertical`; zero-norm quaternions in `add_joint_noise`) | Fix the offending values — they were silent no-ops or opaque errors before. |
| `AugmentationPipeline(cache_quats=True)` (the default) requires a `representation=` declaration | Declare `representation=...` on at least one step, or pass `cache_quats=False`. |
| `harmonize=True` no longer retargets bone offsets to the first clip | Per-actor bone proportions are preserved by default; pass `retarget=True` to restore the 0.4.0 retarget-to-first behavior. |
| Normalization trio moved here from pybvh | `from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array` (was `from pybvh import ...`). |
| `center_root` recorded in preprocessed dataset metadata | New `.npz` key / HDF5 root attribute; older files load with `center_root=None`. Preprocessed arrays saved with `center_root=True` are already centered — don't re-center via `pack_to_*(center_root=True)`. |

### Breaking changes

- **Minimum pybvh version: `>=0.8,<0.9`.** pybvh 0.8.0 renames the representation surface; pybvh-ml adopts the new names atomically with no dual-name support.
- **Representation string `"quaternion"` → `"quat"`** everywhere a representation token is accepted or reported: `extract_repr`, `describe_features`, `convert_arrays`, all augmentation functions and `AugmentationPipeline` step kwargs, and `preprocess_directory(representation=...)`. Follows pybvh 0.8.0's short-token convention (`quat` / `6d` / `axisangle` / `rotmat` / `euler`), including the keys of `pybvh.rotations.REPRESENTATION_CHANNELS`, which pybvh-ml re-exports as `REPR_CHANNELS`.
  - **Migration**: replace `representation="quaternion"` with `representation="quat"` at every call site; datasets preprocessed with the old token name need their stored `representation` metadata read as `"quat"` going forward (re-preprocess or rename the metadata field value).
- **`bvh.to_quaternions()` / `bvh.from_quaternions()` call sites migrated to `bvh.to_quat()` / `bvh.from_quat()`** (silent breaking via pybvh 0.8.0). Affects any user code calling these pybvh methods around pybvh-ml pipelines; pybvh-ml's own internals are updated.
- **`pybvh_ml.augmentation` no longer imports `pybvh.tools.rotX/rotY/rotZ`** (removed in pybvh 0.8.0). The internal cardinal-axis rotation matrix is now built via the public `pybvh.rotations.quat_to_rotmat`; results are identical up to float rounding (~1e-16).
- **Radians everywhere in augmentation parameters.** `rotate_vertical(angle_deg=...)` → `rotate_vertical(angle=...)` and `add_joint_noise(sigma_deg=...)` → `add_joint_noise(sigma=...)`, both in radians; `AugmentationPipeline.standard(noise_sigma_deg=...)` → `noise_sigma=` (radians, default one degree) and `rotate_angle_range` defaults to `(-np.pi, np.pi)`. `sigma_pos` is in positional units and is unchanged. Matches pybvh 0.8.0's radians-first API — the pybvh ecosystem now speaks radians end to end.
  - **Migration**: rename the kwarg and wrap old degree values in `np.radians(...)` — `rotate_vertical(..., angle_deg=90)` becomes `rotate_vertical(..., angle=np.radians(90))`; same for `sigma` and the `standard()` kwargs.
- **Invalid augmentation inputs now raise `ValueError` instead of silently misbehaving.** `add_joint_noise` rejects negative `sigma` / `sigma_pos` and zero-norm input quaternions (previously silent NaN output); `dropout_arrays` rejects `drop_rate` outside the documented `[0, 1)`; all five augmentation functions (and the pipeline) reject `root_pos` / `joint_data` arrays that disagree on frame count (previously silently interpolated or indexed the wrong frames); `rotate_vertical` rejects empty joint arrays (previously an opaque `IndexError`). All checks apply to both the public functions and the staged pipeline variants. Previously a negative `sigma_pos` or `drop_rate` was a silent no-op and a negative `sigma` surfaced as an opaque NumPy error.
- **`harmonize=True` no longer retargets bone offsets to the first clip by default.** Previously the alphabetically first file was silently pinned as the harmonize reference, overwriting every actor's bone proportions with the first actor's (and FK-derived `include_velocities` / `include_foot_contacts` features were computed on the retargeted skeletons). Harmonization is now pure reorientation/resampling; the new `retarget=True` opt-in restores reference pinning for datasets that should share one skeleton geometry. Hierarchy mismatches still raise either way — from the skeleton-compatibility check by default, or from the harmonize drop report under `retarget=True`.
- **`AugmentationPipeline` with `cache_quats=True` (the default) raises `ValueError` when no step declares a `representation` kwarg** instead of silently assuming `"quat"` — the quat-caching path cannot know what representation `joint_data` is in, and guessing would corrupt non-quat inputs run through representation-less custom steps. Declare `representation=...` on at least one step, or pass `cache_quats=False`. Empty pipelines remain valid no-ops.

### Added

- **Normalization trio moved here from pybvh** — `compute_normalization_stats`, `normalize_array`, and `denormalize_array` are now pybvh-ml public API (pybvh 0.8.0 removed them from `pybvh.batch`; dataset z-score normalization is an ML-pipeline concern). Same signatures and `Mean.npy` / `Std.npy`-compatible output as the pybvh originals: `{"mean", "std", "constant_channels"}` with zero-std channels guarded to `1.0`. The Bvh-list entry point extracts through pybvh-ml's own `extract_repr` and uses the representation-aware loose skeleton check (bone-length variation across actors is accepted), so supported representations are `"euler"` / `"quat"` / `"6d"` / `"axisangle"` — `"rotmat"` is not part of pybvh-ml's extraction surface. `preprocess_directory`'s stored stats share the same array-level core.
  - **Migration**: one line — `from pybvh import compute_normalization_stats, normalize_array, denormalize_array` becomes `from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array`.
- **`preprocess_directory(retarget=...)`** — opt-in bone-offset retargeting under `harmonize=True` (see the breaking entry above). The choice and the pinned reference stem are recorded in `uniformity["harmonized_to"]`.
- **`preprocess_directory(foot_joints=...)`** — explicit foot joint names for contact detection, bypassing auto-detection. Required for footless or nonstandard rigs, where auto-detection finds nothing and contact extraction previously had no recovery path through this API.
- **`compute_normalization_stats(center_root=...)`** — centers each clip's root before the stats pass, reproducing exactly the `mean` / `std` that `preprocess_directory` stores under its default `center_root=True`. Previously the two could never agree on the root channels.
- **`get_skeleton_info` carries `world_up`, `rest_forward`, and `rest_up`.** The saved `skeleton_info` now includes the axis strings runtime augmentation needs (`up_axis` for `rotate_vertical` / `AugmentationPipeline.standard`), so a preprocessed dataset is self-sufficient — no reopening source BVHs to configure augmentation.
- **`__all__` on `pybvh_ml` and `pybvh_ml.torch`** — one machine-checkable statement of the public API; `from pybvh_ml import *` no longer leaks submodule names.
- **Dataset ergonomics.** Both Dataset classes support Python negative indexing (`ds[-1]` matches `ds[len(ds)-1]` on the same rng stream — previously it crashed `SeedSequence`, but only when seeded augmentation was on) and raise a clean `IndexError` out of range. `OnTheFlyDataset` coerces `str` paths to `Path` (a `str` entry used to crash only at `label_fn` time) and gains `world_up=` / `lr_mapping=` passthrough to `pybvh.read_bvh_file`, matching `preprocess_directory`. `MotionDataset` gains `center_root=False` for hand-built raw clip dicts, mirroring `OnTheFlyDataset` (clips from `load_preprocessed` are typically already centered — check the loaded `center_root` metadata).
- **The uniformity audit is persisted in the saved dataset.** `preprocess_directory` writes the axis-uniformity audit — including `harmonized_to` (resolved targets, `retarget` choice, per-stage counts, and the full JSON-native `HarmonizeReport`) when harmonizing — as `uniformity_json` in both output formats, and `load_preprocessed` surfaces it under `"uniformity"` (`None` for files written by older versions). The transformation trail is auditable from the dataset file itself.
- **`center_root` recorded in preprocessed dataset metadata.** `preprocess_directory` now writes the flag into both output formats (`.npz` key / HDF5 root attribute `center_root`) and `load_preprocessed` surfaces it in the result dict, so downstream code can tell whether the stored `root_pos` arrays are already centered instead of silently centering them a second time through `pack_to_*(center_root=True)`. Files written by older pybvh-ml versions load with `center_root=None` (unknown). The `pack_to_*` docstrings now state that their flag is for standalone packing of raw extractions — preprocessed arrays saved with `center_root=True` are already centered.
- **Documentation site** at <https://victors-67.github.io/pybvh-ml/> — MkDocs Material + mkdocstrings, the same stack as pybvh's docs: quick start, a user guide (tensor layouts, preprocessing, augmentation, PyTorch integration, skeleton graphs), and a full API reference generated from the docstrings. Deployed to GitHub Pages on push to `main`, with `mkdocs build --strict` gating broken links; a new `docs` extra installs the toolchain. The README slims to a landing page pointing at the site; this CHANGELOG remains the sole migration record.

### Changed

- **Quaternion Hamilton product now comes from pybvh.** The private `_quat_multiply` in `augmentation.py` was deleted in favor of `pybvh.rotations.quat_multiply` (new in pybvh 0.8.0) — verified bit-identical on batched and broadcast inputs (same Hamilton convention, wxyz scalar-first order).
- **`convert_arrays` imports `REPRESENTATION_CHANNELS` directly from `pybvh.rotations`** instead of through `pybvh_ml.metadata`; the `REPR_CHANNELS` re-export in `pybvh_ml.metadata` is unchanged.
- **Both `AugmentationPipeline` dispatch paths use the same probability-check direction** (`draw < prob` applies the step); the staged path previously spelled it as `draw >= prob: continue`. The two forms are exact complements — no behavioral change.
- **Quat datasets no longer store duplicate quaternion arrays.** With `representation="quat"` and `include_quaternions=True`, the primary `joint_data` already is the quaternion array; the on-disk duplicate `joint_quats` copy is gone (~50% smaller quat datasets) and `load_preprocessed` aliases `clip["joint_quats"]` to `clip["joint_data"]` instead. Files written by older versions still load unchanged.

### Fixed

- **`collate_motion_batch` validity masks were wrong for cropped clips — `length` now means valid frames in the returned tensor.** `MotionDataset` / `OnTheFlyDataset` `__getitem__` used to report the pre-standardization clip length, so whenever `target_length` *cropped* a clip (e.g. 150 frames → 100), the collate mask was built from a length larger than the tensor and silently clamped to all-True — correct by accident for cropping, but the batch `lengths` tensor still carried the stale 150. `length` is now `min(original_length, target_length)`: the number of valid frames actually present in the returned tensor (padded frames excluded, cropped clips report `target_length`). User-visible semantics change for code that read `length` / `lengths` as the original clip length — recover the original length from your clip arrays before the dataset if you need it.
- **`collate_motion_batch` raises on mixed label presence.** A batch where some items carry `label` and others don't used to either silently drop all labels or die in a `KeyError`, depending on which item came first; it now raises a `ValueError` naming the first offending batch index.
- **`pybvh_ml.torch` no longer masks broken torch installations.** The import guard now distinguishes "torch not installed" (clear install hint) from "torch present but failing to import" (e.g. a missing CUDA library), which previously had its real traceback rewritten into a misleading "install torch" message.
- **`set_epoch()` had no effect on DataLoader worker processes.** The epoch was a plain attribute, so workers kept the copy pickled at startup: with `persistent_workers=True` every epoch silently replayed the epoch-0 augmentation — and the missing-`set_epoch` warning machinery was pickled as "already emitted", so nothing flagged it. The epoch now lives in a shared-memory `multiprocessing.Value` that workers inherit at creation (fork and spawn both), making the documented DistributedSampler-style contract hold in every DataLoader configuration; a `num_workers=2, persistent_workers=True` regression test asserts bit-equality with the single-process reference per epoch. `set_epoch` now also rejects negative epochs (they would crash `SeedSequence` mid-training). Trade-off, documented on both classes: dataset instances can no longer be `deepcopy`-ed or `torch.save`-ed directly (shared state travels only via process inheritance).
- **`sample_temporal(mode="test", num_samples=N)` returned N identical clips.** The internal rng was re-seeded to 0 on every draw, so "multiple independent samples" were bit-equal copies; the rng is now created once and threaded through every draw — test mode yields distinct samples that are still deterministic across calls. Relatedly, `uniform_temporal_sample` no longer silently ignores a caller-supplied `rng` in test mode (`rng=None` keeps the fixed default, so existing outputs are unchanged).
- **`standardize_length` rejects `target_length < 1`.** A negative target used to silently return a wrong-length array via Python negative-slice truncation (`target_length=-3` on 10 frames returned 7).
- **`unpack_from_flat` validates channel divisibility.** Unpacking quat-packed data (D = 3 + J·4) with the default `joint_channels=3` used to mis-reshape into garbage or die in a raw NumPy reshape error; a clean `ValueError` now names `joint_channels` and the residual column count.
- **`pack_to_ctv` returns a C-contiguous array** instead of a transpose view — `torch.from_numpy(...).view(...)` and other C-contiguity assumptions no longer trip on it.
- **HDF5 export crashed on non-ASCII filenames.** The `filenames` dataset was written with a fixed ASCII bytes dtype (`"S"`), so any non-ASCII clip stem raised `UnicodeEncodeError` at the final save step. Stems are now stored as variable-length UTF-8 strings; loading is unchanged.
- **Unrecognized dataset extensions raise up front instead of writing to a different path.** A typo'd output suffix (e.g. `.dat`) used to fall through to `np.savez`, which silently appends `.npz` — the dataset landed at a path the caller never named and `load_preprocessed` on the original path failed with `FileNotFoundError`. `preprocess_directory` and `load_preprocessed` now validate the extension (`.npz` / `.hdf5` / `.h5`) before any file I/O, and `representation` is validated before the directory is parsed rather than after the full load.
- **Degenerate rigs no longer break harmonize target resolution.** `Bvh.rest_up` is `None` for degenerate skeletons; a `None` key in the uniformity audit used to crash the majority tie-break with a `TypeError` (or, worse, resolve a `None` reorientation target). `None` values are now excluded from majority resolution while staying visible in the returned `uniformity` audit.
- **`mirror` with `representation="euler"` corrupted L/R pairs whose members use different Euler orders.** The left/right swap ran on raw euler triples before the order-aware conversion, so a left joint's angles were decoded with the right joint's Euler order (and vice versa) whenever a pair mixed orders. The convert-through-quats branch now converts first and swaps in quaternion space — matching what the quat-caching pipeline variant always did. Uniform-order datasets are unaffected (the two orderings are equivalent there). Present in shipped 0.4.0.
- **The quat-caching pipeline handed representation-less custom steps quaternion-space data.** With `cache_quats=True` (the default), a user-defined step whose kwargs carried no `representation` received `joint_data` in whatever internal representation the previous built-in step left behind — quaternions after noise/speed/dropout — while `cache_quats=False` handed it the pipeline's declared representation. A shape-agnostic custom step silently corrupted data and the two dispatch paths diverged. Unknown steps now always receive the pipeline's current declared representation (the most recent step carrying a `representation` kwarg), matching the direct path bit for bit.
- **`representation="rotmat"` crashed in every augmentation function and the quat-caching pipeline.** pybvh-ml carries rotation matrices flat as `(F, J, 9)` (the layout `convert_arrays` documents and produces), but the augmentation conversion helpers passed that array straight to `pybvh.rotations.convert`, which expects `(..., 3, 3)` — every advertised rotmat augmentation raised a reshape `ValueError`. The flat↔3×3 adaptation now happens at the conversion boundary, exactly as `convert_arrays` already did, and the staged pipeline state routes through the same shared helpers (one conversion implementation instead of two). Present in shipped 0.4.0.
- **Euler joint data was silently treated as degrees while pybvh stores radians.** All euler↔quaternion conversion sites (the augmentation helpers, the quat-caching pipeline state, and `convert_arrays`) passed `degrees=True` to `pybvh.rotations.convert`, but `Bvh.joint_angles` — what `extract_repr(bvh, "euler")` hands out — has been radians since pybvh 0.7.0. Every augmentation with `representation="euler"` and every euler-side `convert_arrays` call shrank rotations ~57× on the way into quaternion space and re-inflated them on the way out: `rotate_vertical` corrupted the root joint, `add_joint_noise` amplified its perturbations ~57×, and euler→quat/6d/axisangle/rotmat conversions produced wrong rotations. Present in shipped 0.4.0. The bug was invisible to self-inverse round-trip tests (a consistent unit error cancels); the suite now compares euler paths against pybvh ground truth (`bvh.to_quat()`) for all five augmentation functions, the staged pipeline, and `convert_arrays`.
- **The sdist no longer ships a partial, non-runnable `tests/` directory.** The 0.4.0 sdist carried two test files without the fixtures or data they need to run (stale build metadata swept them in). A `MANIFEST.in` now prunes the test, tutorial, data, and docs trees, so the sdist holds exactly the package sources; the wheel was always clean.
- **`AugmentationPipeline` outputs never alias the inputs.** The staged path (`cache_quats=True`, the default) could hand back the caller's own `root_pos` / `joint_data` arrays when no step fired and no representation change ran, and the staged `add_joint_noise` returned the caller's `root_pos` unaliased when `sigma_pos == 0` — in both cases later in-place edits on the "augmented" output would corrupt the caller's (e.g. a Dataset's cached) arrays. Both paths of `pipeline(...)` now guarantee freshly allocated outputs.

### Documentation

- `packing._center` docstring now states explicitly that pybvh-ml's `center_root` subtracts the full 3D first-frame root position — distinct from pybvh 0.8.0's `centered="first"`, which is ground-plane-only.
- `pybvh_ml.packing` docstrings clarify the root zero-padding: in `(C, T, V)` / `(T, V, C)` layouts the root vertex's position fills channels `0:3`, and when the joint representation has more than 3 channels the root's channels `3:C` are zero padding — the position values themselves are unchanged.

## [0.4.0] - 2026-05-14

Heterogeneous-dataset support pass driven by an external maintainer report
on `preprocess_directory`'s UX gaps. Pairs with pybvh 0.7.0 — minimum version
bumped accordingly.

### Breaking changes

- **`require_matching_topology` kwarg removed from `preprocess_directory`.**
  - **Why**: the flag advertised a fallback that didn't exist. Setting it
    to False let the per-clip topology check skip, but the downstream
    `compute_normalization_stats` (which routes through pybvh's
    `batch_to_numpy`) re-ran the same check and raised anyway. The "lenient
    pre-0.3 behavior" the docstring promised was dead code — the only
    observable effect of `False` was a less informative error from a deeper
    stack frame.
  - **Migration**: drop the kwarg. For heterogeneous datasets, pass
    `harmonize=True` instead (see below); for incompatible hierarchies
    that need bone retargeting, pre-call `pybvh.harmonize(reference=...)`.
- **Compatibility check is now representation-aware.** Clips that share
  joint hierarchy but disagree on per-joint Euler orders are accepted for
  rotation-invariant representations (`"6d"` / `"quaternion"` / `"rotmat"`),
  where the saved tensor's channel layout is order-agnostic. Order-sensitive
  representations (`"euler"` / `"axisangle"`) still require channel equality.
  - **Why**: the previous uniform-check rejected legitimately compatible
    datasets (same skeleton, half the clips in XZY, half in ZYX) for users
    on 6D / quaternion representations, who'd get the strictness of the
    least-flexible representation regardless of what they were extracting.
  - **Migration**: most users are net-better off — fewer false rejections.
    Datasets that previously raised on Euler-order mismatch under `"6d"` etc.
    now succeed. For `"euler"` / `"axisangle"` users, the recovery is
    `harmonize=True` or an explicit `target_euler_order=` on
    `pybvh.harmonize` before preprocessing.
- **Minimum pybvh version: 0.7.0.** Required for `Bvh.matches_hierarchy`,
  `Bvh.matches_channels`, `Bvh.source_path`, and
  `pybvh.harmonize(target_euler_order=..., return_report=True)`. Older pybvh
  installs will fail at import.
- **`bvh.joint_angles` is radians-native (silent breaking via pybvh 0.7.0).**
  pybvh 0.7.0 moved the deg↔rad boundary entirely to I/O: `read_bvh_file`
  converts deg→rad on read and `write_bvh_file` converts rad→deg on write,
  so every value in `bvh.joint_angles` is now ~57× smaller than before.
  pybvh-ml's internals already operated in radians (`np.radians(angle_deg)`
  on user input, quaternion math throughout), so this is a no-op
  internally — but user code that did `np.deg2rad(bvh.joint_angles)` before
  feeding pybvh-ml is now a no-op and must be dropped. The unit-test cross-
  checks against `pybvh.transforms.rotate_angles_vertical` / `mirror_angles`
  were updated to compare in radians (`rotmat_to_euler(degrees=False)`).
- **`bvh.joint_velocities()` / `bvh.joint_accelerations()` now return
  `(F, J, 3)`, not `(F, N, 3)`** (silent breaking via pybvh 0.7.0).  The
  `(F, J, 3)` layout is joint-axis aligned with `joint_data` /
  `joint_angles` (no end sites). pybvh-ml's `preprocess_directory(include_velocities=True)`
  now stores arrays of this tightened shape — code that indexed velocities
  by node-axis must switch to `bvh.node_velocities()` upstream or adjust
  axis indexing downstream.

### Added

- **`harmonize=True` kwarg on `preprocess_directory`** — one-line opt-in for
  the common heterogeneous-dataset workflow. Runs `pybvh.harmonize` against
  the first clip with `return_report=True`, resolving each target axis from
  the explicit `target_*` kwarg when set or the majority value from the
  uniformity audit when not. For order-sensitive representations, also
  resolves a `target_euler_order` (most-common per-joint order across the
  dataset). Hierarchy mismatches raise loudly with the dropped filenames
  and pybvh's drop reasons — no silent shrinkage, which was the original
  failure mode the maintainer report flagged.
- **`target_euler_order` kwarg on `preprocess_directory`** — canonical
  Euler order to unify joint angles to, honored only when `harmonize=True`
  and the representation is order-sensitive (silently ignored otherwise).
- **`uniformity["harmonized_to"]`** in the returned summary, populated when
  `harmonize=True` ran. Carries the resolved target signature
  (`{"target_world_up": "+z", ...}`), per-stage modification counts
  (from `HarmonizeReport.applied_stages`), and the full report serialized
  via `dataclasses.asdict` — JSON-native and ready to embed in dataset
  metadata for an auditable transformation trail.

### Changed

- **Compatibility-check error messages** now name both the reference clip
  and the divergent clip, distinguish hierarchy mismatch from channel
  mismatch, and point at the right recovery (`harmonize=True` for channel,
  filter / `harmonize(reference=...)` for hierarchy). Uses each clip's
  `Bvh.source_path` when set (shipped by pybvh 0.7.0), falling back to the
  caller-supplied filename stem.
- **Rest-up disagreement warning recovery advice rewritten.** The previous
  warning suggested `target_rest_up='<axis>'` as the only fix, which fails
  silently when the file's rest pose is authoritative and animation-frame
  inference is the wrong one (the maintainer-report's case). The new text
  names both recovery paths and flags `world_up='<axis>'` (override at
  parse time) as the more common fix.

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
