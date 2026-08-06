# CONTEXT.md — pybvh-ml

> **Purpose of this document**: Give any AI agent (or human contributor) a complete understanding of the pybvh-ml project — its goals, architecture, dependencies, and design decisions — so they can build it correctly from the start.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Name** | pybvh-ml |
| **Language** | Python 3 (>= 3.9) |
| **Dependencies** | `pybvh` (required), `numpy` (required), `h5py` (optional), `torch` (optional) |
| **Primary use-case** | ML bridge layer: convert pybvh motion data into training-ready tensors, provide augmentation pipelines, dataset classes, and skeleton graph metadata |
| **Design principles** | **numpy core, PyTorch extras** (torch is optional), **composable** (each component works standalone), **opinionated but transparent** (makes layout/format choices, documents them clearly) |
| **Package** | Published on PyPI as `pybvh-ml`. Install via `pip install pybvh-ml` or `pip install "pybvh-ml[torch]"` or `pip install "pybvh-ml[hdf5]"` |

---

## 2. Relationship to pybvh

pybvh-ml depends on pybvh. It never reimplements pybvh functionality. The division:

| Concern | Owner |
|---|---|
| BVH file I/O | pybvh |
| Rotation math (euler, quat, 6d, rotmat, axis-angle) | pybvh |
| Forward kinematics | pybvh |
| Bvh object transforms (rotate, mirror, speed perturb, etc.) | pybvh |
| Motion analysis (velocities, foot contacts, trajectory) | pybvh |
| Batch BVH loading | pybvh |
| **Dataset z-score normalization (stats + apply/invert)** | **pybvh-ml** (since 0.5.0; absorbed from pybvh 0.8.0) |
| **Tensor layout packing (CTV, TVC, flat)** | **pybvh-ml** |
| **Array-level augmentation (no Bvh object needed)** | **pybvh-ml** |
| **Skeleton graph metadata (edge lists, partitions)** | **pybvh-ml** |
| **Preprocessing pipeline (BVH dir → on-disk dataset)** | **pybvh-ml** |
| **PyTorch Dataset classes** | **pybvh-ml** |
| **Feature metadata / column descriptors** | **pybvh-ml** |

**Key rule**: If a new feature could be useful to a biomechanics researcher who doesn't do ML, it belongs in pybvh. If it only makes sense in an ML training context, it belongs in pybvh-ml.

---

## 3. Development Environment

**Conda env**: `pybvh_ml`

```bash
conda run -n pybvh_ml pytest tests/test_pybvh_ml.py -v
```

This env has:
- Python 3.9
- numpy, matplotlib, pytest
- torch 2.8.0+cpu (CPU-only)
- h5py 3.14.0
- pybvh (editable install of the sister repo) — pybvh-ml 0.5 pins `pybvh>=0.8.1,<0.9` (0.8.1 publishes `parse_axis`)
- mkdocs + mkdocs-material + mkdocstrings (docs site; `mkdocs serve` for local preview, `mkdocs build --strict` as the link-integrity gate)

**Important**: pybvh has its own separate conda env (`pybvh`) with no torch/h5py. pybvh-ml code must never require torch or h5py for core functionality — they are optional, guarded with try/except.

---

## 4. Architecture Overview

```
pybvh-ml/
├── pybvh_ml/
│   ├── __init__.py              # Public API (32 exports)
│   ├── arrays.py                # MotionArrays — the container every surface takes
│   ├── packing.py               # Tensor layout conversion (CTV, TVC, flat)
│   ├── augmentation.py          # Array-level augmentation (all representations, no Bvh objects)
│   ├── convert.py               # Representation conversion dispatch
│   ├── pipeline.py              # Composable AugmentationPipeline
│   ├── _staged.py               # Internal quat-cache staging for AugmentationPipeline
│   ├── sequences.py             # Sequence length utilities (windowing, standardization)
│   ├── skeleton.py              # Skeleton graph metadata (edges, partitions)
│   ├── preprocessing.py         # Batch BVH → on-disk dataset pipelines + normalization trio
│   ├── metadata.py              # Feature column descriptors
│   └── torch/                   # Optional PyTorch integration
│       ├── __init__.py
│       ├── datasets.py          # Dataset / IterableDataset classes
│       └── collate.py           # Collate functions for variable-length sequences
├── bvh_data/                    # Test BVH files (bvh_test1-3, standard_skeleton)
├── tests/
│   ├── conftest.py              # shared fixtures (bvh_example, bvh_test3, rng)
│   ├── test_pybvh_ml.py         # numpy-core unit tests (44 test classes)
│   ├── test_position_streams.py # joint_pos / node_pos, end to end (0.6.0)
│   ├── test_torch_datasets.py   # torch Dataset/collate tests (skips without torch)
│   ├── test_no_pybvh_deprecation.py  # guards against deprecated pybvh API usage
│   ├── test_no_global_state_mutation.py  # guards principle 6: no process-wide mutation
│   ├── test_docs_api_coverage.py     # docs/api pages ↔ modules two-way sync; __all__ resolution
│   └── integration/             # real-data sweeps, seeding determinism, staging parity
├── tutorials/                   # 3 runnable notebooks (executed in CI via pytest --nbmake)
├── gallery/                     # feature-gallery notebook (jupytext pair, outputs committed) + its plotting helpers
├── scripts/                     # export_gallery.py — generates docs/gallery/ from the committed notebook outputs
├── docs/                        # MkDocs Material site source (deployed to GitHub Pages on push to main)
├── mkdocs.yml                   # site config; exclude_docs keeps gitignored docs/internal_logs/ out of local builds
├── pyproject.toml
└── README.md
```

### Module responsibilities

**`arrays.py`** — The container every array-level surface speaks
- `MotionArrays(root_pos=(F,3), joint_rot=(F,J,C) | None, joint_pos=(F,J,3) | None, node_pos=(F,N,3) | None, position_centering=str | None)` — keyword-only, frozen, validates shapes, the shared frame count, shared `J` between `joint_rot`/`joint_pos` and `N >= J` once at construction. `present_streams` reports which streams a clip carries; `STREAM_NAMES` is the vocabulary
- `MotionArrays.from_bvh(bvh, representation=None, center_root=False, include_positions=False, position_space="joint", position_centering="world")` — the producer edge, so extract → augment → pack never hand-assembles the container. `representation=None` is the positions-only (ST-GCN) journey
- `.replace(**fields)` — the only mutation path, revalidating. Dropping the last position stream must clear `position_centering` in the same call
- **`position_centering` travels with the arrays** (`"world"` / `"skeleton"` / `"first"` / `None`), not only with the dataset, because `add_root_position_noise`, the FK refresh in `add_joint_rotation_noise` and `pack_to_*(center_root=True)` are only correct given it. `None` is legal and fails at *use*: most of the surface is a rigid or temporal operation applied identically to both streams and does not care, and a guessed `"world"` from a caller who does not know is worse than an honest `None`. Anything pybvh-ml writes records it. `center_root_streams()` is the one shared implementation of "subtract the first frame's root from every stream it affects", used by `from_bvh`, the packers, preprocessing and `MotionDataset`
- Deliberately **not** a tuple and not unpackable: the 0.6.0 position streams had to be additive, and a 2-field tuple could not have grown. `__iter__` exists only to raise the migration message for `rp, jd = ...`
- Fields are **read-only views**, not copies: writes through a field raise (so a container over a Dataset cache cannot rewrite the cache), while construction and `replace` stay allocation-free — the alternative, copying in the constructor, would copy every clip on every pipeline step
- **dtype preserved, not promoted**: floating input keeps its dtype (`float32` stays `float32` — the container is what a per-sample Dataset holds), non-floating is promoted to `float64`, and each stream follows its own input
- **Augmentation computes in `float64` and returns the caller's dtype** (`_result` in `augmentation.py`, `_finish` in `pipeline.py`). Not letting the input dtype flow through the math is load-bearing twice over: otherwise a probabilistic pipeline's output dtype depends on which steps fired for that sample, and `cache_quats=True`/`False` stop being bit-identical for `float32` input (the staged 6d fast path writes into a copy of its input, so that step would run in single precision). Preservation stops at the packers and `standardize_length(resample_linear)`, which are `float64` by contract

**`packing.py`** — Tensor layout conversion
- `pack_to_ctv(arrays, center_root=True, *, streams=("root_pos", "joint_rot"))` → `(C, T, V)` ndarray; `pack_to_tvc` / `pack_to_flat` take the same arguments
- `unpack_from_ctv(data, root_channels=3)` → `MotionArrays`; likewise `unpack_from_tvc` / `unpack_from_flat`
- `streams=` names what is packed and in what order — channel order in the graph layouts, column order in flat. `"root_pos"` contributes vertex 0 (`V = 1 + J`); omit it and `V = J`, which is what removes the off-by-one against `skeleton_info["edges"]`. `node_pos` cannot share a vertex axis with a joint-space stream (different `V`) and raises
- Root position is always vertex 0 when packed, zero-padded to C channels if `C > 3`
- `center_root=True` shifts the position vertices too under `"world"` / `"first"`, leaves them alone under `"skeleton"`, and raises on `None` — centering only the root would move vertex 0 away from a body that stayed put
- **The unpackers deliberately take no `streams=`** and invert only the default packing; a streams-aware form is additive whenever it lands

**`augmentation.py`** — Array-level augmentation (operates on pre-extracted numpy arrays, no Bvh object needed)
- Unified functions that accept any representation via a `representation=` kwarg: `rotate_vertical`, `mirror`, `add_joint_rotation_noise`, `speed_perturbation_arrays`, `dropout_arrays`. Plus `add_root_position_noise` and the two keypoint-jitter functions `add_joint_position_noise` / `add_node_position_noise`, which take none — their sigmas are lengths. Supported representations: `"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"` (Euler additionally requires `euler_orders=`); `representation` is required only when the sample carries `joint_rot`
- Each takes a `MotionArrays` positionally and returns a new one; every other parameter is **keyword-only**. `rotate_vertical` and `add_joint_rotation_noise` accept `degrees=True` to read their angle in degrees (radians remain the default)
- **Stream coherence**: `@handles_streams(...)` declares what a step handles and `stream_support(fn)` reads it back; undeclared steps default to `{"root_pos", "joint_rot"}` (the pre-0.6 capability). "Handles" means *left correct*, by transformation or by re-derivation — it does **not** mean positions stay exact FK partners of the rotations beside them, except right after a re-derivation. Two divergences are intrinsic and documented: mirror (world-space for positions, parent-local for rotations) and slerp-vs-lerp under speed perturbation and dropout
- Only the two keypoint-jitter functions decline anything, and they decline `joint_rot` because rotation → position is FK while position → rotation is IK. That also makes the one destructive composition impossible: jitter and the FK refresh can never share a pipeline
- `add_joint_rotation_noise` **re-derives** the position streams via `pybvh.frames_to_node_positions`, so it needs `fk_topology=` (and `world_up=` under `"first"` centering). It hands the quat cache straight to FK, so the conversion is one hop (quat → euler), not two
- Fast paths: `rotate_vertical` and `mirror` skip the quaternion round-trip when `representation="6d"` (direct rotation of the two column vectors / analytic sign mask)
- Quat-internal ops (`add_joint_rotation_noise`, `speed_perturbation_arrays`, `dropout_arrays`) convert to/from quaternion space once; `AugmentationPipeline(cache_quats=True)` amortizes the conversion across consecutive steps

**`convert.py`** — Representation conversion, at both levels
- `convert_arrays(arrays, from_repr, to_repr, euler_orders)` → `MotionArrays` — the container-level form, so conversion composes with augmentation and packing instead of making callers take the container apart. `root_pos` is carried through unchanged
- `convert_rotations(joint_rot, from_repr, to_repr, euler_orders)` → ndarray — the rotation-level primitive for data with no root stream (a model's rotation output, a cached quat array); wraps pybvh's rotation functions and owns the flat-`(F,J,9)`-rotmat adaptation

**`pipeline.py`** — Composable augmentation pipeline
- `AugmentationPipeline` — composable sequence with per-augmentation probabilities and seeded rng. Supports callable kwargs (`lambda rng: value`) for per-sample random parameter sampling. Automatically forwards `rng` to functions that accept it (via signature inspection). `__call__` takes a `MotionArrays` positionally; `rng=` / `return_params=` stay keyword-only.
- Every configured step's preconditions — stream support, and `add_joint_rotation_noise`'s `fk_topology` / `joint_rot` / centering requirements — are checked **at `__call__` entry, before any step fires**, so a `p<1` step's misconfiguration cannot raise stochastically. A step that adds or drops a stream is also refused.
- `AugmentationPipeline.standard(skeleton_info, ...)` classmethod — opinionated factory that builds the canonical rotate + mirror + noise + speed pipeline from a `skeleton_info` dict. Each step is optional (pass `None` or `mirror_prob=0` to skip). For anything beyond the exposed kwargs, build the pipeline directly with the `(fn, prob, kwargs)` constructor. Two resolutions happen at construction because the pipeline predates any sample: `representation=None` skips the rotation-noise step (meaningless on a rotation-free clip), and `position_noise_sigma=` picks joint- or node-space jitter from `position_space` / `skeleton_info` — they are different functions with different stream declarations.
- `cache_quats=True` (default) shares one quaternion cache across consecutive staged steps via `_staged.py`'s `STAGED_DISPATCH` registry. User-defined augmentations are supported transparently (cache flushed, function called normally, staging resumed cold). Set `cache_quats=False` for historical bit-exact behavior.

**`sequences.py`** — Sequence length utilities
- `sliding_window(data, window_size, stride)` → `(num_windows, window_size, ...)` fixed-length windows
- `standardize_length(data, target_length, method)` → crop, pad, or resample to fixed frame count
- `uniform_temporal_sample(num_frames, clip_length, mode, rng)` → PySKL-style uniform segment sampling with three regimes (short/wrapping, dense/gap-insertion, uniform/segment-based)
- `sample_temporal(data, clip_length, num_samples, mode, rng)` → convenience wrapper that applies sampled indices with wraparound

**`skeleton.py`** — Skeleton graph metadata, in two index spaces
- `get_edge_list(bvh, include_end_sites=False)` → `list[(child_idx, parent_idx)]`
- `get_body_partitions(bvh)` → `dict[str, list[int]]` mapping body part names to joint indices
- `get_lr_pairs(bvh)` / `get_node_lr_pairs(bvh)` → `list[(left_idx, right_idx)]` in joint / node space (thin wrappers over pybvh's cached properties)
- `get_fk_topology_dict(bvh)` → the four `FkTopology` fields as JSON-native lists; `build_fk_topology(skeleton_info)` rebuilds one at train time (raises for pre-0.6.0 datasets, whose bone offsets are stored nowhere else)
- `find_mismatched_end_site_pairs(bvh)` → node-space L/R pairs whose two sides carry different numbers of end sites. **Both sides of that comparison must be node-space**: mixing in joint-space `lr_pairs` does not fail loudly, it indexes the end-site counter with the wrong keys and reports a wrong pair. `bvh.node_lr_pairs` *drops* such a pair's end sites (a property filters rather than raises), which is right upstream and wrong for us — we persist the list and mirror at train time, far from any `Bvh`
- `get_skeleton_info(bvh)` → unified dict: joint-space keys, node-space keys, `fk_topology`, `mismatched_end_site_pairs`, and the axis strings. Everything JSON-serializable, because `preprocess_directory` persists the whole dict

**`preprocessing.py`** — Batch preprocessing pipelines
- `preprocess_directory(bvh_dir, output_path, representation, ...)` — BVH directory → on-disk dataset
- Supports output formats: `.npz`, `.hdf5` (if h5py installed)
- Stores arrays + skeleton metadata + normalization stats in a single file
- `include_positions=True` stores `joint_pos` or `node_pos` (`position_space=`) in the frame `position_centering=` names. The two settings live apart deliberately: `position_space` is a topology fact and goes in `skeleton_info` (the `foot_joints` precedent), `position_centering` is a statement about the values and goes in dataset metadata beside `center_root`. `center_root=True` + `"first"` is rejected — they would center the streams differently in the up axis. Positions get their own `position_stats` block; widening `mean`/`std` would make one file format mean two things, since `D = 3 + J*C` is matched by `pack_to_flat`, `describe_features` and HumanML3D's `Mean.npy`/`Std.npy`
- Optional label function `label_fn(filename) → int`
- Optional filter function `filter_fn(filename_stem) → bool` — applied before loading, skipped files are never parsed
- Rep-aware compatibility check: skeleton graph (`matches_hierarchy(match_offsets=False)`) must always agree; per-joint Euler orders must additionally agree for order-sensitive reps (`euler`, `axisangle`).  Bone-length variation across actors is accepted — `joint_data` is a function of rotations, not bone lengths.
- `harmonize=True` runs `pybvh.harmonize` after the uniformity audit — pure reorientation by default (per-actor bone lengths preserved); `retarget=True` pins the first clip and retargets bone offsets to it. Resolves each `target_*` from the explicit kwarg if set, else the audit majority; for order-sensitive reps also picks `target_euler_order` from the most common per-joint order. Hierarchy mismatches raise loudly either way — no silent shrinkage. The resolved targets, the `retarget` choice, per-stage counts, and full `HarmonizeReport` land in `uniformity["harmonized_to"]` (JSON-serializable via `dataclasses.asdict`), persisted in the saved dataset as `uniformity_json`.

**`metadata.py`** — Feature column descriptors
- `FeatureDescriptor` — describes which columns correspond to which features in a packed array
- `describe_features(num_joints, representation="6d", include_root_pos=True, *, streams=None, num_nodes=None)` → `FeatureDescriptor`, with blocks `root_pos` / `joint_rotations` / `joint_positions` / `node_positions`
- Enables programmatic access to feature slices without hardcoded column indices

**`torch/datasets.py`** — PyTorch Dataset classes (optional, only if torch is installed)
- `MotionDataset(Dataset)` — loads preprocessed data from disk, returns tensors
- `OnTheFlyDataset(Dataset)` — loads raw arrays, applies augmentation each epoch
- Both support variable-length sequences with configurable padding/cropping
- `streams=` picks what the single `data` tensor carries (one tensor with explicit streams, not a second tensor, so the batch contract does not depend on preprocessing flags). `MotionDataset.from_preprocessed` threads the stored `position_centering` onto every container it mints — storage metadata alone is not enough, since the steps that depend on it only ever see the container — and raises when a requested stream is absent. `OnTheFlyDataset` has its own `include_positions` / `position_space` / `position_centering`, since it extracts per clip rather than reading preprocessed arrays
- **Per-clip identity**: items carry `name` (the filename stem) — `MotionDataset` when built with `names=` (which `from_preprocessed` fills from the stored `filenames`), `OnTheFlyDataset` always, since it holds the paths. Omitted rather than index-substituted when unavailable, so "no identity provided" is distinguishable from a real name. `labels` / `names` are length-validated against the clip count: a long sequence would otherwise attribute every clip to the wrong entry

**`torch/collate.py`** — Collate functions
- `collate_motion_batch(batch)` — handles variable-length sequences with padding and mask generation
- Returns a dict: `data` `(B, T_max, D)` zero-padded, `lengths` `(B,)` valid-frame counts, `mask` `(B, T_max)` bool (True = valid frame), plus `labels` `(B,)` and `names` (a plain list of `B` strings, matching what `default_collate` does with strings) when those are present; both are all-or-none across the batch. Since 0.5.0 each item's `length` means valid frames in the returned tensor — cropped clips report `target_length`, not the original clip length.

---

## 5. Key Design Decisions

### 5.1 Unified augmentation across representations
Array-level augmentation is a single function per operation, parameterized by a `representation=` kwarg that covers every representation pybvh supports (`"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"`). Internally:
- **Quaternion** — primary internal representation for rotation-space math (SLERP, Hamilton product, unit-sphere noise). Clean, no gimbal lock.
- **6D** — fast paths in `rotate_vertical` and `mirror` operate directly on `(F, J, 6)` without a quat round-trip. Other ops (`add_joint_rotation_noise`, `speed_perturbation_arrays`, `dropout_arrays`) convert once, stay in quat, convert back — or, with `AugmentationPipeline(cache_quats=True)`, share the cache across consecutive steps.
- **Axis-angle / rotmat / euler** — supported uniformly via convert-to-quat, mutate, convert-back. Euler additionally requires `euler_orders=`.

### 5.2 Preprocessing and runtime are separate
- **Preprocessing** (`preprocessing.py`): runs once, converts BVH files to arrays on disk. Slow is OK.
- **Runtime** (`augmentation.py`, `torch/datasets.py`): runs every epoch, must be fast. Operates on pre-extracted arrays, never touches BVH files.

### 5.3 PyTorch is optional
All `torch/` imports are guarded. The core modules (`packing`, `augmentation`, `skeleton`, `preprocessing`, `metadata`) work with pure NumPy. Users who don't use PyTorch can still use pybvh-ml for preprocessing and array manipulation.

### 5.4 pybvh API surface that pybvh-ml relies on
pybvh-ml uses these pybvh entry points:
- `pybvh.read_bvh_file()`, `pybvh.read_bvh_directory()` — loading
- `bvh.root_pos`, `bvh.joint_angles`, `bvh.joint_count`, `bvh.joint_names` — data access
- `bvh.source_path` — on-disk origin used in error messages
- `bvh.to_quat()`, `bvh.to_6d()`, `bvh.to_axisangle()`, `bvh.to_rotmat()` — representation conversion (2-tuple `(root_pos, joint_data)` since pybvh 0.6.0)
- `bvh.euler_orders` — per-joint Euler order strings
- `bvh.change_euler_order(order)` — re-express angles in a uniform Euler order (used by `harmonize(target_euler_order=...)`)
- `bvh.matches_hierarchy(other, match_offsets=False)` and `bvh.matches_channels(other)` — skeleton compatibility predicates (pybvh 0.7.0)
- `bvh.edges` — skeleton edge list as `(child_idx, parent_idx)` tuples
- `bvh.nodes`, `bvh.node_index` — skeleton topology
- `bvh.lr_pairs`, `bvh.lr_mapping`, `bvh.node_lr_pairs` — cached L/R pair detection in both index spaces (pybvh 0.8.2 for the node-space form)
- `bvh.node_edges`, `bvh.joint_positions()`, `bvh.node_positions(centered=...)` — node-space topology and FK-derived positions (world-frame FK cached on the `Bvh`, invalidated on motion writes)
- `bvh.fk_topology` + `pybvh.FkTopology` + `pybvh.frames_to_node_positions(...)` (pybvh 0.8.2) — array-signature forward kinematics, which is what makes pybvh-ml's train-time FK refresh possible with no `Bvh` in sight
- `pybvh.rotations.*` — rotation conversion primitives, `quat_multiply`, `REPRESENTATION_CHANNELS`
- `bvh.joint_velocities()`, `bvh.foot_contacts()` — motion analysis for the optional `include_velocities` / `include_foot_contacts` preprocessing outputs
- `pybvh.harmonize(...)` + `HarmonizeReport` (pybvh 0.7.0) — dataset-level harmonization; pybvh-ml's `preprocess_directory(harmonize=True)` drives it with `return_report=True` and surfaces drops with the report's `dropped_sources` / `drop_reasons`. By default no `reference=` is passed (pure reorientation, per-actor bone lengths preserved; hierarchy mismatches surface in `_check_skeleton_compatibility` right after); `retarget=True` pins `clips[0]` as the reference, enabling pybvh's topology gate + bone-offset retargeting (offsets only — root translations keep each clip's scale)

Normalization is pybvh-ml's own public API since 0.5.0: `compute_normalization_stats` / `normalize_array` / `denormalize_array` live in `preprocessing.py` (absorbed from pybvh 0.8.0, which removed the trio from `pybvh.batch`). The Bvh-list entry point extracts via `extract_repr` and applies pybvh-ml's intentionally loose skeleton check (`_check_skeleton_compatibility`, bone-length variation accepted); `preprocess_directory` shares the same array-level core (`_normalization_stats_from_arrays`) on its already-extracted arrays.

### 5.5 Joint noise is quaternion-internal
`add_joint_rotation_noise` generates noise as random axis-angle perturbations (random axis on the unit sphere, angle from N(0, sigma) in radians), converts to quaternion, and composes via Hamilton product (`pybvh.rotations.quat_multiply`). This avoids gimbal lock sensitivity and gives uniform perturbation regardless of pose. The public `representation=` kwarg controls the input/output format; the math itself is always quat-space.

### 5.6 Callable kwargs and rng forwarding in AugmentationPipeline
Kwargs values can be callables of the form `lambda rng: value`, resolved at each invocation. This enables per-sample random parameter sampling (e.g., random rotation angles) without modifying augmentation function signatures. Static kwargs continue to work unchanged.

The pipeline automatically forwards its `rng` to augmentation functions that accept an `rng` parameter (detected via `inspect.signature`). This ensures reproducibility for functions like `dropout_arrays` and `add_joint_rotation_noise` without requiring explicit `"rng": lambda rng: rng` in kwargs. If the user provides an explicit `rng` kwarg (static or callable), it takes precedence over the auto-forwarded one.

Both dispatch paths route their probability draw and kwarg resolution through one `_resolve_step`, so draw order can't drift between them. It also builds the per-step record `{"name", "applied", "params"}` that `__call__(..., return_params=True)` returns — callables are resolved *only* when the step fires, which is what keeps the flag stream-neutral (asking for records never consumes a draw). `params` reports only the kwargs whose spec was a callable: static kwargs are configuration the caller already has in `augmentations`, and `rng` is machinery. Step names come from `_step_name`, which tolerates steps that carry no `__name__` (`functools.partial`, callable instances) by unwrapping `partial.func` and falling back to the class name.

`MotionDataset.explain_augmentation(idx, epoch=...)` / `OnTheFlyDataset.explain_augmentation(...)` sit on top of this: they re-run one sample's augmentation on a freshly composed `(seed, epoch, idx)` rng — the same one `__getitem__` used — and return its records. Nothing is recorded during training. Both classes feed the replay from the same `_clip_arrays` helper `__getitem__` uses, so the replay cannot drift from what the loader fed the pipeline; the pipeline is genuinely re-run rather than the draws recomputed, because steps like `add_joint_rotation_noise` consume an amount of randomness that depends on the clip's shape. An unseeded dataset raises instead of answering (a fresh draw would be indistinguishable from the real one).

### 5.7 Uniform temporal sampling matches PySKL
`uniform_temporal_sample` reproduces the PySKL/MMAction2 `UniformSampleFrames` algorithm as a stateless function. Three regimes:
- **Short** (`num_frames < clip_length`): sequential `[start..start+clip_length-1]` with random start (train) or start=0 (test). Caller applies `% num_frames`.
- **Dense** (`clip_length <= num_frames < 2*clip_length`): starts with `[0..clip_length-1]`, randomly inserts gaps to spread indices across the full range.
- **Uniform** (`num_frames >= 2*clip_length`): integer segment boundaries (`i * num_frames // clip_length`), discrete random offset per segment.

### 5.8 Implementation-level conventions
1. **Every array-level function takes and returns a `MotionArrays`** — augmentation, the pipeline, the packers, and `convert_arrays` — one clip's `root_pos` plus its rotation and position streams, passed positionally (it is a distinct type, so a swap is not expressible) with every other parameter keyword-only. The container is frozen and validates shapes and frame counts once at construction, which is why no function re-checks them. It exists so a later stream is additive rather than an arity break at every call site — 0.6.0's `joint_pos` / `node_pos` are what cashed that in, at zero call-site cost. All arguments are keyword-only to prevent silent-corruption swaps on shape-compatible ndarrays.
2. **`convert_rotations` routes through rotation matrices** as intermediate (and `convert_arrays` is a thin container-level wrapper over it). Per-joint Euler orders are handled by grouping joints by unique order and batch-converting each group.
3. **Packing zero-pads root only** — root has 3 channels (position), joints have C_joint channels. In CTV/TVC layouts, `C = max(3, C_joint)`. Since C_joint >= 3 for all real representations, joint data is never padded.
4. **Mirror math**: quaternion mirror negates the two imaginary components NOT at the lateral axis. 6D mirror uses `R'[i,j] = s_i * s_j * R[i,j]` where `s[lateral] = -1`. Both derived from `R' = S @ R @ S`.
5. **Quaternion multiplication comes from pybvh** — `pybvh.rotations.quat_multiply` (public since pybvh 0.8.0; Hamilton convention, wxyz scalar-first order). pybvh-ml carried a private bit-identical copy in `augmentation.py` until 0.5.0.
6. **`torch/` subpackage fails hard on import if torch is missing** — `pybvh_ml.torch` raises ImportError (via `importlib.util.find_spec`, so a *broken* torch installation surfaces its real traceback instead of a misleading "install torch"). But `import pybvh_ml` (the top-level) works fine without torch.
7. **The Dataset epoch lives in shared memory** — `EpochState` wraps a `multiprocessing.Value("i", -1)` (−1 = never-set sentinel) so `set_epoch()` in the main process reaches DataLoader workers, including `persistent_workers=True` (workers are created once and never re-receive the dataset — a plain attribute is structurally frozen there). The Value comes from an explicit **spawn** context, not the process default: a fork-context lock is an anonymous unlinked semaphore whose handle unpickles into a spawn-started worker and then segfaults it, and Linux defaults to fork — so `multiprocessing_context="spawn"` was broken until 0.5.0. A spawn-context lock is named and survives both start methods. Deliberately no `__getstate__`/`__setstate__`: swapping the Value during pickling would silently break sharing under spawn. Consequences: dataset instances aren't `deepcopy`/`torch.save`-able (sharing only travels via process inheritance, which is exactly how the DataLoader passes the dataset to workers), and under a spawn loader the whole dataset must be picklable — so pipeline kwargs need module-level callables rather than lambdas. Public alongside `rng_for(seed, epoch, idx)` since 0.5.0: a Dataset that isn't a `MotionDataset` subclass needs both to honor the same contract. `EpochState.is_set` (and `dataset.epoch` / `dataset.epoch_is_set` on both shipped classes) distinguishes never-set from epoch 0, which `current` deliberately collapses — needed because a framework that forks workers before its epoch hook leaves them in the unset state, and the fix is an earlier hook claiming epoch 0 only if nothing has.
8. **Freshly-allocated output is the pipeline's guarantee, not each function's** — `AugmentationPipeline.__call__` never returns arrays sharing storage with its input (it copies on the no-step-fired fall-through), while an individual augmentation may pass an untouched stream through by reference (`add_root_position_noise` returns the input's `joint_rot`; the only one today). Read-only `MotionArrays` fields are what makes the looser function-level contract safe, and copying every untouched stream would allocate a full clip per step for nothing.
9. **The two stochastic `__getitem__` stages share one per-sample generator** — augmentation draws first, then `temporal="resample"` continues on the same stream. That order is what keeps `explain_augmentation` exact (it replays only the augmentation, from the head of an identical stream). `temporal="resample_deterministic"` deliberately gets `rng=None` instead: `uniform_temporal_sample` honors a supplied rng in test mode too, so passing the advanced stream would make "deterministic" drift with the epoch.

---

## 6. Coding Conventions

1. **Follow pybvh conventions**: snake_case, full type annotations, property validation
2. **NumPy first**: All core functions take and return NumPy arrays
3. **Optional imports**: PyTorch and h5py are imported lazily, inside functions or behind `try/except`
4. **Docstrings**: Every public function documents input/output shapes explicitly (e.g., `(F, J, 4)`)
5. **No pybvh internals**: Only use pybvh's public API. Never import private functions or access private attributes
6. **Tests**: pytest, same conventions as pybvh. Test with and without optional dependencies installed
7. **No global state, no import-time side effects**: never touch `torch`'s precision / determinism / threading / seeding knobs or anything under `torch.backends`, `np.seterr` / `np.random.seed` / print options, `os.environ`, `warnings.filters`, or logging config. Those are the *application's* policy, and a data library setting them makes `import pybvh_ml` change a model's numbers with nothing at any call site to show it. Randomness is always passed in (`rng=` / `seed=`), never installed. `tests/test_no_global_state_mutation.py` enforces it (AST scan + fresh-interpreter import diff), so the rule survives a well-meant "let's just enable TF32" patch

---

## 7. What NOT to build

- **Model architectures** (GCN layers, transformer blocks, etc.) — user's responsibility
- **Training loops, loss functions, optimizers** — user's responsibility
- **Visualization** — pybvh already has plotting
- **BVH parsing or writing** — pybvh's job
- **Rotation math** — pybvh's job
- **Features that require specific model assumptions** — keep it model-agnostic

---

## 8. Test Patterns

Unit tests are in `tests/test_pybvh_ml.py` (44 test classes) plus `tests/test_position_streams.py` (12 classes — the 0.6.0 position streams end to end, with purpose-built rigs for the end-site-mismatch check, the asymmetric-rest-offset order dependence, and a name collision), `tests/test_torch_datasets.py` (10 classes, module-level `pytest.importorskip("torch")` so the suite collects without torch), `tests/test_no_pybvh_deprecation.py`, `tests/test_no_global_state_mutation.py`, `tests/test_docs_api_coverage.py` (the API reference stays two-way in sync with the modules; `__all__` names resolve), and `tests/test_gallery_notebook.py` (the gallery jupytext pair stays synced and its committed outputs fresh) — 675 tests total; `tests/integration/` adds real-data sweeps (representation parity, seeding determinism, pipeline staging, end-to-end MLP training) for 1227 with them. Test BVH files are in `bvh_data/` at the project root.

**Fixtures** (shared ones live in `tests/conftest.py`):
- `bvh_example` — loads `bvh_data/bvh_test1.bvh` (24 joints, ZYX)
- `bvh_test3` — loads `bvh_data/bvh_test3.bvh` (60 joints, mixed euler orders)
- `rng` — `np.random.default_rng(42)`

**Conventions**:
- `@pytest.mark.parametrize` for representation names, channel counts, axis indices
- `np.testing.assert_allclose` for numerical comparisons
- Round-trip tests: pack/unpack, convert/convert-back, mirror/mirror, augment/inverse
- Consistency tests: quaternion augmentation must match Euler augmentation (via pybvh's `rotate_angles_vertical`) after conversion
- Real BVH integration: tests use actual BVH data, not synthetic arrays
- Shape assertions: explicit `assert result.shape == (F, J, C)`

**Note**: `bvh_test1` and `standard_skeleton` share the skeleton *graph* (joint names + parent indices; offsets differ), so the loose compatibility check accepts that pair; `bvh_test2` and `bvh_test3` each have distinct skeletons. Tests that batch multiple files usually copy `bvh_test1.bvh` under several names in a `tmp_path` work directory for full control.

---

## 9. Lessons Learned

1. **pybvh's rotation functions originally didn't support 3D batch dims** — `euler_to_rotmat((F, J, 3))` crashed. Fixed in pybvh v0.4.0 by flattening to 2D, processing, reshaping back. If you encounter similar issues with pybvh primitives, the fix pattern is: flatten leading dims, call the function, reshape.

2. **Euler angle round-trips are not unique** — converting Euler → rotmat → Euler may give different angles that represent the same rotation (especially near gimbal lock). Always compare via rotation matrices, not raw Euler angles.

3. **`standardize_length(method="resample_linear")` uses linear interpolation** — correct for positional channels only, not for rotation arrays. Resample rotations with `pybvh.Bvh.resample()` (SLERP-based) before extracting arrays instead; the docstring states this plainly.
