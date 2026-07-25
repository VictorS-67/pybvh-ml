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
- pybvh 0.8.0 (editable install) — pybvh-ml 0.5 pins `pybvh>=0.8,<0.9`
- mkdocs + mkdocs-material + mkdocstrings (docs site; `mkdocs serve` for local preview, `mkdocs build --strict` as the link-integrity gate)

**Important**: pybvh has its own separate conda env (`pybvh`) with no torch/h5py. pybvh-ml code must never require torch or h5py for core functionality — they are optional, guarded with try/except.

---

## 4. Architecture Overview

```
pybvh-ml/
├── pybvh_ml/
│   ├── __init__.py              # Public API (28 exports)
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
│   ├── test_pybvh_ml.py         # numpy-core unit tests (22 test classes)
│   ├── test_torch_datasets.py   # torch Dataset/collate tests (skips without torch)
│   ├── test_no_pybvh_deprecation.py  # guards against deprecated pybvh API usage
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

**`packing.py`** — Tensor layout conversion
- `pack_to_ctv(root_pos, joint_data, center_root=True)` → `(C, T, V)` ndarray
- `pack_to_tvc(root_pos, joint_data, center_root=True)` → `(T, V, C)` ndarray
- `pack_to_flat(root_pos, joint_data, center_root=True)` → `(T, D)` ndarray
- `unpack_from_ctv(data, root_channels=3)` → `(root_pos, joint_data)`
- `unpack_from_tvc(data, root_channels=3)` → `(root_pos, joint_data)`
- Root position is always vertex 0, zero-padded to C channels if `C > 3`

**`augmentation.py`** — Array-level augmentation (operates on pre-extracted numpy arrays, no Bvh object needed)
- Unified functions that accept any representation via a `representation=` kwarg: `rotate_vertical`, `mirror`, `add_joint_noise`, `speed_perturbation_arrays`, `dropout_arrays`. Supported representations: `"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"` (Euler additionally requires `euler_orders=`)
- All public functions are **keyword-only**: `root_pos` and `joint_data` are shape-compatible ndarrays, so positional binding is refused to prevent silent-corruption swaps
- Fast paths: `rotate_vertical` and `mirror` skip the quaternion round-trip when `representation="6d"` (direct rotation of the two column vectors / analytic sign mask)
- Quat-internal ops (`add_joint_noise`, `speed_perturbation_arrays`, `dropout_arrays`) convert to/from quaternion space once; `AugmentationPipeline(cache_quats=True)` amortizes the conversion across consecutive steps

**`convert.py`** — Representation conversion
- `convert_arrays(joint_data, from_repr, to_repr, euler_orders)` — unified representation conversion dispatch for `(F, J, C)` arrays, wrapping pybvh's rotation functions

**`pipeline.py`** — Composable augmentation pipeline
- `AugmentationPipeline` — composable sequence with per-augmentation probabilities and seeded rng. Supports callable kwargs (`lambda rng: value`) for per-sample random parameter sampling. Automatically forwards `rng` to functions that accept it (via signature inspection). `__call__` is keyword-only (`root_pos=`, `joint_data=`, `rng=`).
- `AugmentationPipeline.standard(skeleton_info, ...)` classmethod — opinionated factory that builds the canonical rotate + mirror + noise + speed pipeline from a `skeleton_info` dict. Each step is optional (pass `None` or `mirror_prob=0` to skip). For anything beyond the exposed kwargs, build the pipeline directly with the `(fn, prob, kwargs)` constructor.
- `cache_quats=True` (default) shares one quaternion cache across consecutive staged steps via `_staged.py`'s `STAGED_DISPATCH` registry. User-defined augmentations are supported transparently (cache flushed, function called normally, staging resumed cold). Set `cache_quats=False` for historical bit-exact behavior.

**`sequences.py`** — Sequence length utilities
- `sliding_window(data, window_size, stride)` → `(num_windows, window_size, ...)` fixed-length windows
- `standardize_length(data, target_length, method)` → crop, pad, or resample to fixed frame count
- `uniform_temporal_sample(num_frames, clip_length, mode, rng)` → PySKL-style uniform segment sampling with three regimes (short/wrapping, dense/gap-insertion, uniform/segment-based)
- `sample_temporal(data, clip_length, num_samples, mode, rng)` → convenience wrapper that applies sampled indices with wraparound

**`skeleton.py`** — Skeleton graph metadata
- `get_edge_list(bvh, include_end_sites=False)` → `list[(child_idx, parent_idx)]`
- `get_body_partitions(bvh)` → `dict[str, list[int]]` mapping body part names to joint indices
- `get_lr_pairs(bvh)` → `list[(left_idx, right_idx)]` (returns `list(bvh.lr_pairs)`, pybvh's cached index-space property)
- `get_skeleton_info(bvh)` → unified dict with edges, partitions, L/R pairs, joint names, euler orders

**`preprocessing.py`** — Batch preprocessing pipelines
- `preprocess_directory(bvh_dir, output_path, representation, ...)` — BVH directory → on-disk dataset
- Supports output formats: `.npz`, `.hdf5` (if h5py installed)
- Stores arrays + skeleton metadata + normalization stats in a single file
- Optional label function `label_fn(filename) → int`
- Optional filter function `filter_fn(filename_stem) → bool` — applied before loading, skipped files are never parsed
- Rep-aware compatibility check: skeleton graph (`matches_hierarchy(match_offsets=False)`) must always agree; per-joint Euler orders must additionally agree for order-sensitive reps (`euler`, `axisangle`).  Bone-length variation across actors is accepted — `joint_data` is a function of rotations, not bone lengths.
- `harmonize=True` runs `pybvh.harmonize` after the uniformity audit — pure reorientation by default (per-actor bone lengths preserved); `retarget=True` pins the first clip and retargets bone offsets to it. Resolves each `target_*` from the explicit kwarg if set, else the audit majority; for order-sensitive reps also picks `target_euler_order` from the most common per-joint order. Hierarchy mismatches raise loudly either way — no silent shrinkage. The resolved targets, the `retarget` choice, per-stage counts, and full `HarmonizeReport` land in `uniformity["harmonized_to"]` (JSON-serializable via `dataclasses.asdict`), persisted in the saved dataset as `uniformity_json`.

**`metadata.py`** — Feature column descriptors
- `FeatureDescriptor` — describes which columns correspond to which features in a packed array
- `describe_features(num_joints, representation="6d", include_root_pos=True)` → `FeatureDescriptor`
- Enables programmatic access to feature slices without hardcoded column indices

**`torch/datasets.py`** — PyTorch Dataset classes (optional, only if torch is installed)
- `MotionDataset(Dataset)` — loads preprocessed data from disk, returns tensors
- `OnTheFlyDataset(Dataset)` — loads raw arrays, applies augmentation each epoch
- Both support variable-length sequences with configurable padding/cropping

**`torch/collate.py`** — Collate functions
- `collate_motion_batch(batch)` — handles variable-length sequences with padding and mask generation
- Returns a dict: `data` `(B, T_max, D)` zero-padded, `lengths` `(B,)` valid-frame counts, `mask` `(B, T_max)` bool (True = valid frame), plus `labels` `(B,)` when labels are present. Since 0.5.0 each item's `length` means valid frames in the returned tensor — cropped clips report `target_length`, not the original clip length.

---

## 5. Key Design Decisions

### 5.1 Unified augmentation across representations
Array-level augmentation is a single function per operation, parameterized by a `representation=` kwarg that covers every representation pybvh supports (`"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`, `"euler"`). Internally:
- **Quaternion** — primary internal representation for rotation-space math (SLERP, Hamilton product, unit-sphere noise). Clean, no gimbal lock.
- **6D** — fast paths in `rotate_vertical` and `mirror` operate directly on `(F, J, 6)` without a quat round-trip. Other ops (`add_joint_noise`, `speed_perturbation_arrays`, `dropout_arrays`) convert once, stay in quat, convert back — or, with `AugmentationPipeline(cache_quats=True)`, share the cache across consecutive steps.
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
- `bvh.lr_pairs`, `bvh.lr_mapping` — cached L/R joint pair detection (index pairs / name-keyed dict)
- `pybvh.rotations.*` — rotation conversion primitives, `quat_multiply`, `REPRESENTATION_CHANNELS`
- `bvh.joint_velocities()`, `bvh.foot_contacts()` — motion analysis for the optional `include_velocities` / `include_foot_contacts` preprocessing outputs
- `pybvh.harmonize(...)` + `HarmonizeReport` (pybvh 0.7.0) — dataset-level harmonization; pybvh-ml's `preprocess_directory(harmonize=True)` drives it with `return_report=True` and surfaces drops with the report's `dropped_sources` / `drop_reasons`. By default no `reference=` is passed (pure reorientation, per-actor bone lengths preserved; hierarchy mismatches surface in `_check_skeleton_compatibility` right after); `retarget=True` pins `clips[0]` as the reference, enabling pybvh's topology gate + bone-offset retargeting (offsets only — root translations keep each clip's scale)

Normalization is pybvh-ml's own public API since 0.5.0: `compute_normalization_stats` / `normalize_array` / `denormalize_array` live in `preprocessing.py` (absorbed from pybvh 0.8.0, which removed the trio from `pybvh.batch`). The Bvh-list entry point extracts via `extract_repr` and applies pybvh-ml's intentionally loose skeleton check (`_check_skeleton_compatibility`, bone-length variation accepted); `preprocess_directory` shares the same array-level core (`_normalization_stats_from_arrays`) on its already-extracted arrays.

### 5.5 Joint noise is quaternion-internal
`add_joint_noise` generates noise as random axis-angle perturbations (random axis on the unit sphere, angle from N(0, sigma) in radians), converts to quaternion, and composes via Hamilton product (`pybvh.rotations.quat_multiply`). This avoids gimbal lock sensitivity and gives uniform perturbation regardless of pose. The public `representation=` kwarg controls the input/output format; the math itself is always quat-space.

### 5.6 Callable kwargs and rng forwarding in AugmentationPipeline
Kwargs values can be callables of the form `lambda rng: value`, resolved at each invocation. This enables per-sample random parameter sampling (e.g., random rotation angles) without modifying augmentation function signatures. Static kwargs continue to work unchanged.

The pipeline automatically forwards its `rng` to augmentation functions that accept an `rng` parameter (detected via `inspect.signature`). This ensures reproducibility for functions like `dropout_arrays` and `add_joint_noise` without requiring explicit `"rng": lambda rng: rng` in kwargs. If the user provides an explicit `rng` kwarg (static or callable), it takes precedence over the auto-forwarded one.

### 5.7 Uniform temporal sampling matches PySKL
`uniform_temporal_sample` reproduces the PySKL/MMAction2 `UniformSampleFrames` algorithm as a stateless function. Three regimes:
- **Short** (`num_frames < clip_length`): sequential `[start..start+clip_length-1]` with random start (train) or start=0 (test). Caller applies `% num_frames`.
- **Dense** (`clip_length <= num_frames < 2*clip_length`): starts with `[0..clip_length-1]`, randomly inserts gaps to spread indices across the full range.
- **Uniform** (`num_frames >= 2*clip_length`): integer segment boundaries (`i * num_frames // clip_length`), discrete random offset per segment.

### 5.8 Implementation-level conventions
1. **All augmentation functions take and return `(root_pos, joint_data)`** — root first, matching pybvh's `Bvh.from_*` / `Bvh.to_*` and pybvh-ml's own `pack_to_flat` / `extract_repr`. All arguments are keyword-only to prevent silent-corruption swaps on shape-compatible ndarrays.
2. **`convert_arrays` routes through rotation matrices** as intermediate. Per-joint Euler orders are handled by grouping joints by unique order and batch-converting each group.
3. **Packing zero-pads root only** — root has 3 channels (position), joints have C_joint channels. In CTV/TVC layouts, `C = max(3, C_joint)`. Since C_joint >= 3 for all real representations, joint data is never padded.
4. **Mirror math**: quaternion mirror negates the two imaginary components NOT at the lateral axis. 6D mirror uses `R'[i,j] = s_i * s_j * R[i,j]` where `s[lateral] = -1`. Both derived from `R' = S @ R @ S`.
5. **Quaternion multiplication comes from pybvh** — `pybvh.rotations.quat_multiply` (public since pybvh 0.8.0; Hamilton convention, wxyz scalar-first order). pybvh-ml carried a private bit-identical copy in `augmentation.py` until 0.5.0.
6. **`torch/` subpackage fails hard on import if torch is missing** — `pybvh_ml.torch` raises ImportError (via `importlib.util.find_spec`, so a *broken* torch installation surfaces its real traceback instead of a misleading "install torch"). But `import pybvh_ml` (the top-level) works fine without torch.
7. **The Dataset epoch lives in shared memory** — `_EpochState` wraps a `multiprocessing.Value("i", -1)` (−1 = never-set sentinel) so `set_epoch()` in the main process reaches DataLoader workers, including `persistent_workers=True` (workers are created once and never re-receive the dataset — a plain attribute is structurally frozen there). Deliberately no `__getstate__`/`__setstate__`: swapping the Value during pickling would silently break sharing under spawn. Consequence: dataset instances aren't `deepcopy`/`torch.save`-able; sharing only travels via process inheritance, which is exactly how the DataLoader passes the dataset to workers.

---

## 6. Coding Conventions

1. **Follow pybvh conventions**: snake_case, full type annotations, property validation
2. **NumPy first**: All core functions take and return NumPy arrays
3. **Optional imports**: PyTorch and h5py are imported lazily, inside functions or behind `try/except`
4. **Docstrings**: Every public function documents input/output shapes explicitly (e.g., `(F, J, 4)`)
5. **No pybvh internals**: Only use pybvh's public API. Never import private functions or access private attributes
6. **Tests**: pytest, same conventions as pybvh. Test with and without optional dependencies installed

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

Unit tests are in `tests/test_pybvh_ml.py` (22 test classes) plus `tests/test_torch_datasets.py` (3 classes, module-level `pytest.importorskip("torch")` so the suite collects without torch), `tests/test_no_pybvh_deprecation.py`, `tests/test_docs_api_coverage.py` (the API reference stays two-way in sync with the modules; `__all__` names resolve), and `tests/test_gallery_notebook.py` (the gallery jupytext pair stays synced and its committed outputs fresh) — 357 tests total; `tests/integration/` adds real-data sweeps (representation parity, seeding determinism, pipeline staging, end-to-end MLP training). Test BVH files are in `bvh_data/` at the project root.

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
