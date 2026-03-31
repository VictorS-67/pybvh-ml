# AI_CONTEXT.md — pybvh-ml

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
| Normalization stats computation | pybvh |
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
- pybvh 0.4.0 (editable install)

**Important**: pybvh has its own separate conda env (`pybvh`) with no torch/h5py. pybvh-ml code must never require torch or h5py for core functionality — they are optional, guarded with try/except.

---

## 4. Architecture Overview

```
pybvh-ml/
├── pybvh_ml/
│   ├── __init__.py              # Public API (27 exports)
│   ├── packing.py               # Tensor layout conversion (CTV, TVC, flat)
│   ├── augmentation.py          # Array-level augmentation (quat + 6D, no Bvh objects)
│   ├── convert.py               # Representation conversion dispatch
│   ├── pipeline.py              # Composable AugmentationPipeline
│   ├── sequences.py             # Sequence length utilities (windowing, standardization)
│   ├── skeleton.py              # Skeleton graph metadata (edges, partitions)
│   ├── preprocessing.py         # Batch BVH → on-disk dataset pipelines
│   ├── metadata.py              # Feature column descriptors
│   └── torch/                   # Optional PyTorch integration
│       ├── __init__.py
│       ├── datasets.py          # Dataset / IterableDataset classes
│       └── collate.py           # Collate functions for variable-length sequences
├── bvh_data/                    # Test BVH files (bvh_example, bvh_test1-3, standard_skeleton)
├── tests/
│   └── test_pybvh_ml.py         # 191 tests across 14 test classes
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
- Quaternion-space: `rotate_quaternions_vertical`, `mirror_quaternions`, `speed_perturbation_arrays`, `dropout_arrays`, `add_joint_noise_quaternions` — operate on `(F, J, 4)` arrays using pybvh's `quat_slerp` and rotation primitives
- 6D-space: `rotate_rot6d_vertical`, `mirror_rot6d` — operate directly on `(F, J, 6)` arrays. Apply rotation matrix to the two 3D column vectors, re-orthogonalize via Gram-Schmidt. Avoids unnecessary 6D → quat → 6D round-trip in hot data loader paths

**`convert.py`** — Representation conversion
- `convert_arrays(joint_data, from_repr, to_repr, euler_orders)` — unified representation conversion dispatch for `(F, J, C)` arrays, wrapping pybvh's rotation functions

**`pipeline.py`** — Composable augmentation pipeline
- `AugmentationPipeline` — composable sequence with per-augmentation probabilities and seeded rng. Supports callable kwargs (`lambda rng: value`) for per-sample random parameter sampling. Automatically forwards `rng` to functions that accept it (via signature inspection)

**`sequences.py`** — Sequence length utilities
- `sliding_window(data, window_size, stride)` → `(num_windows, window_size, ...)` fixed-length windows
- `standardize_length(data, target_length, method)` → crop, pad, or resample to fixed frame count
- `uniform_temporal_sample(num_frames, clip_length, mode, rng)` → PySKL-style uniform segment sampling with three regimes (short/wrapping, dense/gap-insertion, uniform/segment-based)
- `sample_temporal(data, clip_length, num_samples, mode, rng)` → convenience wrapper that applies sampled indices with wraparound

**`skeleton.py`** — Skeleton graph metadata
- `get_edge_list(bvh, include_end_sites=False)` → `list[(child_idx, parent_idx)]`
- `get_body_partitions(bvh)` → `dict[str, list[int]]` mapping body part names to joint indices
- `get_lr_pairs(bvh)` → `list[(left_idx, right_idx)]` (wraps pybvh's `auto_detect_lr_mapping` into index pairs)
- `get_skeleton_info(bvh)` → unified dict with edges, partitions, L/R pairs, joint names, euler orders

**`preprocessing.py`** — Batch preprocessing pipelines
- `preprocess_directory(bvh_dir, output_path, representation, ...)` — BVH directory → on-disk dataset
- Supports output formats: `.npz`, `.hdf5` (if h5py installed)
- Stores arrays + skeleton metadata + normalization stats in a single file
- Optional label function `label_fn(filename) → int`
- Optional filter function `filter_fn(filename_stem) → bool` — applied before loading, skipped files are never parsed

**`metadata.py`** — Feature column descriptors
- `FeatureDescriptor` — describes which columns correspond to which features in a packed array
- `describe_features(representation, include_root_pos, include_velocities, ...)` → `FeatureDescriptor`
- Enables programmatic access to feature slices without hardcoded column indices

**`torch/datasets.py`** — PyTorch Dataset classes (optional, only if torch is installed)
- `MotionDataset(Dataset)` — loads preprocessed data from disk, returns tensors
- `OnTheFlyDataset(Dataset)` — loads raw arrays, applies augmentation each epoch
- Both support variable-length sequences with configurable padding/cropping

**`torch/collate.py`** — Collate functions
- `collate_motion_batch(batch)` — handles variable-length sequences with padding and mask generation
- Returns `(data_tensor, lengths_tensor, mask_tensor, labels_tensor)`

---

## 5. Key Design Decisions

### 5.1 Supported augmentation representations: quaternion and 6D
Array-level augmentation supports two representation families:
- **Quaternion `(F, J, 4)`** — full augmentation suite (rotation, mirror, speed perturbation, dropout). Quaternion math is clean: no gimbal lock, SLERP is well-defined.
- **6D `(F, J, 6)`** — rotation and mirror only. 6D is the dominant training representation in recent motion ML. Avoiding the 6D → quat → augment → quat → 6D round-trip saves compute in hot data loader paths. The round-trip is lossless (unlike Euler), so this is a performance choice, not a correctness one.
- **Euler arrays are NOT directly augmented** by pybvh-ml. Euler augmentation lives in pybvh itself (`transforms.rotate_angles_vertical`, `mirror_angles`). pybvh-ml focuses on the representations ML pipelines actually train with.
- **Axis-angle** — deferred. Niche (mainly SMPL/SMPL-X). Users can convert to quaternion and back.

### 5.2 Preprocessing and runtime are separate
- **Preprocessing** (`preprocessing.py`): runs once, converts BVH files to arrays on disk. Slow is OK.
- **Runtime** (`augmentation.py`, `torch/datasets.py`): runs every epoch, must be fast. Operates on pre-extracted arrays, never touches BVH files.

### 5.3 PyTorch is optional
All `torch/` imports are guarded. The core modules (`packing`, `augmentation`, `skeleton`, `preprocessing`, `metadata`) work with pure NumPy. Users who don't use PyTorch can still use pybvh-ml for preprocessing and array manipulation.

### 5.4 pybvh API surface that pybvh-ml relies on
pybvh-ml uses these pybvh entry points:
- `pybvh.read_bvh_file()`, `pybvh.read_bvh_directory()` — loading
- `bvh.root_pos`, `bvh.joint_angles`, `bvh.joint_count`, `bvh.joint_names` — data access
- `bvh.get_frames_as_quaternion()`, `bvh.get_frames_as_6d()`, etc. — representation conversion
- `bvh.euler_orders` — per-joint Euler order strings
- `bvh.edges` — skeleton edge list as `(child_idx, parent_idx)` tuples
- `bvh.nodes`, `bvh.node_index` — skeleton topology
- `pybvh.transforms.auto_detect_lr_pairs()` — L/R joint pair detection as index tuples
- `pybvh.rotations.*` — rotation conversion primitives
- `pybvh.features.*` — motion analysis features (velocities, foot contacts, etc.)
- `pybvh.batch.compute_normalization_stats()` — normalization
- `pybvh.tools.rotX`, `rotY`, `rotZ` — elementary rotation matrices

### 5.5 Joint noise is quaternion-only
`add_joint_noise_quaternions` generates noise as random axis-angle perturbations (random axis on the unit sphere, angle from N(0, sigma_deg)), converts to quaternion, and composes via Hamilton product. This avoids gimbal lock sensitivity and gives uniform perturbation regardless of pose. No 6D variant — same rationale as speed perturbation and dropout (the math is naturally quaternion-based).

### 5.6 Callable kwargs and rng forwarding in AugmentationPipeline
Kwargs values can be callables of the form `lambda rng: value`, resolved at each invocation. This enables per-sample random parameter sampling (e.g., random rotation angles) without modifying augmentation function signatures. Static kwargs continue to work unchanged.

The pipeline automatically forwards its `rng` to augmentation functions that accept an `rng` parameter (detected via `inspect.signature`). This ensures reproducibility for functions like `dropout_arrays` and `add_joint_noise_quaternions` without requiring explicit `"rng": lambda rng: rng` in kwargs. If the user provides an explicit `rng` kwarg (static or callable), it takes precedence over the auto-forwarded one.

### 5.7 Uniform temporal sampling matches PySKL
`uniform_temporal_sample` reproduces the PySKL/MMAction2 `UniformSampleFrames` algorithm as a stateless function. Three regimes:
- **Short** (`num_frames < clip_length`): sequential `[start..start+clip_length-1]` with random start (train) or start=0 (test). Caller applies `% num_frames`.
- **Dense** (`clip_length <= num_frames < 2*clip_length`): starts with `[0..clip_length-1]`, randomly inserts gaps to spread indices across the full range.
- **Uniform** (`num_frames >= 2*clip_length`): integer segment boundaries (`i * num_frames // clip_length`), discrete random offset per segment.

### 5.8 Implementation-level conventions
1. **All augmentation functions return `(joint_data, root_pos)`** — joint data first, matching pybvh's convention in `rotate_angles_vertical` and `mirror_angles`.
2. **`convert_arrays` routes through rotation matrices** as intermediate. Per-joint Euler orders are handled by grouping joints by unique order and batch-converting each group.
3. **Packing zero-pads root only** — root has 3 channels (position), joints have C_joint channels. In CTV/TVC layouts, `C = max(3, C_joint)`. Since C_joint >= 3 for all real representations, joint data is never padded.
4. **Mirror math**: quaternion mirror negates the two imaginary components NOT at the lateral axis. 6D mirror uses `R'[i,j] = s_i * s_j * R[i,j]` where `s[lateral] = -1`. Both derived from `R' = S @ R @ S`.
5. **`_quat_multiply` is local to `augmentation.py`** — pybvh doesn't export quaternion multiplication. If pybvh adds one later, switch to it.
6. **`torch/` subpackage fails hard on import if torch is missing** — `pybvh_ml.torch` raises ImportError. But `import pybvh_ml` (the top-level) works fine without torch.

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

Tests are in `tests/test_pybvh_ml.py` (201 tests, 16 test classes). Test BVH files are in `bvh_data/` at the project root.

**Fixtures**:
- `bvh_example` — loads `bvh_data/bvh_example.bvh` (24 joints, 56 frames, ZYX)
- `bvh_test3` — loads `bvh_data/bvh_test3.bvh` (60 joints, mixed euler orders)
- `rng` — `np.random.default_rng(42)`

**Conventions**:
- `@pytest.mark.parametrize` for representation names, channel counts, axis indices
- `np.testing.assert_allclose` for numerical comparisons
- Round-trip tests: pack/unpack, convert/convert-back, mirror/mirror, augment/inverse
- Consistency tests: quaternion augmentation must match Euler augmentation (via pybvh's `rotate_angles_vertical`) after conversion
- Real BVH integration: tests use actual BVH data, not synthetic arrays
- Shape assertions: explicit `assert result.shape == (F, J, C)`

**Note**: `bvh_example` and `bvh_test1` share a skeleton (24 joints, ZYX). `bvh_test2` and `bvh_test3` have different skeletons. `compute_normalization_stats` rejects mixed skeletons — tests that batch multiple files must use compatible ones.

---

## 9. Lessons Learned

1. **pybvh's rotation functions originally didn't support 3D batch dims** — `euler_to_rotmat((F, J, 3))` crashed. Fixed in pybvh v0.4.0 by flattening to 2D, processing, reshaping back. If you encounter similar issues with pybvh primitives, the fix pattern is: flatten leading dims, call the function, reshape.

2. **Euler angle round-trips are not unique** — converting Euler → rotmat → Euler may give different angles that represent the same rotation (especially near gimbal lock). Always compare via rotation matrices, not raw Euler angles.

3. **`standardize_length(method="resample")` uses linear interpolation** — not suitable for rotation data. A `warnings.warn` is emitted at runtime. Users should call `pybvh.Bvh.resample()` (SLERP-based) before extracting arrays instead.
