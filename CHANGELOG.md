# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`AugmentationPipeline` now auto-forwards `rng`** to augmentation functions that accept it (via signature inspection). Previously, functions like `add_joint_noise_quaternions` and `dropout_arrays` received `rng=None` and created unseeded generators, breaking reproducibility. User-provided `rng` kwargs still take precedence.

### Added

- **`filter_fn` parameter in `preprocess_directory`** — `filter_fn(filename_stem) -> bool` allows excluding files before they are loaded and processed. Skipped files are never parsed, saving I/O and memory when preprocessing a subset of a large directory.

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
