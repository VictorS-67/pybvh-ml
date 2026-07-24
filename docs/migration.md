# Migrating to 0.5

pybvh-ml 0.5.0 is a coordinated release with pybvh 0.8.0 — one atomic migration, no shims, no dual names. **Upgrade pybvh first** (it ships its own [consolidated migration table](https://github.com/VictorS-67/pybvh/blob/main/CHANGELOG.md)), then pybvh-ml; the 0.5 pin `pybvh>=0.8,<0.9` enforces the pairing.

This page covers the changes that need action in your code. The full record — including the bug fixes that change outputs without changing call sites — is in the [CHANGELOG](https://github.com/VictorS-67/pybvh-ml/blob/main/CHANGELOG.md).

## At a glance

| Change | Migration |
|---|---|
| Requires **pybvh >= 0.8, < 0.9** | Upgrade pybvh first. |
| Representation token `"quaternion"` → `"quat"` | Rename at every call site; stored dataset metadata now reads back as `"quat"`. |
| Angles are radians: `angle_deg=` → `angle=`, `sigma_deg=` → `sigma=`, `noise_sigma_deg=` → `noise_sigma=` | Rename the kwarg and wrap old degree values in `np.radians(...)`. |
| Dataset `length` / batch `lengths` now mean valid frames in the returned tensor | Cropped clips report `target_length`; recover original lengths upstream if you need them. |
| Invalid augmentation inputs raise `ValueError` | Fix the offending values — they were silent no-ops or opaque errors before. |
| `AugmentationPipeline(cache_quats=True)` (the default) requires a `representation=` declaration | Declare it on at least one step, or pass `cache_quats=False`. |
| `harmonize=True` no longer retargets bone offsets to the first clip | Pass `retarget=True` to restore the 0.4.0 behavior. |
| Normalization trio moved here from pybvh | Import from `pybvh_ml` instead of `pybvh`. |

## Representation tokens

pybvh 0.8 shortened its representation tokens; pybvh-ml adopts them everywhere a token is accepted or reported (`extract_repr`, `describe_features`, `convert_arrays`, augmentation, `preprocess_directory`):

```python
# 0.4.0
preprocess_directory("data/", "train.npz", representation="quaternion")

# 0.5.0
preprocess_directory("data/", "train.npz", representation="quat")
```

The full token set is `"euler"`, `"quat"`, `"6d"`, `"axisangle"`, `"rotmat"`. Datasets preprocessed under 0.4.0 carry the old token in their stored metadata — re-preprocess, or read the stored value as `"quat"` going forward.

## Radians everywhere

Augmentation parameters are radians, matching pybvh 0.8:

```python
# 0.4.0
rotate_vertical(..., angle_deg=90)
add_joint_noise(..., sigma_deg=1.0)
AugmentationPipeline.standard(skel, noise_sigma_deg=1.0)

# 0.5.0
rotate_vertical(..., angle=np.radians(90))
add_joint_noise(..., sigma=np.radians(1.0))
AugmentationPipeline.standard(skel, noise_sigma=np.radians(1.0))
```

`rotate_angle_range` defaults to `(-np.pi, np.pi)`; `sigma_pos` is in positional units and is unchanged.

!!! warning "Euler outputs change even without touching your code"
    0.4.0 silently treated euler *joint data* as degrees during internal conversions while pybvh has stored radians since 0.7 — every euler-representation augmentation and euler-side `convert_arrays` call shrank rotations ~57× into quaternion space and re-inflated them on the way out. 0.5.0 fixes this, so euler-path outputs are different — and now correct, verified against pybvh ground truth. Rotmat augmentation (which raised in 0.4.0) and mixed-Euler-order mirroring are likewise fixed.

## Harmonize no longer retargets by default

In 0.4.0, `preprocess_directory(harmonize=True)` silently pinned the alphabetically first file as the harmonize reference, overwriting every actor's bone proportions. In 0.5.0 harmonization is pure reorientation/resampling; per-actor bone lengths are preserved:

```python
# 0.5.0 — restore the 0.4.0 retarget-to-first behavior explicitly:
preprocess_directory("raw/", "train.npz", harmonize=True, retarget=True)
```

Hierarchy mismatches raise loudly either way. The choice is recorded in the persisted [uniformity audit](guide/preprocessing.md#what-the-file-stores).

## Dataset `length` semantics

`MotionDataset` / `OnTheFlyDataset` items and `collate_motion_batch` batches now report the number of **valid frames actually present in the returned tensor**: padded clips report their original length, cropped clips report `target_length` (0.4.0 reported the pre-crop length, producing masks larger than the tensor). Code that used `lengths` to recover original clip lengths must read them from the clip arrays before the dataset.

## Normalization trio import

pybvh 0.8 removed `compute_normalization_stats` / `normalize_array` / `denormalize_array` from `pybvh.batch`; they live in pybvh-ml now, same signatures:

```python
# 0.4.0
from pybvh import compute_normalization_stats, normalize_array, denormalize_array

# 0.5.0
from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array
```

## Fail-loud validation

Inputs that used to silently misbehave now raise `ValueError`: negative `sigma` / `sigma_pos`, `drop_rate` outside `[0, 1)`, mismatched `root_pos` / `joint_data` frame counts, empty joint arrays in `rotate_vertical`, zero-norm quaternions in `add_joint_noise`, `standardize_length(target_length < 1)`, non-divisible `unpack_from_flat` channel counts, and unrecognized dataset file extensions (which previously wrote to a *different path* via `np.savez`'s silent `.npz` suffixing).

## `set_epoch` now actually reaches DataLoader workers

Not an API change — a contract now honored. In 0.4.0, with `num_workers > 0` (and always with `persistent_workers=True`), workers kept the epoch pickled at startup and silently replayed epoch-0 augmentation forever. The epoch now lives in shared memory and the documented contract holds in every DataLoader configuration. Consequence: dataset instances can't be `deepcopy`-ed or `torch.save`-ed — details in the [PyTorch guide](guide/pytorch.md#reproducible-per-epoch-augmentation). `set_epoch` also rejects negative epochs now.
