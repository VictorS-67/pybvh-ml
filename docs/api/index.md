# API Reference

Everything public is importable from the top-level namespace (`import pybvh_ml`) — see `pybvh_ml.__all__`. The PyTorch layer lives in `pybvh_ml.torch` and imports only if torch is installed. The pages in this section group the reference by module.

## Find a function

The fastest route from "I want to…" to the exact call.

| I want to… | Call | Reference |
|---|---|---|
| Preprocess a BVH directory to one dataset file | `preprocess_directory("data/", "train.npz")` | [Preprocessing](preprocessing.md) |
| Load a preprocessed dataset | `load_preprocessed("train.npz")` | [Preprocessing](preprocessing.md) |
| Reconcile mixed skeletons / fps / up-axes | `preprocess_directory(..., harmonize=True)` | [Preprocessing guide](../guide/preprocessing.md#harmonizing-heterogeneous-datasets) |
| Resample a corpus to a training frame rate | `preprocess_directory(..., target_fps=30)` | [Preprocessing guide](../guide/preprocessing.md#frame-rate) |
| Compute / apply z-score normalization | `compute_normalization_stats(bvhs)`, `normalize_array(x, stats)` | [Preprocessing](preprocessing.md) |
| Pack arrays into a model layout | `pack_to_ctv(root_pos, jd)`, `pack_to_tvc(...)`, `pack_to_flat(...)` | [Packing](packing.md) |
| Unpack a model layout back to arrays | `unpack_from_ctv(x)`, `unpack_from_tvc(x)`, `unpack_from_flat(x)` | [Packing](packing.md) |
| Hold one clip's arrays | `MotionArrays(root_pos=…, joint_rot=…)` | [Motion Arrays](arrays.md) |
| Rotate a clip around the up axis | `rotate_vertical(arrays, angle=…, up_axis=…)` | [Augmentation](augmentation.md) |
| Mirror a clip left/right | `mirror(arrays, lr_joint_pairs=…, lateral_axis=…)` | [Augmentation](augmentation.md) |
| Perturb speed / drop frames / add noise | `speed_perturbation_arrays(...)`, `dropout_arrays(...)`, `add_joint_rotation_noise(...)`, `add_root_position_noise(...)` | [Augmentation](augmentation.md) |
| Compose augmentations with probabilities | `AugmentationPipeline([...])` / `AugmentationPipeline.standard(skel)` | [Pipeline](pipeline.md) |
| Convert rotation representations on arrays | `convert_arrays(jd, "euler", "6d", euler_orders=…)` | [Conversion](convert.md) |
| Get graph edges / L/R pairs / body parts | `get_edge_list(bvh)`, `get_lr_pairs(bvh)`, `get_body_partitions(bvh)` | [Skeleton](skeleton.md) |
| Get all skeleton metadata at once | `get_skeleton_info(bvh)` | [Skeleton](skeleton.md) |
| Cut sliding windows | `sliding_window(data, window_size=64, stride=32)` | [Sequences](sequences.md) |
| Pad / crop to a fixed length | `standardize_length(data, target_length=128)` | [Sequences](sequences.md) |
| Sample frames PySKL-style | `sample_temporal(data, clip_length=64, mode="train")` | [Sequences](sequences.md) |
| Know what each packed column means | `describe_features(num_joints=24, representation="6d")` | [Metadata](metadata.md) |
| Build a PyTorch Dataset from a preprocessed file | `MotionDataset.from_preprocessed(loaded)` | [PyTorch](torch.md) |
| Build a PyTorch Dataset from clip dicts | `MotionDataset(clips, target_length=128)` | [PyTorch](torch.md) |
| Build a Dataset straight from BVH paths | `OnTheFlyDataset(paths, representation="6d")` | [PyTorch](torch.md) |
| Feed a GCN fixed-budget `(C, T, V)` clips | `MotionDataset(..., layout="ctv", temporal="resample")` | [PyTorch guide](../guide/pytorch.md#shaping-the-clip-temporal-and-layout) |
| Batch variable-length clips with masks | `DataLoader(ds, collate_fn=collate_motion_batch)` | [PyTorch](torch.md) |
| Seed your own Dataset the same way | `rng_for(seed, epoch, idx)`, `EpochState()` | [PyTorch guide](../guide/pytorch.md#using-the-seeding-scheme-in-your-own-dataset) |

## Modules at a glance

| Module | Owns | Page |
|---|---|---|
| `pybvh_ml.packing` | `(C, T, V)` / `(T, V, C)` / flat `(T, D)` layout conversion, both directions | [Packing](packing.md) |
| `pybvh_ml.augmentation` | the five array-level augmentation functions | [Augmentation](augmentation.md) |
| `pybvh_ml.pipeline` | `AugmentationPipeline` — composition, probabilities, quat caching | [Pipeline](pipeline.md) |
| `pybvh_ml.preprocessing` | directory → dataset file, harmonization, normalization stats | [Preprocessing](preprocessing.md) |
| `pybvh_ml.sequences` | sliding windows, length standardization, temporal sampling | [Sequences](sequences.md) |
| `pybvh_ml.skeleton` | graph metadata: edges, L/R pairs, partitions, `skeleton_info` | [Skeleton](skeleton.md) |
| `pybvh_ml.convert` | `convert_arrays` — representation conversion on `(F, J, C)` arrays | [Conversion](convert.md) |
| `pybvh_ml.metadata` | `FeatureDescriptor` / `describe_features` column maps | [Metadata](metadata.md) |
| `pybvh_ml.torch` | optional: Dataset classes and the collate function | [PyTorch](torch.md) |
