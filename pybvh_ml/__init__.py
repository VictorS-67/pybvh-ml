"""pybvh-ml: ML bridge layer for pybvh motion capture data."""
from __future__ import annotations

__version__ = "0.5.0"

from .arrays import MotionArrays
from .packing import (
    pack_to_ctv,
    pack_to_tvc,
    pack_to_flat,
    unpack_from_ctv,
    unpack_from_tvc,
    unpack_from_flat,
)
from .skeleton import (
    get_edge_list,
    get_lr_pairs,
    get_skeleton_info,
    get_body_partitions,
)
from .sequences import (
    sliding_window,
    standardize_length,
    uniform_temporal_sample,
    sample_temporal,
)
from .metadata import (
    FeatureDescriptor,
    describe_features,
)
from .augmentation import (
    rotate_vertical,
    mirror,
    add_joint_rotation_noise,
    add_root_position_noise,
    speed_perturbation_arrays,
    dropout_arrays,
)
from .convert import convert_arrays, convert_rotations
from .pipeline import AugmentationPipeline, AugmentationStep
from .preprocessing import (
    preprocess_directory,
    load_preprocessed,
    compute_normalization_stats,
    normalize_array,
    denormalize_array,
)

# torch subpackage is imported by user as: from pybvh_ml.torch import ...

__all__ = [
    "__version__",

    # arrays
    "MotionArrays",
    # packing
    "pack_to_ctv",
    "pack_to_tvc",
    "pack_to_flat",
    "unpack_from_ctv",
    "unpack_from_tvc",
    "unpack_from_flat",
    # skeleton
    "get_edge_list",
    "get_lr_pairs",
    "get_skeleton_info",
    "get_body_partitions",
    # sequences
    "sliding_window",
    "standardize_length",
    "uniform_temporal_sample",
    "sample_temporal",
    # metadata
    "FeatureDescriptor",
    "describe_features",
    # augmentation
    "rotate_vertical",
    "mirror",
    "add_joint_rotation_noise",
    "add_root_position_noise",
    "speed_perturbation_arrays",
    "dropout_arrays",
    "convert_arrays",
    "convert_rotations",
    "AugmentationPipeline",
    "AugmentationStep",
    # preprocessing
    "preprocess_directory",
    "load_preprocessed",
    "compute_normalization_stats",
    "normalize_array",
    "denormalize_array",
]
