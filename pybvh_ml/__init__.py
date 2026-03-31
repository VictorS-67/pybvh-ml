"""pybvh-ml: ML bridge layer for pybvh motion capture data."""
from __future__ import annotations

__version__ = "0.2.0"

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
    rotate_quaternions_vertical,
    mirror_quaternions,
    speed_perturbation_arrays,
    dropout_arrays,
    add_joint_noise_quaternions,
    rotate_rot6d_vertical,
    mirror_rot6d,
)
from .convert import convert_arrays
from .pipeline import AugmentationPipeline
from .preprocessing import preprocess_directory, load_preprocessed

# torch subpackage is imported by user as: from pybvh_ml.torch import ...
