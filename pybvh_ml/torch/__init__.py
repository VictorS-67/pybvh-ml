"""PyTorch integration for pybvh-ml.

Requires ``torch`` to be installed.  All exports are conditional —
importing ``pybvh_ml`` without torch will not raise an error, but
importing ``pybvh_ml.torch`` without torch will.
"""
from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ImportError:
    raise ImportError(
        "pybvh_ml.torch requires PyTorch. Install with: "
        "pip install torch")

from .datasets import MotionDataset, OnTheFlyDataset
from .collate import collate_motion_batch
