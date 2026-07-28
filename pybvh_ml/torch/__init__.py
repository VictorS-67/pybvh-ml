"""PyTorch integration for pybvh-ml.

Requires ``torch`` to be installed.  All exports are conditional —
importing ``pybvh_ml`` without torch will not raise an error, but
importing ``pybvh_ml.torch`` without torch will.
"""
from __future__ import annotations

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise ImportError(
        "pybvh_ml.torch requires PyTorch. Install with: "
        "pip install torch") from None

# torch is installed — import it plainly so a *broken* installation
# (e.g. a missing CUDA library) surfaces its real traceback instead of
# a misleading "install torch" message.
import torch as _torch  # noqa: F401, E402

from .datasets import EpochState, MotionDataset, OnTheFlyDataset, rng_for
from .collate import collate_motion_batch

__all__ = [
    "MotionDataset",
    "OnTheFlyDataset",
    "collate_motion_batch",
    # Seeding primitives, usable without subclassing either Dataset.
    "EpochState",
    "rng_for",
]
