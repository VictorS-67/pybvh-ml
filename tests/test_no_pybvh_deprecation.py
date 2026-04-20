"""Regression guard: pybvh-ml must not trigger pybvh DeprecationWarnings.

Exercises the code paths most likely to drift back to deprecated names
(``get_frames_as_*``, ``get_rest_pose``, ``add_joint_noise``, etc.) and
asserts that no warnings with ``pybvh`` in the module path were emitted.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import pybvh
from pybvh import read_bvh_file

from pybvh_ml.preprocessing import (
    preprocess_directory, load_preprocessed, extract_repr,
)
from pybvh_ml.skeleton import get_edge_list, get_lr_pairs, get_skeleton_info


BVH_DIR = Path(__file__).parent.parent / "bvh_data"


def _pybvh_deprecations(records):
    """Filter warning records to those originating from pybvh sources."""
    pybvh_pkg = Path(pybvh.__file__).parent
    out = []
    for r in records:
        if not issubclass(r.category, DeprecationWarning):
            continue
        try:
            fname = Path(r.filename).resolve()
        except OSError:
            continue
        if str(fname).startswith(str(pybvh_pkg)):
            out.append(r)
    return out


@pytest.fixture
def bvh_example():
    return read_bvh_file(BVH_DIR / "bvh_test1.bvh")


def test_extract_repr_no_pybvh_deprecation(bvh_example):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for rep in ("euler", "6d", "quaternion", "axisangle"):
            extract_repr(bvh_example, rep)
        assert _pybvh_deprecations(w) == []


def test_skeleton_helpers_no_pybvh_deprecation(bvh_example):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_edge_list(bvh_example)
        get_edge_list(bvh_example, include_end_sites=True)
        get_lr_pairs(bvh_example)
        get_skeleton_info(bvh_example, include_partitions=True)
        assert _pybvh_deprecations(w) == []


def test_preprocess_roundtrip_no_pybvh_deprecation(tmp_path):
    out = tmp_path / "dataset.npz"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        preprocess_directory(
            BVH_DIR, out, file_pattern="bvh_test1.bvh",
            include_quaternions=True,
        )
        load_preprocessed(out)
        assert _pybvh_deprecations(w) == []


def test_torch_dataset_no_pybvh_deprecation(tmp_path):
    """Pull a sample through OnTheFlyDataset (exercises extract_repr path)."""
    torch = pytest.importorskip("torch")
    from pybvh_ml.torch import OnTheFlyDataset

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = OnTheFlyDataset(
            [BVH_DIR / "bvh_test1.bvh"],
            representation="6d",
            target_length=32,
        )
        sample = ds[0]
        assert "data" in sample
        assert _pybvh_deprecations(w) == []
