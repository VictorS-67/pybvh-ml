"""Shared fixtures for the pybvh-ml unit test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pybvh import read_bvh_file  # noqa: E402


@pytest.fixture
def bvh_example():
    return read_bvh_file(
        Path(__file__).parent.parent / "bvh_data" / "bvh_test1.bvh")


@pytest.fixture
def bvh_test3():
    return read_bvh_file(
        Path(__file__).parent.parent / "bvh_data" / "bvh_test3.bvh")


@pytest.fixture
def rng():
    return np.random.default_rng(42)
