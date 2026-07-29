"""Shared fixtures for the pybvh-ml unit test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
# tests/ itself, so both this suite and tests/integration can import
# `helpers` without a package dance.
sys.path.insert(0, str(Path(__file__).parent))

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
def bvh_dir():
    return Path(__file__).parent.parent / "bvh_data"


@pytest.fixture
def bvh_paths():
    """Single-file path list for the on-the-fly Dataset tests."""
    bvh_dir = Path(__file__).parent.parent / "bvh_data"
    return sorted(bvh_dir.glob("bvh_test1.bvh"))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# A rig whose every offset is zero: the rest pose spans no direction, so
# ``Bvh.rest_up`` is None ("degenerate") while ``world_up`` still falls
# back to a string.  Written inline rather than committed as a fixture
# file because it is a deliberately broken skeleton, not sample data.
DEGENERATE_RIG_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  End Site
  {
    OFFSET 0.0 0.0 0.0
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0
"""


@pytest.fixture
def degenerate_rig_dir(tmp_path):
    """Directory of two rigs with an unmeasurable rest-pose up axis."""
    d = tmp_path / "degenerate"
    d.mkdir()
    for stem in ("degen_a", "degen_b"):
        (d / f"{stem}.bvh").write_text(DEGENERATE_RIG_BVH)
    return d
