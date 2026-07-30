"""
Tests for pybvh-ml library.

Run with: pytest tests/test_pybvh_ml.py -v
"""

import functools
import warnings

import pytest
import numpy as np
from pathlib import Path

from pybvh import read_bvh_file

from pybvh_ml.packing import (
    pack_to_ctv, pack_to_tvc, pack_to_flat,
    unpack_from_ctv, unpack_from_tvc, unpack_from_flat,
)
from pybvh_ml.skeleton import get_edge_list, get_lr_pairs, get_skeleton_info
from pybvh_ml.sequences import sliding_window, standardize_length
from pybvh_ml.metadata import FeatureDescriptor, describe_features
from pybvh_ml import MotionArrays
from helpers import as_pair, as_triple


# Shared fixtures (bvh_example, bvh_test3, rng) live in conftest.py.

# =============================================================================
# Packing
# =============================================================================

class TestPacking:
    """Tests for tensor layout packing and unpacking."""

    # --- Shape tests ---

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_ctv_shape(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        C = max(3, C_joint)
        assert packed.shape == (C, F, 1 + J)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_tvc_shape(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_tvc(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        C = max(3, C_joint)
        assert packed.shape == (F, 1 + J, C)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_flat_shape(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_flat(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        assert packed.shape == (F, 3 + J * C_joint)

    # --- Round-trip tests ---

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_ctv_roundtrip(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        rp_out, jd_out = as_pair(unpack_from_ctv(packed))
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)
        np.testing.assert_allclose(jd_out, joint_data, atol=1e-12)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_tvc_roundtrip(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_tvc(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        rp_out, jd_out = as_pair(unpack_from_tvc(packed))
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)
        np.testing.assert_allclose(jd_out[:, :, :C_joint], joint_data, atol=1e-12)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_flat_roundtrip(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_flat(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        rp_out, jd_out = as_pair(unpack_from_flat(
            packed, root_channels=3, joint_channels=C_joint))
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)
        np.testing.assert_allclose(jd_out, joint_data, atol=1e-12)

    # --- center_root tests ---

    def test_center_root_subtracts_first_frame(self, rng):
        F, J = 30, 10
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, 3))
        packed = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=True)
        rp_out, _ = as_pair(unpack_from_ctv(packed))
        # First frame root should be zero
        np.testing.assert_allclose(rp_out[0], 0.0, atol=1e-12)
        # Subsequent frames should be relative
        expected = root_pos - root_pos[0:1]
        np.testing.assert_allclose(rp_out, expected, atol=1e-12)

    def test_center_root_false_preserves_values(self, rng):
        F, J = 30, 10
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, 3))
        packed = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        rp_out, _ = as_pair(unpack_from_ctv(packed))
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)

    # --- Zero-padding tests ---

    def test_ctv_output_contiguous(self, rng):
        """(C, T, V) output is C-contiguous — consumers hand it to
        torch.from_numpy(...).view(...)."""
        root_pos = rng.normal(size=(10, 3))
        joint_data = rng.normal(size=(10, 5, 6))
        ctv = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=joint_data))
        assert ctv.flags["C_CONTIGUOUS"]

    def test_unpack_from_flat_indivisible_raises(self, rng):
        """Quat data (D = 3 + J*4) unpacked with the default
        joint_channels=3 must fail loudly, not mis-reshape."""
        root_pos = rng.normal(size=(10, 3))
        quats = rng.normal(size=(10, 5, 4))
        flat = pack_to_flat(MotionArrays(root_pos=root_pos, joint_rot=quats))
        with pytest.raises(ValueError, match="joint_channels"):
            unpack_from_flat(flat)  # default joint_channels=3; 20 % 3 != 0

    def test_ctv_root_zero_padded_for_6d(self, rng):
        """When C_joint=6, root occupies channels 0:3, channels 3:6 are zero."""
        F, J = 20, 10
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, 6))
        packed = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        # packed shape: (6, 20, 11). Root is vertex 0.
        root_vertex = packed[:, :, 0]  # (6, 20)
        # Channels 0:3 should have root data
        np.testing.assert_allclose(root_vertex[:3, :], root_pos.T, atol=1e-12)
        # Channels 3:6 should be zero (padding)
        np.testing.assert_allclose(root_vertex[3:, :], 0.0, atol=1e-12)

    # --- Integration with pybvh ---

    def test_pack_from_bvh_euler(self, bvh_example):
        """Pack actual BVH data in Euler representation."""
        packed = pack_to_ctv(MotionArrays(root_pos=bvh_example.root_pos, joint_rot=bvh_example.joint_angles), center_root=True)
        F = bvh_example.frame_count
        J = bvh_example.joint_count
        assert packed.shape == (3, F, 1 + J)

    def test_pack_from_bvh_6d(self, bvh_example):
        """Pack actual BVH data in 6D representation."""
        root_pos, rot6d = bvh_example.to_6d()
        packed = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=rot6d), center_root=True)
        F = bvh_example.frame_count
        J = bvh_example.joint_count
        assert packed.shape == (6, F, 1 + J)

    def test_pack_from_bvh_quaternion(self, bvh_example):
        """Pack actual BVH data in quaternion representation."""
        root_pos, quats = bvh_example.to_quat()
        packed = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=quats), center_root=True)
        F = bvh_example.frame_count
        J = bvh_example.joint_count
        assert packed.shape == (4, F, 1 + J)


# =============================================================================
# Skeleton
# =============================================================================

class TestSkeleton:
    """Tests for skeleton graph metadata."""

    def test_edge_list_count(self, bvh_example):
        edges = get_edge_list(bvh_example)
        assert len(edges) == bvh_example.joint_count - 1

    def test_edge_list_valid_indices(self, bvh_example):
        edges = get_edge_list(bvh_example)
        J = bvh_example.joint_count
        for child, parent in edges:
            assert 0 <= child < J
            assert 0 <= parent < J
            assert child != parent

    def test_edge_list_matches_bvh_edges(self, bvh_example):
        """get_edge_list(bvh) should match bvh.edges."""
        assert get_edge_list(bvh_example) == bvh_example.edges

    def test_edge_list_with_end_sites(self, bvh_example):
        edges = get_edge_list(bvh_example, include_end_sites=True)
        # More edges when end sites are included
        assert len(edges) > len(get_edge_list(bvh_example))
        # Total edges = total nodes - 1 (root has no parent)
        assert len(edges) == len(bvh_example.nodes) - 1

    def test_lr_pairs(self, bvh_example):
        pairs = get_lr_pairs(bvh_example)
        assert isinstance(pairs, list)
        assert len(pairs) > 0  # the fixture has Left/Right joints

    def test_lr_pairs_consistency(self, bvh_example):
        """get_lr_pairs should match pybvh's auto_detect_lr_pairs."""
        from pybvh.transforms import auto_detect_lr_pairs
        assert get_lr_pairs(bvh_example) == auto_detect_lr_pairs(bvh_example)

    def test_skeleton_info_keys(self, bvh_example):
        info = get_skeleton_info(bvh_example)
        assert set(info.keys()) == {
            'num_joints', 'joint_names', 'edges', 'euler_orders',
            'lr_pairs', 'lr_mapping', 'world_up', 'rest_forward',
            'rest_up'}

    def test_skeleton_info_values(self, bvh_example):
        info = get_skeleton_info(bvh_example)
        assert info['num_joints'] == bvh_example.joint_count
        assert info['joint_names'] == bvh_example.joint_names
        assert info['edges'] == bvh_example.edges
        assert info['euler_orders'] == bvh_example.euler_orders

    def test_different_skeletons(self, bvh_example, bvh_test3):
        info1 = get_skeleton_info(bvh_example)
        info3 = get_skeleton_info(bvh_test3)
        assert info1['num_joints'] != info3['num_joints']

    def test_skeleton_info_lr_mapping(self, bvh_example):
        """lr_mapping entry should mirror bvh.lr_mapping."""
        info = get_skeleton_info(bvh_example)
        if bvh_example.lr_mapping is None:
            assert info['lr_mapping'] is None
        else:
            assert info['lr_mapping'] == dict(bvh_example.lr_mapping)

    def test_lr_pairs_via_cached_mapping(self, bvh_example):
        """get_lr_pairs should agree with bvh.lr_pairs (the index-space cache).

        bvh.lr_mapping is bidirectional in pybvh 0.7.0 (each pair appears
        once as L→R and once as R→L), but bvh.lr_pairs remains
        single-direction — that's what get_lr_pairs delegates to.
        """
        if bvh_example.lr_pairs is None:
            pytest.skip("fixture has no L/R mapping")
        assert get_lr_pairs(bvh_example) == list(bvh_example.lr_pairs)
        # Sanity: every (l, r) in get_lr_pairs maps to a name pair present
        # in lr_mapping (the bidirectional dict).
        for li, ri in get_lr_pairs(bvh_example):
            l_name = bvh_example.joint_names[li]
            r_name = bvh_example.joint_names[ri]
            assert bvh_example.lr_mapping[l_name] == r_name
            assert bvh_example.lr_mapping[r_name] == l_name

    def test_lr_pairs_fallback(self, bvh_example):
        """With lr_mapping cleared, get_lr_pairs still returns the auto
        detection result."""
        from pybvh.transforms import auto_detect_lr_pairs
        bvh = bvh_example.copy()
        bvh.lr_mapping = None
        assert get_lr_pairs(bvh) == auto_detect_lr_pairs(bvh)


# =============================================================================
# Sequences
# =============================================================================

class TestSequences:
    """Tests for sequence length utilities."""

    @pytest.mark.parametrize("bad_target", [0, -3])
    def test_standardize_length_invalid_target_raises(self, bad_target):
        """Regression: negative targets used to silently return a
        wrong-length array via Python negative-slice truncation."""
        data = np.zeros((10, 5))
        with pytest.raises(ValueError, match="target_length must be >= 1"):
            standardize_length(data, bad_target)

    # --- sliding_window ---

    def test_window_shape_1d(self):
        data = np.arange(100, dtype=np.float64)
        result = sliding_window(data, window_size=10)
        assert result.shape == (91, 10)

    def test_window_shape_2d(self):
        data = np.zeros((100, 5), dtype=np.float64)
        result = sliding_window(data, window_size=10)
        assert result.shape == (91, 10, 5)

    def test_window_shape_3d(self):
        data = np.zeros((100, 24, 3), dtype=np.float64)
        result = sliding_window(data, window_size=20, stride=5)
        num_windows = (100 - 20) // 5 + 1
        assert result.shape == (num_windows, 20, 24, 3)

    def test_window_stride(self):
        data = np.arange(20, dtype=np.float64)
        result = sliding_window(data, window_size=5, stride=5)
        assert result.shape == (4, 5)
        np.testing.assert_array_equal(result[0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result[1], [5, 6, 7, 8, 9])

    def test_window_exact_fit(self):
        data = np.arange(10, dtype=np.float64)
        result = sliding_window(data, window_size=10)
        assert result.shape == (1, 10)
        np.testing.assert_array_equal(result[0], data)

    def test_window_too_large(self):
        data = np.arange(5, dtype=np.float64)
        with pytest.raises(ValueError, match="exceeds data length"):
            sliding_window(data, window_size=10)

    def test_window_is_copy(self):
        """Modifying the result should not affect the original."""
        data = np.arange(10, dtype=np.float64)
        result = sliding_window(data, window_size=5)
        result[0, 0] = 999.0
        assert data[0] == 0.0

    # --- standardize_length ---

    def test_pad_shorter(self):
        data = np.ones((5, 3), dtype=np.float64)
        result = standardize_length(data, target_length=10, method="pad")
        assert result.shape == (10, 3)
        np.testing.assert_array_equal(result[:5], 1.0)
        np.testing.assert_array_equal(result[5:], 0.0)

    def test_pad_longer(self):
        data = np.ones((20, 3), dtype=np.float64)
        result = standardize_length(data, target_length=10, method="pad")
        assert result.shape == (10, 3)
        np.testing.assert_array_equal(result, 1.0)

    def test_pad_exact(self):
        data = np.ones((10, 3), dtype=np.float64)
        result = standardize_length(data, target_length=10, method="pad")
        assert result.shape == (10, 3)
        np.testing.assert_array_equal(result, 1.0)

    def test_pad_custom_value(self):
        data = np.ones((5,), dtype=np.float64)
        result = standardize_length(data, target_length=10, method="pad",
                                    pad_value=-1.0)
        np.testing.assert_array_equal(result[5:], -1.0)

    def test_crop_center(self):
        data = np.arange(20, dtype=np.float64)
        result = standardize_length(data, target_length=10, method="crop")
        assert result.shape == (10,)
        # Center crop: start = (20 - 10) // 2 = 5
        np.testing.assert_array_equal(result, np.arange(5, 15, dtype=np.float64))

    def test_crop_shorter_pads(self):
        data = np.ones((5, 3), dtype=np.float64)
        result = standardize_length(data, target_length=10, method="crop")
        assert result.shape == (10, 3)
        np.testing.assert_array_equal(result[:5], 1.0)
        np.testing.assert_array_equal(result[5:], 0.0)

    def test_resample_double(self):
        """Resample 10 frames to 20 — linear interpolation."""
        data = np.linspace(0, 1, 10, dtype=np.float64).reshape(-1, 1)
        result = standardize_length(data, target_length=20, method="resample_linear")
        assert result.shape == (20, 1)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1], 1.0, atol=1e-10)

    def test_resample_half(self):
        """Resample 20 frames to 10."""
        data = np.linspace(0, 1, 20, dtype=np.float64).reshape(-1, 1)
        result = standardize_length(data, target_length=10, method="resample_linear")
        assert result.shape == (10, 1)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1], 1.0, atol=1e-10)

    def test_resample_preserves_3d_shape(self):
        data = np.zeros((50, 24, 3), dtype=np.float64)
        result = standardize_length(data, target_length=30, method="resample_linear")
        assert result.shape == (30, 24, 3)

    def test_resample_same_length(self):
        data = np.ones((10, 3), dtype=np.float64)
        result = standardize_length(data, target_length=10, method="resample_linear")
        np.testing.assert_array_equal(result, data)

    def test_unknown_method(self):
        data = np.ones((10,), dtype=np.float64)
        with pytest.raises(ValueError, match="Unknown method"):
            standardize_length(data, target_length=5, method="invalid")


# =============================================================================
# Metadata
# =============================================================================

class TestMetadata:
    """Tests for feature column descriptors."""

    def test_euler_descriptor(self):
        desc = describe_features(24, representation="euler")
        assert desc.total_dim == 3 + 24 * 3  # 75
        assert desc["root_pos"] == (0, 3)
        assert desc["joint_rotations"] == (3, 75)

    def test_6d_descriptor(self):
        desc = describe_features(24, representation="6d")
        assert desc.total_dim == 3 + 24 * 6  # 147
        assert desc["root_pos"] == (0, 3)
        assert desc["joint_rotations"] == (3, 147)

    def test_quaternion_descriptor(self):
        desc = describe_features(24, representation="quat")
        assert desc.total_dim == 3 + 24 * 4  # 99

    def test_no_root_pos(self):
        desc = describe_features(24, representation="euler", include_root_pos=False)
        assert desc.total_dim == 24 * 3  # 72
        assert "root_pos" not in desc
        assert desc["joint_rotations"] == (0, 72)

    def test_contiguous_ranges(self):
        """All ranges should be contiguous — no gaps."""
        desc = describe_features(24, representation="6d")
        sorted_ranges = sorted(desc.ranges.values())
        for i in range(len(sorted_ranges) - 1):
            assert sorted_ranges[i][1] == sorted_ranges[i + 1][0]
        assert sorted_ranges[-1][1] == desc.total_dim

    def test_slice_method(self):
        desc = describe_features(24, representation="euler")
        s = desc.slice("root_pos")
        assert s == slice(0, 3)

    def test_contains(self):
        desc = describe_features(24, representation="euler")
        assert "root_pos" in desc
        assert "joint_rotations" in desc
        assert "nonexistent" not in desc

    def test_unknown_representation(self):
        with pytest.raises(ValueError, match="Unknown representation"):
            describe_features(24, representation="invalid")

    @pytest.mark.parametrize("repr_name,expected_c", [
        ("euler", 3), ("axisangle", 3), ("quat", 4),
        ("6d", 6), ("rotmat", 9),
    ])
    def test_all_representations(self, repr_name, expected_c):
        desc = describe_features(10, representation=repr_name)
        assert desc.total_dim == 3 + 10 * expected_c


# =============================================================================
# Phase 2: Augmentation
# =============================================================================

from pybvh_ml.augmentation import (
    rotate_vertical, mirror,
    speed_perturbation_arrays, dropout_arrays,
    add_joint_rotation_noise,
    add_root_position_noise,
)
from pybvh_ml.sequences import uniform_temporal_sample, sample_temporal
from pybvh_ml.convert import convert_rotations
from pybvh_ml.pipeline import AugmentationPipeline, AugmentationStep


def _get_quat_data(bvh):
    """Helper: extract quaternion arrays from a Bvh."""
    return bvh.to_quat()


def _get_6d_data(bvh):
    """Helper: extract 6D arrays from a Bvh."""
    return bvh.to_6d()


def _get_mirror_metadata(bvh):
    """Helper: get L/R pairs and signed lateral / up axis strings."""
    from pybvh.transforms import auto_detect_lr_pairs
    return auto_detect_lr_pairs(bvh), bvh.left_at(0), bvh.world_up


# =============================================================================
# Quaternion augmentation
# =============================================================================

class TestQuaternionAugmentation:
    """Tests for quaternion-space augmentation functions."""

    # --- rotate_vertical (quaternion) ---

    def test_rotate_quat_shape(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(45.0), up_axis="+y", representation="quat"))
        assert new_quats.shape == quats.shape
        assert new_pos.shape == pos.shape

    def test_rotate_quat_zero_is_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=0.0, up_axis="+y", representation="quat"))
        np.testing.assert_allclose(new_quats, quats, atol=1e-10)
        np.testing.assert_allclose(new_pos, pos, atol=1e-10)

    def test_rotate_quat_360_is_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(360.0), up_axis="+y", representation="quat"))
        np.testing.assert_allclose(new_pos, pos, atol=1e-10)
        # Quaternions: q and -q represent same rotation
        for f in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                q_orig = quats[f, j]
                q_new = new_quats[f, j]
                match = (np.allclose(q_new, q_orig, atol=1e-10)
                         or np.allclose(q_new, -q_orig, atol=1e-10))
                assert match, f"Frame {f}, joint {j}: rotation mismatch"

    def test_rotate_quat_nonroot_unchanged(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        _, new_quats = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="+y", representation="quat"))
        np.testing.assert_allclose(new_quats[:, 1:], quats[:, 1:], atol=1e-10)

    def test_rotate_quat_root_pos_rotated(self, bvh_example):
        """Root position should be transformed by the rotation matrix."""
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="+y", representation="quat"))
        # 90° around Y: (x, y, z) → (z, y, -x)
        np.testing.assert_allclose(new_pos[:, 0], pos[:, 2], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 1], pos[:, 1], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 2], -pos[:, 0], atol=1e-10)

    def test_rotate_quat_negative_axis_flips_direction(self, bvh_example):
        """'+y' and '-y' of the same angle should rotate in opposite directions."""
        pos, quats = _get_quat_data(bvh_example)
        pos_plus, _ = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="+y", representation="quat"))
        pos_minus, _ = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="-y", representation="quat"))
        # Same magnitude, opposite sign on the non-up components.
        np.testing.assert_allclose(pos_plus[:, 1], pos_minus[:, 1], atol=1e-10)
        np.testing.assert_allclose(pos_plus[:, 0], -pos_minus[:, 0], atol=1e-10)
        np.testing.assert_allclose(pos_plus[:, 2], -pos_minus[:, 2], atol=1e-10)

    def test_rotate_quat_bad_axis_raises(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        # Message comes from pybvh.parse_axis since 0.5.0 ("Axis must be
        # one of [...]"); an unsigned letter states no direction and is
        # rejected as firmly as a nonexistent one.
        with pytest.raises(ValueError, match="(?i)axis must be"):
            rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="y", representation="quat")
        with pytest.raises(ValueError, match="(?i)axis must be"):
            rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="+w", representation="quat")

    @pytest.mark.parametrize("up_idx", [0, 1, 2])
    def test_rotate_quat_consistency_with_euler(self, bvh_example, up_idx):
        """Quaternion rotation should match pybvh's Euler rotation after conversion.

        The whole chain is radians: bvh.joint_angles is radians-native
        (pybvh 0.7.0), rotate_angles_vertical takes radians (pybvh 0.8.0),
        and pybvh-ml's rotate_vertical takes radians (pybvh-ml 0.5.0).
        """
        from pybvh.transforms import rotate_angles_vertical
        angle = np.radians(73.0)
        # Euler-level rotation (pybvh's int-axis API)
        root_order = ''.join(bvh_example.nodes[0].rot_channels)
        euler_angles, euler_pos = rotate_angles_vertical(
            bvh_example.joint_angles, bvh_example.root_pos,
            angle, up_idx, root_order)
        # Quaternion-level rotation (pybvh-ml's signed-axis API)
        up_axis = "+" + "xyz"[up_idx]
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=angle, up_axis=up_axis, representation="quat"))
        # Compare root positions
        np.testing.assert_allclose(new_pos, euler_pos, atol=1e-6)
        # Convert quaternion result to radians-Euler and compare
        from pybvh import rotations
        for j_idx in range(bvh_example.joint_count):
            order = bvh_example.euler_orders[j_idx]
            euler_from_quat = rotations.rotmat_to_euler(
                rotations.quat_to_rotmat(new_quats[:, j_idx]),
                order, degrees=False)
            np.testing.assert_allclose(
                euler_from_quat, euler_angles[:, j_idx], atol=1e-4)

    # --- mirror (quaternion) ---

    def test_mirror_quat_shape(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        new_pos, new_quats = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat"))
        assert new_quats.shape == quats.shape
        assert new_pos.shape == pos.shape

    def test_mirror_quat_lateral_negated(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        new_pos, _ = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat"))
        lat_idx = "xyz".index(lateral_axis[1])
        np.testing.assert_allclose(
            new_pos[:, lat_idx], -pos[:, lat_idx], atol=1e-10)

    def test_mirror_quat_double_is_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        p1, q1 = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat"))
        p2, q2 = as_pair(mirror(MotionArrays(root_pos=p1, joint_rot=q1), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat"))
        np.testing.assert_allclose(p2, pos, atol=1e-10)
        np.testing.assert_allclose(q2, quats, atol=1e-10)

    def test_mirror_quat_sign_invariant(self, bvh_example):
        """'+x' and '-x' should produce identical mirror (sign-invariant)."""
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        flipped = ("-" if lateral_axis[0] == "+" else "+") + lateral_axis[1]
        p1, q1 = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat"))
        p2, q2 = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis=flipped, representation="quat"))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)

    @pytest.mark.parametrize("lateral_idx", [0, 1, 2])
    def test_mirror_quat_consistency_with_euler(self, bvh_example, lateral_idx):
        """Quaternion mirror should produce same spatial result as Euler mirror.

        pybvh 0.7.0 made bvh.joint_angles radians-native — comparison is in
        radians (rotmat_to_euler(..., degrees=False)).
        """
        from pybvh.transforms import mirror_angles, auto_detect_lr_pairs
        pairs = auto_detect_lr_pairs(bvh_example)
        rot_ch = [list(n.rot_channels) for n in bvh_example.nodes
                   if not n.is_end_site()]
        # Euler mirror — radians-in, radians-out.
        euler_m, pos_m = mirror_angles(
            bvh_example.joint_angles, bvh_example.root_pos,
            pairs, lateral_idx, rot_ch)
        # Quaternion mirror (pybvh-ml's signed-axis API)
        lateral_axis = "+" + "xyz"[lateral_idx]
        pos, quats = _get_quat_data(bvh_example)
        quat_pos_m, quat_m = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat"))
        # Root positions should match
        np.testing.assert_allclose(quat_pos_m, pos_m, atol=1e-6)
        # Convert quaternion result to radians-Euler and compare
        from pybvh import rotations
        for j_idx in range(bvh_example.joint_count):
            order = bvh_example.euler_orders[j_idx]
            euler_from_quat = rotations.rotmat_to_euler(
                rotations.quat_to_rotmat(quat_m[:, j_idx]),
                order, degrees=False)
            np.testing.assert_allclose(
                euler_from_quat, euler_m[:, j_idx], atol=1e-4)

    # --- speed_perturbation_arrays ---

    def test_speed_frame_count(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        F = pos.shape[0]
        new_p, new_q = as_pair(speed_perturbation_arrays(MotionArrays(root_pos=pos, joint_rot=quats), factor=2.0, representation="quat"))
        assert new_p.shape[0] == max(2, round(F / 2.0))
        assert new_q.shape[0] == new_p.shape[0]

    def test_speed_factor_one(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(speed_perturbation_arrays(MotionArrays(root_pos=pos, joint_rot=quats), factor=1.0, representation="quat"))
        assert new_p.shape[0] == pos.shape[0]
        np.testing.assert_allclose(new_p, pos, atol=1e-10)
        # Quaternions should match (q or -q)
        for f in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                match = (np.allclose(new_q[f, j], quats[f, j], atol=1e-10)
                         or np.allclose(new_q[f, j], -quats[f, j], atol=1e-10))
                assert match

    def test_speed_endpoints(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(speed_perturbation_arrays(MotionArrays(root_pos=pos, joint_rot=quats), factor=1.5, representation="quat"))
        np.testing.assert_allclose(new_p[0], pos[0], atol=1e-10)
        np.testing.assert_allclose(new_p[-1], pos[-1], atol=1e-10)

    def test_speed_invalid_factor(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="factor must be > 0"):
            speed_perturbation_arrays(MotionArrays(root_pos=pos, joint_rot=quats), factor=0.0, representation="quat")

    # --- dropout_arrays ---

    def test_dropout_shape(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(dropout_arrays(MotionArrays(root_pos=pos, joint_rot=quats), drop_rate=0.3, representation="quat", rng=np.random.default_rng(42)))
        assert new_q.shape == quats.shape
        assert new_p.shape == pos.shape

    def test_dropout_first_last_kept(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(dropout_arrays(MotionArrays(root_pos=pos, joint_rot=quats), drop_rate=0.5, representation="quat", rng=np.random.default_rng(42)))
        np.testing.assert_allclose(new_p[0], pos[0], atol=1e-10)
        np.testing.assert_allclose(new_p[-1], pos[-1], atol=1e-10)
        np.testing.assert_allclose(new_q[0], quats[0], atol=1e-10)
        np.testing.assert_allclose(new_q[-1], quats[-1], atol=1e-10)

    def test_dropout_zero_rate(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(dropout_arrays(MotionArrays(root_pos=pos, joint_rot=quats), drop_rate=0.0, representation="quat"))
        np.testing.assert_allclose(new_q, quats, atol=1e-10)
        np.testing.assert_allclose(new_p, pos, atol=1e-10)

    def test_dropout_reproducible(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        p1, q1 = as_pair(dropout_arrays(MotionArrays(root_pos=pos, joint_rot=quats), drop_rate=0.3, representation="quat", rng=np.random.default_rng(99)))
        p2, q2 = as_pair(dropout_arrays(MotionArrays(root_pos=pos, joint_rot=quats), drop_rate=0.3, representation="quat", rng=np.random.default_rng(99)))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)


# =============================================================================
# 6D augmentation
# =============================================================================

class TestRot6dAugmentation:
    """Tests for 6D-space augmentation functions."""

    def test_rotate_6d_shape(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        new_pos, new_6d = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=np.radians(45.0), up_axis="+y", representation="6d"))
        assert new_6d.shape == rot6d.shape
        assert new_pos.shape == pos.shape

    def test_rotate_6d_zero_identity(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        new_pos, new_6d = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=0.0, up_axis="+y", representation="6d"))
        np.testing.assert_allclose(new_6d, rot6d, atol=1e-10)
        np.testing.assert_allclose(new_pos, pos, atol=1e-10)

    def test_rotate_6d_nonroot_unchanged(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        _, new_6d = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=np.radians(90.0), up_axis="+y", representation="6d"))
        np.testing.assert_allclose(new_6d[:, 1:], rot6d[:, 1:], atol=1e-10)

    def test_rotate_6d_root_pos_rotated(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        new_pos, _ = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=np.radians(90.0), up_axis="+y", representation="6d"))
        np.testing.assert_allclose(new_pos[:, 0], pos[:, 2], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 1], pos[:, 1], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 2], -pos[:, 0], atol=1e-10)

    def test_rotate_6d_negative_axis_flips_direction(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pos_plus, _ = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=np.radians(90.0), up_axis="+y", representation="6d"))
        pos_minus, _ = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=np.radians(90.0), up_axis="-y", representation="6d"))
        np.testing.assert_allclose(pos_plus[:, 1], pos_minus[:, 1], atol=1e-10)
        np.testing.assert_allclose(pos_plus[:, 0], -pos_minus[:, 0], atol=1e-10)
        np.testing.assert_allclose(pos_plus[:, 2], -pos_minus[:, 2], atol=1e-10)

    @pytest.mark.parametrize("up_idx", [0, 1, 2])
    def test_rotate_6d_consistency_with_quat(self, bvh_example, up_idx):
        """6D rotation should match quaternion rotation after conversion."""
        from pybvh import rotations
        angle = np.radians(73.0)
        up_axis = "+" + "xyz"[up_idx]
        pos, quats = _get_quat_data(bvh_example)
        _, rot6d = _get_6d_data(bvh_example)
        # Quaternion rotation
        new_pos_q, new_quats = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=angle, up_axis=up_axis, representation="quat"))
        # 6D rotation
        new_pos_6d, new_6d = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=angle, up_axis=up_axis, representation="6d"))
        # Root positions should match
        np.testing.assert_allclose(new_pos_6d, new_pos_q, atol=1e-10)
        # Convert both to rotmat and compare
        R_from_quat = rotations.quat_to_rotmat(new_quats)
        R_from_6d = rotations.rot6d_to_rotmat(new_6d)
        np.testing.assert_allclose(R_from_6d, R_from_quat, atol=1e-6)

    def test_mirror_6d_shape(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        new_pos, new_6d = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=rot6d), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d"))
        assert new_6d.shape == rot6d.shape
        assert new_pos.shape == pos.shape

    def test_mirror_6d_lateral_negated(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        new_pos, _ = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=rot6d), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d"))
        lat_idx = "xyz".index(lateral_axis[1])
        np.testing.assert_allclose(
            new_pos[:, lat_idx], -pos[:, lat_idx], atol=1e-10)

    def test_mirror_6d_double_is_identity(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        p1, r1 = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=rot6d), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d"))
        p2, r2 = as_pair(mirror(MotionArrays(root_pos=p1, joint_rot=r1), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d"))
        np.testing.assert_allclose(p2, pos, atol=1e-10)
        np.testing.assert_allclose(r2, rot6d, atol=1e-10)

    def test_mirror_6d_consistency_with_quat(self, bvh_example):
        """6D mirror should match quaternion mirror after conversion."""
        from pybvh import rotations
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        pos, quats = _get_quat_data(bvh_example)
        _, rot6d = _get_6d_data(bvh_example)
        quat_pos, quat_m = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat"))
        r6d_pos, r6d_m = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=rot6d), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d"))
        np.testing.assert_allclose(r6d_pos, quat_pos, atol=1e-10)
        R_from_quat = rotations.quat_to_rotmat(quat_m)
        R_from_6d = rotations.rot6d_to_rotmat(r6d_m)
        np.testing.assert_allclose(R_from_6d, R_from_quat, atol=1e-6)

    def test_rotate_6d_orthogonal(self, bvh_example):
        """Output 6D should decode to valid rotation matrices."""
        from pybvh import rotations
        pos, rot6d = _get_6d_data(bvh_example)
        _, new_6d = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=rot6d), angle=np.radians(73.0), up_axis="+y", representation="6d"))
        R = rotations.rot6d_to_rotmat(new_6d)
        # Check orthogonality: R @ R.T ≈ I
        I = np.eye(3)
        for f in range(R.shape[0]):
            for j in range(R.shape[1]):
                np.testing.assert_allclose(
                    R[f, j] @ R[f, j].T, I, atol=1e-10)

    def test_mirror_6d_orthogonal(self, bvh_example):
        """Mirrored 6D should decode to valid rotation matrices."""
        from pybvh import rotations
        pos, rot6d = _get_6d_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        _, new_6d = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=rot6d), lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d"))
        R = rotations.rot6d_to_rotmat(new_6d)
        I = np.eye(3)
        for f in range(R.shape[0]):
            for j in range(R.shape[1]):
                np.testing.assert_allclose(
                    R[f, j] @ R[f, j].T, I, atol=1e-10)


# =============================================================================
# Representation conversion
# =============================================================================

class TestConvertArraysContainer:
    """The container-level form: `MotionArrays` in, `MotionArrays` out.

    Conversion is the natural neighbour of augment-and-pack, so it has to
    compose with the container instead of making callers take it apart.
    """

    def test_container_in_container_out(self, bvh_example):
        from pybvh_ml import convert_arrays
        arrays = MotionArrays.from_bvh(bvh_example, "euler")
        out = convert_arrays(arrays, "euler", "6d",
                             euler_orders=bvh_example.euler_orders)
        assert isinstance(out, MotionArrays)
        assert out.joint_rot.shape[-1] == 6

    def test_agrees_with_the_rotation_level_form(self, bvh_example):
        from pybvh_ml import convert_arrays
        orders = bvh_example.euler_orders
        arrays = MotionArrays.from_bvh(bvh_example, "euler")
        out = convert_arrays(arrays, "euler", "quat", euler_orders=orders)
        expected = convert_rotations(arrays.joint_rot, "euler", "quat",
                                     euler_orders=orders)
        np.testing.assert_array_equal(out.joint_rot, expected)

    def test_root_pos_is_carried_through_unchanged(self, bvh_example):
        """A translation has no rotation representation, so the only field
        that may differ from the input is `joint_rot`."""
        from pybvh_ml import convert_arrays
        arrays = MotionArrays.from_bvh(bvh_example, "6d")
        out = convert_arrays(arrays, "6d", "quat")
        np.testing.assert_array_equal(out.root_pos, arrays.root_pos)

    def test_same_representation_still_allocates(self, bvh_example):
        arrays = MotionArrays.from_bvh(bvh_example, "quat")
        from pybvh_ml import convert_arrays
        out = convert_arrays(arrays, "quat", "quat")
        assert not np.shares_memory(out.joint_rot, arrays.joint_rot)

    def test_bare_array_names_the_migration(self, bvh_example):
        """The pre-0.5.0 call took a loose `joint_data` array; an
        `AttributeError` on ndarray would not say what to do about it."""
        from pybvh_ml import convert_arrays
        _, quats = _get_quat_data(bvh_example)
        with pytest.raises(TypeError, match="takes a MotionArrays"):
            convert_arrays(quats, "quat", "6d")
        with pytest.raises(TypeError, match="convert_rotations"):
            convert_arrays(quats, "quat", "6d")

    def test_missing_joint_rot_names_the_caller(self, bvh_example):
        from pybvh_ml import convert_arrays
        arrays = MotionArrays(root_pos=bvh_example.root_pos)
        with pytest.raises(ValueError,
                           match="convert_arrays needs joint rotations"):
            convert_arrays(arrays, "quat", "6d")


class TestConvertRotations:
    """Representation conversion on a bare `(F, J, C)` rotation array."""

    def test_identity(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        result = convert_rotations(quats, "quat", "quat")
        np.testing.assert_allclose(result, quats, atol=1e-12)

    def test_euler_to_quat_shape(self, bvh_example):
        result = convert_rotations(
            bvh_example.joint_angles, "euler", "quat",
            euler_orders=bvh_example.euler_orders)
        assert result.shape == (bvh_example.frame_count, bvh_example.joint_count, 4)

    def test_euler_to_6d_shape(self, bvh_example):
        result = convert_rotations(
            bvh_example.joint_angles, "euler", "6d",
            euler_orders=bvh_example.euler_orders)
        assert result.shape == (bvh_example.frame_count, bvh_example.joint_count, 6)

    def test_roundtrip_euler_quat(self, bvh_example):
        orders = bvh_example.euler_orders
        q = convert_rotations(bvh_example.joint_angles, "euler", "quat",
                           euler_orders=orders)
        back = convert_rotations(q, "quat", "euler", euler_orders=orders)
        np.testing.assert_allclose(back, bvh_example.joint_angles, atol=1e-4)

    def test_roundtrip_euler_6d(self, bvh_example):
        orders = bvh_example.euler_orders
        r6d = convert_rotations(bvh_example.joint_angles, "euler", "6d",
                             euler_orders=orders)
        back = convert_rotations(r6d, "6d", "euler", euler_orders=orders)
        np.testing.assert_allclose(back, bvh_example.joint_angles, atol=1e-4)

    def test_roundtrip_quat_6d(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        r6d = convert_rotations(quats, "quat", "6d")
        back = convert_rotations(r6d, "6d", "quat")
        # q and -q represent same rotation
        for f in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                match = (np.allclose(back[f, j], quats[f, j], atol=1e-6)
                         or np.allclose(back[f, j], -quats[f, j], atol=1e-6))
                assert match

    def test_roundtrip_quat_axisangle(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        aa = convert_rotations(quats, "quat", "axisangle")
        back = convert_rotations(aa, "axisangle", "quat")
        for f in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                match = (np.allclose(back[f, j], quats[f, j], atol=1e-6)
                         or np.allclose(back[f, j], -quats[f, j], atol=1e-6))
                assert match

    def test_roundtrip_6d_rotmat(self, bvh_example):
        _, rot6d = _get_6d_data(bvh_example)
        rm = convert_rotations(rot6d, "6d", "rotmat")
        assert rm.shape[-1] == 9
        back = convert_rotations(rm, "rotmat", "6d")
        np.testing.assert_allclose(back, rot6d, atol=1e-6)

    def test_rotmat_flat_shape(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        rm = convert_rotations(quats, "quat", "rotmat")
        F, J = quats.shape[:2]
        assert rm.shape == (F, J, 9)

    def test_euler_orders_required(self, bvh_example):
        with pytest.raises(ValueError, match="euler_orders is required"):
            convert_rotations(bvh_example.joint_angles, "euler", "quat")

    def test_euler_orders_not_required_for_non_euler(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        # Should not raise
        convert_rotations(quats, "quat", "6d")

    def test_unknown_repr(self):
        data = np.zeros((10, 5, 3))
        with pytest.raises(ValueError, match="Unknown"):
            convert_rotations(data, "invalid", "quat")

    def test_per_joint_mixed_orders(self, bvh_test3):
        """bvh_test3 has mixed Euler orders."""
        orders = bvh_test3.euler_orders
        assert len(set(orders)) >= 1  # may have mixed orders
        q = convert_rotations(bvh_test3.joint_angles, "euler", "quat",
                           euler_orders=orders)
        back = convert_rotations(q, "quat", "euler", euler_orders=orders)
        np.testing.assert_allclose(back, bvh_test3.joint_angles, atol=1e-4)

    @pytest.mark.parametrize("repr_name,expected_c", [
        ("euler", 3), ("axisangle", 3), ("quat", 4),
        ("6d", 6), ("rotmat", 9),
    ])
    def test_convert_shapes(self, bvh_example, repr_name, expected_c):
        orders = bvh_example.euler_orders
        q = convert_rotations(bvh_example.joint_angles, "euler", repr_name,
                           euler_orders=orders)
        assert q.shape[-1] == expected_c


# =============================================================================
# Euler radians ground truth
# =============================================================================

def _align_quat_signs(q, reference):
    """Flip per-element quaternion signs onto the reference hemisphere."""
    dots = np.sum(q * reference, axis=-1, keepdims=True)
    return q * np.where(dots < 0, -1.0, 1.0)


class TestEulerRadians:
    """Euler joint data is radians end to end, checked against pybvh.

    Regression tests for the pre-0.5.0 bug where the euler<->quat
    conversion sites declared ``degrees=True`` while pybvh (0.7.0+)
    stores ``joint_angles`` in radians: euler augmentation shrank
    rotations ~57x on the way in and re-inflated them on the way out,
    corrupting the root joint and amplifying noise.  These tests compare
    against pybvh ground truth instead of self-inverse round trips,
    which cancel a consistent unit error.
    """

    @pytest.mark.parametrize("fixture", ["bvh_example", "bvh_test3"])
    def test_convert_rotations_matches_to_quat(self, request, fixture):
        bvh = request.getfixturevalue(fixture)
        _, quats_gt = bvh.to_quat()
        result = convert_rotations(bvh.joint_angles, "euler", "quat",
                                euler_orders=bvh.euler_orders)
        np.testing.assert_allclose(
            _align_quat_signs(result, quats_gt), quats_gt, atol=1e-6)

    def _euler_vs_quat(self, bvh, fn, kwargs, rng_seed=None):
        """Run fn on euler and quat inputs; return both results as rotmats."""
        orders = bvh.euler_orders
        pos_q, quats = bvh.to_quat()
        kw_euler = dict(kwargs, representation="euler", euler_orders=orders)
        kw_quat = dict(kwargs, representation="quat")
        if rng_seed is not None:
            kw_euler["rng"] = np.random.default_rng(rng_seed)
            kw_quat["rng"] = np.random.default_rng(rng_seed)
        pos_e, jd_e = as_pair(fn(MotionArrays(root_pos=bvh.root_pos, joint_rot=bvh.joint_angles), **kw_euler))
        pos_qr, jd_q = as_pair(fn(MotionArrays(root_pos=pos_q, joint_rot=quats), **kw_quat))
        R_e = convert_rotations(jd_e, "euler", "rotmat", euler_orders=orders)
        R_q = convert_rotations(jd_q, "quat", "rotmat")
        return (pos_e, R_e), (pos_qr, R_q)

    def test_rotate_vertical_euler_matches_quat(self, bvh_example):
        (pos_e, R_e), (pos_q, R_q) = self._euler_vs_quat(
            bvh_example, rotate_vertical,
            {"angle": np.radians(73.0), "up_axis": "+y"})
        np.testing.assert_allclose(pos_e, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)

    def test_mirror_euler_matches_quat(self, bvh_example):
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        (pos_e, R_e), (pos_q, R_q) = self._euler_vs_quat(
            bvh_example, mirror,
            {"lr_joint_pairs": pairs, "lateral_axis": lateral_axis})
        np.testing.assert_allclose(pos_e, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)

    def test_add_joint_rotation_noise_euler_matches_quat(self, bvh_example):
        (pos_e, R_e), (pos_q, R_q) = self._euler_vs_quat(
            bvh_example, add_joint_rotation_noise,
            {"sigma": np.radians(5.0)}, rng_seed=7)
        np.testing.assert_allclose(pos_e, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)

    def test_speed_perturbation_euler_matches_quat(self, bvh_example):
        (pos_e, R_e), (pos_q, R_q) = self._euler_vs_quat(
            bvh_example, speed_perturbation_arrays, {"factor": 1.3})
        np.testing.assert_allclose(pos_e, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)

    def test_dropout_euler_matches_quat(self, bvh_example):
        (pos_e, R_e), (pos_q, R_q) = self._euler_vs_quat(
            bvh_example, dropout_arrays,
            {"drop_rate": 0.3}, rng_seed=11)
        np.testing.assert_allclose(pos_e, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)

    def test_mirror_mixed_euler_orders_matches_quat(self, bvh_example):
        """L/R pairs whose members use different Euler orders mirror correctly.

        Regression: mirror used to swap raw euler triples before the
        order-aware conversion, decoding a left joint's angles with the
        right joint's order.
        """
        pos, quats = bvh_example.to_quat()
        J = quats.shape[1]
        orders = ["ZYX"] * J
        orders[1] = "XYZ"  # pair (1, 2) deliberately mixes orders
        euler = convert_rotations(quats, "quat", "euler", euler_orders=orders)
        pairs = [(1, 2)]
        pos_e, jd_e = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=euler), lr_joint_pairs=pairs, lateral_axis="+x", representation="euler", euler_orders=orders))
        pos_q, jd_q = as_pair(mirror(MotionArrays(root_pos=pos, joint_rot=quats), lr_joint_pairs=pairs, lateral_axis="+x", representation="quat"))
        R_e = convert_rotations(jd_e, "euler", "rotmat", euler_orders=orders)
        R_q = convert_rotations(jd_q, "quat", "rotmat")
        np.testing.assert_allclose(pos_e, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)

    def test_mirror_mixed_euler_orders_staged_matches_direct(self, bvh_example):
        pos, quats = bvh_example.to_quat()
        J = quats.shape[1]
        orders = ["ZYX"] * J
        orders[1] = "XYZ"
        euler = convert_rotations(quats, "quat", "euler", euler_orders=orders)
        steps = [(mirror, 1.0,
                  {"lr_joint_pairs": [(1, 2)], "lateral_axis": "+x",
                   "representation": "euler", "euler_orders": orders})]
        pos_s, jd_s = as_pair(AugmentationPipeline(steps, cache_quats=True)(MotionArrays(root_pos=pos, joint_rot=euler), rng=np.random.default_rng(0)))
        pos_d, jd_d = as_pair(AugmentationPipeline(steps, cache_quats=False)(MotionArrays(root_pos=pos, joint_rot=euler), rng=np.random.default_rng(0)))
        np.testing.assert_allclose(pos_s, pos_d, atol=1e-12)
        np.testing.assert_allclose(jd_s, jd_d, atol=1e-12)

    def test_staged_pipeline_euler_matches_quat(self, bvh_example):
        """The quat-caching pipeline path uses the same radians convention."""
        orders = bvh_example.euler_orders
        pos_q, quats = bvh_example.to_quat()

        def build(representation, **extra):
            steps = [
                (rotate_vertical, 1.0,
                 dict({"angle": np.radians(30.0), "up_axis": "+y",
                       "representation": representation}, **extra)),
                (add_joint_rotation_noise, 1.0,
                 dict({"sigma": np.radians(2.0),
                       "representation": representation}, **extra)),
            ]
            return AugmentationPipeline(steps, cache_quats=True)

        pos_e, jd_e = as_pair(build("euler", euler_orders=orders)(MotionArrays(root_pos=bvh_example.root_pos, joint_rot=bvh_example.joint_angles), rng=np.random.default_rng(42)))
        pos_qr, jd_q = as_pair(build("quat")(MotionArrays(root_pos=pos_q, joint_rot=quats), rng=np.random.default_rng(42)))
        R_e = convert_rotations(jd_e, "euler", "rotmat", euler_orders=orders)
        R_q = convert_rotations(jd_q, "quat", "rotmat")
        np.testing.assert_allclose(pos_e, pos_qr, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)


# =============================================================================
# Rotmat augmentation
# =============================================================================

class TestRotmatAugmentation:
    """Flat (F, J, 9) rotmat joint data through every augmentation function.

    Regression tests for the pre-0.5.0 crash where augmentation passed
    pybvh-ml's flat rotmat layout straight to pybvh's
    ``rotations.convert``, which expects ``(..., 3, 3)``.
    """

    def _rotmat_vs_quat(self, bvh, fn, kwargs, rng_seed=None):
        """Run fn on rotmat and quat inputs; return positions + rotmats."""
        pos, quats = bvh.to_quat()
        rm = convert_rotations(quats, "quat", "rotmat")
        kw_rm = dict(kwargs, representation="rotmat")
        kw_quat = dict(kwargs, representation="quat")
        if rng_seed is not None:
            kw_rm["rng"] = np.random.default_rng(rng_seed)
            kw_quat["rng"] = np.random.default_rng(rng_seed)
        pos_r, jd_r = as_pair(fn(MotionArrays(root_pos=pos, joint_rot=rm), **kw_rm))
        pos_q, jd_q = as_pair(fn(MotionArrays(root_pos=pos, joint_rot=quats), **kw_quat))
        assert jd_r.shape[-1] == 9
        R_q = convert_rotations(jd_q, "quat", "rotmat")
        return (pos_r, jd_r), (pos_q, R_q)

    def test_rotate_vertical_rotmat_matches_quat(self, bvh_example):
        (pos_r, R_r), (pos_q, R_q) = self._rotmat_vs_quat(
            bvh_example, rotate_vertical,
            {"angle": np.radians(73.0), "up_axis": "+y"})
        np.testing.assert_allclose(pos_r, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_r, R_q, atol=1e-6)

    def test_mirror_rotmat_matches_quat(self, bvh_example):
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        (pos_r, R_r), (pos_q, R_q) = self._rotmat_vs_quat(
            bvh_example, mirror,
            {"lr_joint_pairs": pairs, "lateral_axis": lateral_axis})
        np.testing.assert_allclose(pos_r, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_r, R_q, atol=1e-6)

    def test_add_joint_rotation_noise_rotmat_matches_quat(self, bvh_example):
        (pos_r, R_r), (pos_q, R_q) = self._rotmat_vs_quat(
            bvh_example, add_joint_rotation_noise,
            {"sigma": np.radians(5.0)}, rng_seed=7)
        np.testing.assert_allclose(pos_r, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_r, R_q, atol=1e-6)

    def test_speed_perturbation_rotmat_matches_quat(self, bvh_example):
        (pos_r, R_r), (pos_q, R_q) = self._rotmat_vs_quat(
            bvh_example, speed_perturbation_arrays, {"factor": 1.3})
        np.testing.assert_allclose(pos_r, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_r, R_q, atol=1e-6)

    def test_dropout_rotmat_matches_quat(self, bvh_example):
        (pos_r, R_r), (pos_q, R_q) = self._rotmat_vs_quat(
            bvh_example, dropout_arrays,
            {"drop_rate": 0.3}, rng_seed=11)
        np.testing.assert_allclose(pos_r, pos_q, atol=1e-10)
        np.testing.assert_allclose(R_r, R_q, atol=1e-6)

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_pipeline_rotmat(self, bvh_example, cache_quats):
        """A rotmat pipeline runs on both dispatch paths and they agree."""
        pos, quats = bvh_example.to_quat()
        rm = convert_rotations(quats, "quat", "rotmat")
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0,
             {"angle": np.radians(30.0), "up_axis": "+y",
              "representation": "rotmat"}),
            (add_joint_rotation_noise, 1.0,
             {"sigma": np.radians(2.0), "representation": "rotmat"}),
        ], cache_quats=cache_quats)
        new_pos, new_rm = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=rm), rng=np.random.default_rng(42)))
        assert new_rm.shape == rm.shape

    def test_pipeline_rotmat_staged_matches_direct(self, bvh_example):
        pos, quats = bvh_example.to_quat()
        rm = convert_rotations(quats, "quat", "rotmat")
        steps = [
            (rotate_vertical, 1.0,
             {"angle": np.radians(30.0), "up_axis": "+y",
              "representation": "rotmat"}),
            (add_joint_rotation_noise, 1.0,
             {"sigma": np.radians(2.0), "representation": "rotmat"}),
        ]
        pos_s, rm_s = as_pair(AugmentationPipeline(steps, cache_quats=True)(MotionArrays(root_pos=pos, joint_rot=rm), rng=np.random.default_rng(42)))
        pos_d, rm_d = as_pair(AugmentationPipeline(steps, cache_quats=False)(MotionArrays(root_pos=pos, joint_rot=rm), rng=np.random.default_rng(42)))
        np.testing.assert_allclose(pos_s, pos_d, atol=1e-12)
        np.testing.assert_allclose(rm_s, rm_d, atol=1e-12)


# =============================================================================
# Augmentation pipeline
# =============================================================================

class TestAugmentationPipeline:
    """Tests for AugmentationPipeline."""

    def test_empty_pipeline(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats)))
        np.testing.assert_array_equal(new_q, quats)
        np.testing.assert_array_equal(new_p, pos)

    def test_prob_zero_skips(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 0.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        np.testing.assert_array_equal(new_q, quats)
        np.testing.assert_array_equal(new_p, pos)

    def test_prob_one_applies(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        # Should NOT be identical
        assert not np.allclose(new_p, pos)

    def test_reproducibility(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, up_axis = _get_mirror_metadata(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 0.5,
                {"angle": np.radians(45), "up_axis": up_axis, "representation": "quat"}),
            (mirror, 0.5,
                {"lr_joint_pairs": pairs, "lateral_axis": lateral_axis, "representation": "quat"}),
        ])
        p1, q1 = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(123)))
        p2, q2 = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(123)))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)

    def test_chain_multiple(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, up_axis = _get_mirror_metadata(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0,
                {"angle": np.radians(45), "up_axis": up_axis, "representation": "quat"}),
            (mirror, 1.0,
                {"lr_joint_pairs": pairs, "lateral_axis": lateral_axis, "representation": "quat"}),
        ])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        # Both should have been applied
        assert not np.allclose(new_p, pos)
        assert new_q.shape == quats.shape

    def test_len(self):
        pipeline = AugmentationPipeline([
            (rotate_vertical, 0.5, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
            (mirror, 0.5, {"lr_joint_pairs": [], "lateral_axis": "+x", "representation": "quat"}),
        ])
        assert len(pipeline) == 2

    def test_repr(self):
        pipeline = AugmentationPipeline([
            (rotate_vertical, 0.5, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        r = repr(pipeline)
        assert "rotate_vertical" in r

    @pytest.mark.parametrize("custom_position", ["first", "middle", "last"])
    def test_custom_step_without_representation_gets_declared_repr(
            self, bvh_example, custom_position):
        """A representation-less custom step sees the pipeline's declared representation on both dispatch paths.

        Regression: the staged path used to hand such steps whatever
        internal representation the previous built-in step left behind
        (quaternions), so cache_quats=True/False silently diverged.
        """
        pos, rot6d = _get_6d_data(bvh_example)
        seen_shapes = []

        def custom_scale(arrays):
            seen_shapes.append(arrays.joint_rot.shape)
            assert arrays.joint_rot.shape[-1] == 6, (
                f"custom step expected 6d data, got trailing dim "
                f"{arrays.joint_rot.shape[-1]}")
            return arrays.replace(root_pos=arrays.root_pos * 1.01,
                                  joint_rot=arrays.joint_rot.copy())

        custom = (custom_scale, 1.0, {})
        builtin_a = (rotate_vertical, 1.0,
                     {"angle": np.radians(30.0), "up_axis": "+y",
                      "representation": "6d"})
        builtin_b = (add_joint_rotation_noise, 1.0,
                     {"sigma": np.radians(2.0), "representation": "6d"})
        order = {
            "first": [custom, builtin_a, builtin_b],
            "middle": [builtin_a, custom, builtin_b],
            "last": [builtin_a, builtin_b, custom],
        }[custom_position]

        p_staged, jd_staged = as_pair(AugmentationPipeline(order, cache_quats=True)(MotionArrays(root_pos=pos, joint_rot=rot6d), rng=np.random.default_rng(42)))
        p_direct, jd_direct = as_pair(AugmentationPipeline(order, cache_quats=False)(MotionArrays(root_pos=pos, joint_rot=rot6d), rng=np.random.default_rng(42)))
        np.testing.assert_array_equal(p_staged, p_direct)
        np.testing.assert_array_equal(jd_staged, jd_direct)
        assert all(s[-1] == 6 for s in seen_shapes)

    def test_default_rng(self, bvh_example):
        """Pipeline should work without explicit rng."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(45), "up_axis": "+y", "representation": "quat"}),
        ])
        # Should not raise
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats)))

    def test_positional_call_raises(self, bvh_example):
        """Positional binding of root_pos/joint_data is refused."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([])
        with pytest.raises(TypeError):
            pipeline(pos, quats)

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_no_fire_outputs_are_fresh_arrays(self, bvh_example, cache_quats):
        """All probabilities 0: outputs equal the inputs but never alias them.

        Regression: the staged path used to hand back the caller's own
        arrays when no step fired and no representation change ran, so
        in-place edits on the output would corrupt the caller's data.
        """
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 0.0,
                {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
            (add_joint_rotation_noise, 0.0,
                {"sigma": np.radians(2.0), "representation": "quat"}),
        ], cache_quats=cache_quats)
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        np.testing.assert_array_equal(new_q, quats)
        np.testing.assert_array_equal(new_p, pos)
        assert not np.shares_memory(new_p, pos)
        assert not np.shares_memory(new_q, quats)

    def test_staged_requires_declared_representation(self, bvh_example):
        """cache_quats=True raises when no step declares representation=.

        The quat-caching path has to know what representation joint_data
        is in; it used to silently assume "quat", corrupting non-quat
        inputs run through representation-less custom steps.
        """
        pos, quats = _get_quat_data(bvh_example)

        def _shift_root(arrays):
            return arrays.replace(root_pos=arrays.root_pos + 1.0)

        pipeline = AugmentationPipeline([(_shift_root, 1.0, {})])
        with pytest.raises(ValueError, match="representation"):
            pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0))

        # cache_quats=False has no cache to manage — no declaration needed.
        direct = AugmentationPipeline(
            [(_shift_root, 1.0, {})], cache_quats=False)
        new_p, _ = as_pair(direct(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0)))
        np.testing.assert_allclose(new_p, pos + 1.0)


class TestKeywordOnlyAugmentation:
    """Augmentation functions refuse positional root_pos / joint_data."""

    def test_rotate_vertical_refuses_positional(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(TypeError):
            rotate_vertical(
                pos, quats, angle=np.radians(45.0), up_axis="+y",
                representation="quat")

    def test_mirror_refuses_positional(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(TypeError):
            mirror(
                pos, quats, lr_joint_pairs=[], lateral_axis="+x",
                representation="quat")

    def test_add_joint_rotation_noise_refuses_positional(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(TypeError):
            add_joint_rotation_noise(
                pos, quats, sigma=np.radians(1.0), representation="quat")


class TestAugmentationParamValidation:
    """Out-of-range parameters raise instead of silently no-oping."""

    def test_negative_sigma_raises(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="sigma must be"):
            add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=-0.1, representation="quat")

    def test_negative_root_position_sigma_raises(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="sigma must be"):
            add_root_position_noise(
                MotionArrays(root_pos=pos, joint_rot=quats), sigma=-0.5)

    @pytest.mark.parametrize("drop_rate", [-0.1, 1.0, 1.5])
    def test_drop_rate_out_of_range_raises(self, bvh_example, drop_rate):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match=r"drop_rate must be in \[0, 1\)"):
            dropout_arrays(MotionArrays(root_pos=pos, joint_rot=quats), drop_rate=drop_rate, representation="quat")

    def test_negative_sigma_raises_staged(self, bvh_example):
        from pybvh_ml._staged import _StagingState, _add_joint_rotation_noise_staged
        pos, quats = _get_quat_data(bvh_example)
        state = _StagingState(quats, "quat", None)
        with pytest.raises(ValueError, match="sigma must be"):
            _add_joint_rotation_noise_staged(
                pos, state, sigma=-0.1, representation="quat",
                rng=np.random.default_rng(0))

    @pytest.mark.parametrize("drop_rate", [-0.1, 1.0])
    def test_drop_rate_out_of_range_raises_staged(self, bvh_example, drop_rate):
        from pybvh_ml._staged import _StagingState, _dropout_staged
        pos, quats = _get_quat_data(bvh_example)
        state = _StagingState(quats, "quat", None)
        with pytest.raises(ValueError, match=r"drop_rate must be in \[0, 1\)"):
            _dropout_staged(
                pos, state, drop_rate=drop_rate, representation="quat",
                rng=np.random.default_rng(0))

    @pytest.mark.parametrize("fn,kwargs", [
        (rotate_vertical, {"angle": 0.5, "up_axis": "+y"}),
        (mirror, {"lr_joint_pairs": [], "lateral_axis": "+x"}),
        (add_joint_rotation_noise, {"sigma": 0.01}),
        (speed_perturbation_arrays, {"factor": 1.2}),
        (dropout_arrays, {"drop_rate": 0.2}),
    ])
    def test_frame_count_mismatch_raises(self, bvh_example, fn, kwargs):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="disagree on frame count"):
            fn(MotionArrays(root_pos=pos[:-1], joint_rot=quats), representation="quat", **kwargs)

    def test_frame_count_mismatch_raises_staged_pipeline(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0,
             {"angle": 0.5, "up_axis": "+y", "representation": "quat"}),
        ], cache_quats=True)
        with pytest.raises(ValueError, match="disagree on frame count"):
            pipeline(MotionArrays(root_pos=pos[:-1], joint_rot=quats), rng=np.random.default_rng(0))

    @pytest.mark.parametrize("cache_quats", [False, True])
    def test_rotate_vertical_no_joints_raises(self, bvh_example, cache_quats):
        pos, quats = _get_quat_data(bvh_example)
        empty = quats[:, :0, :]
        if cache_quats:
            pipeline = AugmentationPipeline([
                (rotate_vertical, 1.0,
                 {"angle": 0.5, "up_axis": "+y", "representation": "quat"}),
            ], cache_quats=True)
            with pytest.raises(ValueError, match="at least one joint"):
                pipeline(MotionArrays(root_pos=pos, joint_rot=empty), rng=np.random.default_rng(0))
        else:
            with pytest.raises(ValueError, match="at least one joint"):
                rotate_vertical(MotionArrays(root_pos=pos, joint_rot=empty), angle=0.5, up_axis="+y", representation="quat")

    @pytest.mark.parametrize("cache_quats", [False, True])
    def test_zero_norm_quat_raises(self, bvh_example, cache_quats):
        pos, quats = _get_quat_data(bvh_example)
        bad = quats.copy()
        bad[0, 0] = 0.0
        if cache_quats:
            pipeline = AugmentationPipeline([
                (add_joint_rotation_noise, 1.0,
                 {"sigma": 0.01, "representation": "quat"}),
            ], cache_quats=True)
            with pytest.raises(ValueError, match="zero-norm quaternion"):
                pipeline(MotionArrays(root_pos=pos, joint_rot=bad), rng=np.random.default_rng(0))
        else:
            with pytest.raises(ValueError, match="zero-norm quaternion"):
                add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=bad), sigma=0.01, representation="quat", rng=np.random.default_rng(0))

    def test_speed_perturbation_single_frame_noop(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        p1, q1 = pos[:1], quats[:1]
        new_pos, new_quats = as_pair(speed_perturbation_arrays(MotionArrays(root_pos=p1, joint_rot=q1), factor=2.0, representation="quat"))
        np.testing.assert_array_equal(new_pos, p1)
        np.testing.assert_array_equal(new_quats, q1)
        assert not np.shares_memory(new_pos, p1)
        assert not np.shares_memory(new_quats, q1)


class TestPipelineStandardFactory:
    """Tests for AugmentationPipeline.standard()."""

    def test_builds_four_steps_by_default(self, bvh_example):
        from pybvh_ml.skeleton import get_skeleton_info
        skel = get_skeleton_info(bvh_example)
        p = AugmentationPipeline.standard(skel, representation="quat")
        # rotate + mirror (iff lr_pairs) + noise + speed
        expected = 4 if skel.get("lr_pairs") else 3
        assert len(p) == expected

    def test_runs_end_to_end(self, bvh_example):
        from pybvh_ml.skeleton import get_skeleton_info
        pos, quats = _get_quat_data(bvh_example)
        skel = get_skeleton_info(bvh_example)
        p = AugmentationPipeline.standard(
            skel, representation="quat",
            up_axis=bvh_example.world_up)
        new_p, new_q = as_pair(p(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0)))
        assert new_q.shape[1] == quats.shape[1]  # joints unchanged

    def test_disabling_steps(self, bvh_example):
        from pybvh_ml.skeleton import get_skeleton_info
        skel = get_skeleton_info(bvh_example)
        p = AugmentationPipeline.standard(
            skel, representation="quat",
            rotate_angle_range=None,
            mirror_prob=0.0,
            noise_sigma=None,
            speed_factor_range=None,
        )
        assert len(p) == 0

    def test_skips_mirror_when_no_pairs(self, bvh_example):
        # Skeleton info with empty lr_pairs should skip mirror silently.
        from pybvh_ml.skeleton import get_skeleton_info
        skel = get_skeleton_info(bvh_example)
        skel_no_pairs = {**skel, "lr_pairs": []}
        p = AugmentationPipeline.standard(
            skel_no_pairs, representation="quat")
        # rotate + noise + speed only (no mirror)
        assert len(p) == 3


# =============================================================================
# Phase 3: Preprocessing
# =============================================================================

from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array
from pybvh_ml.preprocessing import preprocess_directory, load_preprocessed, extract_repr
from pybvh_ml.skeleton import get_body_partitions


class TestPreprocessing:
    """Tests for batch preprocessing and loading."""

    @pytest.fixture
    def bvh_dir(self):
        return Path(__file__).parent.parent / "bvh_data"

    def test_preprocess_npz(self, bvh_dir, tmp_path):
        out = tmp_path / "dataset.npz"
        result = preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        assert out.exists()
        assert result["num_clips"] == 1
        assert result["representation"] == "6d"

    def test_load_roundtrip_npz(self, bvh_dir, tmp_path):
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        loaded = load_preprocessed(out)
        assert len(loaded["clips"]) == 1
        assert "root_pos" in loaded["clips"][0]
        assert "joint_rot" in loaded["clips"][0]
        assert loaded["mean"] is not None
        assert loaded["std"] is not None
        assert loaded["skeleton_info"]["num_joints"] > 0

    def test_preprocess_hdf5(self, bvh_dir, tmp_path):
        pytest.importorskip("h5py")
        out = tmp_path / "dataset.hdf5"
        result = preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        assert out.exists()
        assert result["num_clips"] == 1

    def test_load_roundtrip_hdf5(self, bvh_dir, tmp_path):
        pytest.importorskip("h5py")
        out = tmp_path / "dataset.hdf5"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        loaded = load_preprocessed(out)
        assert len(loaded["clips"]) == 1
        assert "root_pos" in loaded["clips"][0]

    def test_label_fn(self, bvh_dir, tmp_path):
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                              label_fn=lambda s: 42)
        loaded = load_preprocessed(out)
        assert loaded["labels"] is not None
        assert loaded["labels"][0] == 42

    def test_include_quaternions(self, bvh_dir, tmp_path):
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                              include_quaternions=True)
        loaded = load_preprocessed(out)
        assert "joint_quats" in loaded["clips"][0]
        assert loaded["clips"][0]["joint_quats"].shape[-1] == 4

    def test_center_root(self, bvh_dir, tmp_path):
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                              center_root=True)
        loaded = load_preprocessed(out)
        root_pos = loaded["clips"][0]["root_pos"]
        np.testing.assert_allclose(root_pos[0], 0.0, atol=1e-10)

    @pytest.mark.parametrize("fmt", ["npz", "hdf5"])
    @pytest.mark.parametrize("center", [True, False])
    def test_center_root_flag_roundtrip(self, bvh_dir, tmp_path, fmt, center):
        """The center_root flag is recorded in the saved metadata so downstream packing can avoid double-centering."""
        out = tmp_path / f"dataset.{fmt}"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                              center_root=center)
        loaded = load_preprocessed(out)
        assert loaded["center_root"] is center

    def test_center_root_flag_absent_in_legacy_files(self, bvh_dir, tmp_path):
        """Datasets written before 0.5.0 carry no flag: it loads as None."""
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        legacy_arrays = dict(np.load(out, allow_pickle=False))
        del legacy_arrays["center_root"]
        legacy = tmp_path / "legacy.npz"
        np.savez(legacy, **legacy_arrays)
        loaded = load_preprocessed(legacy)
        assert loaded["center_root"] is None

    def test_multiple_files(self, bvh_dir, tmp_path):
        """Two clips sharing a skeleton should preprocess into one dataset."""
        import shutil
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "clip_a.bvh")
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "clip_b.bvh")
        out = tmp_path / "dataset.npz"
        result = preprocess_directory(work_dir, out)
        assert result["num_clips"] == 2

    def test_representations(self, bvh_dir, tmp_path):
        for repr_name in ["euler", "quat", "6d", "axisangle"]:
            out = tmp_path / f"dataset_{repr_name}.npz"
            result = preprocess_directory(
                bvh_dir, out, representation=repr_name,
                file_pattern="bvh_test1.bvh")
            assert result["representation"] == repr_name

    def test_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out = tmp_path / "dataset.npz"
        with pytest.raises(ValueError, match="No BVH files found"):
            preprocess_directory(empty_dir, out)

    # --- skip_errors, world_up, lr_mapping, constant_channels,
    #     skeleton compatibility check ---

    def test_skip_errors_skips_malformed(self, bvh_dir, tmp_path):
        """A malformed file should be skipped with a UserWarning."""
        import shutil
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "bvh_test1.bvh")
        (work_dir / "broken.bvh").write_text("this is not a bvh file\n")

        out = tmp_path / "dataset.npz"
        with pytest.warns(UserWarning, match="skipping"):
            result = preprocess_directory(work_dir, out, skip_errors=True)
        assert result["num_clips"] == 1
        assert result["filenames"] == ["bvh_test1"]

    def test_skip_errors_false_propagates(self, bvh_dir, tmp_path):
        """Without skip_errors, a malformed file should raise."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "broken.bvh").write_text("not a bvh file\n")
        out = tmp_path / "dataset.npz"
        with pytest.raises(Exception):
            preprocess_directory(work_dir, out, skip_errors=False)

    def test_world_up_passthrough(self, bvh_dir, tmp_path):
        """world_up kwarg should override the clip's detected up axis."""
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                             world_up="+y")
        # We can't read bvh.world_up off the saved file, but the absence of
        # a warning and successful save is the observable behaviour; also
        # verify by loading a BVH directly with the same override.
        from pybvh import read_bvh_file
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh", world_up="+y")
        assert bvh.world_up == "+y"

    def test_constant_channels_persisted_npz(self, bvh_dir, tmp_path):
        """constant_channels mask from compute_normalization_stats round-trips."""
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        loaded = load_preprocessed(out)
        assert "constant_channels" in loaded
        assert loaded["constant_channels"].dtype == np.bool_
        assert loaded["constant_channels"].shape == loaded["mean"].shape

    def test_constant_channels_persisted_hdf5(self, bvh_dir, tmp_path):
        """Same as above for HDF5 backend."""
        pytest.importorskip("h5py")
        out = tmp_path / "dataset.hdf5"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        loaded = load_preprocessed(out)
        assert "constant_channels" in loaded
        assert loaded["constant_channels"].shape == loaded["mean"].shape

    def test_skeleton_mismatch_raises_with_hierarchy_message(
            self, bvh_dir, tmp_path):
        """Mixing bvh_test1 (24 joints) and bvh_test3 (60 joints) raises with a
        hierarchy-mismatch error that names both clips."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        import shutil
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "a.bvh")
        shutil.copy(bvh_dir / "bvh_test3.bvh", work_dir / "b.bvh")
        out = tmp_path / "dataset.npz"
        with pytest.warns(UserWarning, match="(world_up|Rest-pose)"):
            with pytest.raises(ValueError, match="graph is incompatible"):
                preprocess_directory(work_dir, out)

    def test_mixed_euler_orders_6d_succeeds(self, bvh_dir, tmp_path):
        """For rotation-invariant 6d, clips that share hierarchy but differ in
        Euler order are batchable — the saved tensor is order-agnostic."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        # Re-express joint angles in a different Euler order — same
        # skeleton, same motion, different channel order.
        bvh_zxy = bvh.change_euler_order("ZXY")
        write_bvh_file(bvh_zxy, work_dir / "b.bvh")
        out = tmp_path / "dataset.npz"
        result = preprocess_directory(
            work_dir, out, representation="6d")
        assert result["num_clips"] == 2

    def test_mixed_euler_orders_euler_raises_with_recovery_hint(
            self, bvh_dir, tmp_path):
        """For order-sensitive 'euler', same setup must raise — and the error
        should mention harmonize=True as the recovery."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        bvh_zxy = bvh.change_euler_order("ZXY")
        write_bvh_file(bvh_zxy, work_dir / "b.bvh")
        out = tmp_path / "dataset.npz"
        with pytest.raises(ValueError, match="Euler orders"):
            preprocess_directory(work_dir, out, representation="euler")
        with pytest.raises(ValueError, match="harmonize=True"):
            preprocess_directory(work_dir, out, representation="euler")

    def test_require_matching_topology_kwarg_removed(self, bvh_dir, tmp_path):
        """Passing the dropped kwarg should be a TypeError, not silently ignored."""
        out = tmp_path / "dataset.npz"
        with pytest.raises(TypeError, match="require_matching_topology"):
            preprocess_directory(
                bvh_dir, out, file_pattern="bvh_test1.bvh",
                require_matching_topology=False)

    def test_parallel_matches_serial(self, bvh_dir, tmp_path):
        """parallel=True must produce byte-identical clip arrays to serial."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        import shutil
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "a.bvh")
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "b.bvh")

        out_s = tmp_path / "serial.npz"
        out_p = tmp_path / "parallel.npz"
        preprocess_directory(work_dir, out_s, parallel=False)
        preprocess_directory(work_dir, out_p, parallel=True, max_workers=2)

        s = load_preprocessed(out_s)
        p = load_preprocessed(out_p)
        assert s["filenames"] == p["filenames"]
        for cs, cp in zip(s["clips"], p["clips"]):
            np.testing.assert_array_equal(cs["root_pos"], cp["root_pos"])
            np.testing.assert_array_equal(cs["joint_rot"], cp["joint_rot"])

    def test_include_velocities(self, bvh_dir, tmp_path):
        """Velocities array is (F, J, 3) — joint-axis aligned with joint_data,
        not node-axis (pybvh 0.7.0 dropped end sites from joint_velocities)."""
        out = tmp_path / "vel.npz"
        preprocess_directory(
            bvh_dir, out, file_pattern="bvh_test1.bvh",
            include_velocities=True)
        loaded = load_preprocessed(out)
        vel = loaded["clips"][0]["velocities"]
        jd = loaded["clips"][0]["joint_rot"]
        assert vel.ndim == 3
        assert vel.shape[0] == jd.shape[0], "F axis must match joint_data"
        assert vel.shape[1] == jd.shape[1], (
            "J axis must match joint_data (per-joint, no end sites)")
        assert vel.shape[-1] == 3

    def test_include_foot_contacts(self, bvh_dir, tmp_path):
        """Foot-contact labels are (F, num_feet) and foot_joints is recorded."""
        out = tmp_path / "fc.npz"
        preprocess_directory(
            bvh_dir, out, file_pattern="bvh_test1.bvh",
            include_foot_contacts=True)
        loaded = load_preprocessed(out)
        fc = loaded["clips"][0]["foot_contacts"]
        assert fc.ndim == 2
        assert fc.shape[0] == loaded["clips"][0]["joint_rot"].shape[0]
        assert "foot_joints" in loaded["skeleton_info"]
        assert fc.shape[1] == len(loaded["skeleton_info"]["foot_joints"])

    def test_harmonize_unifies_euler_orders_for_euler_rep(
            self, bvh_dir, tmp_path):
        """For representation='euler', clips with mismatched Euler orders should
        be batchable after harmonize=True, and the saved tensor's per-joint
        channel layout becomes the harmonized order."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        bvh_zxy = bvh.change_euler_order("ZXY")
        write_bvh_file(bvh_zxy, work_dir / "b.bvh")

        out = tmp_path / "harmonized.npz"
        result = preprocess_directory(
            work_dir, out, representation="euler",
            harmonize=True, target_euler_order="ZYX",
        )
        assert result["num_clips"] == 2
        h = result["uniformity"]["harmonized_to"]
        assert h["targets"].get("target_euler_order") == "ZYX"
        # One of the clips was already ZYX, so only the ZXY one should
        # have had an euler_order stage applied.
        assert h["stage_counts"].get("euler_order", 0) >= 1

    def test_harmonize_picks_majority_when_target_omitted(
            self, bvh_dir, tmp_path):
        """When target_euler_order is omitted under harmonize=True, the
        most-common per-joint order across clips is chosen.

        bvh_test1's per-joint distribution is {XZY: 16, XYZ: 4, YZX: 2,
        ZYX: 2}. Two native copies + one ZXY-reoriented copy →
        per-joint mode XZY (32 occurrences vs ZXY's 24).
        """
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        write_bvh_file(bvh, work_dir / "b.bvh")
        write_bvh_file(bvh.change_euler_order("ZXY"), work_dir / "c.bvh")

        out = tmp_path / "harmonized.npz"
        result = preprocess_directory(
            work_dir, out, representation="euler", harmonize=True)
        assert result["num_clips"] == 3
        targets = result["uniformity"]["harmonized_to"]["targets"]
        assert targets.get("target_euler_order") == "XZY"

    def test_harmonize_skips_euler_order_for_rotation_invariant_reps(
            self, bvh_dir, tmp_path):
        """For 6d / quaternion / rotmat, harmonize=True should not include
        target_euler_order in the resolved signature — channel layout is
        order-agnostic so unifying orders is wasted work."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        write_bvh_file(bvh.change_euler_order("ZXY"), work_dir / "b.bvh")

        out = tmp_path / "harmonized.npz"
        result = preprocess_directory(
            work_dir, out, representation="6d", harmonize=True)
        targets = result["uniformity"]["harmonized_to"]["targets"]
        assert "target_euler_order" not in targets

    def test_harmonize_hierarchy_mismatch_raises(self, bvh_dir, tmp_path):
        """Hierarchy mismatches under harmonize=True must raise loudly —
        not silently shrink the dataset (the original maintainer-report
        failure).  Without retarget the check happens post-harmonize in
        the skeleton-compatibility gate."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        import shutil
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "a.bvh")
        shutil.copy(bvh_dir / "bvh_test3.bvh", work_dir / "b.bvh")
        out = tmp_path / "harmonized.npz"
        with pytest.warns(UserWarning):
            with pytest.raises(
                    ValueError,
                    match="skeleton graph is incompatible"):
                preprocess_directory(
                    work_dir, out, representation="6d", harmonize=True)

    def test_harmonize_retarget_hierarchy_mismatch_raises(
            self, bvh_dir, tmp_path):
        """With retarget=True the reference gate inside pybvh.harmonize
        reports the drop, and preprocess_directory surfaces it with the
        dropped filename and reason."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        import shutil
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "a.bvh")
        shutil.copy(bvh_dir / "bvh_test3.bvh", work_dir / "b.bvh")
        out = tmp_path / "harmonized.npz"
        with pytest.warns(UserWarning):
            with pytest.raises(
                    ValueError,
                    match=r"pybvh\.harmonize dropped \d+ clip"):
                preprocess_directory(
                    work_dir, out, representation="6d",
                    harmonize=True, retarget=True)

    def test_harmonize_default_preserves_actor_offsets(
            self, bvh_dir, tmp_path):
        """harmonize=True alone must not touch bone offsets — per-actor
        proportions are intrinsic data.  Regression: the reference clip
        used to be pinned unconditionally, silently retargeting every
        actor to the alphabetically first file."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "actor_a.bvh")
        write_bvh_file(bvh.scale(0.5), work_dir / "actor_b.bvh")

        out = tmp_path / "dataset.npz"
        result = preprocess_directory(
            work_dir, out, representation="6d", harmonize=True)
        harmonized_to = result["uniformity"]["harmonized_to"]
        assert harmonized_to["retarget"] is False
        assert harmonized_to["reference"] is None
        assert "retarget" not in harmonized_to["stage_counts"]
        # Root translations keep their per-actor scale: actor_b's
        # de-centered trajectory spans half of actor_a's.
        loaded = load_preprocessed(out)
        span = [np.ptp(c["root_pos"], axis=0) for c in loaded["clips"]]
        np.testing.assert_allclose(span[1], span[0] * 0.5, atol=1e-6)

    def test_harmonize_retarget_unifies_offsets(self, bvh_dir, tmp_path):
        """retarget=True pins the first clip and pybvh applies the
        retarget stage (bone offsets copied from the reference; root
        translation is deliberately untouched)."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "actor_a.bvh")
        write_bvh_file(bvh.scale(0.5), work_dir / "actor_b.bvh")

        out = tmp_path / "dataset.npz"
        result = preprocess_directory(
            work_dir, out, representation="6d",
            harmonize=True, retarget=True)
        harmonized_to = result["uniformity"]["harmonized_to"]
        assert harmonized_to["retarget"] is True
        assert harmonized_to["reference"] == "actor_a"
        assert harmonized_to["stage_counts"].get("retarget", 0) >= 1

    @pytest.mark.parametrize("suffix", [".npz", ".hdf5"])
    def test_uniformity_persisted_and_loaded(self, bvh_dir, tmp_path, suffix):
        """The uniformity audit (incl. harmonized_to) round-trips through
        the saved dataset."""
        if suffix == ".hdf5":
            pytest.importorskip("h5py")
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        write_bvh_file(bvh.change_euler_order("ZXY"), work_dir / "b.bvh")

        out = tmp_path / ("dataset" + suffix)
        result = preprocess_directory(
            work_dir, out, representation="euler",
            harmonize=True, target_euler_order="ZYX")
        loaded = load_preprocessed(out)
        assert loaded["uniformity"] is not None
        # Compared directly, not through a json round trip on both sides:
        # laundering each dict through json hid a key that only survives
        # the trip in one direction (a None key persisting as "null").
        assert loaded["uniformity"] == result["uniformity"]
        assert loaded["uniformity"]["harmonized_to"]["retarget"] is False

    @pytest.mark.parametrize("harmonize_on,expected", [
        (False, "Pass target_world_up='<axis>' to harmonize."),
        (True, "harmonize=True will unify these to the majority value"),
    ])
    def test_heterogeneous_warning_wording(self, harmonize_on, expected):
        """The advice changes with harmonize: required fix vs override."""
        import warnings as _warnings
        from pybvh_ml.preprocessing import _warn_if_heterogeneous
        uniformity = {
            "fps": {"30": ["a", "b", "c"]},
            "world_up": {"+y": ["a", "b"], "+z": ["c"]},
            "rest_forward": {"+x": ["a", "b", "c"]},
            "rest_up": {"+y": ["a", "b", "c"]},
            "rest_anim_mismatch": [],
        }
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _warn_if_heterogeneous(
                uniformity, None, None, None, harmonize=harmonize_on)
        messages = [str(w.message) for w in caught]
        assert len(messages) == 1
        assert expected in messages[0]

    @pytest.mark.parametrize("target_fps,expect_warning", [
        (None, True),
        (30.0, False),
    ])
    def test_mixed_frame_rate_warns_unless_targeted(
            self, target_fps, expect_warning):
        """A mixed-rate dataset is a silent training bug — every
        frame-indexed feature is sampled at its own clip's rate — so the
        audit flags it, and passing target_fps signals intent and
        silences it (same contract as the target_* axis kwargs)."""
        import warnings as _warnings
        from pybvh_ml.preprocessing import _warn_if_heterogeneous
        uniformity = {
            "fps": {"30": ["a", "b"], "120": ["c"]},
            "world_up": {"+y": ["a", "b", "c"]},
            "rest_forward": {"+x": ["a", "b", "c"]},
            "rest_up": {"+y": ["a", "b", "c"]},
            "rest_anim_mismatch": [],
        }
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _warn_if_heterogeneous(
                uniformity, None, None, None, target_fps, harmonize=False)
        messages = [str(w.message) for w in caught]
        assert len(messages) == (1 if expect_warning else 0)
        if expect_warning:
            assert "Frame rate is not uniform" in messages[0]
            assert "target_fps=<hz>" in messages[0]

    def test_bad_output_extension_raises_before_parsing(self, tmp_path):
        """Unknown extensions fail up front — np.savez used to silently
        append .npz and write to a path the caller never named.  The
        nonexistent input dir proves validation runs before any I/O."""
        with pytest.raises(ValueError, match="Unrecognized dataset extension"):
            preprocess_directory(
                tmp_path / "does_not_exist", tmp_path / "out.dat")

    def test_bad_extension_raises_on_load(self, tmp_path):
        with pytest.raises(ValueError, match="Unrecognized dataset extension"):
            load_preprocessed(tmp_path / "dataset.dat")

    def test_bad_representation_raises_before_parsing(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown representation"):
            preprocess_directory(
                tmp_path / "does_not_exist", tmp_path / "out.npz",
                representation="rotmat")

    def test_majority_value_ignores_unknown_key(self):
        """Degenerate rigs are filed under the 'unknown' sentinel; it must
        neither crash the tie-break nor win the majority."""
        from pybvh_ml.preprocessing import _majority_value, _REST_UP_UNKNOWN
        U = _REST_UP_UNKNOWN
        assert _majority_value({"+y": ["a"], U: ["b", "c"]}) == "+y"
        assert _majority_value({U: ["a", "b"]}) is None
        assert _majority_value({}) is None
        assert _majority_value({"+y": ["a"], U: ["b"]}) == "+y"

    def test_resolve_targets_skips_all_unknown_rest_up(self):
        from pybvh_ml.preprocessing import (
            _resolve_harmonize_targets, _REST_UP_UNKNOWN)
        uniformity = {
            "fps": {"30": ["a", "b"]},
            "world_up": {"+y": ["a", "b"]},
            "rest_forward": {"+x": ["a", "b"]},
            "rest_up": {_REST_UP_UNKNOWN: ["a", "b"]},
            "rest_anim_mismatch": ["a", "b"],
        }
        targets = _resolve_harmonize_targets(
            [], uniformity, "6d", None, None, None, None)
        assert "target_rest_up" not in targets

    def test_explicit_foot_joints_roundtrip(self, bvh_dir, tmp_path):
        """An explicit foot_joints list bypasses auto-detection, drives
        contact extraction, and lands in skeleton_info."""
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        explicit = bvh.auto_detect_foot_joints()
        assert explicit, "fixture should have detectable feet"
        out = tmp_path / "dataset.npz"
        preprocess_directory(
            bvh_dir, out, file_pattern="bvh_test1.bvh",
            include_foot_contacts=True, foot_joints=explicit)
        loaded = load_preprocessed(out)
        assert loaded["skeleton_info"]["foot_joints"] == explicit
        fc = loaded["clips"][0]["foot_contacts"]
        assert fc.shape[1] == len(explicit)

    def test_hdf5_non_ascii_stems(self, bvh_dir, tmp_path):
        """Non-ASCII filenames survive the HDF5 round trip (dtype='S'
        used to crash with UnicodeEncodeError)."""
        pytest.importorskip("h5py")
        import shutil
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "mocap_é.bvh")
        out = tmp_path / "dataset.hdf5"
        preprocess_directory(work_dir, out)
        loaded = load_preprocessed(out)
        assert loaded["filenames"] == ["mocap_é"]

    @pytest.mark.parametrize("suffix", [".npz", ".hdf5"])
    def test_quat_dataset_stores_no_duplicate_quats(
            self, bvh_dir, tmp_path, suffix):
        """representation='quat' + include_quaternions=True stores the
        quaternions once; the loader aliases joint_quats to joint_data."""
        if suffix == ".hdf5":
            pytest.importorskip("h5py")
        out = tmp_path / ("dataset" + suffix)
        preprocess_directory(
            bvh_dir, out, file_pattern="bvh_test1.bvh",
            representation="quat", include_quaternions=True)
        if suffix == ".npz":
            raw = np.load(out)
            assert "clip_0_joint_quats" not in raw.files
        else:
            import h5py
            with h5py.File(out, "r") as f:
                assert "joint_quats" not in f["clip_0"]
        loaded = load_preprocessed(out)
        clip = loaded["clips"][0]
        assert clip["joint_quats"] is clip["joint_rot"]
        assert clip["joint_quats"].shape[-1] == 4

    def test_all_names_resolve(self):
        import pybvh_ml
        for name in pybvh_ml.__all__:
            assert hasattr(pybvh_ml, name), name
        assert "preprocess_directory" in pybvh_ml.__all__
        assert "__version__" in pybvh_ml.__all__

    def test_skeleton_info_axis_keys_roundtrip(self, bvh_dir, tmp_path):
        """world_up / rest_forward / rest_up ride along in skeleton_info
        so augmentation can be configured without reopening BVHs."""
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        loaded = load_preprocessed(out)
        info = loaded["skeleton_info"]
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        assert info["world_up"] == bvh.world_up
        assert info["rest_forward"] == bvh.rest_forward
        assert info["rest_up"] == bvh.rest_up

    def test_same_graph_different_offsets_accepted(self, bvh_dir, tmp_path):
        """Multi-actor case: clips share the skeleton graph (same names +
        parent indices) but have different bone offsets. The compatibility
        check accepts them — joint_data tensors don't depend on bone
        lengths, and harmonize(reference=...) is the right tool when
        offset uniformity matters."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "actor_a.bvh")
        write_bvh_file(bvh.scale(0.5), work_dir / "actor_b.bvh")

        out = tmp_path / "dataset.npz"
        result = preprocess_directory(work_dir, out, representation="6d")
        assert result["num_clips"] == 2

    def test_harmonize_explicit_target_wins_over_majority(
            self, bvh_dir, tmp_path):
        """When harmonize=True receives an explicit target_euler_order, that
        value wins even when the dataset majority is different."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        # Majority would be XZY (bvh_test1's native); force XYZ.
        write_bvh_file(bvh, work_dir / "a.bvh")
        write_bvh_file(bvh, work_dir / "b.bvh")
        write_bvh_file(bvh.change_euler_order("ZXY"), work_dir / "c.bvh")

        out = tmp_path / "dataset.npz"
        result = preprocess_directory(
            work_dir, out, representation="euler",
            harmonize=True, target_euler_order="XYZ",
        )
        targets = result["uniformity"]["harmonized_to"]["targets"]
        assert targets["target_euler_order"] == "XYZ"

    def test_harmonize_roundtrip_saved_arrays_load_back(
            self, bvh_dir, tmp_path):
        """End-to-end: after harmonize=True, the saved .npz loads with
        consistent shapes — extraction actually happened post-harmonize."""
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        write_bvh_file(bvh.change_euler_order("ZXY"), work_dir / "b.bvh")

        out = tmp_path / "dataset.npz"
        preprocess_directory(
            work_dir, out, representation="euler",
            harmonize=True, target_euler_order="ZYX",
        )
        loaded = load_preprocessed(out)
        a = loaded["clips"][0]["joint_rot"]
        b = loaded["clips"][1]["joint_rot"]
        assert a.shape == b.shape, (
            "After harmonize=True with target_euler_order='ZYX', both "
            "clips should have the same (F, J, 3) channel layout")

    def test_harmonize_report_is_json_serializable(self, bvh_dir, tmp_path):
        """The harmonized_to summary must be JSON-serializable so it can be
        embedded in dataset metadata downstream."""
        import json
        from pybvh import read_bvh_file, write_bvh_file
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work_dir / "a.bvh")
        write_bvh_file(bvh.change_euler_order("ZXY"), work_dir / "b.bvh")

        out = tmp_path / "harmonized.npz"
        result = preprocess_directory(
            work_dir, out, representation="euler",
            harmonize=True, target_euler_order="ZYX",
        )
        # Must not raise — proves dataclasses.asdict produced JSON-native data.
        encoded = json.dumps(result["uniformity"]["harmonized_to"])
        assert "report" in json.loads(encoded)

    def test_rest_anim_mismatch_warning_offers_both_recoveries(self):
        """When rest_up != world_up, the warning names both world_up= and
        target_rest_up= recovery paths (the file's parsed rest_up vs the
        animation-inferred world_up). Old text mentioned only target_rest_up,
        which doesn't help when animation inference is the wrong one."""
        import warnings as _warnings
        from pybvh_ml.preprocessing import _warn_if_heterogeneous

        uniformity = {
            "fps": {"30": ["clip_a", "clip_b"]},
            "world_up": {"+y": ["clip_a", "clip_b"]},
            "rest_forward": {"+z": ["clip_a", "clip_b"]},
            "rest_up": {"+y": ["clip_a", "clip_b"]},
            "rest_anim_mismatch": ["clip_a", "clip_b"],
        }
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _warn_if_heterogeneous(
                uniformity,
                target_world_up=None,
                target_rest_forward=None,
                target_rest_up=None,
            )
        mismatch_warnings = [
            w for w in caught
            if "Rest-pose up disagrees" in str(w.message)
        ]
        assert len(mismatch_warnings) == 1
        msg = str(mismatch_warnings[0].message)
        assert "world_up='<axis>'" in msg, (
            "warning should recommend world_up= override at parse time")
        assert "target_rest_up='<axis>'" in msg, (
            "warning should keep the target_rest_up= path as alternative")


# =============================================================================
# Normalization
# =============================================================================

def _flat_features(bvh, representation="euler"):
    """Flat ``(F, 3 + J*C)`` features in compute_normalization_stats' layout."""
    root_pos, joint_data = extract_repr(bvh, representation)
    return pack_to_flat(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)


class TestNormalization:
    """Tests for the normalization trio (absorbed from pybvh 0.8.0)."""

    def test_round_trip(self, bvh_example):
        """Normalize then denormalize should recover original."""
        stats = compute_normalization_stats([bvh_example])
        original = _flat_features(bvh_example)
        normalized = normalize_array(original, stats)
        recovered = denormalize_array(normalized, stats)
        np.testing.assert_allclose(recovered, original, atol=1e-10)

    def test_stats_shapes(self, bvh_example):
        stats = compute_normalization_stats([bvh_example])
        D = 3 + bvh_example.joint_count * 3  # root_pos + euler
        assert stats["mean"].shape == (D,)
        assert stats["std"].shape == (D,)

    def test_stats_shapes_6d(self, bvh_example):
        stats = compute_normalization_stats(
            [bvh_example], representation="6d")
        D = 3 + bvh_example.joint_count * 6
        assert stats["mean"].shape == (D,)
        assert stats["std"].shape == (D,)

    def test_zero_std_guard(self, bvh_example):
        """Constant channels should get std=1.0, not 0.0."""
        static = bvh_example.copy()
        static.root_pos = np.broadcast_to(
            static.root_pos[0:1], static.root_pos.shape).copy()
        static.joint_angles = np.broadcast_to(
            static.joint_angles[0:1], static.joint_angles.shape).copy()
        stats = compute_normalization_stats([static])
        assert np.all(stats["std"] >= 1e-8)

    def test_normalized_mean_zero(self, bvh_example):
        """After normalization, mean should be ~0."""
        stats = compute_normalization_stats([bvh_example])
        normalized = normalize_array(_flat_features(bvh_example), stats)
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-10)

    def test_multiple_files(self, bvh_example):
        """Stats from multiple clips sharing a skeleton have correct shape."""
        bvh2 = bvh_example.copy()
        stats = compute_normalization_stats([bvh_example, bvh2])
        D = 3 + bvh_example.joint_count * 3
        assert stats["mean"].shape == (D,)

    def test_quaternion_round_trip(self, bvh_example):
        stats = compute_normalization_stats(
            [bvh_example], representation="quat")
        data = _flat_features(bvh_example, representation="quat")
        recovered = denormalize_array(normalize_array(data, stats), stats)
        np.testing.assert_allclose(recovered, data, atol=1e-10)

    def test_no_root_pos(self, bvh_example):
        stats = compute_normalization_stats(
            [bvh_example], include_root_pos=False)
        D = bvh_example.joint_count * 3
        assert stats["mean"].shape == (D,)

    def test_constant_channels_key_present(self, bvh_example):
        stats = compute_normalization_stats([bvh_example])
        assert "constant_channels" in stats

    def test_constant_channels_shape_and_dtype(self, bvh_example):
        stats = compute_normalization_stats([bvh_example])
        D = 3 + bvh_example.joint_count * 3
        assert stats["constant_channels"].shape == (D,)
        assert stats["constant_channels"].dtype == bool

    def test_constant_channels_content(self, bvh_example):
        """Known-constant channels should be flagged True."""
        static = bvh_example.copy()
        static.root_pos = np.broadcast_to(
            static.root_pos[0:1], static.root_pos.shape).copy()
        static.joint_angles = np.broadcast_to(
            static.joint_angles[0:1], static.joint_angles.shape).copy()
        stats = compute_normalization_stats([static])
        # Every channel is constant — all flagged
        assert stats["constant_channels"].all()

    def test_constant_channels_mixed(self, bvh_example):
        """Only channels that are actually constant should be flagged."""
        partial = bvh_example.copy()
        # Freeze only channel 0 (root X position) across frames
        rp = partial.root_pos.copy()
        rp[:, 0] = rp[0, 0]
        partial.root_pos = rp
        stats = compute_normalization_stats([partial])
        assert stats["constant_channels"][0]
        # Other channels generally vary across the 75-frame clip
        assert not stats["constant_channels"][1:].all()

    def test_roundtrip_through_npz(self, bvh_example, tmp_path):
        """Bool arrays round-trip cleanly through np.savez/np.load."""
        stats = compute_normalization_stats([bvh_example])
        path = tmp_path / "stats.npz"
        np.savez(path, **stats)
        loaded = dict(np.load(path))
        np.testing.assert_array_equal(
            loaded["constant_channels"], stats["constant_channels"])
        assert loaded["constant_channels"].dtype == bool

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_normalization_stats([])

    def test_mixed_skeletons_raise(self, bvh_example, bvh_test3):
        """Clips with different skeleton graphs cannot share stats."""
        with pytest.raises(ValueError, match="graph is incompatible"):
            compute_normalization_stats([bvh_example, bvh_test3])

    def test_matches_preprocess_directory_stats(self, tmp_path):
        """center_root=True reproduces preprocess_directory's stored stats
        exactly (including root channels); center_root=False still agrees
        on rotation channels but differs on the centered root."""
        bvh_dir = Path(__file__).parent.parent / "bvh_data"
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                             representation="6d")
        loaded = load_preprocessed(out)
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")

        centered = compute_normalization_stats(
            [bvh], representation="6d", center_root=True)
        np.testing.assert_allclose(centered["mean"], loaded["mean"],
                                   atol=1e-10)
        np.testing.assert_allclose(centered["std"], loaded["std"],
                                   atol=1e-10)

        raw = compute_normalization_stats([bvh], representation="6d")
        np.testing.assert_allclose(raw["mean"][3:], loaded["mean"][3:],
                                   atol=1e-10)
        np.testing.assert_allclose(raw["std"][3:], loaded["std"][3:],
                                   atol=1e-10)
        assert not np.allclose(raw["mean"][:3], loaded["mean"][:3])


# =============================================================================
# Body partitions
# =============================================================================

class TestBodyPartitions:
    """Tests for heuristic body-part grouping."""

    def test_expected_keys(self, bvh_example):
        parts = get_body_partitions(bvh_example)
        expected = {"torso", "head", "left_arm", "right_arm",
                    "left_leg", "right_leg", "other"}
        assert set(parts.keys()) == expected

    def test_valid_indices(self, bvh_example):
        parts = get_body_partitions(bvh_example)
        J = bvh_example.joint_count
        for group, indices in parts.items():
            for idx in indices:
                assert 0 <= idx < J, f"Index {idx} out of range in group '{group}'"

    def test_complete_coverage(self, bvh_example):
        """Every joint should appear in exactly one group."""
        parts = get_body_partitions(bvh_example)
        all_indices = []
        for indices in parts.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(bvh_example.joint_count))

    def test_no_overlap(self, bvh_example):
        parts = get_body_partitions(bvh_example)
        seen = set()
        for group, indices in parts.items():
            for idx in indices:
                assert idx not in seen, f"Index {idx} in multiple groups"
                seen.add(idx)

    def test_different_skeletons(self, bvh_example, bvh_test3):
        parts1 = get_body_partitions(bvh_example)
        parts3 = get_body_partitions(bvh_test3)
        total1 = sum(len(v) for v in parts1.values())
        total3 = sum(len(v) for v in parts3.values())
        assert total1 == bvh_example.joint_count
        assert total3 == bvh_test3.joint_count

    def test_lr_symmetry(self, bvh_example):
        """Left and right arm/leg groups should have same size."""
        parts = get_body_partitions(bvh_example)
        assert len(parts["left_arm"]) == len(parts["right_arm"])
        assert len(parts["left_leg"]) == len(parts["right_leg"])


# =============================================================================
# Uniform temporal sampling
# =============================================================================

class TestUniformTemporalSample:
    """Tests for uniform_temporal_sample and sample_temporal."""

    # --- uniform_temporal_sample ---

    def test_output_length(self):
        indices = uniform_temporal_sample(100, 20, mode="test")
        assert indices.shape == (20,)

    def test_short_sequence_wraps(self):
        """When num_frames < clip_length, indices exceed num_frames."""
        indices = uniform_temporal_sample(5, 20, mode="test")
        assert indices.shape == (20,)
        # Test mode starts at 0
        np.testing.assert_array_equal(indices, np.arange(20))
        # Some indices must be >= num_frames (wrapping)
        assert np.any(indices >= 5)
        # After modulo, all indices should be valid
        assert np.all(indices % 5 < 5)

    def test_short_sequence_uniform_coverage(self):
        """Short wrapping gives each frame equal representation."""
        indices = uniform_temporal_sample(5, 20, mode="test") % 5
        # Each of 0..4 should appear exactly 4 times
        for i in range(5):
            assert np.sum(indices == i) == 4

    def test_short_train_random_start(self):
        """Train mode short sequences should have varying start offsets."""
        starts = set()
        for seed in range(20):
            indices = uniform_temporal_sample(
                5, 20, mode="train", rng=np.random.default_rng(seed))
            starts.add(indices[0])
        # Should see multiple different start positions
        assert len(starts) > 1

    def test_dense_regime(self):
        """clip_length <= num_frames < 2*clip_length: scattered dense sampling."""
        indices = uniform_temporal_sample(30, 20, mode="test")
        assert indices.shape == (20,)
        assert np.all(indices < 30)
        assert np.all(indices >= 0)
        # Indices should be non-decreasing
        diffs = np.diff(indices)
        assert np.all(diffs >= 0)
        # Steps are either 1 (consecutive) or 2 (gap inserted)
        assert np.all((diffs == 1) | (diffs == 2))
        # Span should cover most of the range
        assert indices[-1] - indices[0] >= 19

    def test_uniform_regime(self):
        """num_frames >= 2*clip_length: uniform segment sampling."""
        indices = uniform_temporal_sample(200, 20, mode="test")
        assert indices.shape == (20,)
        assert np.all(indices < 200)
        assert np.all(indices >= 0)

    def test_uniform_regime_sorted(self):
        """Uniform segment indices should be non-decreasing."""
        indices = uniform_temporal_sample(500, 50, mode="train",
                                          rng=np.random.default_rng(42))
        assert np.all(np.diff(indices) >= 0)

    def test_uniform_integer_boundaries(self):
        """Segment boundaries should use integer division."""
        # 100 frames, 7 clips → segments of 14 or 15 frames
        indices = uniform_temporal_sample(100, 7, mode="test")
        assert indices.shape == (7,)
        # Each index must be within its integer-division segment
        for i in range(7):
            seg_start = i * 100 // 7
            seg_end = (i + 1) * 100 // 7
            assert seg_start <= indices[i] < seg_end

    def test_boundary_dense_to_uniform(self):
        """num_frames == 2*clip_length is the uniform regime boundary."""
        # 2*clip_length: should be uniform (not dense)
        indices = uniform_temporal_sample(40, 20, mode="train",
                                          rng=np.random.default_rng(42))
        assert indices.shape == (20,)
        # Uniform regime: indices should NOT necessarily be consecutive
        # (seg_size=2, so gaps are possible)
        assert np.all(indices < 40)

    def test_boundary_short_to_dense(self):
        """num_frames == clip_length is the dense regime boundary."""
        indices = uniform_temporal_sample(20, 20, mode="test")
        assert indices.shape == (20,)
        assert np.all(indices < 20)
        assert np.all(indices >= 0)
        # num_frames == clip_length → 0 gaps, so exactly [0..19]
        np.testing.assert_array_equal(indices, np.arange(20))

    def test_uniform_covers_full_range(self):
        """Indices should span most of the sequence, not cluster."""
        indices = uniform_temporal_sample(1000, 20, mode="test")
        assert indices[-1] > 900  # last segment should be near the end
        assert indices[0] < 100   # first segment should be near the start

    def test_train_mode_varies(self):
        """Different rng seeds should produce different indices."""
        i1 = uniform_temporal_sample(200, 20, mode="train", rng=np.random.default_rng(1))
        i2 = uniform_temporal_sample(200, 20, mode="train", rng=np.random.default_rng(2))
        assert not np.array_equal(i1, i2)

    def test_test_mode_deterministic(self):
        """Test mode always produces the same indices."""
        i1 = uniform_temporal_sample(200, 20, mode="test")
        i2 = uniform_temporal_sample(200, 20, mode="test")
        np.testing.assert_array_equal(i1, i2)

    def test_test_mode_default_rng_is_stable(self):
        """rng=None in test mode is deterministic across calls (the
        caller-supplied-rng case is covered separately — it is honored
        since 0.5.0, no longer ignored)."""
        i1 = uniform_temporal_sample(200, 20, mode="test")
        i2 = uniform_temporal_sample(200, 20, mode="test")
        np.testing.assert_array_equal(i1, i2)

    def test_single_frame(self):
        indices = uniform_temporal_sample(1, 10, mode="test")
        assert indices.shape == (10,)
        assert np.all(indices % 1 == 0)

    def test_clip_equals_frames(self):
        indices = uniform_temporal_sample(20, 20, mode="test")
        assert indices.shape == (20,)
        assert np.all(indices < 20)

    def test_invalid_num_frames(self):
        with pytest.raises(ValueError, match="num_frames"):
            uniform_temporal_sample(0, 10)

    def test_invalid_clip_length(self):
        with pytest.raises(ValueError, match="clip_length"):
            uniform_temporal_sample(10, 0)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            uniform_temporal_sample(10, 5, mode="invalid")

    # --- sample_temporal ---

    def test_sample_temporal_shape(self):
        data = np.random.randn(100, 24, 4)
        result = sample_temporal(data, clip_length=20, mode="test")
        assert result.shape == (20, 24, 4)

    def test_sample_temporal_multi(self):
        data = np.random.randn(100, 24, 4)
        result = sample_temporal(data, clip_length=20, num_samples=5, mode="test")
        assert result.shape == (5, 20, 24, 4)

    def test_sample_temporal_short_wraps(self):
        """Short sequences should wrap around, not error."""
        data = np.arange(15, dtype=np.float64).reshape(5, 3)
        result = sample_temporal(data, clip_length=20, mode="test")
        assert result.shape == (20, 3)
        # All values should come from the original data
        for row in result:
            assert any(np.array_equal(row, data[i]) for i in range(5))
        # Consecutive result rows should be consecutive source frames (mod 5)
        for i in range(1, 20):
            prev_idx = int(result[i - 1, 0]) // 3  # which source row
            curr_idx = int(result[i, 0]) // 3
            assert curr_idx == (prev_idx + 1) % 5

    def test_sample_temporal_reproducible(self):
        data = np.random.randn(100, 10)
        r1 = sample_temporal(data, 20, mode="train", rng=np.random.default_rng(42))
        r2 = sample_temporal(data, 20, mode="train", rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_sample_temporal_test_mode_samples_are_distinct(self):
        """Test mode with num_samples > 1 yields distinct (yet
        deterministic) samples.  Regression: the internal rng used to
        be re-seeded to 0 on every draw, so all samples were
        bit-identical despite the 'independent samples' promise."""
        data = np.random.default_rng(3).normal(size=(100, 10))
        result = sample_temporal(data, 20, num_samples=3, mode="test")
        assert not np.array_equal(result[0], result[1])
        assert not np.array_equal(result[1], result[2])
        # Deterministic across calls.
        again = sample_temporal(data, 20, num_samples=3, mode="test")
        np.testing.assert_array_equal(result, again)

    def test_uniform_temporal_sample_test_mode_honors_rng(self):
        """A caller-supplied rng drives test mode too (previously it
        was silently ignored); rng=None keeps the fixed default."""
        default_1 = uniform_temporal_sample(100, 20, mode="test")
        default_2 = uniform_temporal_sample(100, 20, mode="test")
        np.testing.assert_array_equal(default_1, default_2)
        np.testing.assert_array_equal(
            default_1,
            uniform_temporal_sample(
                100, 20, mode="test", rng=np.random.default_rng(0)))
        custom = uniform_temporal_sample(
            100, 20, mode="test", rng=np.random.default_rng(7))
        assert not np.array_equal(custom, default_1)


# =============================================================================
# Joint noise augmentation
# =============================================================================

class TestJointNoise:
    """Tests for add_joint_rotation_noise."""

    def test_shape_preserved(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(1.0), representation="quat", rng=np.random.default_rng(42)))
        assert new_q.shape == quats.shape
        assert new_p.shape == pos.shape

    def test_zero_noise_is_near_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=0.0, representation="quat", rng=np.random.default_rng(42)))
        # sigma=0 means angle is always 0, so noise quat ≈ identity
        # but axis is still random, so cos(0)=1, sin(0)=0 → q_noise = [1,0,0,0]
        np.testing.assert_allclose(new_p, pos, atol=1e-10)
        # Quaternions should be very close (numerical noise only)
        for f in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                match = (np.allclose(new_q[f, j], quats[f, j], atol=1e-6)
                         or np.allclose(new_q[f, j], -quats[f, j], atol=1e-6))
                assert match, f"Frame {f}, joint {j}: unexpected change"

    def test_output_unit_quaternions(self, bvh_example):
        """Output quaternions should be unit length."""
        pos, quats = _get_quat_data(bvh_example)
        _, new_q = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42)))
        norms = np.linalg.norm(new_q, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_noise_changes_values(self, bvh_example):
        """Non-zero sigma should produce different quaternions."""
        pos, quats = _get_quat_data(bvh_example)
        _, new_q = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42)))
        assert not np.allclose(new_q, quats, atol=1e-4)

    def test_small_noise_stays_close(self, bvh_example):
        """Small sigma should produce quaternions close to originals."""
        from pybvh import rotations
        pos, quats = _get_quat_data(bvh_example)
        _, new_q = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(0.1), representation="quat", rng=np.random.default_rng(42)))
        # Geodesic distance: angle = 2 * arccos(|q1 . q2|)
        dots = np.abs(np.sum(quats * new_q, axis=-1))
        dots = np.clip(dots, 0, 1)
        angles_deg = np.degrees(2 * np.arccos(dots))
        # With sigma=0.1 deg, angles should be very small
        assert np.mean(angles_deg) < 1.0

    def test_root_position_noise_moves_the_root(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = as_pair(add_root_position_noise(
            MotionArrays(root_pos=pos, joint_rot=quats), sigma=0.5,
            rng=np.random.default_rng(42)))
        assert not np.allclose(new_p, pos, atol=1e-4)
        # The split's point: rotations are untouched by positional noise.
        np.testing.assert_array_equal(new_q, quats)

    def test_rotation_noise_leaves_the_root_alone(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, _ = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42)))
        np.testing.assert_array_equal(new_p, pos)

    def test_reproducible(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        p1, q1 = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(2.0), representation="quat", rng=np.random.default_rng(42)))
        p2, q2 = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(2.0), representation="quat", rng=np.random.default_rng(42)))
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(p1, p2)

    def test_valid_rotations(self, bvh_example):
        """Noisy quaternions should convert to valid rotation matrices."""
        from pybvh import rotations
        pos, quats = _get_quat_data(bvh_example)
        _, new_q = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos, joint_rot=quats), sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42)))
        R = rotations.quat_to_rotmat(new_q)
        I = np.eye(3)
        for f in range(R.shape[0]):
            for j in range(R.shape[1]):
                np.testing.assert_allclose(
                    R[f, j] @ R[f, j].T, I, atol=1e-10)

    def test_pipeline_integration(self, bvh_example):
        """Joint noise should work inside AugmentationPipeline."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {"sigma": np.radians(2.0), "representation": "quat"}),
        ])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        assert new_q.shape == quats.shape

    def test_staged_zero_sigma_pos_root_not_aliased(self, bvh_example):
        """Regression: the staged variant used to return the caller's own root_pos when sigma_pos=0, so later in-place edits could mutate the input."""
        from pybvh_ml._staged import _StagingState, _add_joint_rotation_noise_staged
        pos, quats = _get_quat_data(bvh_example)
        state = _StagingState(quats, "quat", None)
        new_p = _add_joint_rotation_noise_staged(
            pos, state, sigma=np.radians(1.0), representation="quat",
            rng=np.random.default_rng(42))
        assert not np.shares_memory(new_p, pos)
        # sigma_pos=0 leaves the values themselves unchanged.
        np.testing.assert_array_equal(new_p, pos)


# =============================================================================
# Callable kwargs in AugmentationPipeline
# =============================================================================

class TestPipelineCallableKwargs:
    """Tests for callable kwargs support in AugmentationPipeline."""

    def test_callable_kwarg_resolved(self, bvh_example):
        """A callable kwarg should be called with rng."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y",
                "representation": "quat",
            }),
        ])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        # Should have been rotated by some angle
        assert not np.allclose(new_p, pos)

    def test_callable_produces_different_values(self, bvh_example):
        """Successive calls should sample different random values."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y",
                "representation": "quat",
            }),
        ])
        p1, _ = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(1)))
        p2, _ = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(2)))
        assert not np.allclose(p1, p2)

    def test_mixed_callable_and_static(self, bvh_example):
        """Callable and static kwargs should coexist."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (speed_perturbation_arrays, 1.0, {
                "factor": lambda rng: rng.uniform(0.8, 1.2),
                "representation": "quat",
            }),
        ])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        # Frame count may differ due to speed perturbation
        assert new_q.shape[1] == quats.shape[1]  # joints unchanged

    def test_reproducible_with_callable(self, bvh_example):
        """Same rng seed should produce identical results."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y",
                "representation": "quat",
            }),
        ])
        p1, q1 = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(99)))
        p2, q2 = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(99)))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)

    def test_static_kwargs_still_work(self, bvh_example):
        """Existing static kwargs should not be broken."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        # Should match direct call
        ref_p, ref_q = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="+y", representation="quat"))
        np.testing.assert_allclose(new_p, ref_p, atol=1e-12)
        np.testing.assert_allclose(new_q, ref_q, atol=1e-12)


# =============================================================================
# Pipeline rng forwarding (bug fix)
# =============================================================================

class TestPipelineRngForwarding:
    """Tests for automatic rng forwarding to augmentation functions."""

    def test_noise_reproducible_via_pipeline(self, bvh_example):
        """add_joint_rotation_noise should be deterministic in pipeline."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        _, q1 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(42)))
        _, q2 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(42)))
        np.testing.assert_array_equal(q1, q2)

    def test_dropout_reproducible_via_pipeline(self, bvh_example):
        """dropout_arrays should be deterministic in pipeline."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (dropout_arrays, 1.0, {"drop_rate": 0.3, "representation": "quat"}),
        ])
        p1, q1 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(42)))
        p2, q2 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(42)))
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(p1, p2)

    def test_no_rng_functions_unaffected(self, bvh_example):
        """Functions without rng param should still work."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        # Should not raise TypeError
        new_p, new_q = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(42)))
        ref_p, ref_q = as_pair(rotate_vertical(MotionArrays(root_pos=pos, joint_rot=quats), angle=np.radians(90.0), up_axis="+y", representation="quat"))
        np.testing.assert_allclose(new_q, ref_q, atol=1e-12)

    def test_explicit_rng_kwarg_takes_precedence(self, bvh_example):
        """User-provided rng kwarg should not be overwritten."""
        pos, quats = _get_quat_data(bvh_example)
        custom_rng = np.random.default_rng(999)
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {
                "sigma": np.radians(5.0),
                "representation": "quat",
                "rng": lambda rng: custom_rng,  # explicit override
            }),
        ])
        _, q1 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(42)))
        # Should use custom_rng(999), not pipeline's rng(42)
        custom_rng2 = np.random.default_rng(999)
        _, q2 = as_pair(add_joint_rotation_noise(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), sigma=np.radians(5.0), representation="quat", rng=custom_rng2))
        np.testing.assert_array_equal(q1, q2)

    def test_mixed_rng_and_no_rng_functions(self, bvh_example):
        """Pipeline with both rng and non-rng functions should work."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y",
                "representation": "quat",
            }),
            (add_joint_rotation_noise, 1.0, {"sigma": np.radians(2.0), "representation": "quat"}),
        ])
        p1, q1 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(42)))
        p2, q2 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(42)))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)


# =============================================================================
# Pipeline return_params
# =============================================================================

class TestPipelineReturnParams:
    """Tests for AugmentationPipeline(..., return_params=True)."""

    @staticmethod
    def _sampling_pipeline(cache_quats=True, mirror_prob=1.0):
        """rotate (sampled angle) + mirror (probabilistic) + noise (static)."""
        return AugmentationPipeline([
            (rotate_vertical, 1.0, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y",
                "representation": "quat",
            }),
            (mirror, mirror_prob, {
                "lr_joint_pairs": [(1, 2)],
                "lateral_axis": "+x",
                "representation": "quat",
            }),
            (add_joint_rotation_noise, 1.0, {
                "sigma": np.radians(2.0), "representation": "quat"}),
        ], cache_quats=cache_quats)

    def test_default_returns_two_values(self, bvh_example):
        """The flag is opt-in: the 2-tuple contract is unchanged."""
        pos, quats = _get_quat_data(bvh_example)
        result = self._sampling_pipeline()(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0))
        assert isinstance(result, MotionArrays)

    def test_records_one_entry_per_step_in_order(self, bvh_example):
        """Records are index-aligned with pipeline.augmentations."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = self._sampling_pipeline()
        _, _, params = as_triple(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0), return_params=True))
        assert len(params) == len(pipeline)
        assert [p["name"] for p in params] == [
            fn.__name__ for fn, _, _ in pipeline.augmentations]

    def test_sampled_values_are_what_the_function_received(self, bvh_example):
        """The reported angle is the angle the augmentation was called with."""
        pos, quats = _get_quat_data(bvh_example)
        seen = {}

        def spy(arrays, *, angle, representation):
            seen["angle"] = angle
            return arrays

        pipeline = AugmentationPipeline([
            (spy, 1.0, {"angle": lambda rng: rng.uniform(-np.pi, np.pi),
                        "representation": "quat"}),
        ])
        _, _, params = as_triple(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(7), return_params=True))
        assert params[0]["params"]["angle"] == seen["angle"]

    def test_static_kwargs_and_rng_excluded(self, bvh_example):
        """Only sampled kwargs are reported; config and machinery are not."""
        pos, quats = _get_quat_data(bvh_example)
        _, _, params = as_triple(self._sampling_pipeline()(
            MotionArrays(root_pos=pos, joint_rot=quats),
            rng=np.random.default_rng(0), return_params=True))
        assert set(params[0]["params"]) == {"angle"}
        assert params[2]["params"] == {}          # noise: sigma is static
        for record in params:
            assert "rng" not in record["params"]
            assert "representation" not in record["params"]

    def test_applied_reflects_probability(self, bvh_example):
        """prob=0 never fires, prob=1 always does."""
        pos, quats = _get_quat_data(bvh_example)
        for prob, expected in ((0.0, False), (1.0, True)):
            _, _, params = as_triple(self._sampling_pipeline(mirror_prob=prob)(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(3), return_params=True))
            assert params[1]["applied"] is expected

    def test_skipped_step_reports_no_params(self, bvh_example):
        """A step that never fires resolves nothing."""
        pos, quats = _get_quat_data(bvh_example)
        drawn = []
        pipeline = AugmentationPipeline([
            (rotate_vertical, 0.0, {
                "angle": lambda rng: drawn.append(1) or 0.5,
                "up_axis": "+y", "representation": "quat"}),
        ])
        _, _, params = as_triple(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0), return_params=True))
        assert params[0] == {"name": "rotate_vertical", "applied": False,
                             "params": {}}
        assert drawn == []          # the callable was never invoked

    def test_does_not_perturb_the_random_stream(self, bvh_example):
        """Asking for params must not change what the pipeline produces."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = self._sampling_pipeline(mirror_prob=0.5)
        p1, q1 = as_pair(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(11)))
        p2, q2, _ = as_triple(pipeline(MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()), rng=np.random.default_rng(11), return_params=True))
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(q1, q2)

    def test_staged_and_direct_paths_agree(self, bvh_example):
        """cache_quats=True/False report identical records for one seed."""
        pos, quats = _get_quat_data(bvh_example)
        records = []
        for cache in (True, False):
            _, _, params = as_triple(self._sampling_pipeline(
                cache_quats=cache, mirror_prob=0.5)(
                    MotionArrays(root_pos=pos.copy(), joint_rot=quats.copy()),
                    rng=np.random.default_rng(5), return_params=True))
            records.append(params)
        assert records[0] == records[1]

    def test_custom_step_in_staged_path_recorded(self, bvh_example):
        """An unregistered function still reports its sampled kwargs."""
        pos, quats = _get_quat_data(bvh_example)

        def custom(arrays, *, scale, representation):
            return arrays.replace(root_pos=arrays.root_pos * scale)

        pipeline = AugmentationPipeline([
            (custom, 1.0, {"scale": lambda rng: rng.uniform(1.0, 2.0),
                           "representation": "quat"}),
        ])
        _, _, params = as_triple(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(2), return_params=True))
        assert params[0]["name"] == "custom"
        assert 1.0 <= params[0]["params"]["scale"] <= 2.0

    def test_empty_pipeline_returns_empty_records(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q, params = as_triple(AugmentationPipeline([])(MotionArrays(root_pos=pos, joint_rot=quats), return_params=True))
        assert params == []
        np.testing.assert_array_equal(new_q, quats)

    def test_standard_pipeline_records_are_json_serializable(self, bvh_example):
        """The logging use case: dump per-sample draws straight to JSON."""
        import json

        pos, quats = _get_quat_data(bvh_example)
        skel = get_skeleton_info(bvh_example)
        pipeline = AugmentationPipeline.standard(
            skel, representation="quat", up_axis=bvh_example.world_up)
        _, _, params = as_triple(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0), return_params=True))
        assert json.loads(json.dumps(params)) == params


class TestPipelineStepsWithoutName:
    """Steps that are not plain functions still run and still report.

    ``functools.partial`` and callable instances are the two natural
    ways to write a custom step with baked-in configuration, and
    neither carries ``__name__`` — reading it unguarded broke every
    call, not just ``return_params=True`` ones.
    """

    @staticmethod
    def _scale_root(arrays, representation="quat", scale=1.0):
        return arrays.replace(root_pos=arrays.root_pos * scale)

    @pytest.fixture
    def steps(self):
        class ScalerStep:
            def __call__(self, arrays, representation="quat"):
                return arrays.replace(root_pos=arrays.root_pos * 3.0)

        partial_step = functools.partial(
            TestPipelineStepsWithoutName._scale_root, scale=2.0)
        return {"partial": (partial_step, 2.0), "callable": (ScalerStep(), 3.0)}

    @pytest.mark.parametrize("kind", ["partial", "callable"])
    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_step_runs(self, bvh_example, steps, kind, cache_quats):
        pos, quats = _get_quat_data(bvh_example)
        step, scale = steps[kind]
        pipeline = AugmentationPipeline(
            [(step, 1.0, {"representation": "quat"})], cache_quats=cache_quats)

        new_pos, _ = as_pair(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0)))

        np.testing.assert_allclose(new_pos, pos * scale)

    @pytest.mark.parametrize("kind, expected", [
        ("partial", "_scale_root"),      # unwrapped to the wrapped function
        ("callable", "ScalerStep"),      # falls back to the class name
    ])
    def test_record_name(self, bvh_example, steps, kind, expected):
        pos, quats = _get_quat_data(bvh_example)
        step, _ = steps[kind]
        pipeline = AugmentationPipeline([(step, 1.0, {"representation": "quat"})])

        _, _, params = as_triple(pipeline(MotionArrays(root_pos=pos, joint_rot=quats), rng=np.random.default_rng(0), return_params=True))

        assert params[0]["name"] == expected

    @pytest.mark.parametrize("kind", ["partial", "callable"])
    def test_repr(self, steps, kind):
        pipeline = AugmentationPipeline(
            [(steps[kind][0], 1.0, {"representation": "quat"})])
        assert "AugmentationPipeline" in repr(pipeline)


# =============================================================================
# Preprocessing filter_fn
# =============================================================================

class TestPreprocessingFilter:
    """Tests for filter_fn in preprocess_directory."""

    @pytest.fixture
    def bvh_dir(self):
        return Path(__file__).parent.parent / "bvh_data"

    def test_filter_reduces_clips(self, bvh_dir, tmp_path):
        """filter_fn should exclude files before loading."""
        out = tmp_path / "filtered.npz"
        # Only include bvh_example
        result = preprocess_directory(
            bvh_dir, out,
            filter_fn=lambda stem: stem == "bvh_test1",
        )
        assert result["num_clips"] == 1
        assert result["filenames"] == ["bvh_test1"]

    def test_filter_none_includes_all(self, bvh_dir, tmp_path):
        """filter_fn=None should include all matching files."""
        out = tmp_path / "all.npz"
        # Use single-file pattern to avoid mixed-skeleton errors
        result = preprocess_directory(
            bvh_dir, out, filter_fn=None, file_pattern="bvh_test1.bvh")
        assert result["num_clips"] == 1

    def test_filter_with_label_fn(self, bvh_dir, tmp_path):
        """filter_fn and label_fn should compose correctly."""
        import shutil
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "clip_a.bvh")
        shutil.copy(bvh_dir / "bvh_test1.bvh", work_dir / "clip_b.bvh")
        allowed = {"clip_a", "clip_b"}
        labels_map = {"clip_a": 0, "clip_b": 1}
        out = tmp_path / "labeled.npz"
        result = preprocess_directory(
            work_dir, out,
            filter_fn=lambda stem: stem in allowed,
            label_fn=lambda stem: labels_map[stem],
        )
        assert result["num_clips"] == 2
        loaded = load_preprocessed(out)
        np.testing.assert_array_equal(loaded["labels"], [0, 1])

    def test_filter_excludes_all_raises(self, bvh_dir, tmp_path):
        """Filtering out everything should raise ValueError."""
        out = tmp_path / "empty.npz"
        with pytest.raises(ValueError, match="No BVH files found.*after filtering"):
            preprocess_directory(
                bvh_dir, out,
                filter_fn=lambda stem: False,
            )

    def test_filter_roundtrip(self, bvh_dir, tmp_path):
        """Filtered output should load correctly."""
        out = tmp_path / "filtered.npz"
        preprocess_directory(
            bvh_dir, out,
            filter_fn=lambda stem: stem == "bvh_test1",
            representation="quat",
        )
        loaded = load_preprocessed(out)
        assert len(loaded["clips"]) == 1
        assert loaded["representation"] == "quat"
        assert loaded["filenames"] == ["bvh_test1"]


# =============================================================================
# Version floor
# =============================================================================


class TestVersionFloor:
    """pybvh-ml 0.5 requires pybvh >= 0.8.1 (pin: pybvh>=0.8.1,<0.9)."""

    def test_pybvh_provides_the_api_the_floor_exists_for(self):
        """Asserted by capability, not by version string.

        The floor exists to guarantee these symbols; a string
        comparison would additionally go red during the coordinated
        release window, when the sister repo carries the API but has
        not yet bumped its ``__version__`` — a release-ordering fact,
        not a correctness one.
        """
        import pybvh
        assert hasattr(pybvh, "parse_axis")        # 0.8.1
        assert hasattr(pybvh.rotations, "quat_multiply")   # 0.8.0
        assert hasattr(pybvh.Bvh, "to_quat")               # 0.8.0
        assert not hasattr(pybvh.Bvh, "to_quaternions")    # removed in 0.8.0

    def test_pyproject_floor_covers_parse_axis(self):
        import re
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        match = re.search(r'"pybvh>=([\d.]+),', pyproject.read_text())
        assert match is not None
        floor = tuple(int(x) for x in match.group(1).split("."))
        assert floor >= (0, 8, 1), (
            "pybvh.parse_axis ships in pybvh 0.8.1; a lower floor would "
            "let pybvh-ml install against a pybvh that lacks it")

    def test_public_parse_axis_is_used(self):
        """The private _parse_axis copy is gone: pybvh made it public,
        so per the charter pybvh-ml consumes it rather than keeping a
        reimplementation."""
        from pybvh_ml import augmentation, _staged
        assert not hasattr(augmentation, "_parse_axis")
        assert augmentation.parse_axis is _staged.parse_axis

    def test_version_matches_pyproject(self):
        # Guards the pyproject/__init__ version drift that shipped in 0.3/0.4.
        import re
        import pybvh_ml
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        match = re.search(
            r'^version = "(.+)"$', pyproject.read_text(), re.MULTILINE)
        assert match is not None
        assert match.group(1) == pybvh_ml.__version__


class TestAugmentationStep:
    """Pipeline steps are named tuples — introspection reads, and index
    access keeps working."""

    def _step(self):
        return (dropout_arrays, 0.25, {"drop_rate": 0.1,
                                       "representation": "quat"})

    def test_named_fields(self):
        pipeline = AugmentationPipeline([self._step()])
        step = pipeline.augmentations[0]
        assert step.fn is dropout_arrays
        assert step.prob == 0.25
        assert step.kwargs["drop_rate"] == 0.1

    def test_index_access_still_works(self):
        """Downstream tests assert on augmentations[i][2] — a NamedTuple
        is a tuple, so nothing that read positionally breaks."""
        pipeline = AugmentationPipeline([self._step()])
        step = pipeline.augmentations[0]
        assert step[0] is dropout_arrays
        assert step[1] == 0.25
        assert step[2]["drop_rate"] == 0.1
        fn, prob, kwargs = step
        assert (fn, prob) == (dropout_arrays, 0.25)

    def test_exported(self):
        import pybvh_ml
        assert pybvh_ml.AugmentationStep is AugmentationStep
        assert "AugmentationStep" in pybvh_ml.__all__

    @pytest.mark.parametrize("bad,match", [
        ((dropout_arrays, 0.5), "must be a \\(fn, probability, kwargs\\) triple"),
        ((None, 0.5, {}), "must be callable"),
        ((dropout_arrays, 1.5, {}), "probability in \\[0, 1\\]"),
        ((dropout_arrays, -0.1, {}), "probability in \\[0, 1\\]"),
        ((dropout_arrays, "x", {}), "probability in \\[0, 1\\]"),
        ((dropout_arrays, 0.5, "nope"), "must be a dict of kwargs"),
    ])
    def test_malformed_steps_raise_at_construction(self, bad, match):
        """A bad step used to surface as an opaque unpacking error inside
        __getitem__, or as a step that silently always fired."""
        with pytest.raises(ValueError, match=match):
            AugmentationPipeline([bad])


class TestPipelineRepresentationDefault:
    """A pipeline is homogeneous in practice — declare the
    representation once instead of on every step."""

    def _arrays(self, bvh_example):
        root_pos, quats = bvh_example.to_quat()
        return root_pos[:10].copy(), quats[:10].copy()

    def test_pipeline_level_representation_feeds_every_step(
            self, bvh_example, rng):
        rp, jd = self._arrays(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": 0.3,
                                    "up_axis": bvh_example.world_up}),
            (add_joint_rotation_noise, 1.0, {"sigma": 0.01}),
        ], representation="quat")
        new_rp, new_jd = as_pair(pipeline(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng))
        assert new_jd.shape == jd.shape

    def test_matches_per_step_declaration(self, bvh_example):
        """Same draws, same result — the default is a spelling of what
        the per-step kwarg already said."""
        rp, jd = self._arrays(bvh_example)
        steps_bare = [(rotate_vertical, 1.0,
                       {"angle": 0.3, "up_axis": bvh_example.world_up})]
        steps_full = [(rotate_vertical, 1.0,
                       {"angle": 0.3, "up_axis": bvh_example.world_up,
                        "representation": "quat"})]
        a = AugmentationPipeline(steps_bare, representation="quat")(MotionArrays(root_pos=rp, joint_rot=jd), rng=np.random.default_rng(0))
        b = AugmentationPipeline(steps_full)(MotionArrays(root_pos=rp, joint_rot=jd), rng=np.random.default_rng(0))
        np.testing.assert_allclose(a.root_pos, b.root_pos, rtol=0, atol=0)
        np.testing.assert_allclose(a.joint_rot, b.joint_rot, rtol=0, atol=0)

    def test_per_step_override_wins(self, bvh_example, rng):
        """The step's own token beats the pipeline default."""
        rp, rot6d = bvh_example.to_6d()
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": 0.3,
                                    "up_axis": bvh_example.world_up,
                                    "representation": "6d"}),
        ], representation="quat")
        _, new_jd = as_pair(pipeline(MotionArrays(root_pos=rp[:10].copy(), joint_rot=rot6d[:10].copy()), rng=rng))
        assert new_jd.shape[-1] == 6

    def test_satisfies_the_cache_quats_requirement(self, bvh_example, rng):
        """cache_quats=True needs *something* to declare the
        representation; the pipeline-level default is that something."""
        rp, jd = self._arrays(bvh_example)
        pipeline = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.01})], representation="quat")
        new_rp, new_jd = as_pair(pipeline(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng))
        assert new_jd.shape == jd.shape

    def test_still_raises_when_nothing_declares_one(self, bvh_example, rng):
        rp, jd = self._arrays(bvh_example)
        pipeline = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.01})])
        with pytest.raises(ValueError, match="No representation declared"):
            pipeline(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng)

    def test_custom_step_without_the_parameter_is_untouched(
            self, bvh_example, rng):
        """Injection is signature-aware: a custom step that doesn't take
        `representation` is still called with exactly its own kwargs."""
        seen = {}

        def custom(arrays, *, scale):
            seen["kwargs"] = {"scale": scale}
            return arrays.replace(root_pos=arrays.root_pos * scale)

        rp, jd = self._arrays(bvh_example)
        pipeline = AugmentationPipeline(
            [(custom, 1.0, {"scale": 2.0})], representation="quat")
        pipeline(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng)
        assert seen["kwargs"] == {"scale": 2.0}

    def test_euler_orders_default(self, bvh_example, rng):
        orders = list(bvh_example.euler_orders)
        rp = bvh_example.root_pos[:10].copy()
        jd = bvh_example.joint_angles[:10].copy()
        pipeline = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.01})],
            representation="euler", euler_orders=orders)
        _, new_jd = as_pair(pipeline(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng))
        assert new_jd.shape == jd.shape

    def test_standard_factory_uses_the_pipeline_level_defaults(
            self, bvh_example):
        """The factory builds exactly the homogeneous pipeline these
        defaults exist for — it should not repeat the token five times."""
        info = get_skeleton_info(bvh_example)
        pipeline = AugmentationPipeline.standard(info, representation="6d")
        assert pipeline.representation == "6d"
        assert all("representation" not in step.kwargs
                   for step in pipeline.augmentations)

    def test_repr_shows_the_defaults(self, bvh_example):
        pipeline = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.01})], representation="quat")
        assert "representation='quat'" in repr(pipeline)


class TestPreprocessFrameRate:
    """`target_fps=` — resampling before extraction is what makes every
    derived feature describe the motion at the target rate."""

    @pytest.fixture
    def mixed_dir(self, tmp_path, bvh_dir):
        from pybvh import read_bvh_file, write_bvh_file
        work = tmp_path / "work"
        work.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work / "a.bvh")
        write_bvh_file(bvh.resample(1.0 / bvh.frame_time / 4), work / "b.bvh")
        return work

    def test_uniformity_audit_records_frame_rate(self, mixed_dir, tmp_path):
        out = tmp_path / "ds.npz"
        with pytest.warns(UserWarning, match="Frame rate is not uniform"):
            result = preprocess_directory(mixed_dir, out)
        assert len(result["uniformity"]["fps"]) == 2

    def test_target_fps_resamples_before_extraction(self, tmp_path, bvh_dir):
        from pybvh import read_bvh_file
        source = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        original_fps = 1.0 / source.frame_time
        out = tmp_path / "ds.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                             target_fps=original_fps / 4)
        loaded = load_preprocessed(out)
        frames = loaded["clips"][0]["root_pos"].shape[0]
        assert frames == pytest.approx(source.frame_count / 4, abs=2)

    def test_velocities_are_recomputed_not_decimated(self, tmp_path, bvh_dir):
        """The reason resampling must precede extraction: velocities are
        finite differences whose stencil baseline is the *original*
        frame_time, so decimating a saved velocity array cannot
        reproduce a genuine resample."""
        from pybvh import read_bvh_file
        source = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        original_fps = 1.0 / source.frame_time
        full = tmp_path / "full.npz"
        resampled = tmp_path / "resampled.npz"
        preprocess_directory(bvh_dir, full, file_pattern="bvh_test1.bvh",
                             include_velocities=True)
        preprocess_directory(bvh_dir, resampled, file_pattern="bvh_test1.bvh",
                             include_velocities=True,
                             target_fps=original_fps / 4)
        decimated = load_preprocessed(full)["clips"][0]["velocities"][::4]
        genuine = load_preprocessed(resampled)["clips"][0]["velocities"]
        n = min(len(decimated), len(genuine))
        assert not np.allclose(decimated[:n], genuine[:n], rtol=1e-3)

    def test_target_fps_silences_the_warning(self, mixed_dir, tmp_path):
        import warnings as _warnings
        out = tmp_path / "ds.npz"
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            preprocess_directory(mixed_dir, out, target_fps=30.0)
        assert not any("Frame rate is not uniform" in str(w.message)
                       for w in caught)

    def test_target_fps_unifies_clip_rates(self, mixed_dir, tmp_path):
        """Two clips of the same motion stored at rates 4x apart come out
        covering the same span of frames.  Not bit-identical: a
        downsample-then-upsample round trip drops the tail frame that no
        longer falls on a sample time."""
        untouched = tmp_path / "raw.npz"
        with pytest.warns(UserWarning, match="Frame rate is not uniform"):
            preprocess_directory(mixed_dir, untouched)
        before = sorted(c["root_pos"].shape[0]
                        for c in load_preprocessed(untouched)["clips"])
        assert before[1] > 3 * before[0]

        out = tmp_path / "ds.npz"
        preprocess_directory(mixed_dir, out, target_fps=30.0)
        after = sorted(c["root_pos"].shape[0]
                       for c in load_preprocessed(out)["clips"])
        assert after[1] - after[0] <= 4

    def test_harmonize_fills_frame_rate_from_the_majority(
            self, tmp_path, bvh_dir):
        """harmonize=True promises to unify every axis the dataset
        disagrees on — frame rate included."""
        from pybvh import read_bvh_file, write_bvh_file
        work = tmp_path / "work"
        work.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        fast = 1.0 / bvh.frame_time
        write_bvh_file(bvh, work / "a.bvh")
        write_bvh_file(bvh, work / "b.bvh")
        write_bvh_file(bvh.resample(fast / 4), work / "c.bvh")

        out = tmp_path / "ds.npz"
        with pytest.warns(UserWarning, match="Frame rate is not uniform"):
            result = preprocess_directory(work, out, harmonize=True)
        harmonized_to = result["uniformity"]["harmonized_to"]
        assert harmonized_to["targets"]["target_fps"] == pytest.approx(fast)
        # Only the minority clip needed resampling.
        assert harmonized_to["stage_counts"]["resample"] == 1

    def test_explicit_target_fps_overrides_the_majority(
            self, mixed_dir, tmp_path):
        out = tmp_path / "ds.npz"
        result = preprocess_directory(mixed_dir, out, harmonize=True,
                                      target_fps=25.0)
        targets = result["uniformity"]["harmonized_to"]["targets"]
        assert targets["target_fps"] == pytest.approx(25.0)


class TestLoadedSkeletonInfoShape:
    """`load_preprocessed` guarantees the full get_skeleton_info key set
    whatever version wrote the file."""

    def test_axis_keys_present_on_a_fresh_dataset(self, tmp_path, bvh_dir):
        out = tmp_path / "ds.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        info = load_preprocessed(out)["skeleton_info"]
        for key in ("world_up", "rest_forward", "rest_up"):
            assert info[key] is not None

    def test_keys_an_older_dataset_never_recorded_read_back_as_none(
            self, tmp_path, bvh_dir):
        """Pre-0.5.0 files carry no axis strings.  They must load as None
        rather than being absent, so consumers can index them directly
        instead of doing a .get() dance."""
        import json as _json
        out = tmp_path / "ds.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")

        # Rewrite the file with the 0.4-era skeleton_info: axis keys absent.
        with np.load(out, allow_pickle=False) as data:
            contents = dict(data)
        legacy = _json.loads(str(contents["skeleton_info_json"]))
        for key in ("world_up", "rest_forward", "rest_up"):
            legacy.pop(key)
        contents["skeleton_info_json"] = np.array(_json.dumps(legacy))
        np.savez(out, **contents)

        info = load_preprocessed(out)["skeleton_info"]
        for key in ("world_up", "rest_forward", "rest_up"):
            assert key in info
            assert info[key] is None
        # Keys that were recorded still survive.
        assert info["joint_names"] == legacy["joint_names"]

    def test_optional_keys_stay_absent(self, tmp_path, bvh_dir):
        """foot_joints signals a preprocessing choice — inventing a None
        would make "not requested" look like "requested and empty"."""
        out = tmp_path / "ds.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh")
        assert "foot_joints" not in load_preprocessed(out)["skeleton_info"]


class TestTestModeRngContract:
    """Pins what the docstring now promises about mode="test" — the fix
    for the one 0.5.0 change that broke evaluation silently."""

    def test_rng_none_is_reproducible_across_calls(self):
        a = uniform_temporal_sample(100, 16, mode="test")
        b = uniform_temporal_sample(100, 16, mode="test")
        np.testing.assert_array_equal(a, b)

    def test_a_shared_generator_makes_test_mode_vary(self):
        """Not a bug — the documented consequence of supplying an rng.
        A generator shared with other draws has advanced by the next
        call, so repeated reads of one clip differ."""
        shared = np.random.default_rng(0)
        a = uniform_temporal_sample(100, 16, mode="test", rng=shared)
        b = uniform_temporal_sample(100, 16, mode="test", rng=shared)
        assert not np.array_equal(a, b)

    def test_a_freshly_seeded_generator_is_reproducible(self):
        a = uniform_temporal_sample(100, 16, mode="test",
                                    rng=np.random.default_rng(7))
        b = uniform_temporal_sample(100, 16, mode="test",
                                    rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


# =============================================================================
# Harmonize target consistency and degenerate rigs
# =============================================================================

class TestHarmonizeTargetConsistency:
    """Independently-resolved axis majorities must still be mutually valid.

    Regression: world-up and rest-forward majorities come from different
    clips and need not co-occur in any single one, so the pair could be
    parallel — which pybvh rejects per clip, naming no file and no kwarg.
    """

    @staticmethod
    def _uniformity(world_up, rest_forward):
        return {
            "fps": {"30": ["a"]},
            "world_up": world_up,
            "rest_forward": rest_forward,
            "rest_up": {"+y": ["a"]},
            "rest_anim_mismatch": [],
        }

    def test_majority_forward_parallel_to_up_is_excluded(self):
        from pybvh_ml.preprocessing import _resolve_harmonize_targets
        uniformity = self._uniformity(
            world_up={"+z": ["a", "b", "c", "d"], "+x": ["e", "f", "g"]},
            rest_forward={"+z": ["a", "b", "c"], "+y": ["d", "e"],
                          "+x": ["f", "g"]})
        targets = _resolve_harmonize_targets(
            [], uniformity, "6d", None, None, None, None)
        assert targets["target_world_up"] == "+z"
        # '+z' wins on count but is parallel to the resolved up axis;
        # the runner-up perpendicular candidate takes it instead.
        assert targets["target_rest_forward"] == "+y"

    def test_sign_flipped_forward_counts_as_parallel(self):
        """Parallel is an axis test, not a string test: '-z' vs '+z'."""
        from pybvh_ml.preprocessing import _resolve_harmonize_targets
        uniformity = self._uniformity(
            world_up={"+z": ["a", "b"], "+x": ["c"]},
            rest_forward={"-z": ["a", "b", "c"], "+y": ["d"]})
        targets = _resolve_harmonize_targets(
            [], uniformity, "6d", None, None, None, None)
        assert targets["target_rest_forward"] == "+y"

    def test_all_candidates_parallel_warns_and_skips(self):
        from pybvh_ml.preprocessing import _resolve_harmonize_targets
        uniformity = self._uniformity(
            world_up={"+z": ["a", "b"], "+x": ["c"]},
            rest_forward={"+z": ["a"], "-z": ["b"]})
        with pytest.warns(UserWarning, match="left unharmonized"):
            targets = _resolve_harmonize_targets(
                [], uniformity, "6d", None, None, None, None)
        assert "target_rest_forward" not in targets

    def test_explicit_parallel_pair_raises_naming_both_kwargs(self):
        from pybvh_ml.preprocessing import _resolve_harmonize_targets
        uniformity = self._uniformity(
            world_up={"+z": ["a"]}, rest_forward={"+y": ["a"]})
        with pytest.raises(ValueError) as exc:
            _resolve_harmonize_targets(
                [], uniformity, "6d", "+z", "-z", None, None)
        assert "target_rest_forward" in str(exc.value)
        assert "target_world_up" in str(exc.value)

    def test_explicit_parallel_pair_raises_without_harmonize(
            self, bvh_dir, tmp_path):
        """The same unreachable target is validated on the direct path."""
        from pybvh import write_bvh_file
        work = tmp_path / "work"
        work.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        write_bvh_file(bvh, work / "a.bvh")
        with pytest.raises(ValueError) as exc:
            preprocess_directory(
                work, tmp_path / "out.npz", representation="6d",
                target_rest_forward=bvh.world_up)
        assert "target_rest_forward" in str(exc.value)
        assert "a" in str(exc.value)

    def test_mixed_world_up_dataset_harmonizes(self, bvh_dir, tmp_path):
        """End-to-end: a corpus whose axis majorities collide completes."""
        from pybvh import write_bvh_file
        work = tmp_path / "work"
        work.mkdir()
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        # Four clips keep the fixture's up axis; three are rotated to a
        # different one, so world_up and rest_forward majorities are
        # measured over different subsets.
        for i in range(4):
            write_bvh_file(bvh, work / f"keep_{i}.bvh")
        rotated = bvh.reorient_world_up("+y")
        for i in range(3):
            write_bvh_file(rotated, work / f"rot_{i}.bvh")

        out = tmp_path / "dataset.npz"
        result = preprocess_directory(
            work, out, representation="6d", harmonize=True)
        targets = result["uniformity"]["harmonized_to"]["targets"]
        if "target_rest_forward" in targets:
            from pybvh_ml.preprocessing import _is_parallel
            assert not _is_parallel(
                targets["target_rest_forward"], targets["target_world_up"])
        assert out.exists()


class TestDegenerateRigAudit:
    """A rig with no measurable rest pose is 'unknown', not 'mismatched'."""

    def test_not_counted_as_rest_anim_mismatch(self, degenerate_rig_dir):
        from pybvh_ml.preprocessing import (
            _compute_uniformity, _REST_UP_UNKNOWN)
        stems = ["degen_a", "degen_b"]
        clips = [read_bvh_file(degenerate_rig_dir / f"{s}.bvh")
                 for s in stems]
        assert all(c.rest_up is None for c in clips)
        uniformity = _compute_uniformity(clips, stems)
        # 'unknown' is not a disagreement — the axis was never measured.
        assert uniformity["rest_anim_mismatch"] == []
        assert uniformity["rest_up"] == {_REST_UP_UNKNOWN: stems}

    def test_no_corruption_warning_for_degenerate_rigs(
            self, degenerate_rig_dir, tmp_path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preprocess_directory(
                degenerate_rig_dir, tmp_path / "out.npz",
                representation="quat")
        assert not [w for w in caught
                    if "Rest-pose up disagrees" in str(w.message)]

    @pytest.mark.parametrize("suffix", [".npz", ".hdf5"])
    def test_unknown_key_roundtrips_exactly(
            self, degenerate_rig_dir, tmp_path, suffix):
        """JSON object keys are strings: a None key would come back
        as the string 'null' and the saved audit would not equal the
        returned one."""
        if suffix == ".hdf5":
            pytest.importorskip("h5py")
        out = tmp_path / ("dataset" + suffix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = preprocess_directory(
                degenerate_rig_dir, out, representation="quat")
        loaded = load_preprocessed(out)
        assert loaded["uniformity"] == result["uniformity"]
        assert "null" not in loaded["uniformity"]["rest_up"]

    @pytest.mark.parametrize("harmonize", [False, True])
    def test_rest_up_target_on_degenerate_rig_raises_named(
            self, degenerate_rig_dir, tmp_path, harmonize):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc:
                preprocess_directory(
                    degenerate_rig_dir, tmp_path / "out.npz",
                    representation="quat", target_rest_up="+y",
                    harmonize=harmonize)
        assert "degen_a" in str(exc.value)
        assert "degenerate rest pose" in str(exc.value)

    def test_guard_fires_for_majority_filled_target(self, degenerate_rig_dir):
        """The guard covers an auto-resolved target, not just an explicit
        one — that is the case where the user never wrote the kwarg."""
        from pybvh_ml.preprocessing import _reject_degenerate_rest_up_targets
        stems = ["degen_a", "degen_b"]
        clips = [read_bvh_file(degenerate_rig_dir / f"{s}.bvh")
                 for s in stems]
        with pytest.raises(ValueError, match="degenerate rest pose"):
            _reject_degenerate_rest_up_targets(clips, stems, "+y")


class TestAppliedTargetsRecord:
    """Direct-mode transforms are recorded the way harmonize's are."""

    @pytest.fixture
    def one_clip_dir(self, bvh_dir, tmp_path):
        from pybvh import write_bvh_file
        work = tmp_path / "work"
        work.mkdir()
        write_bvh_file(read_bvh_file(bvh_dir / "bvh_test1.bvh"),
                       work / "a.bvh")
        return work

    def test_target_fps_recorded_and_roundtrips(self, one_clip_dir, tmp_path):
        out = tmp_path / "dataset.npz"
        result = preprocess_directory(
            one_clip_dir, out, representation="6d", target_fps=15)
        assert result["uniformity"]["applied_targets"] == {"target_fps": 15.0}
        loaded = load_preprocessed(out)
        assert loaded["uniformity"] == result["uniformity"]

    def test_absent_when_nothing_applied(self, one_clip_dir, tmp_path):
        result = preprocess_directory(
            one_clip_dir, tmp_path / "d.npz", representation="6d")
        assert "applied_targets" not in result["uniformity"]

    def test_absent_under_harmonize(self, one_clip_dir, tmp_path):
        """harmonized_to owns the record there; two would contradict."""
        result = preprocess_directory(
            one_clip_dir, tmp_path / "d.npz", representation="6d",
            harmonize=True, target_fps=15)
        assert "applied_targets" not in result["uniformity"]
        assert (result["uniformity"]["harmonized_to"]["targets"]["target_fps"]
                == 15.0)

    def test_no_op_axis_target_still_recorded(self, one_clip_dir, tmp_path):
        """The record is the requested policy, not the clips it moved."""
        bvh = read_bvh_file(one_clip_dir / "a.bvh")
        result = preprocess_directory(
            one_clip_dir, tmp_path / "d.npz", representation="6d",
            target_world_up=bvh.world_up)
        assert (result["uniformity"]["applied_targets"]["target_world_up"]
                == bvh.world_up)


# =============================================================================
# Pipeline construction and entry validation
# =============================================================================

class TestPipelineStepValidation:
    """Both spellings of a step get the same checks."""

    @pytest.mark.parametrize("fn,prob,kwargs,match", [
        ("not-callable", 1.0, {}, "must be callable"),
        (mirror, 7.5, {}, r"probability in \[0, 1\]"),
        (mirror, -0.1, {}, r"probability in \[0, 1\]"),
        (mirror, 1.0, None, "must be a dict"),
    ])
    def test_prebuilt_step_is_validated(self, fn, prob, kwargs, match):
        """Regression: an AugmentationStep built by hand skipped every
        check, so a bad probability silently always fired and bad kwargs
        died deep inside the call."""
        step = AugmentationStep(fn=fn, prob=prob, kwargs=kwargs)
        with pytest.raises(ValueError, match=match) as exc:
            AugmentationPipeline([step], representation="quat")
        assert "augmentations[0]" in str(exc.value)

    def test_valid_prebuilt_step_still_accepted(self):
        step = AugmentationStep(fn=mirror, prob=0.5, kwargs={})
        pipeline = AugmentationPipeline([step], representation="quat")
        assert pipeline.augmentations[0] == step

    def test_index_named_for_later_step(self):
        good = AugmentationStep(fn=mirror, prob=1.0, kwargs={})
        bad = AugmentationStep(fn=mirror, prob=3.0, kwargs={})
        with pytest.raises(ValueError, match=r"augmentations\[1\]"):
            AugmentationPipeline([good, bad], representation="quat")


class TestPipelineFrameCountValidationAtEntry:
    """Mismatched frame counts raise on both dispatch paths, always."""

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_mismatch_raises_even_when_no_step_fires(self, cache_quats):
        """Regression: the direct path only validated inside a firing
        step, so a p=0 pipeline returned mismatched arrays untouched
        and a p<1 pipeline raised at random."""
        root_pos = np.zeros((8, 3))
        joint_data = np.zeros((9, 3, 4))
        joint_data[..., 0] = 1.0
        pipeline = AugmentationPipeline(
            [(mirror, 0.0, {"lr_joint_pairs": [(0, 1)]})],
            cache_quats=cache_quats, representation="quat")
        with pytest.raises(ValueError, match="frame"):
            pipeline(MotionArrays(root_pos=root_pos, joint_rot=joint_data), rng=np.random.default_rng(0))

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_matching_frame_counts_still_pass(self, cache_quats):
        root_pos = np.zeros((8, 3))
        joint_data = np.zeros((8, 3, 4))
        joint_data[..., 0] = 1.0
        pipeline = AugmentationPipeline(
            [(mirror, 0.0, {"lr_joint_pairs": [(0, 1)]})],
            cache_quats=cache_quats, representation="quat")
        out_pos, out_jd = as_pair(pipeline(MotionArrays(root_pos=root_pos, joint_rot=joint_data), rng=np.random.default_rng(0)))
        assert out_pos.shape == root_pos.shape
        assert out_jd.shape == joint_data.shape


class TestPipelineRepresentationConflict:
    """cache_quats must not be able to change results.

    Regression: two built-ins declaring different representations gave
    different arrays under cache_quats True vs False — the staged path
    converted through the quat cache, the direct path reinterpreted the
    previous step's bytes under the new token.
    """

    def test_conflicting_builtin_steps_raise(self):
        with pytest.raises(ValueError) as exc:
            AugmentationPipeline([
                (add_joint_rotation_noise, 1.0, {"sigma": 0.1,
                                        "representation": "euler",
                                        "euler_orders": ["XYZ"] * 3}),
                (add_joint_rotation_noise, 1.0, {"sigma": 0.1,
                                        "representation": "axisangle"}),
            ])
        assert "augmentations[0]" in str(exc.value)
        assert "augmentations[1]" in str(exc.value)

    def test_step_conflicting_with_pipeline_default_raises(self):
        with pytest.raises(ValueError, match="must agree"):
            AugmentationPipeline([
                (add_joint_rotation_noise, 1.0, {"sigma": 0.1}),
                (add_joint_rotation_noise, 1.0, {"sigma": 0.1,
                                        "representation": "axisangle"}),
            ], representation="quat")

    def test_custom_step_between_lifts_the_check(self):
        """A custom step may legitimately convert mid-pipeline."""
        def convert_step(arrays, **kwargs):
            return arrays

        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {"sigma": 0.1,
                                    "representation": "quat"}),
            (convert_step, 1.0, {}),
            (add_joint_rotation_noise, 1.0, {"sigma": 0.1,
                                    "representation": "axisangle"}),
        ])
        assert len(pipeline.augmentations) == 3

    def test_homogeneous_pipeline_unaffected(self):
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {"sigma": 0.1}),
            (mirror, 0.5, {"lr_joint_pairs": [(0, 1)]}),
        ], representation="6d")
        assert len(pipeline.augmentations) == 2

    def test_standard_factory_still_constructs(self, bvh_example):
        skel = get_skeleton_info(bvh_example)
        pipeline = AugmentationPipeline.standard(
            skel, representation="6d", up_axis=bvh_example.world_up)
        assert pipeline.augmentations


# =============================================================================
# Fail-loud unpacking, dtype preservation, flat-rotmat layout
# =============================================================================

class TestUnpackRootChannelsValidation:
    """Slicing past the channel axis is silent in NumPy; these raise."""

    @pytest.mark.parametrize("unpack,shape", [
        (unpack_from_ctv, (4, 5, 3)),   # (C, T, V)
        (unpack_from_tvc, (5, 3, 4)),   # (T, V, C)
    ])
    def test_root_channels_beyond_array_raises(self, unpack, shape):
        data = np.zeros(shape)
        with pytest.raises(ValueError, match="root_channels=7"):
            unpack(data, root_channels=7)

    @pytest.mark.parametrize("unpack,shape", [
        (unpack_from_ctv, (4, 5, 3)),
        (unpack_from_tvc, (5, 3, 4)),
    ])
    def test_zero_root_channels_raises(self, unpack, shape):
        with pytest.raises(ValueError, match="root_channels must be >= 1"):
            unpack(np.zeros(shape), root_channels=0)

    def test_valid_root_channels_unaffected(self):
        root_pos = np.random.default_rng(0).normal(size=(5, 3))
        joint_data = np.random.default_rng(1).normal(size=(5, 4, 4))
        ctv = pack_to_ctv(MotionArrays(root_pos=root_pos, joint_rot=joint_data), center_root=False)
        rp, jd = as_pair(unpack_from_ctv(ctv, root_channels=3))
        np.testing.assert_allclose(rp, root_pos)
        np.testing.assert_allclose(jd, joint_data)


class TestStandardizeLengthDtype:
    """pad/crop only select and append frames — they must not upcast."""

    @pytest.mark.parametrize("method", ["pad", "crop"])
    @pytest.mark.parametrize("target", [5, 20])
    def test_float32_preserved(self, method, target):
        data = np.zeros((10, 3), dtype=np.float32)
        out = standardize_length(data, target, method=method)
        assert out.dtype == np.float32
        assert out.shape == (target, 3)

    def test_pad_value_cast_into_input_dtype(self):
        data = np.ones((4, 2), dtype=np.float32)
        out = standardize_length(data, 6, method="pad", pad_value=2.5)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[4:], 2.5)

    def test_resample_linear_returns_float64(self):
        """It computes new values; the interpolation runs in double."""
        data = np.zeros((10, 3), dtype=np.float32)
        out = standardize_length(data, 7, method="resample_linear")
        assert out.dtype == np.float64

    def test_float64_input_unchanged(self):
        data = np.zeros((10, 3), dtype=np.float64)
        for method in ("pad", "crop", "resample_linear"):
            assert standardize_length(data, 7, method=method).dtype == np.float64


class TestRotmatLayoutGuard:
    """rotmat joint data is flat (F, J, 9); the 4-D form must fail loud."""

    @staticmethod
    def _rotmat_arrays(F=6, J=3):
        rng = np.random.default_rng(0)
        quats = rng.normal(size=(F, J, 4))
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        flat = convert_rotations(quats, "quat", "rotmat")
        return rng.normal(size=(F, 3)), flat

    def test_nested_rotmat_raises(self):
        root_pos, flat = self._rotmat_arrays()
        F, J = flat.shape[:2]
        nested = flat.reshape(F, J, 3, 3)
        with pytest.raises(ValueError, match=r"\(F, J, 9\)"):
            rotate_vertical(MotionArrays(root_pos=root_pos, joint_rot=nested), angle=0.5, up_axis="+y", representation="rotmat")

    def test_nested_rotmat_raises_in_staged_pipeline(self):
        root_pos, flat = self._rotmat_arrays()
        F, J = flat.shape[:2]
        nested = flat.reshape(F, J, 3, 3)
        pipeline = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.1})],
            representation="rotmat")
        with pytest.raises(ValueError, match=r"\(F, J, 9\)"):
            pipeline(MotionArrays(root_pos=root_pos, joint_rot=nested), rng=np.random.default_rng(0))

    def test_flat_rotmat_still_works(self):
        root_pos, flat = self._rotmat_arrays()
        out_pos, out_jd = as_pair(rotate_vertical(MotionArrays(root_pos=root_pos, joint_rot=flat), angle=0.5, up_axis="+y", representation="rotmat"))
        assert out_jd.shape == flat.shape


class TestMotionArrays:
    """The container itself: construction, validation, frozen semantics."""

    @staticmethod
    def _arrays(F=10, J=4, C=6):
        rng = np.random.default_rng(0)
        return rng.normal(size=(F, 3)), rng.normal(size=(F, J, C))

    def test_fields_round_trip(self):
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr)
        np.testing.assert_array_equal(a.root_pos, rp)
        np.testing.assert_array_equal(a.joint_rot, jr)
        assert a.frame_count == 10

    def test_joint_rot_is_optional(self):
        rp, _ = self._arrays()
        assert MotionArrays(root_pos=rp).joint_rot is None

    def test_construction_is_keyword_only(self):
        rp, jr = self._arrays()
        with pytest.raises(TypeError):
            MotionArrays(rp, jr)

    def test_frame_count_mismatch_raises(self):
        rp, jr = self._arrays()
        with pytest.raises(ValueError, match="disagree on frame count"):
            MotionArrays(root_pos=rp[:5], joint_rot=jr)

    def test_bad_root_shape_raises(self):
        _, jr = self._arrays()
        with pytest.raises(ValueError, match=r"root_pos must have shape \(F, 3\)"):
            MotionArrays(root_pos=np.zeros((10, 4)), joint_rot=jr)

    def test_nested_rotmat_rejected_with_the_flat_layout_named(self):
        rp, _ = self._arrays()
        with pytest.raises(ValueError, match=r"\(F, J, 9\)"):
            MotionArrays(root_pos=rp, joint_rot=np.zeros((10, 4, 3, 3)))

    def test_is_frozen(self):
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr)
        with pytest.raises(AttributeError, match="frozen"):
            a.root_pos = rp[:5]
        with pytest.raises(AttributeError, match="frozen"):
            del a.joint_rot

    def test_replace_revalidates(self):
        """Frozen fields plus a revalidating replace() is what lets the rest
        of the package assume the invariant holds — not just at birth."""
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr)
        with pytest.raises(ValueError, match="disagree on frame count"):
            a.replace(root_pos=rp[:5])

    def test_replace_keeps_the_untouched_field(self):
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr)
        b = a.replace(root_pos=rp * 2.0)
        np.testing.assert_array_equal(b.joint_rot, jr)
        np.testing.assert_array_equal(b.root_pos, rp * 2.0)

    def test_repr_shows_shapes_not_values(self):
        rp, jr = self._arrays()
        r = repr(MotionArrays(root_pos=rp, joint_rot=jr))
        assert "(10, 3)" in r and "(10, 4, 6)" in r

    def test_equality_compares_arrays(self):
        rp, jr = self._arrays()
        assert (MotionArrays(root_pos=rp, joint_rot=jr)
                == MotionArrays(root_pos=rp.copy(), joint_rot=jr.copy()))
        assert (MotionArrays(root_pos=rp, joint_rot=jr)
                != MotionArrays(root_pos=rp))

    def test_from_bvh_matches_extract_repr(self, bvh_example):
        from pybvh_ml.preprocessing import extract_repr
        rp, jd = extract_repr(bvh_example, "6d")
        a = MotionArrays.from_bvh(bvh_example, "6d")
        np.testing.assert_array_equal(a.root_pos, rp)
        np.testing.assert_array_equal(a.joint_rot, jd)

    def test_from_bvh_center_root(self, bvh_example):
        a = MotionArrays.from_bvh(bvh_example, "6d", center_root=True)
        np.testing.assert_allclose(a.root_pos[0], np.zeros(3), atol=0)

    def test_missing_joint_rot_names_the_caller(self):
        rp, _ = self._arrays()
        with pytest.raises(ValueError, match="rotate_vertical needs joint rotations"):
            rotate_vertical(MotionArrays(root_pos=rp), angle=0.1,
                            up_axis="+y", representation="6d")

    def test_float32_stays_float32(self):
        """A per-sample container that silently doubled a cached clip's
        memory would make float32 storage pointless."""
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp.astype(np.float32),
                         joint_rot=jr.astype(np.float32))
        assert a.root_pos.dtype == np.float32
        assert a.joint_rot.dtype == np.float32

    def test_streams_keep_their_own_dtypes(self):
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr.astype(np.float32))
        assert a.root_pos.dtype == np.float64
        assert a.joint_rot.dtype == np.float32

    def test_non_floating_is_promoted_to_float64(self):
        """Integer rotations are never the intent; promote rather than
        carry a dtype the rotation math cannot use."""
        a = MotionArrays(root_pos=np.zeros((4, 3), dtype=np.int32),
                         joint_rot=[[[1, 0, 0, 0]]] * 4)
        assert a.root_pos.dtype == np.float64
        assert a.joint_rot.dtype == np.float64

    def test_fields_are_read_only(self):
        """Frozen has to cover the buffers too, or a container built over a
        Dataset's cache could rewrite the cache through the field."""
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr)
        with pytest.raises(ValueError, match="read-only"):
            a.root_pos[0, 0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            a.joint_rot[0, 0, 0] = 1.0

    def test_fields_are_views_and_leave_the_source_writable(self):
        """Read-only views, not copies: the container costs no allocation,
        and `setflags` on the view must not reach the caller's array."""
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr)
        assert np.shares_memory(a.root_pos, rp)
        assert np.shares_memory(a.joint_rot, jr)
        rp[0, 0] = 42.0            # the source stayed writable...
        assert a.root_pos[0, 0] == 42.0   # ...and the container sees the write

    def test_replace_keeps_the_carried_field_read_only(self):
        rp, jr = self._arrays()
        b = MotionArrays(root_pos=rp, joint_rot=jr).replace(root_pos=rp * 2.0)
        with pytest.raises(ValueError, match="read-only"):
            b.joint_rot[0, 0, 0] = 1.0

    def test_deepcopy_detaches_the_storage(self):
        """The one thing read-only views cannot express, and the natural
        reach for it — so the frozen guard must not break `copy`."""
        import copy
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp, joint_rot=jr)
        b = copy.deepcopy(a)
        assert b == a
        assert not np.shares_memory(b.root_pos, rp)
        assert not np.shares_memory(b.joint_rot, jr)

    def test_pickles_through_the_constructor(self):
        """A DataLoader boundary pickles whatever crosses it; the frozen
        `__setattr__` would otherwise fire on unpickling."""
        import pickle
        rp, jr = self._arrays()
        a = MotionArrays(root_pos=rp.astype(np.float32), joint_rot=jr)
        b = pickle.loads(pickle.dumps(a))
        assert b == a
        assert b.root_pos.dtype == np.float32          # dtype survives
        assert not b.root_pos.flags.writeable          # so does read-only
        assert pickle.loads(pickle.dumps(
            MotionArrays(root_pos=rp))).joint_rot is None

    def test_unpacking_names_the_migration(self):
        """`rp, jd = pipeline(...)` was every downstream's shape; a stock
        'cannot unpack non-iterable' would not say what to read instead."""
        rp, jr = self._arrays()
        with pytest.raises(TypeError, match="not iterable"):
            _rp, _jr = MotionArrays(root_pos=rp, joint_rot=jr)
        with pytest.raises(TypeError, match=r"out\.root_pos"):
            list(MotionArrays(root_pos=rp, joint_rot=jr))


class TestNoiseSplit:
    """add_joint_noise became two functions, split by unit."""

    def test_rotation_noise_leaves_root_untouched(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        out = add_joint_rotation_noise(
            MotionArrays(root_pos=pos, joint_rot=quats), sigma=0.1,
            representation="quat", rng=np.random.default_rng(0))
        np.testing.assert_array_equal(out.root_pos, pos)

    def test_position_noise_leaves_rotations_untouched(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        out = add_root_position_noise(
            MotionArrays(root_pos=pos, joint_rot=quats), sigma=0.5,
            rng=np.random.default_rng(0))
        np.testing.assert_array_equal(out.joint_rot, quats)

    def test_position_noise_needs_no_representation(self, bvh_example):
        """Its sigma is a length, so it has no rotation-format opinion —
        and it works on a container carrying no rotations at all."""
        pos, _ = _get_quat_data(bvh_example)
        out = add_root_position_noise(
            MotionArrays(root_pos=pos), sigma=0.5,
            rng=np.random.default_rng(0))
        assert out.joint_rot is None
        assert not np.allclose(out.root_pos, pos)

    def test_zero_position_sigma_draws_nothing(self, bvh_example):
        """Matches pybvh's add_position_noise: 0 consumes no randomness."""
        pos, quats = _get_quat_data(bvh_example)
        rng = np.random.default_rng(3)
        add_root_position_noise(
            MotionArrays(root_pos=pos, joint_rot=quats), sigma=0.0, rng=rng)
        after_noop = rng.random()
        rng2 = np.random.default_rng(3)
        assert after_noop == rng2.random()

    def test_zero_rotation_sigma_still_draws(self, bvh_example):
        """Ours does consume randomness at sigma=0, so a seeded pipeline's
        stream does not depend on the sigma it was configured with."""
        pos, quats = _get_quat_data(bvh_example)
        rng = np.random.default_rng(3)
        add_joint_rotation_noise(
            MotionArrays(root_pos=pos, joint_rot=quats), sigma=0.0,
            representation="quat", rng=rng)
        rng2 = np.random.default_rng(3)
        assert rng.random() != rng2.random()

    def test_chaining_reproduces_the_fused_call(self, bvh_example):
        """The documented migration: one generator, rotation first."""
        pos, quats = _get_quat_data(bvh_example)
        rng = np.random.default_rng(11)
        a = MotionArrays(root_pos=pos, joint_rot=quats)
        a = add_joint_rotation_noise(a, sigma=0.05, representation="quat",
                                     rng=rng)
        a = add_root_position_noise(a, sigma=0.3, rng=rng)
        assert not np.allclose(a.root_pos, pos)
        assert not np.allclose(a.joint_rot, quats)

    def test_staged_position_noise_never_materializes_quats(self, bvh_example):
        """The split's performance payoff: positional jitter no longer
        forces a quaternion conversion it does not need."""
        from pybvh_ml._staged import (_StagingState,
                                      _add_root_position_noise_staged)
        pos, rot6d = bvh_example.to_6d()
        state = _StagingState(rot6d, "6d", None)
        _add_root_position_noise_staged(
            pos, state, sigma=0.1, rng=np.random.default_rng(0))
        assert state.quats is None

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_both_paths_agree_on_position_noise(self, bvh_example, cache_quats):
        pos, rot6d = bvh_example.to_6d()
        pipe = AugmentationPipeline(
            [(add_root_position_noise, 1.0, {"sigma": 0.4})],
            representation="6d", cache_quats=cache_quats)
        out = pipe(MotionArrays(root_pos=pos, joint_rot=rot6d),
                   rng=np.random.default_rng(5))
        np.testing.assert_allclose(out.joint_rot, rot6d, rtol=0, atol=0)


class TestDegreesFlag:
    """`degrees=` on the three angle-taking surfaces."""

    def test_rotate_vertical_degrees_matches_radians(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        a = MotionArrays(root_pos=pos, joint_rot=quats)
        kw = dict(up_axis="+y", representation="quat")
        deg = rotate_vertical(a, angle=45.0, degrees=True, **kw)
        rad = rotate_vertical(a, angle=np.radians(45.0), **kw)
        np.testing.assert_allclose(deg.root_pos, rad.root_pos, rtol=0, atol=0)
        np.testing.assert_allclose(deg.joint_rot, rad.joint_rot, rtol=0, atol=0)

    def test_rotation_noise_degrees_matches_radians(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        a = MotionArrays(root_pos=pos, joint_rot=quats)
        deg = add_joint_rotation_noise(
            a, sigma=2.0, degrees=True, representation="quat",
            rng=np.random.default_rng(4))
        rad = add_joint_rotation_noise(
            a, sigma=np.radians(2.0), representation="quat",
            rng=np.random.default_rng(4))
        np.testing.assert_allclose(deg.joint_rot, rad.joint_rot, rtol=0, atol=0)

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_standard_degrees_matches_radians(self, bvh_example, cache_quats):
        skel = get_skeleton_info(bvh_example)
        pos, rot6d = bvh_example.to_6d()
        a = MotionArrays(root_pos=pos, joint_rot=rot6d)
        deg = AugmentationPipeline.standard(
            skel, representation="6d", cache_quats=cache_quats,
            rotate_angle_range=(-180.0, 180.0), noise_sigma=1.0, degrees=True)
        rad = AugmentationPipeline.standard(
            skel, representation="6d", cache_quats=cache_quats,
            rotate_angle_range=(-np.pi, np.pi), noise_sigma=np.radians(1.0))
        # Same seed, and degrees only rescales the drawn value, so the two
        # pipelines must agree bit-for-bit.
        out_d = deg(a, rng=np.random.default_rng(9))
        out_r = rad(a, rng=np.random.default_rng(9))
        np.testing.assert_allclose(out_d.root_pos, out_r.root_pos,
                                   rtol=1e-12, atol=1e-12)

    def test_radians_remain_the_default(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        a = MotionArrays(root_pos=pos, joint_rot=quats)
        small = rotate_vertical(a, angle=1.0, up_axis="+y",
                                representation="quat")
        big = rotate_vertical(a, angle=1.0, degrees=True, up_axis="+y",
                              representation="quat")
        assert not np.allclose(small.root_pos, big.root_pos)


class TestLegacyStepContract:
    """Steps and calls written against the pre-0.5.0 signature fail loudly."""

    def test_pipeline_rejects_the_old_keyword_form(self, bvh_example):
        """Python's own binding error names the offending kwarg here, which
        is why `__call__` grows no `**legacy` catch-all just to reword it."""
        pos, quats = _get_quat_data(bvh_example)
        pipe = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.1})],
            representation="quat")
        with pytest.raises(TypeError, match="root_pos"):
            pipe(root_pos=pos, joint_data=quats)

    def test_pipeline_rejects_a_bare_tuple(self, bvh_example):
        """Passing the old pair positionally *is* worth a migration message:
        nothing in Python's error would say what to build instead."""
        pos, quats = _get_quat_data(bvh_example)
        pipe = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.1})],
            representation="quat")
        with pytest.raises(TypeError, match="takes a MotionArrays"):
            pipe((pos, quats))

    def test_legacy_step_signature_names_the_migration(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)

        def legacy(*, root_pos, joint_data):
            return root_pos, joint_data

        pipe = AugmentationPipeline([(legacy, 1.0, {})],
                                    representation="quat")
        with pytest.raises(TypeError, match="pre-0.5.0 signature"):
            pipe(MotionArrays(root_pos=pos, joint_rot=quats),
                 rng=np.random.default_rng(0))

    def test_step_returning_a_tuple_names_the_migration(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)

        def tuple_step(arrays):
            return arrays.root_pos, arrays.joint_rot

        pipe = AugmentationPipeline([(tuple_step, 1.0, {})],
                                    representation="quat")
        with pytest.raises(TypeError, match="expected MotionArrays"):
            pipe(MotionArrays(root_pos=pos, joint_rot=quats),
                 rng=np.random.default_rng(0))

    def test_unpacking_the_pipeline_result_names_the_migration(self, bvh_example):
        """`rp, jd = pipeline(...)` is the most common downstream shape, so it
        earns the same migration message as the other two halves — not a bare
        'cannot unpack non-iterable MotionArrays object'."""
        pos, quats = _get_quat_data(bvh_example)
        pipe = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.1})],
            representation="quat")
        with pytest.raises(TypeError, match="not iterable"):
            _rp, _jd = pipe(MotionArrays(root_pos=pos, joint_rot=quats),
                            rng=np.random.default_rng(0))

    def test_uninspectable_step_does_not_break_kwarg_filling(self, bvh_example):
        """`inspect.signature` raises on some callables; the pipeline must
        degrade to 'pass exactly the configured kwargs', not propagate it.

        The step declares an unreadable signature explicitly rather than
        borrowing a builtin, because which builtins are inspectable moved
        between Python versions — `print` raises on 3.9 and does not on
        3.12, so a test written against it passes on half the CI matrix.
        """
        import inspect

        seen = {}

        class Unsignable:
            def __call__(self, arrays, **kwargs):
                seen["kwargs"] = kwargs
                return arrays

            @property
            def __signature__(self):
                raise ValueError("signature unavailable")

        step = Unsignable()
        with pytest.raises(ValueError):
            inspect.signature(step)      # the premise of the test

        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline(
            [(step, 1.0, {"scale": 2.0})], representation="quat")
        pipeline(MotionArrays(root_pos=pos, joint_rot=quats),
                 rng=np.random.default_rng(0))
        # Neither representation, euler_orders nor rng was injected: with no
        # readable signature the pipeline passes exactly what was configured.
        assert seen["kwargs"] == {"scale": 2.0}
