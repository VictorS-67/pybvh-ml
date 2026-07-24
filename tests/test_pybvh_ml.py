"""
Tests for pybvh-ml library.

Run with: pytest tests/test_pybvh_ml.py -v
"""

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
        packed = pack_to_ctv(root_pos, joint_data, center_root=False)
        C = max(3, C_joint)
        assert packed.shape == (C, F, 1 + J)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_tvc_shape(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_tvc(root_pos, joint_data, center_root=False)
        C = max(3, C_joint)
        assert packed.shape == (F, 1 + J, C)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_flat_shape(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_flat(root_pos, joint_data, center_root=False)
        assert packed.shape == (F, 3 + J * C_joint)

    # --- Round-trip tests ---

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_ctv_roundtrip(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_ctv(root_pos, joint_data, center_root=False)
        rp_out, jd_out = unpack_from_ctv(packed)
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)
        np.testing.assert_allclose(jd_out, joint_data, atol=1e-12)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_tvc_roundtrip(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_tvc(root_pos, joint_data, center_root=False)
        rp_out, jd_out = unpack_from_tvc(packed)
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)
        np.testing.assert_allclose(jd_out[:, :, :C_joint], joint_data, atol=1e-12)

    @pytest.mark.parametrize("C_joint", [3, 4, 6])
    def test_flat_roundtrip(self, rng, C_joint):
        F, J = 50, 24
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, C_joint))
        packed = pack_to_flat(root_pos, joint_data, center_root=False)
        rp_out, jd_out = unpack_from_flat(
            packed, root_channels=3, joint_channels=C_joint)
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)
        np.testing.assert_allclose(jd_out, joint_data, atol=1e-12)

    # --- center_root tests ---

    def test_center_root_subtracts_first_frame(self, rng):
        F, J = 30, 10
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, 3))
        packed = pack_to_ctv(root_pos, joint_data, center_root=True)
        rp_out, _ = unpack_from_ctv(packed)
        # First frame root should be zero
        np.testing.assert_allclose(rp_out[0], 0.0, atol=1e-12)
        # Subsequent frames should be relative
        expected = root_pos - root_pos[0:1]
        np.testing.assert_allclose(rp_out, expected, atol=1e-12)

    def test_center_root_false_preserves_values(self, rng):
        F, J = 30, 10
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, 3))
        packed = pack_to_ctv(root_pos, joint_data, center_root=False)
        rp_out, _ = unpack_from_ctv(packed)
        np.testing.assert_allclose(rp_out, root_pos, atol=1e-12)

    # --- Zero-padding tests ---

    def test_ctv_root_zero_padded_for_6d(self, rng):
        """When C_joint=6, root occupies channels 0:3, channels 3:6 are zero."""
        F, J = 20, 10
        root_pos = rng.standard_normal((F, 3))
        joint_data = rng.standard_normal((F, J, 6))
        packed = pack_to_ctv(root_pos, joint_data, center_root=False)
        # packed shape: (6, 20, 11). Root is vertex 0.
        root_vertex = packed[:, :, 0]  # (6, 20)
        # Channels 0:3 should have root data
        np.testing.assert_allclose(root_vertex[:3, :], root_pos.T, atol=1e-12)
        # Channels 3:6 should be zero (padding)
        np.testing.assert_allclose(root_vertex[3:, :], 0.0, atol=1e-12)

    # --- Integration with pybvh ---

    def test_pack_from_bvh_euler(self, bvh_example):
        """Pack actual BVH data in Euler representation."""
        packed = pack_to_ctv(
            bvh_example.root_pos, bvh_example.joint_angles, center_root=True)
        F = bvh_example.frame_count
        J = bvh_example.joint_count
        assert packed.shape == (3, F, 1 + J)

    def test_pack_from_bvh_6d(self, bvh_example):
        """Pack actual BVH data in 6D representation."""
        root_pos, rot6d = bvh_example.to_6d()
        packed = pack_to_ctv(root_pos, rot6d, center_root=True)
        F = bvh_example.frame_count
        J = bvh_example.joint_count
        assert packed.shape == (6, F, 1 + J)

    def test_pack_from_bvh_quaternion(self, bvh_example):
        """Pack actual BVH data in quaternion representation."""
        root_pos, quats = bvh_example.to_quat()
        packed = pack_to_ctv(root_pos, quats, center_root=True)
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
            'lr_pairs', 'lr_mapping'}

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
    add_joint_noise,
)
from pybvh_ml.sequences import uniform_temporal_sample, sample_temporal
from pybvh_ml.convert import convert_arrays
from pybvh_ml.pipeline import AugmentationPipeline


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
        new_pos, new_quats = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(45.0), up_axis="+y", representation="quat")
        assert new_quats.shape == quats.shape
        assert new_pos.shape == pos.shape

    def test_rotate_quat_zero_is_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = rotate_vertical(root_pos=pos, joint_data=quats, angle=0.0, up_axis="+y", representation="quat")
        np.testing.assert_allclose(new_quats, quats, atol=1e-10)
        np.testing.assert_allclose(new_pos, pos, atol=1e-10)

    def test_rotate_quat_360_is_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(360.0), up_axis="+y", representation="quat")
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
        _, new_quats = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="+y", representation="quat")
        np.testing.assert_allclose(new_quats[:, 1:], quats[:, 1:], atol=1e-10)

    def test_rotate_quat_root_pos_rotated(self, bvh_example):
        """Root position should be transformed by the rotation matrix."""
        pos, quats = _get_quat_data(bvh_example)
        new_pos, new_quats = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="+y", representation="quat")
        # 90° around Y: (x, y, z) → (z, y, -x)
        np.testing.assert_allclose(new_pos[:, 0], pos[:, 2], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 1], pos[:, 1], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 2], -pos[:, 0], atol=1e-10)

    def test_rotate_quat_negative_axis_flips_direction(self, bvh_example):
        """'+y' and '-y' of the same angle should rotate in opposite directions."""
        pos, quats = _get_quat_data(bvh_example)
        pos_plus, _ = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="+y", representation="quat")
        pos_minus, _ = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="-y", representation="quat")
        # Same magnitude, opposite sign on the non-up components.
        np.testing.assert_allclose(pos_plus[:, 1], pos_minus[:, 1], atol=1e-10)
        np.testing.assert_allclose(pos_plus[:, 0], -pos_minus[:, 0], atol=1e-10)
        np.testing.assert_allclose(pos_plus[:, 2], -pos_minus[:, 2], atol=1e-10)

    def test_rotate_quat_bad_axis_raises(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="axis must be"):
            rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="y", representation="quat")
        with pytest.raises(ValueError, match="axis must be"):
            rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="+w", representation="quat")

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
        new_pos, new_quats = rotate_vertical(root_pos=pos, joint_data=quats, angle=angle, up_axis=up_axis, representation="quat")
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
        new_pos, new_quats = mirror(root_pos=pos, joint_data=quats, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat")
        assert new_quats.shape == quats.shape
        assert new_pos.shape == pos.shape

    def test_mirror_quat_lateral_negated(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        new_pos, _ = mirror(root_pos=pos, joint_data=quats, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat")
        lat_idx = "xyz".index(lateral_axis[1])
        np.testing.assert_allclose(
            new_pos[:, lat_idx], -pos[:, lat_idx], atol=1e-10)

    def test_mirror_quat_double_is_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        p1, q1 = mirror(root_pos=pos, joint_data=quats, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat")
        p2, q2 = mirror(root_pos=p1, joint_data=q1, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat")
        np.testing.assert_allclose(p2, pos, atol=1e-10)
        np.testing.assert_allclose(q2, quats, atol=1e-10)

    def test_mirror_quat_sign_invariant(self, bvh_example):
        """'+x' and '-x' should produce identical mirror (sign-invariant)."""
        pos, quats = _get_quat_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        flipped = ("-" if lateral_axis[0] == "+" else "+") + lateral_axis[1]
        p1, q1 = mirror(root_pos=pos, joint_data=quats, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat")
        p2, q2 = mirror(root_pos=pos, joint_data=quats, lr_joint_pairs=pairs, lateral_axis=flipped, representation="quat")
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
        quat_pos_m, quat_m = mirror(root_pos=pos, joint_data=quats, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat")
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
        new_p, new_q = speed_perturbation_arrays(root_pos=pos, joint_data=quats, factor=2.0, representation="quat")
        assert new_p.shape[0] == max(2, round(F / 2.0))
        assert new_q.shape[0] == new_p.shape[0]

    def test_speed_factor_one(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = speed_perturbation_arrays(root_pos=pos, joint_data=quats, factor=1.0, representation="quat")
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
        new_p, new_q = speed_perturbation_arrays(root_pos=pos, joint_data=quats, factor=1.5, representation="quat")
        np.testing.assert_allclose(new_p[0], pos[0], atol=1e-10)
        np.testing.assert_allclose(new_p[-1], pos[-1], atol=1e-10)

    def test_speed_invalid_factor(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="factor must be > 0"):
            speed_perturbation_arrays(root_pos=pos, joint_data=quats, factor=0.0, representation="quat")

    # --- dropout_arrays ---

    def test_dropout_shape(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = dropout_arrays(root_pos=pos, joint_data=quats, drop_rate=0.3, representation="quat", rng=np.random.default_rng(42))
        assert new_q.shape == quats.shape
        assert new_p.shape == pos.shape

    def test_dropout_first_last_kept(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = dropout_arrays(root_pos=pos, joint_data=quats, drop_rate=0.5, representation="quat", rng=np.random.default_rng(42))
        np.testing.assert_allclose(new_p[0], pos[0], atol=1e-10)
        np.testing.assert_allclose(new_p[-1], pos[-1], atol=1e-10)
        np.testing.assert_allclose(new_q[0], quats[0], atol=1e-10)
        np.testing.assert_allclose(new_q[-1], quats[-1], atol=1e-10)

    def test_dropout_zero_rate(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = dropout_arrays(root_pos=pos, joint_data=quats, drop_rate=0.0, representation="quat")
        np.testing.assert_allclose(new_q, quats, atol=1e-10)
        np.testing.assert_allclose(new_p, pos, atol=1e-10)

    def test_dropout_reproducible(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        p1, q1 = dropout_arrays(root_pos=pos, joint_data=quats, drop_rate=0.3, representation="quat", rng=np.random.default_rng(99))
        p2, q2 = dropout_arrays(root_pos=pos, joint_data=quats, drop_rate=0.3, representation="quat", rng=np.random.default_rng(99))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)


# =============================================================================
# 6D augmentation
# =============================================================================

class TestRot6dAugmentation:
    """Tests for 6D-space augmentation functions."""

    def test_rotate_6d_shape(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        new_pos, new_6d = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=np.radians(45.0), up_axis="+y", representation="6d")
        assert new_6d.shape == rot6d.shape
        assert new_pos.shape == pos.shape

    def test_rotate_6d_zero_identity(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        new_pos, new_6d = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=0.0, up_axis="+y", representation="6d")
        np.testing.assert_allclose(new_6d, rot6d, atol=1e-10)
        np.testing.assert_allclose(new_pos, pos, atol=1e-10)

    def test_rotate_6d_nonroot_unchanged(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        _, new_6d = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=np.radians(90.0), up_axis="+y", representation="6d")
        np.testing.assert_allclose(new_6d[:, 1:], rot6d[:, 1:], atol=1e-10)

    def test_rotate_6d_root_pos_rotated(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        new_pos, _ = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=np.radians(90.0), up_axis="+y", representation="6d")
        np.testing.assert_allclose(new_pos[:, 0], pos[:, 2], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 1], pos[:, 1], atol=1e-10)
        np.testing.assert_allclose(new_pos[:, 2], -pos[:, 0], atol=1e-10)

    def test_rotate_6d_negative_axis_flips_direction(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pos_plus, _ = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=np.radians(90.0), up_axis="+y", representation="6d")
        pos_minus, _ = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=np.radians(90.0), up_axis="-y", representation="6d")
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
        new_pos_q, new_quats = rotate_vertical(root_pos=pos, joint_data=quats, angle=angle, up_axis=up_axis, representation="quat")
        # 6D rotation
        new_pos_6d, new_6d = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=angle, up_axis=up_axis, representation="6d")
        # Root positions should match
        np.testing.assert_allclose(new_pos_6d, new_pos_q, atol=1e-10)
        # Convert both to rotmat and compare
        R_from_quat = rotations.quat_to_rotmat(new_quats)
        R_from_6d = rotations.rot6d_to_rotmat(new_6d)
        np.testing.assert_allclose(R_from_6d, R_from_quat, atol=1e-6)

    def test_mirror_6d_shape(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        new_pos, new_6d = mirror(root_pos=pos, joint_data=rot6d, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d")
        assert new_6d.shape == rot6d.shape
        assert new_pos.shape == pos.shape

    def test_mirror_6d_lateral_negated(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        new_pos, _ = mirror(root_pos=pos, joint_data=rot6d, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d")
        lat_idx = "xyz".index(lateral_axis[1])
        np.testing.assert_allclose(
            new_pos[:, lat_idx], -pos[:, lat_idx], atol=1e-10)

    def test_mirror_6d_double_is_identity(self, bvh_example):
        pos, rot6d = _get_6d_data(bvh_example)
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        p1, r1 = mirror(root_pos=pos, joint_data=rot6d, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d")
        p2, r2 = mirror(root_pos=p1, joint_data=r1, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d")
        np.testing.assert_allclose(p2, pos, atol=1e-10)
        np.testing.assert_allclose(r2, rot6d, atol=1e-10)

    def test_mirror_6d_consistency_with_quat(self, bvh_example):
        """6D mirror should match quaternion mirror after conversion."""
        from pybvh import rotations
        pairs, lateral_axis, _ = _get_mirror_metadata(bvh_example)
        pos, quats = _get_quat_data(bvh_example)
        _, rot6d = _get_6d_data(bvh_example)
        quat_pos, quat_m = mirror(root_pos=pos, joint_data=quats, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="quat")
        r6d_pos, r6d_m = mirror(root_pos=pos, joint_data=rot6d, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d")
        np.testing.assert_allclose(r6d_pos, quat_pos, atol=1e-10)
        R_from_quat = rotations.quat_to_rotmat(quat_m)
        R_from_6d = rotations.rot6d_to_rotmat(r6d_m)
        np.testing.assert_allclose(R_from_6d, R_from_quat, atol=1e-6)

    def test_rotate_6d_orthogonal(self, bvh_example):
        """Output 6D should decode to valid rotation matrices."""
        from pybvh import rotations
        pos, rot6d = _get_6d_data(bvh_example)
        _, new_6d = rotate_vertical(root_pos=pos, joint_data=rot6d, angle=np.radians(73.0), up_axis="+y", representation="6d")
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
        _, new_6d = mirror(root_pos=pos, joint_data=rot6d, lr_joint_pairs=pairs, lateral_axis=lateral_axis, representation="6d")
        R = rotations.rot6d_to_rotmat(new_6d)
        I = np.eye(3)
        for f in range(R.shape[0]):
            for j in range(R.shape[1]):
                np.testing.assert_allclose(
                    R[f, j] @ R[f, j].T, I, atol=1e-10)


# =============================================================================
# Representation conversion
# =============================================================================

class TestConvertArrays:
    """Tests for representation conversion."""

    def test_identity(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        result = convert_arrays(quats, "quat", "quat")
        np.testing.assert_allclose(result, quats, atol=1e-12)

    def test_euler_to_quat_shape(self, bvh_example):
        result = convert_arrays(
            bvh_example.joint_angles, "euler", "quat",
            euler_orders=bvh_example.euler_orders)
        assert result.shape == (bvh_example.frame_count, bvh_example.joint_count, 4)

    def test_euler_to_6d_shape(self, bvh_example):
        result = convert_arrays(
            bvh_example.joint_angles, "euler", "6d",
            euler_orders=bvh_example.euler_orders)
        assert result.shape == (bvh_example.frame_count, bvh_example.joint_count, 6)

    def test_roundtrip_euler_quat(self, bvh_example):
        orders = bvh_example.euler_orders
        q = convert_arrays(bvh_example.joint_angles, "euler", "quat",
                           euler_orders=orders)
        back = convert_arrays(q, "quat", "euler", euler_orders=orders)
        np.testing.assert_allclose(back, bvh_example.joint_angles, atol=1e-4)

    def test_roundtrip_euler_6d(self, bvh_example):
        orders = bvh_example.euler_orders
        r6d = convert_arrays(bvh_example.joint_angles, "euler", "6d",
                             euler_orders=orders)
        back = convert_arrays(r6d, "6d", "euler", euler_orders=orders)
        np.testing.assert_allclose(back, bvh_example.joint_angles, atol=1e-4)

    def test_roundtrip_quat_6d(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        r6d = convert_arrays(quats, "quat", "6d")
        back = convert_arrays(r6d, "6d", "quat")
        # q and -q represent same rotation
        for f in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                match = (np.allclose(back[f, j], quats[f, j], atol=1e-6)
                         or np.allclose(back[f, j], -quats[f, j], atol=1e-6))
                assert match

    def test_roundtrip_quat_axisangle(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        aa = convert_arrays(quats, "quat", "axisangle")
        back = convert_arrays(aa, "axisangle", "quat")
        for f in range(quats.shape[0]):
            for j in range(quats.shape[1]):
                match = (np.allclose(back[f, j], quats[f, j], atol=1e-6)
                         or np.allclose(back[f, j], -quats[f, j], atol=1e-6))
                assert match

    def test_roundtrip_6d_rotmat(self, bvh_example):
        _, rot6d = _get_6d_data(bvh_example)
        rm = convert_arrays(rot6d, "6d", "rotmat")
        assert rm.shape[-1] == 9
        back = convert_arrays(rm, "rotmat", "6d")
        np.testing.assert_allclose(back, rot6d, atol=1e-6)

    def test_rotmat_flat_shape(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        rm = convert_arrays(quats, "quat", "rotmat")
        F, J = quats.shape[:2]
        assert rm.shape == (F, J, 9)

    def test_euler_orders_required(self, bvh_example):
        with pytest.raises(ValueError, match="euler_orders is required"):
            convert_arrays(bvh_example.joint_angles, "euler", "quat")

    def test_euler_orders_not_required_for_non_euler(self, bvh_example):
        _, quats = _get_quat_data(bvh_example)
        # Should not raise
        convert_arrays(quats, "quat", "6d")

    def test_unknown_repr(self):
        data = np.zeros((10, 5, 3))
        with pytest.raises(ValueError, match="Unknown"):
            convert_arrays(data, "invalid", "quat")

    def test_per_joint_mixed_orders(self, bvh_test3):
        """bvh_test3 has mixed Euler orders."""
        orders = bvh_test3.euler_orders
        assert len(set(orders)) >= 1  # may have mixed orders
        q = convert_arrays(bvh_test3.joint_angles, "euler", "quat",
                           euler_orders=orders)
        back = convert_arrays(q, "quat", "euler", euler_orders=orders)
        np.testing.assert_allclose(back, bvh_test3.joint_angles, atol=1e-4)

    @pytest.mark.parametrize("repr_name,expected_c", [
        ("euler", 3), ("axisangle", 3), ("quat", 4),
        ("6d", 6), ("rotmat", 9),
    ])
    def test_convert_shapes(self, bvh_example, repr_name, expected_c):
        orders = bvh_example.euler_orders
        q = convert_arrays(bvh_example.joint_angles, "euler", repr_name,
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
    def test_convert_arrays_matches_to_quat(self, request, fixture):
        bvh = request.getfixturevalue(fixture)
        _, quats_gt = bvh.to_quat()
        result = convert_arrays(bvh.joint_angles, "euler", "quat",
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
        pos_e, jd_e = fn(root_pos=bvh.root_pos, joint_data=bvh.joint_angles,
                         **kw_euler)
        pos_qr, jd_q = fn(root_pos=pos_q, joint_data=quats, **kw_quat)
        R_e = convert_arrays(jd_e, "euler", "rotmat", euler_orders=orders)
        R_q = convert_arrays(jd_q, "quat", "rotmat")
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

    def test_add_joint_noise_euler_matches_quat(self, bvh_example):
        (pos_e, R_e), (pos_q, R_q) = self._euler_vs_quat(
            bvh_example, add_joint_noise,
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

    def test_staged_pipeline_euler_matches_quat(self, bvh_example):
        """The quat-caching pipeline path uses the same radians convention."""
        orders = bvh_example.euler_orders
        pos_q, quats = bvh_example.to_quat()

        def build(representation, **extra):
            steps = [
                (rotate_vertical, 1.0,
                 dict({"angle": np.radians(30.0), "up_axis": "+y",
                       "representation": representation}, **extra)),
                (add_joint_noise, 1.0,
                 dict({"sigma": np.radians(2.0),
                       "representation": representation}, **extra)),
            ]
            return AugmentationPipeline(steps, cache_quats=True)

        pos_e, jd_e = build("euler", euler_orders=orders)(
            root_pos=bvh_example.root_pos,
            joint_data=bvh_example.joint_angles,
            rng=np.random.default_rng(42))
        pos_qr, jd_q = build("quat")(
            root_pos=pos_q, joint_data=quats,
            rng=np.random.default_rng(42))
        R_e = convert_arrays(jd_e, "euler", "rotmat", euler_orders=orders)
        R_q = convert_arrays(jd_q, "quat", "rotmat")
        np.testing.assert_allclose(pos_e, pos_qr, atol=1e-10)
        np.testing.assert_allclose(R_e, R_q, atol=1e-6)


# =============================================================================
# Augmentation pipeline
# =============================================================================

class TestAugmentationPipeline:
    """Tests for AugmentationPipeline."""

    def test_empty_pipeline(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([])
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats)
        np.testing.assert_array_equal(new_q, quats)
        np.testing.assert_array_equal(new_p, pos)

    def test_prob_zero_skips(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 0.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(new_q, quats)
        np.testing.assert_array_equal(new_p, pos)

    def test_prob_one_applies(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
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
        p1, q1 = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(123))
        p2, q2 = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(123))
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
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
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

    def test_default_rng(self, bvh_example):
        """Pipeline should work without explicit rng."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(45), "up_axis": "+y", "representation": "quat"}),
        ])
        # Should not raise
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats)

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
            (add_joint_noise, 0.0,
                {"sigma": np.radians(2.0), "representation": "quat"}),
        ], cache_quats=cache_quats)
        new_p, new_q = pipeline(
            root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
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

        def _shift_root(*, root_pos, joint_data):
            return root_pos + 1.0, joint_data.copy()

        pipeline = AugmentationPipeline([(_shift_root, 1.0, {})])
        with pytest.raises(ValueError, match="representation"):
            pipeline(root_pos=pos, joint_data=quats,
                     rng=np.random.default_rng(0))

        # cache_quats=False has no cache to manage — no declaration needed.
        direct = AugmentationPipeline(
            [(_shift_root, 1.0, {})], cache_quats=False)
        new_p, _ = direct(root_pos=pos, joint_data=quats,
                          rng=np.random.default_rng(0))
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

    def test_add_joint_noise_refuses_positional(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(TypeError):
            add_joint_noise(
                pos, quats, sigma=np.radians(1.0), representation="quat")


class TestAugmentationParamValidation:
    """Out-of-range parameters raise instead of silently no-oping."""

    def test_negative_sigma_raises(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="sigma must be"):
            add_joint_noise(
                root_pos=pos, joint_data=quats, sigma=-0.1,
                representation="quat")

    def test_negative_sigma_pos_raises(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match="sigma_pos must be"):
            add_joint_noise(
                root_pos=pos, joint_data=quats, sigma=0.1, sigma_pos=-0.5,
                representation="quat")

    @pytest.mark.parametrize("drop_rate", [-0.1, 1.0, 1.5])
    def test_drop_rate_out_of_range_raises(self, bvh_example, drop_rate):
        pos, quats = _get_quat_data(bvh_example)
        with pytest.raises(ValueError, match=r"drop_rate must be in \[0, 1\)"):
            dropout_arrays(
                root_pos=pos, joint_data=quats, drop_rate=drop_rate,
                representation="quat")

    def test_negative_sigma_raises_staged(self, bvh_example):
        from pybvh_ml._staged import _StagingState, _add_joint_noise_staged
        pos, quats = _get_quat_data(bvh_example)
        state = _StagingState(quats, "quat", None)
        with pytest.raises(ValueError, match="sigma must be"):
            _add_joint_noise_staged(
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
        new_p, new_q = p(
            root_pos=pos, joint_data=quats,
            rng=np.random.default_rng(0))
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
        assert "joint_data" in loaded["clips"][0]
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
            np.testing.assert_array_equal(cs["joint_data"], cp["joint_data"])

    def test_include_velocities(self, bvh_dir, tmp_path):
        """Velocities array is (F, J, 3) — joint-axis aligned with joint_data,
        not node-axis (pybvh 0.7.0 dropped end sites from joint_velocities)."""
        out = tmp_path / "vel.npz"
        preprocess_directory(
            bvh_dir, out, file_pattern="bvh_test1.bvh",
            include_velocities=True)
        loaded = load_preprocessed(out)
        vel = loaded["clips"][0]["velocities"]
        jd = loaded["clips"][0]["joint_data"]
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
        assert fc.shape[0] == loaded["clips"][0]["joint_data"].shape[0]
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
        """Hierarchy mismatches under harmonize=True must raise loudly with
        the dropped filename and pybvh's drop reason — not silently shrink
        the dataset (the original maintainer-report failure)."""
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
                    work_dir, out, representation="6d", harmonize=True)

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
        a = loaded["clips"][0]["joint_data"]
        b = loaded["clips"][1]["joint_data"]
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
    return pack_to_flat(root_pos, joint_data, center_root=False)


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
        """Uncentered stats agree with preprocess_directory's stored stats on
        rotation channels (root_pos differs intentionally: preprocess centers
        it by default)."""
        bvh_dir = Path(__file__).parent.parent / "bvh_data"
        out = tmp_path / "dataset.npz"
        preprocess_directory(bvh_dir, out, file_pattern="bvh_test1.bvh",
                             representation="6d")
        loaded = load_preprocessed(out)
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        stats = compute_normalization_stats([bvh], representation="6d")
        np.testing.assert_allclose(
            stats["mean"][3:], loaded["mean"][3:], atol=1e-10)
        np.testing.assert_allclose(
            stats["std"][3:], loaded["std"][3:], atol=1e-10)


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

    def test_test_mode_ignores_rng(self):
        """Test mode ignores the provided rng."""
        i1 = uniform_temporal_sample(200, 20, mode="test", rng=np.random.default_rng(999))
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


# =============================================================================
# Joint noise augmentation
# =============================================================================

class TestJointNoise:
    """Tests for add_joint_noise."""

    def test_shape_preserved(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(1.0), representation="quat", rng=np.random.default_rng(42))
        assert new_q.shape == quats.shape
        assert new_p.shape == pos.shape

    def test_zero_noise_is_near_identity(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=0.0, representation="quat", rng=np.random.default_rng(42))
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
        _, new_q = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42))
        norms = np.linalg.norm(new_q, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_noise_changes_values(self, bvh_example):
        """Non-zero sigma should produce different quaternions."""
        pos, quats = _get_quat_data(bvh_example)
        _, new_q = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42))
        assert not np.allclose(new_q, quats, atol=1e-4)

    def test_small_noise_stays_close(self, bvh_example):
        """Small sigma should produce quaternions close to originals."""
        from pybvh import rotations
        pos, quats = _get_quat_data(bvh_example)
        _, new_q = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(0.1), representation="quat", rng=np.random.default_rng(42))
        # Geodesic distance: angle = 2 * arccos(|q1 . q2|)
        dots = np.abs(np.sum(quats * new_q, axis=-1))
        dots = np.clip(dots, 0, 1)
        angles_deg = np.degrees(2 * np.arccos(dots))
        # With sigma=0.1 deg, angles should be very small
        assert np.mean(angles_deg) < 1.0

    def test_root_pos_noise(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, new_q = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(1.0), representation="quat", sigma_pos=0.5,
            rng=np.random.default_rng(42))
        assert not np.allclose(new_p, pos, atol=1e-4)

    def test_no_root_pos_noise_by_default(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        new_p, _ = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42))
        np.testing.assert_array_equal(new_p, pos)

    def test_reproducible(self, bvh_example):
        pos, quats = _get_quat_data(bvh_example)
        p1, q1 = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(2.0), representation="quat", rng=np.random.default_rng(42))
        p2, q2 = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(2.0), representation="quat", rng=np.random.default_rng(42))
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(p1, p2)

    def test_valid_rotations(self, bvh_example):
        """Noisy quaternions should convert to valid rotation matrices."""
        from pybvh import rotations
        pos, quats = _get_quat_data(bvh_example)
        _, new_q = add_joint_noise(
            root_pos=pos, joint_data=quats, sigma=np.radians(5.0), representation="quat", rng=np.random.default_rng(42))
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
            (add_joint_noise, 1.0, {"sigma": np.radians(2.0), "representation": "quat"}),
        ])
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
        assert new_q.shape == quats.shape

    def test_staged_zero_sigma_pos_root_not_aliased(self, bvh_example):
        """Regression: the staged variant used to return the caller's own root_pos when sigma_pos=0, so later in-place edits could mutate the input."""
        from pybvh_ml._staged import _StagingState, _add_joint_noise_staged
        pos, quats = _get_quat_data(bvh_example)
        state = _StagingState(quats, "quat", None)
        new_p = _add_joint_noise_staged(
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
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
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
        p1, _ = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(1))
        p2, _ = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(2))
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
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
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
        p1, q1 = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(99))
        p2, q2 = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(99))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)

    def test_static_kwargs_still_work(self, bvh_example):
        """Existing static kwargs should not be broken."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
        # Should match direct call
        ref_p, ref_q = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="+y", representation="quat")
        np.testing.assert_allclose(new_p, ref_p, atol=1e-12)
        np.testing.assert_allclose(new_q, ref_q, atol=1e-12)


# =============================================================================
# Pipeline rng forwarding (bug fix)
# =============================================================================

class TestPipelineRngForwarding:
    """Tests for automatic rng forwarding to augmentation functions."""

    def test_noise_reproducible_via_pipeline(self, bvh_example):
        """add_joint_noise should be deterministic in pipeline."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (add_joint_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        _, q1 = pipeline(root_pos=pos.copy(), joint_data=quats.copy(), rng=np.random.default_rng(42))
        _, q2 = pipeline(root_pos=pos.copy(), joint_data=quats.copy(), rng=np.random.default_rng(42))
        np.testing.assert_array_equal(q1, q2)

    def test_dropout_reproducible_via_pipeline(self, bvh_example):
        """dropout_arrays should be deterministic in pipeline."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (dropout_arrays, 1.0, {"drop_rate": 0.3, "representation": "quat"}),
        ])
        p1, q1 = pipeline(root_pos=pos.copy(), joint_data=quats.copy(), rng=np.random.default_rng(42))
        p2, q2 = pipeline(root_pos=pos.copy(), joint_data=quats.copy(), rng=np.random.default_rng(42))
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(p1, p2)

    def test_no_rng_functions_unaffected(self, bvh_example):
        """Functions without rng param should still work."""
        pos, quats = _get_quat_data(bvh_example)
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {"angle": np.radians(90), "up_axis": "+y", "representation": "quat"}),
        ])
        # Should not raise TypeError
        new_p, new_q = pipeline(root_pos=pos, joint_data=quats, rng=np.random.default_rng(42))
        ref_p, ref_q = rotate_vertical(root_pos=pos, joint_data=quats, angle=np.radians(90.0), up_axis="+y", representation="quat")
        np.testing.assert_allclose(new_q, ref_q, atol=1e-12)

    def test_explicit_rng_kwarg_takes_precedence(self, bvh_example):
        """User-provided rng kwarg should not be overwritten."""
        pos, quats = _get_quat_data(bvh_example)
        custom_rng = np.random.default_rng(999)
        pipeline = AugmentationPipeline([
            (add_joint_noise, 1.0, {
                "sigma": np.radians(5.0),
                "representation": "quat",
                "rng": lambda rng: custom_rng,  # explicit override
            }),
        ])
        _, q1 = pipeline(root_pos=pos.copy(), joint_data=quats.copy(), rng=np.random.default_rng(42))
        # Should use custom_rng(999), not pipeline's rng(42)
        custom_rng2 = np.random.default_rng(999)
        _, q2 = add_joint_noise(
            root_pos=pos.copy(), joint_data=quats.copy(), sigma=np.radians(5.0), representation="quat", rng=custom_rng2)
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
            (add_joint_noise, 1.0, {"sigma": np.radians(2.0), "representation": "quat"}),
        ])
        p1, q1 = pipeline(root_pos=pos.copy(), joint_data=quats.copy(), rng=np.random.default_rng(42))
        p2, q2 = pipeline(root_pos=pos.copy(), joint_data=quats.copy(), rng=np.random.default_rng(42))
        np.testing.assert_allclose(q1, q2, atol=1e-12)
        np.testing.assert_allclose(p1, p2, atol=1e-12)


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
    """pybvh-ml 0.5 requires pybvh >= 0.8 (pyproject pin: pybvh>=0.8,<0.9)."""

    def test_pybvh_version_floor(self):
        import pybvh
        major, minor = (int(x) for x in pybvh.__version__.split(".")[:2])
        assert (major, minor) >= (0, 8), (
            f"pybvh-ml >= 0.5 requires pybvh >= 0.8.0, "
            f"got {pybvh.__version__}")

    def test_version_matches_pyproject(self):
        # Guards the pyproject/__init__ version drift that shipped in 0.3/0.4.
        import re
        import pybvh_ml
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        match = re.search(
            r'^version = "(.+)"$', pyproject.read_text(), re.MULTILINE)
        assert match is not None
        assert match.group(1) == pybvh_ml.__version__
