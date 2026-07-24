"""PyTorch dataset and collate tests.

Split from ``test_pybvh_ml.py`` so the main suite collects without
torch: this module skips entirely when PyTorch is not installed.

Run with: pytest tests/test_torch_datasets.py -v
"""

import pytest
import numpy as np
from pathlib import Path

torch = pytest.importorskip("torch")

from pybvh_ml.torch import MotionDataset, OnTheFlyDataset, collate_motion_batch
from pybvh_ml.augmentation import rotate_vertical, add_joint_noise
from pybvh_ml.convert import convert_arrays
from pybvh_ml.pipeline import AugmentationPipeline


class TestTorchDatasets:
    """Tests for PyTorch Dataset classes and collate function."""

    @pytest.fixture
    def sample_clips(self, bvh_example):
        """Create sample preprocessed clips."""
        root_pos, rot6d = bvh_example.to_6d()
        # Create 3 clips with different lengths
        clips = [
            {"root_pos": root_pos[:30].copy(), "joint_data": rot6d[:30].copy()},
            {"root_pos": root_pos[:20].copy(), "joint_data": rot6d[:20].copy()},
            {"root_pos": root_pos[:40].copy(), "joint_data": rot6d[:40].copy()},
        ]
        return clips

    @pytest.fixture
    def bvh_paths(self):
        bvh_dir = Path(__file__).parent.parent / "bvh_data"
        return sorted(bvh_dir.glob("bvh_test1.bvh"))

    # --- MotionDataset ---

    def test_motion_dataset_len(self, sample_clips):
        ds = MotionDataset(sample_clips)
        assert len(ds) == 3

    def test_motion_dataset_getitem(self, sample_clips):
        ds = MotionDataset(sample_clips)
        item = ds[0]
        assert "data" in item
        assert "length" in item
        assert isinstance(item["data"], torch.Tensor)

    def test_motion_dataset_dtype(self, sample_clips):
        ds = MotionDataset(sample_clips)
        item = ds[0]
        assert item["data"].dtype == torch.float32

    def test_motion_dataset_with_labels(self, sample_clips):
        labels = np.array([0, 1, 2])
        ds = MotionDataset(sample_clips, labels=labels)
        item = ds[1]
        assert item["label"] == 1

    def test_motion_dataset_target_length(self, sample_clips):
        ds = MotionDataset(sample_clips, target_length=50)
        item = ds[0]
        assert item["data"].shape[0] == 50
        assert item["length"] == 30  # valid frames (clip shorter than target)

    def test_motion_dataset_target_length_crops(self, sample_clips):
        """Regression: a cropped clip reports the frames actually present, not its pre-crop length."""
        ds = MotionDataset(sample_clips, target_length=25)
        item = ds[0]  # 30-frame clip cropped to 25
        assert item["data"].shape[0] == 25
        assert item["length"] == 25

    # --- OnTheFlyDataset ---

    def test_onthefly_len(self, bvh_paths):
        ds = OnTheFlyDataset(bvh_paths)
        assert len(ds) == len(bvh_paths)

    def test_onthefly_getitem(self, bvh_paths):
        ds = OnTheFlyDataset(bvh_paths, representation="6d")
        item = ds[0]
        assert "data" in item
        assert isinstance(item["data"], torch.Tensor)
        assert item["data"].dtype == torch.float32

    def test_onthefly_target_length_reports_valid_frames(self, bvh_paths):
        """Same length semantics as MotionDataset: valid frames in the returned tensor."""
        num_frames = OnTheFlyDataset(bvh_paths, representation="6d")[0]["length"]
        assert num_frames > 10  # sanity: the fixture clip must be croppable

        cropped = OnTheFlyDataset(
            bvh_paths, representation="6d", target_length=10)[0]
        assert cropped["data"].shape[0] == 10
        assert cropped["length"] == 10

        padded = OnTheFlyDataset(
            bvh_paths, representation="6d", target_length=num_frames + 5)[0]
        assert padded["data"].shape[0] == num_frames + 5
        assert padded["length"] == num_frames

    # --- collate_motion_batch ---

    def test_collate_shapes(self, sample_clips):
        ds = MotionDataset(sample_clips, labels=np.array([0, 1, 2]))
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        assert collated["data"].shape[0] == 3  # batch size
        assert collated["data"].shape[1] == 40  # max length
        assert collated["lengths"].shape == (3,)
        assert collated["mask"].shape == (3, 40)

    def test_collate_padding(self, sample_clips):
        ds = MotionDataset(sample_clips)
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        # Clip 1 has 20 frames — padding should be zero after that
        assert torch.all(collated["data"][1, 20:] == 0)

    def test_collate_mask(self, sample_clips):
        ds = MotionDataset(sample_clips)
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        # Clip 0: 30 frames → mask[0, :30] = True, mask[0, 30:] = False
        assert collated["mask"][0, :30].all()
        assert not collated["mask"][0, 30:].any()

    def test_collate_mask_with_cropped_and_padded_clips(self, sample_clips):
        """Regression: the mask must reflect the frames in the tensor, not the pre-standardization clip lengths."""
        ds = MotionDataset(sample_clips, target_length=25)
        batch = [ds[0], ds[1]]  # clip 0: 30→25 cropped; clip 1: 20→25 padded
        collated = collate_motion_batch(batch)
        assert collated["lengths"].tolist() == [25, 20]
        # Cropped clip: every frame in the tensor is valid.
        assert collated["mask"][0].all()
        # Padded clip: True exactly for the valid prefix.
        assert collated["mask"][1, :20].all()
        assert not collated["mask"][1, 20:].any()

    def test_collate_labels(self, sample_clips):
        labels = np.array([5, 3, 7])
        ds = MotionDataset(sample_clips, labels=labels)
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        assert "labels" in collated
        assert collated["labels"].tolist() == [5, 3, 7]

    def test_collate_with_dataloader(self, sample_clips):
        from torch.utils.data import DataLoader
        labels = np.array([0, 1, 2])
        ds = MotionDataset(sample_clips, labels=labels)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_motion_batch)
        batch = next(iter(loader))
        assert batch["data"].shape[0] == 2
        assert "lengths" in batch
        assert "mask" in batch

    # --- set_epoch + seeded augmentation (0.3) ---

    def _make_seeded_dataset(self, sample_clips, seed):
        from pybvh_ml.augmentation import add_joint_noise
        from pybvh_ml.pipeline import AugmentationPipeline
        # Build a fresh quat-primary dataset so noise aug can run in place.
        from pybvh_ml.convert import convert_arrays
        quat_clips = []
        for c in sample_clips:
            rot6d = c["joint_data"]
            quats = convert_arrays(rot6d, from_repr="6d", to_repr="quat")
            quat_clips.append({"root_pos": c["root_pos"].copy(),
                               "joint_data": quats})
        pipeline = AugmentationPipeline([
            (add_joint_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        return MotionDataset(
            quat_clips, target_length=30, augmentation=pipeline, seed=seed)

    def test_seed_reproducible_same_epoch(self, sample_clips):
        """Two datasets with the same seed + epoch produce identical tensors."""
        ds1 = self._make_seeded_dataset(sample_clips, seed=42)
        ds2 = self._make_seeded_dataset(sample_clips, seed=42)
        ds1.set_epoch(3)
        ds2.set_epoch(3)
        torch.testing.assert_close(ds1[0]["data"], ds2[0]["data"])

    def test_set_epoch_changes_output(self, sample_clips):
        """Different epochs produce different augmented tensors."""
        ds = self._make_seeded_dataset(sample_clips, seed=42)
        ds.set_epoch(0)
        a = ds[0]["data"].clone()
        ds.set_epoch(1)
        b = ds[0]["data"].clone()
        assert not torch.allclose(a, b)

    def test_seed_none_is_nondeterministic(self, sample_clips):
        """With seed=None, repeated __getitem__ uses fresh entropy."""
        from pybvh_ml.augmentation import add_joint_noise
        from pybvh_ml.pipeline import AugmentationPipeline
        from pybvh_ml.convert import convert_arrays
        quat_clips = [
            {"root_pos": c["root_pos"].copy(),
             "joint_data": convert_arrays(
                 c["joint_data"], from_repr="6d", to_repr="quat")}
            for c in sample_clips
        ]
        pipeline = AugmentationPipeline([
            (add_joint_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        ds = MotionDataset(
            quat_clips, target_length=30, augmentation=pipeline, seed=None)
        a = ds[0]["data"].clone()
        b = ds[0]["data"].clone()
        # Vanishingly unlikely to collide.
        assert not torch.allclose(a, b)

    def test_warns_when_seeded_without_set_epoch(self, sample_clips):
        """Silently seeding without set_epoch is a quiet correctness bug."""
        ds = self._make_seeded_dataset(sample_clips, seed=42)
        with pytest.warns(UserWarning, match="set_epoch"):
            ds[0]
        # Warning should only fire once per instance.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds[1]  # must not raise — already warned

    def test_no_warn_when_set_epoch_called(self, sample_clips):
        ds = self._make_seeded_dataset(sample_clips, seed=42)
        ds.set_epoch(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ds[0]  # must not raise — contract acknowledged

    def test_no_warn_when_seed_none(self, sample_clips):
        """seed=None means fresh entropy per call — no epoch contract needed."""
        from pybvh_ml.augmentation import add_joint_noise
        from pybvh_ml.pipeline import AugmentationPipeline
        from pybvh_ml.convert import convert_arrays
        quat_clips = [
            {"root_pos": c["root_pos"].copy(),
             "joint_data": convert_arrays(
                 c["joint_data"], from_repr="6d", to_repr="quat")}
            for c in sample_clips
        ]
        pipeline = AugmentationPipeline([
            (add_joint_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        ds = MotionDataset(
            quat_clips, target_length=30, augmentation=pipeline, seed=None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ds[0]

    def test_onthefly_set_epoch(self, bvh_paths):
        """OnTheFlyDataset also honors set_epoch when seeded."""
        from pybvh_ml.augmentation import rotate_vertical
        from pybvh_ml.pipeline import AugmentationPipeline
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y",
                "representation": "quat",
            }),
        ])
        ds = OnTheFlyDataset(
            bvh_paths, representation="quat", target_length=30,
            augmentation=pipeline, seed=7)
        ds.set_epoch(0)
        a = ds[0]["data"].clone()
        ds.set_epoch(5)
        b = ds[0]["data"].clone()
        assert not torch.allclose(a, b)
        ds.set_epoch(0)
        a2 = ds[0]["data"].clone()
        torch.testing.assert_close(a, a2)




class TestSetEpochWorkers:
    """set_epoch must reach DataLoader worker processes.

    Regression: the epoch used to be a plain int attribute — workers
    held a pickled copy, so with persistent_workers=True every epoch
    silently replayed the epoch-0 augmentation (and the
    missing-set_epoch warning was pickled as already-emitted).  The
    epoch now lives in shared memory.
    """

    @pytest.fixture
    def seeded_pair(self, bvh_example):
        """Two identically-constructed seeded datasets (loader + reference)."""
        root_pos, rot6d = bvh_example.to_6d()
        def make():
            clips = [
                {"root_pos": root_pos[:30].copy(),
                 "joint_data": rot6d[:30].copy()},
                {"root_pos": root_pos[:20].copy(),
                 "joint_data": rot6d[:20].copy()},
                {"root_pos": root_pos[:40].copy(),
                 "joint_data": rot6d[:40].copy()},
            ]
            pipeline = AugmentationPipeline([
                (add_joint_noise, 1.0,
                 {"sigma": np.radians(5.0), "representation": "6d"}),
            ])
            return MotionDataset(clips, target_length=30,
                                 augmentation=pipeline, seed=7)
        return make(), make()

    def test_set_epoch_reaches_persistent_workers(self, seeded_pair):
        from torch.utils.data import DataLoader
        ds, ref = seeded_pair
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=2,
            persistent_workers=True, collate_fn=collate_motion_batch)
        per_epoch = []
        for epoch in (0, 1):
            ds.set_epoch(epoch)
            ref.set_epoch(epoch)
            samples = [batch["data"][0] for batch in loader]
            # Bit-equal to the single-process reference: the
            # (seed, epoch, idx) design is worker-count-invariant.
            for i, sample in enumerate(samples):
                torch.testing.assert_close(
                    sample, ref[i]["data"], rtol=0, atol=0)
            per_epoch.append(torch.stack(samples))
        assert not torch.equal(per_epoch[0], per_epoch[1])

    def test_set_epoch_negative_raises(self, seeded_pair):
        ds, _ = seeded_pair
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            ds.set_epoch(-1)
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            ds.set_epoch(-5)
