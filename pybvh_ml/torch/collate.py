"""Collate function for variable-length motion sequences."""
from __future__ import annotations

import torch


def collate_motion_batch(
    batch: list[dict],
) -> dict[str, torch.Tensor]:
    """Collate variable-length motion clips into a padded batch.

    Parameters
    ----------
    batch : list of dict
        Each dict must have ``data`` — a 2-D ``(T, D)`` tensor, the flat layout — and ``length`` (int), the number of valid frames in ``data``, i.e. ``length <= data.shape[0]`` with any frames beyond it being padding (the contract :class:`~pybvh_ml.torch.MotionDataset` and :class:`~pybvh_ml.torch.OnTheFlyDataset` provide under their default ``layout="flat"``).
        Optionally ``label`` (int).

    Returns
    -------
    dict
        ``data`` : ``(B, T_max, D)`` float tensor, zero-padded.
        ``lengths`` : ``(B,)`` long tensor of valid frame counts.
        ``mask`` : ``(B, T_max)`` bool tensor (True = valid frame).
        ``labels`` : ``(B,)`` long tensor (if labels present).

    Raises
    ------
    ValueError
        If any ``data`` is not 2-D.  Padding is time-major on axis 0,
        which the graph layouts don't satisfy: ``(C, T, V)`` puts
        channels there, so padding and masking would silently run along
        the wrong axis.  Those layouts are fixed-size by construction —
        stack them with :func:`torch.utils.data.default_collate`
        instead.
    """
    for i, item in enumerate(batch):
        if item["data"].dim() != 2:
            raise ValueError(
                f"collate_motion_batch expects 2-D (T, D) data — the flat "
                f"layout — but batch item {i} has shape "
                f"{tuple(item['data'].shape)}. Datasets built with "
                f"layout='ctv' / 'tvc' produce fixed-size tensors that "
                f"need no padding; collate them with "
                f"torch.utils.data.default_collate.")

    labeled = [i for i, item in enumerate(batch) if "label" in item]
    if labeled and len(labeled) != len(batch):
        missing = next(i for i in range(len(batch)) if i not in labeled)
        raise ValueError(
            f"'label' present in some batch items but not all (first "
            f"missing at batch index {missing}) — check that labels / "
            f"label_fn cover every clip in the dataset")
    has_labels = bool(labeled)
    D = batch[0]["data"].shape[-1]
    lengths = [item["length"] for item in batch]
    T_max = max(item["data"].shape[0] for item in batch)
    B = len(batch)

    data = torch.zeros(B, T_max, D, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        T_i = item["data"].shape[0]
        data[i, :T_i] = item["data"][:T_i]
        mask[i, :lengths[i]] = True

    result: dict[str, torch.Tensor] = {
        "data": data,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mask": mask,
    }
    if has_labels:
        result["labels"] = torch.tensor(
            [item["label"] for item in batch], dtype=torch.long)

    return result
