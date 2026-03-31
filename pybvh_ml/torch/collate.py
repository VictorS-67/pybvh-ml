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
        Each dict must have ``data`` (tensor), ``length`` (int).
        Optionally ``label`` (int).

    Returns
    -------
    dict
        ``data`` : ``(B, T_max, D)`` float tensor, zero-padded.
        ``lengths`` : ``(B,)`` long tensor of original lengths.
        ``mask`` : ``(B, T_max)`` bool tensor (True = valid frame).
        ``labels`` : ``(B,)`` long tensor (if labels present).
    """
    has_labels = "label" in batch[0]
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
