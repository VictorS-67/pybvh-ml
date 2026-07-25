"""Plotting helpers for the pybvh-ml feature gallery (``feature_gallery.ipynb``).

The notebook stays focused on *which feature is being called*: each cell
computes real pybvh-ml outputs and hands them to the matching ``fig_*``
function here. Skeleton renders go through pybvh's plotting (``bvhplot``) —
pybvh owns visualization; this module only adds the diagram-style figures
(layout heatmaps, timelines, masks) that pybvh has no reason to draw.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pybvh.bvhplot import get_skeleton_lines

# One accent per role, reused across figures so the gallery reads as one set.
ACCENT = "#00897b"          # teal — matches the docs site palette
LEFT_COLOR, RIGHT_COLOR = "#1e88e5", "#e53935"
NEUTRAL = "0.45"

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------
#  Skeleton drawing (2D frontal projections)
# ----------------------------------------------------------------

def _joint_nodes(bvh) -> list[int]:
    """Node-space index of every joint (joint arrays exclude end sites)."""
    return [bvh.index(name, space="node") for name in bvh.joint_names]


def _frontal_axes(bvh, pos: np.ndarray) -> tuple[int, int]:
    """(horizontal, vertical) coordinate indices for a frontal 2D view."""
    up = _AXIS_INDEX[bvh.world_up[-1]]
    lateral = [i for i in range(3) if i != up]
    spans = np.ptp(pos[:, lateral], axis=0)
    return lateral[int(np.argmax(spans))], up


def draw_skeleton_2d(ax, bvh, frame: int = 0, color=NEUTRAL, lw=2.0,
                     joint_colors=None, joint_size=28):
    """Frontal 2D projection of one frame; optional per-joint colors."""
    pos = bvh.node_positions()[frame]
    h, v = _frontal_axes(bvh, pos)
    for p, c in get_skeleton_lines(bvh):
        ax.plot([pos[p, h], pos[c, h]], [pos[p, v], pos[c, v]],
                color=color, lw=lw, zorder=1)
    if joint_colors is not None:
        nodes = _joint_nodes(bvh)
        xy = pos[nodes][:, [h, v]]
        ax.scatter(xy[:, 0], xy[:, 1], c=joint_colors, s=joint_size, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    return pos, h, v


# ----------------------------------------------------------------
#  1 · Tensor layouts
# ----------------------------------------------------------------

def fig_layouts(ctv, tvc, flat, desc, frame: int):
    """CTV / TVC single-frame slices plus the full flat clip, annotated."""
    lim = np.percentile(np.abs(flat), 98)
    kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.45,
                            wspace=0.25)

    C, _, V = ctv.shape
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(ctv[:, frame, :], aspect="auto", **kw)
    ax.add_patch(Rectangle((-0.5, 2.5), 1, C - 3, fill=False,
                           edgecolor=ACCENT, lw=2))
    ax.annotate("root zero-pad\n(channels 3:C)", xy=(0, C - 1),
                xytext=(3.2, C - 0.8), color=ACCENT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.set_title(f"(C, T, V) — frame {frame} slice", fontsize=11)
    ax.set_xlabel("V  (vertex 0 = root, 1..J = joints)")
    ax.set_ylabel("C (channels)")

    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(tvc[frame].T, aspect="auto", **kw)
    ax.set_title(f"(T, V, C) — frame {frame} slice (transposed view)",
                 fontsize=11)
    ax.set_xlabel("V")
    ax.set_ylabel("C")

    ax = fig.add_subplot(grid[1, :])
    ax.imshow(flat.T, aspect="auto", **kw)
    for name in desc:
        start, stop = desc[name]
        ax.axhline(start - 0.5, color="k", lw=0.8)
        ax.annotate(f"{name}  [{start}:{stop}]", xy=(flat.shape[0] * 0.99,
                    (start + stop) / 2), ha="right", va="center", fontsize=9,
                    color="k", bbox=dict(boxstyle="round,pad=0.25",
                                         fc="white", alpha=0.85))
    ax.set_title("flat (T, D) — whole clip, columns mapped by "
                 "describe_features", fontsize=11)
    ax.set_xlabel("T (frames)")
    ax.set_ylabel("D (feature columns)")
    fig.suptitle("One clip, three layouts", y=0.98)
    return fig


def fig_center_hazard(root_pos, up_index: int, window: int, stride: int):
    """Whole-clip centering vs per-window re-centering, top view."""
    plane = [i for i in range(3) if i != up_index]
    path = root_pos[:, plane] - root_pos[0, plane]
    starts = list(range(0, len(path) - window + 1, stride))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(starts)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=False)
    ax = axes[0]
    ax.plot(path[:, 0], path[:, 1], color="0.75", lw=4, zorder=1)
    for s, color in zip(starts, colors):
        seg = path[s:s + window]
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2, zorder=2)
    ax.set_title("center the clip once, then cut windows\n"
                 "(global trajectory preserved)", fontsize=10)

    ax = axes[1]
    for s, color in zip(starts, colors):
        seg = path[s:s + window]
        seg = seg - seg[0]          # what center_root=True does per window
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=2)
    ax.scatter([0], [0], color="k", zorder=3, s=25)
    ax.set_title("pack each window with center_root=True\n"
                 "(every window re-based to its own origin)", fontsize=10)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("ground plane")
        ax.set_yticks([])
        ax.set_xticks([])
    return fig


# ----------------------------------------------------------------
#  2 · Skeleton graph metadata
# ----------------------------------------------------------------

def fig_graph(bvh, edges, lr_pairs):
    """Edges + L/R pairs on the frame-0 pose."""
    fig, ax = plt.subplots(figsize=(5.5, 6))
    pos, h, v = draw_skeleton_2d(ax, bvh, color="0.8", lw=1.5)
    nodes = _joint_nodes(bvh)
    xy = pos[nodes][:, [h, v]]
    for child, parent in edges:
        ax.plot([xy[child, 0], xy[parent, 0]], [xy[child, 1], xy[parent, 1]],
                color=NEUTRAL, lw=2.2, zorder=2)
    side = {}
    for left, right in lr_pairs:
        side[left], side[right] = LEFT_COLOR, RIGHT_COLOR
    colors = [side.get(j, "0.25") for j in range(bvh.joint_count)]
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=42, zorder=3)
    for left, right in lr_pairs:
        ax.plot([xy[left, 0], xy[right, 0]], [xy[left, 1], xy[right, 1]],
                color="0.6", lw=0.7, ls=":", zorder=1)
    ax.set_title(f"get_edge_list: {len(edges)} edges · "
                 f"get_lr_pairs: {len(lr_pairs)} pairs\n"
                 "left=blue, right=red, unpaired=dark", fontsize=10)
    return fig


def fig_partitions(bvh, partitions):
    """Joints colored by body-part partition."""
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(5.5, 6))
    pos, h, v = draw_skeleton_2d(ax, bvh, color="0.8", lw=2.2)
    nodes = _joint_nodes(bvh)
    xy = pos[nodes][:, [h, v]]
    for i, (name, joints) in enumerate(partitions.items()):
        if not joints:
            continue
        ax.scatter(xy[joints, 0], xy[joints, 1], color=palette[i % 10],
                   s=46, zorder=3, label=f"{name} ({len(joints)})")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9,
              frameon=False)
    ax.set_title("get_body_partitions", fontsize=11)
    return fig


# ----------------------------------------------------------------
#  3 · Augmentation (temporal effects)
# ----------------------------------------------------------------

def fig_speed_dropout(track_label, original, slowed, sped, dropped,
                      factors=(0.75, 1.25), drop_rate=0.3):
    """One joint coordinate over time under speed perturbation and dropout."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(original, color="0.6", lw=2.5, label=f"original ({len(original)}f)")
    ax.plot(slowed, color=ACCENT, lw=1.8,
            label=f"factor={factors[0]} ({len(slowed)}f)")
    ax.plot(sped, color="#8e24aa", lw=1.8,
            label=f"factor={factors[1]} ({len(sped)}f)")
    ax.set_title("speed_perturbation_arrays — time axis resampled",
                 fontsize=10)
    ax = axes[1]
    ax.plot(original, color="0.6", lw=2.5, label="original")
    ax.plot(dropped, color=ACCENT, lw=1.8,
            label=f"drop_rate={drop_rate} (re-interpolated)")
    ax.set_title("dropout_arrays — dropped frames re-interpolated",
                 fontsize=10)
    for ax in axes:
        ax.set_xlabel("frame")
        ax.legend(fontsize=8, frameon=False)
        ax.set_yticks([])
    axes[0].set_ylabel(track_label)
    return fig


def fig_epoch_determinism(curves: dict):
    """Per-frame deviation from the raw clip: runs overlay, epochs differ.

    ``curves[(run, epoch)]`` is a per-frame deviation array; runs "A" and "B"
    share a seed, so per epoch the two curves must coincide exactly.
    """
    epochs = sorted({e for _, e in curves})
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, epoch in enumerate(epochs):
        color = plt.cm.viridis(0.15 + 0.7 * i / max(1, len(epochs) - 1))
        ax.plot(curves[("A", epoch)], color=color, lw=3.2, alpha=0.9,
                label=f"epoch {epoch} — run A")
        ax.plot(curves[("B", epoch)], color="k", lw=1.0, ls="--",
                label=f"epoch {epoch} — run B" if i == 0 else None)
    ax.set_xlabel("frame")
    ax.set_ylabel("‖augmented − raw‖ per frame")
    ax.set_title("same (seed, epoch, idx) ⇒ bit-identical draw: dashed run B "
                 "lies exactly on solid run A; epochs differ", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    return fig


# ----------------------------------------------------------------
#  5 · Sequence tools
# ----------------------------------------------------------------

def fig_windows(num_frames, window, stride, pad_to, crop_to):
    """Sliding-window spans plus standardize_length pad/crop bars."""
    starts = list(range(0, num_frames - window + 1, stride))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.broken_barh([(0, num_frames)], (len(starts) + 0.6, 0.6), color="0.85")
    ax.text(num_frames / 2, len(starts) + 1.55, f"clip ({num_frames} frames)",
            ha="center", fontsize=9)
    for i, s in enumerate(starts):
        ax.broken_barh([(s, window)], (len(starts) - 1 - i, 0.8),
                       color=plt.cm.viridis(0.15 + 0.7 * i / max(1, len(starts) - 1)))
    ax.set_title(f"sliding_window(window_size={window}, stride={stride}) → "
                 f"{len(starts)} windows", fontsize=10)
    ax.set_xlabel("frame")
    ax.set_yticks([])

    ax = axes[1]
    bars = [("original", num_frames, num_frames),
            (f'pad → {pad_to}', pad_to, num_frames),
            (f'crop → {crop_to}', crop_to, crop_to)]
    for i, (label, total, valid) in enumerate(bars):
        y = len(bars) - 1 - i
        ax.broken_barh([(0, valid)], (y, 0.7), color=ACCENT)
        if total > valid:
            ax.broken_barh([(valid, total - valid)], (y, 0.7),
                           facecolors="none", edgecolors=ACCENT,
                           hatch="///", lw=1)
        ax.text(-2, y + 0.35, label, ha="right", va="center", fontsize=9)
    ax.set_xlim(-pad_to * 0.45, pad_to * 1.05)
    ax.set_title("standardize_length (hatched = padding)", fontsize=10)
    ax.set_xlabel("frame")
    ax.set_yticks([])
    return fig


def fig_temporal_sample(num_frames, clip_length, train_draws, test_draw):
    """Segment shading + picked frame indices, train vs test mode."""
    bounds = np.linspace(0, num_frames, clip_length + 1)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for i in range(clip_length):
        if i % 2 == 0:
            ax.axvspan(bounds[i], bounds[i + 1], color="0.93", zorder=0)
    rows = [(f"train (draw {i + 1})", d, plt.cm.viridis(0.2 + 0.3 * i))
            for i, d in enumerate(train_draws)] + [("test", test_draw, "k")]
    for y, (label, indices, color) in enumerate(reversed(rows)):
        ax.scatter(indices, np.full(len(indices), y), color=color, s=26,
                   zorder=2)
        ax.text(-1.5, y, label, ha="right", va="center", fontsize=9)
    ax.set_xlim(-num_frames * 0.22, num_frames)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("frame index")
    ax.set_title(f"uniform_temporal_sample: one frame per segment "
                 f"({clip_length} segments, shaded)", fontsize=10)
    return fig


# ----------------------------------------------------------------
#  6 · Preprocessing
# ----------------------------------------------------------------

def fig_normalization(flat, flat_norm, constant_channels):
    """Flat features before/after z-score, plus the constant-channel mask.

    No sharex: the heatmaps' x-axis is T (frames) while the mask strip's is
    D (columns) — sharing would stretch the heatmaps over the wrong range.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 6),
                             gridspec_kw={"height_ratios": [1, 1, 0.12],
                                          "hspace": 0.55})
    lim_raw = np.percentile(np.abs(flat), 98)
    axes[0].imshow(flat.T, aspect="auto", cmap="RdBu_r",
                   vmin=-lim_raw, vmax=lim_raw)
    axes[0].set_title(f"raw (T, D) — per-channel std spans "
                      f"[{flat.std(0).min():.2g}, {flat.std(0).max():.2g}]",
                      fontsize=10)
    axes[1].imshow(flat_norm.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    axes[1].set_title("normalize_array(x, stats) — every varying channel at "
                      "std 1", fontsize=10)
    axes[2].imshow(constant_channels[None, :], aspect="auto", cmap="Greys",
                   vmin=0, vmax=1)
    axes[2].set_title(f"constant_channels mask "
                      f"({int(constant_channels.sum())} of "
                      f"{len(constant_channels)} columns; std guarded to 1)",
                      fontsize=10)
    for ax in axes[:2]:
        ax.set_ylabel("D")
        ax.set_xlabel("T (frames)")
    axes[2].set_yticks([])
    axes[2].set_xlabel("D (columns)")
    return fig


def fig_harmonize(before, after):
    """Frame-0 poses with their up axes, before/after harmonization."""
    n = len(before)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    for row, clips, title in ((0, before, "before"), (1, after, "after")):
        for col, bvh in enumerate(clips):
            ax = axes[row, col]
            pos, h, v = draw_skeleton_2d(ax, bvh, color=NEUTRAL, lw=2.0)
            height = np.ptp(pos[:, v])
            root = pos[0]
            ax.annotate("", xy=(root[h], root[v] + 0.65 * height),
                        xytext=(root[h], root[v]),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
            ax.set_title(f"world_up = {bvh.world_up}", fontsize=10,
                         color=ACCENT if row == 0 else "k")
        axes[row, 0].set_ylabel(title, fontsize=12)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)
    return fig


# ----------------------------------------------------------------
#  7 · PyTorch batching
# ----------------------------------------------------------------

def fig_collate_mask(batch, channel: int = 2):
    """The padded batch tensor and its validity mask."""
    data = batch["data"][:, :, channel].numpy()
    mask = batch["mask"].numpy()
    lengths = batch["lengths"].tolist()
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    lim = np.percentile(np.abs(data), 98)
    axes[0].imshow(data, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_title(f'batch["data"][:, :, {channel}] — zero-padded to '
                      f"T_max = {mask.shape[1]}", fontsize=10)
    axes[1].imshow(mask, aspect="auto", cmap="Greys_r", vmin=0, vmax=1)
    axes[1].set_title('batch["mask"] — True (light) = valid frame', fontsize=10)
    for ax in axes:
        ax.set_yticks(range(len(lengths)))
        ax.set_yticklabels([f"len {ln}" for ln in lengths], fontsize=8)
    axes[1].set_xlabel("T")
    return fig
