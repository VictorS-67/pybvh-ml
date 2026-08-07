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

from pybvh.bvhplot import compute_unified_limits, get_skeleton_lines, render

# One accent per role, reused across figures so the gallery reads as one set.
ACCENT = "#00897b"          # teal — matches the docs site palette
LEFT_COLOR, RIGHT_COLOR = "#1e88e5", "#e53935"
NEUTRAL = "0.45"
DROPPED_COLOR = "#c62828"
# Distinct series colors for pipeline draws (teal is reserved for "the
# augmented result" in single-comparison figures).
DRAW_COLORS = ["#00897b", "#8e24aa", "#fb8c00"]

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
    """CTV / TVC single-frame slices plus the full flat clip, annotated.

    The CTV and TVC slices hold the same numbers — what differs is the
    axis order, so the TVC panel is drawn untransposed (V rows, C
    columns) instead of pretending the layouts look different.  One
    shared color scale across all panels, spelled out by the colorbar.
    """
    lim = np.percentile(np.abs(flat), 98)
    kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05],
                            width_ratios=[3.0, 1.0, 0.09],
                            hspace=0.42, wspace=0.3)

    C, _, V = ctv.shape
    ax = fig.add_subplot(grid[0, 0])
    im = ax.imshow(ctv[:, frame, :], aspect="auto", **kw)
    ax.add_patch(Rectangle((-0.5, 2.5), 1, C - 3, fill=False,
                           edgecolor=ACCENT, lw=2))
    ax.annotate("root zero-pad\n(channels 3:C)", xy=(0, C - 1),
                xytext=(3.2, C - 0.8), color=ACCENT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.set_title(f"(C, T, V) — frame {frame} slice: rows are channels",
                 fontsize=11)
    ax.set_xlabel("V  (vertex 0 = root, 1..J = joints)")
    ax.set_ylabel("C (channels)")

    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(tvc[frame], aspect="auto", **kw)
    ax.add_patch(Rectangle((2.5, -0.5), C - 3, 1, fill=False,
                           edgecolor=ACCENT, lw=2))
    ax.set_title(f"(T, V, C) — frame {frame} slice:\n"
                 "same numbers, axes swapped", fontsize=10)
    ax.set_xlabel("C")
    ax.set_ylabel("V")

    cax = fig.add_subplot(grid[0, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("feature value  (clipped at 98th pct —\n"
                   "root positions saturate)", fontsize=8)

    ax = fig.add_subplot(grid[1, :2])
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


def fig_center_hazard(clip_tvc, window_tvcs, stride: int, up_index: int):
    """Whole-clip centering vs per-window re-centering, top view.

    Both inputs come straight from ``pack_to_tvc(..., center_root=True)``:
    *clip_tvc* packs the whole clip once, *window_tvcs* packs each window
    separately — the hazard the right panel demonstrates.  Root path is
    vertex 0, channels 0:3.
    """
    plane = [i for i in range(3) if i != up_index]
    path = clip_tvc[:, 0, plane]
    window = window_tvcs[0].shape[0]
    starts = [i * stride for i in range(len(window_tvcs))]
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
    for win, color in zip(window_tvcs, colors):
        seg = win[:, 0, plane]
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
                 "character's left=blue, right=red, unpaired=dark",
                 fontsize=10)
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

def clip_gif(bvh, path="feature_gallery_hero.gif", fps=20):
    """The hero clip, rendered to a real-time GIF; returns the written path.

    Resampled to the GIF rate first so playback is true real time (see
    ``speed_comparison_gif``); 20 fps sits on the GIF format's
    centisecond frame-delay grid.

    Returns the path rather than an ``IPython.display.Image``: GitHub's
    notebook renderer displays image/png outputs but silently drops
    image/gif ones, so the notebook shows the committed GIF from a
    markdown cell instead (which also keeps the base64 out of the .ipynb).
    """
    clip = bvh.resample(fps)
    out = render(clip, path, fps=fps, backend="matplotlib", camera="front",
                 resolution=(480, 360))
    return str(out)


def speed_comparison_gif(bvhs, labels, path="feature_gallery_speed.gif",
                         fps=20):
    """Side-by-side real-time playback of speed-perturbed clips; returns the written path (see ``clip_gif`` for why not an ``Image``).

    Each clip is resampled to the GIF rate first — ``render()`` writes
    every frame at the requested playback rate, never resampling — so
    wall-clock speed is true: the factor<1 clip lags behind the original
    and the factor>1 clip finishes early, freezing on its last frame
    (``sync="pad"``).  20 fps sits exactly on the GIF format's
    centisecond frame-delay grid, so playback is real time.
    """
    clips = [b.resample(fps) for b in bvhs]
    out = render(clips, path, labels=labels, fps=fps, sync="pad",
                 backend="matplotlib", camera="front", resolution=(960, 360))
    return str(out)


def fig_dropout(track_label, original, dropped, kept_mask, drop_rate):
    """Dropout on one joint coordinate: kept frames, re-interpolated gaps.

    The re-interpolated curve hugs the original almost everywhere —
    Bernoulli drops leave mostly 1–2-frame gaps, which SLERP bridges
    nearly exactly on smooth mocap.  So the mechanism is drawn
    explicitly: markers on the surviving frames, a rug of dropped frame
    indices, and an inset zooming on the longest gap, where the
    interpolation visibly cuts the corner.
    """
    frames = np.arange(len(original))
    dropped_idx = frames[~kept_mask]
    lo = min(original.min(), dropped.min())
    hi = max(original.max(), dropped.max())
    span = hi - lo

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(original, color="0.6", lw=2.5, label="original", zorder=1)
    ax.scatter(frames[kept_mask], original[kept_mask], s=16, color="0.35",
               zorder=3, label=f"kept frames ({int(kept_mask.sum())})")
    ax.plot(dropped, color=ACCENT, lw=1.6, zorder=2,
            label=f"drop_rate={drop_rate} (re-interpolated)")
    ax.eventplot(dropped_idx, lineoffsets=lo - 0.1 * span,
                 linelengths=0.06 * span, colors=DROPPED_COLOR, lw=1.2,
                 label=f"dropped frames ({len(dropped_idx)})")

    # Inset: the longest run of consecutive dropped frames, where the
    # SLERP bridge deviates most from the original.
    runs = np.split(dropped_idx, np.where(np.diff(dropped_idx) > 1)[0] + 1)
    gap = max(runs, key=len)
    a = max(0, int(gap[0]) - 4)
    b = min(len(original) - 1, int(gap[-1]) + 4)
    seg = slice(a, b + 1)
    seg_lo = min(original[seg].min(), dropped[seg].min())
    seg_hi = max(original[seg].max(), dropped[seg].max())
    pad = 0.15 * (seg_hi - seg_lo)
    axins = ax.inset_axes([0.03, 0.58, 0.3, 0.4])
    axins.plot(frames[seg], original[seg], color="0.6", lw=2.5)
    kept_seg = kept_mask[seg]
    axins.scatter(frames[seg][kept_seg], original[seg][kept_seg], s=22,
                  color="0.35", zorder=3)
    axins.plot(frames[seg], dropped[seg], color=ACCENT, lw=1.6)
    axins.set_xlim(a, b)
    axins.set_ylim(seg_lo - pad, seg_hi + pad)
    axins.set_xticks([]), axins.set_yticks([])
    axins.set_title(f"longest gap ({len(gap)} frames)", fontsize=8)
    ax.indicate_inset_zoom(axins, edgecolor="0.5")

    ax.set_ylim(lo - 0.16 * span, hi + 0.06 * span)
    ax.set_xlabel("frame")
    ax.set_ylabel(track_label)
    ax.set_yticks([])
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("dropout_arrays — dropped frames re-interpolated (SLERP)",
                 fontsize=10)
    return fig


def share_3d_limits(axes, bvhs, frame: int = 0):
    """Give side-by-side 3D panels one shared bounding box.

    ``bvhplot.frame`` auto-scales each panel separately, which silently
    absorbs part of the translation/rotation differences between
    augmentation draws.
    """
    coords = [b.node_positions()[frame] for b in bvhs]
    center, half = compute_unified_limits(coords)
    for ax in axes:
        ax.set_xlim3d(center[0] - half, center[0] + half)
        ax.set_ylim3d(center[1] - half, center[1] + half)
        ax.set_zlim3d(center[2] - half, center[2] + half)


def describe_draw(steps) -> str:
    """One-line summary of a pipeline draw, from ``return_params=True``.

    Turns the per-step records into the caption a still can't otherwise
    carry: which steps fired, and with which sampled values.
    """
    shown = {"rotate_vertical": ("yaw", "angle", np.degrees, "{:+.0f}°"),
             "speed_perturbation_arrays": ("speed", "factor", float, "×{:.2f}")}
    parts = []
    for step in steps:
        if not step["applied"]:
            continue
        if step["name"] in shown:
            label, key, cast, fmt = shown[step["name"]]
            parts.append(f"{label} {fmt.format(cast(step['params'][key]))}")
        elif step["name"] == "mirror":
            parts.append("mirrored")
    return " · ".join(parts)


def fig_draw_tracks(track_label, original, draw_tracks):
    """One tracked joint coordinate under each pipeline draw.

    The effects a frame-0 still can't show: the x-extent is each draw's
    sampled speed factor (frame counts differ) and the jitter is the
    rotation noise.  Rotation and mirror barely touch a height track —
    they show in the skeleton stills above instead.
    """
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(original, color="0.6", lw=2.5,
            label=f"original ({len(original)}f)")
    for i, (track, color) in enumerate(zip(draw_tracks, DRAW_COLORS)):
        ax.plot(track, color=color, lw=1.6, label=f"draw {i} ({len(track)}f)")
    ax.set_xlabel("frame")
    ax.set_ylabel(track_label)
    ax.set_yticks([])
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("per-draw joint track — x-extent = sampled speed factor, "
                 "jitter = noise", fontsize=10)
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
                label=f"epoch {epoch} — run B")
    ax.set_xlabel("frame")
    ax.set_ylabel("‖augmented − raw‖ per frame")
    ax.set_title("same (seed, epoch, idx) ⇒ bit-identical draw: dashed run B "
                 "lies exactly on solid run A; epochs differ", fontsize=10)
    # legend fills column-major, so ncol=3 pairs each epoch's A/B vertically
    ax.legend(fontsize=8, frameon=False, ncol=3)
    return fig


# ----------------------------------------------------------------
#  5 · Sequence tools
# ----------------------------------------------------------------

def fig_windows(num_frames, windows_shape, padded_len, cropped_len, stride):
    """Sliding-window spans plus standardize_length pad/crop bars.

    Geometry comes from the actual output shapes (``windows.shape``,
    ``len(padded)``, ``len(cropped)``), not re-derived formulas.
    """
    n_win, window = windows_shape[0], windows_shape[1]
    starts = [i * stride for i in range(n_win)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.broken_barh([(0, num_frames)], (n_win + 0.6, 0.6), color="0.85")
    ax.text(num_frames / 2, n_win + 0.9, f"clip · {num_frames} frames",
            ha="center", va="center", fontsize=8.5, color="0.25")
    for i, s in enumerate(starts):
        ax.broken_barh([(s, window)], (n_win - 1 - i, 0.8),
                       color=plt.cm.viridis(0.15 + 0.7 * i / max(1, n_win - 1)))
    ax.set_ylim(-0.4, n_win + 1.5)
    ax.set_title(f"sliding_window(window_size={window}, stride={stride}) → "
                 f"{n_win} windows", fontsize=10)
    ax.set_xlabel("frame")
    ax.set_yticks([])

    ax = axes[1]
    bars = [("original", num_frames, num_frames),
            (f'pad → {padded_len}', padded_len, num_frames),
            (f'crop → {cropped_len}', cropped_len, cropped_len)]
    for i, (label, total, valid) in enumerate(bars):
        y = len(bars) - 1 - i
        ax.broken_barh([(0, valid)], (y, 0.7), color=ACCENT)
        if total > valid:
            ax.broken_barh([(valid, total - valid)], (y, 0.7),
                           facecolors="none", edgecolors=ACCENT,
                           hatch="///", lw=1)
        ax.text(-2, y + 0.35, label, ha="right", va="center", fontsize=9)
    ax.set_xlim(-padded_len * 0.45, padded_len * 1.05)
    ax.set_title("standardize_length (hatched = padding)", fontsize=10)
    ax.set_xlabel("frame")
    ax.set_yticks([])
    return fig


def fig_temporal_sample(num_frames, clip_length, train_draws, test_draw):
    """Segment shading + picked frame indices, train vs test mode.

    Shading uses the sampler's own integer boundaries
    (``i * num_frames // clip_length``) — with float ``linspace`` bounds
    the dots drift across the drawn segments and some segments appear to
    hold two picks.  Segment sizes genuinely alternate when
    *clip_length* doesn't divide *num_frames*.  Spans and boundary
    lines sit at half-integers: frame indices are discrete, so segment
    ``[b, b')`` is drawn as ``[b - 0.5, b' - 0.5)`` — its first frame
    lands half a unit inside the shade instead of on the dividing line.
    """
    bounds = np.array([i * num_frames // clip_length
                       for i in range(clip_length + 1)])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for i in range(clip_length):
        if i % 2 == 0:
            ax.axvspan(bounds[i] - 0.5, bounds[i + 1] - 0.5, color="0.93",
                       zorder=0)
    for b in bounds:
        ax.axvline(b - 0.5, color="0.82", lw=0.6, ls=":", zorder=0)
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
    """Flat features before/after z-score, with the constant-channel strip.

    D runs vertically in every panel: the ``constant_channels`` strip is
    drawn as a vertical band sharing the heatmaps' row axis (dark row =
    constant channel), so no mental rotation is needed to line a black
    block up with its feature rows.  Each heatmap gets its own colorbar —
    the raw panel spans its own 98th percentile, the normalized panel is
    fixed at ±3 z-scores.
    """
    fig = plt.figure(figsize=(10, 6.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[40, 1.6, 1.6],
                            hspace=0.5, wspace=0.08)
    strip = constant_channels[:, None].astype(float)
    n_const = int(constant_channels.sum())

    lim_raw = np.percentile(np.abs(flat), 98)
    panels = [
        (flat, dict(cmap="RdBu_r", vmin=-lim_raw, vmax=lim_raw),
         f"raw (T, D) — every channel at its own offset and scale "
         f"(std spans [{flat.std(0).min():.2g}, {flat.std(0).max():.2g}])",
         "raw value"),
        (flat_norm, dict(cmap="RdBu_r", vmin=-3, vmax=3),
         "normalize_array(x, stats) — shared scale, temporal structure "
         "visible; constant channels exactly 0",
         "z-score (±3)"),
    ]
    for row, (data, kw, title, cbar_label) in enumerate(panels):
        ax = fig.add_subplot(grid[row, 0])
        im = ax.imshow(data.T, aspect="auto", **kw)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("D")
        ax.set_xlabel("T (frames)")
        ax_strip = fig.add_subplot(grid[row, 1], sharey=ax)
        ax_strip.imshow(strip, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax_strip.set_xticks([])
        plt.setp(ax_strip.get_yticklabels(), visible=False)
        ax_strip.tick_params(left=False)
        if row == 0:
            ax_strip.set_title("const", fontsize=8)
        cax = fig.add_subplot(grid[row, 2])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=8)
    fig.text(0.5, 0.015,
             f"const strip: dark row = constant channel — raw std < 1e-8, "
             f"guarded to 1 ({n_const} of {len(constant_channels)} columns)",
             ha="center", fontsize=8.5, color="0.25")
    return fig


def fig_harmonize(before, after):
    """Frame-0 poses with their up axes, before/after harmonization.

    Only the clips harmonize actually changes get the accent-colored
    title — coloring every "before" title implied the already-aligned
    clips needed fixing too.
    """
    n = len(before)
    target = after[0].world_up
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
            changed = row == 0 and bvh.world_up != target
            ax.set_title(f"world_up = {bvh.world_up}", fontsize=10,
                         color=ACCENT if changed else "k")
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

def fig_collate_mask(batch, channel: int, channel_desc: str):
    """The padded batch tensor and its validity mask.

    Light means padding in BOTH panels (the mask uses ``Greys``: dark =
    True = valid), and a teal step at each row's true length ties the
    two staircases together.  Pick a *centered* channel — an uncentered
    root coordinate sits at ~constant tens of units and saturates the
    whole valid region into one dark slab.
    """
    data = batch["data"][:, :, channel].numpy()
    mask = batch["mask"].numpy()
    lengths = batch["lengths"].tolist()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.8), sharex=True)
    lim = np.percentile(np.abs(data), 98)
    im = axes[0].imshow(data, aspect="auto", cmap="RdBu_r",
                        vmin=-lim, vmax=lim)
    axes[0].set_title(f'batch["data"][:, :, {channel}] ({channel_desc}) — '
                      f"zero-padded to T_max = {mask.shape[1]}", fontsize=10)
    axes[1].imshow(mask, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title('batch["mask"] — dark = True = valid frame, '
                      "light = padding", fontsize=10)
    for ax in axes:
        for i, ln in enumerate(lengths):
            ax.plot([ln - 0.5, ln - 0.5], [i - 0.5, i + 0.5],
                    color=ACCENT, lw=2)
        ax.set_yticks(range(len(lengths)))
        ax.set_yticklabels([f"len {ln}" for ln in lengths], fontsize=8)
    axes[1].set_xlabel("T")
    fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02,
                 label=channel_desc)
    return fig
