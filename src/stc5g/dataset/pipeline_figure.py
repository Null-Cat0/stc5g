"""Paper figure illustrating data extraction and processing.

The figure goes from the capture in the 5G testbed to the input tensors of the
classifier and has two parts:

* Top row: block diagram of the five pipeline stages.
* Bottom row: example with real dataset values (throughput per service, zoom on the
  sliding window and the normalised input matrix).

Sizes are computed in inches for a double-column width (7.16 in) and the text is
shrunk automatically to fit each block. Labels are in English (the only language).

Example:
    python -m stc5g.dataset.pipeline_figure
    python -m stc5g.dataset.pipeline_figure --output docs/figures/data_pipeline --font serif
"""

from __future__ import annotations

import argparse
import os
import pickle
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from stc5g.config import DOCS_FIG_DIR, LOOKBACK, ROOT_DIR, SERVICES, TIME_DELTA_FIELD
from stc5g.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

#: Matplotlib settings for the figure (applied by :func:`apply_style`).
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # TrueType: selectable/editable text in the PDF
    "ps.fonttype": 42,
}

INK = "#1c1f26"  # main text
MUTED = "#616b7a"  # secondary text
EDGE = "#a7b0bd"  # box borders
FILL_ALT = "#e4ecf5"  # output-box fill
ACCENT = "#2f5d8a"  # accent (headers, arrows)

# Geometry (inches)
FIG_W = 7.16  # double-column width
MARGIN_X = 0.06
DIAG_W = FIG_W - 2 * MARGIN_X
UNIT = DIAG_W / 100.0  # one diagram unit, in inches
PT = 1 / 72.0  # one point, in inches

FS_TITLE = 6.6  # stage header
FS_BULLET = 5.9  # bullets
FS_OUT = 6.0  # output box
WRAP = 22  # characters per line inside a block

# ---------------------------------------------------------------------------
# Figure text
# ---------------------------------------------------------------------------

TEXT = {
    "en": {
        "stages": [
            {
                "n": "1",
                "title": "5G capture",
                "bullets": [
                    "SDN switch, OF 1.3",
                    "IPv4 + GTP-U (2152)",
                    "YouTube/Twitch/TikTok",
                ],
                "out": "{n_traces} .pcap traces",
            },
            {
                "n": "2",
                "title": "Extraction (tshark)",
                "bullets": [
                    "UE filter 10.45.0.0/16",
                    "time_epoch, len, IAT",
                    "GTP-U: inner IP",
                ],
                "out": "per-packet record",
            },
            {
                "n": "3",
                "title": "1 s windows",
                "bullets": [
                    "Σ bytes, mean IAT",
                    r"$T=8\sum b_i/(\Delta t\,10^{6})$ Mbps",
                    "no packets: IAT = 0",
                ],
                "out": "{n_windows} windows ({hours} h)",
            },
            {
                "n": "4",
                "title": "Labelling",
                "bullets": [
                    "label = service (0-2)",
                    "last 30 s dropped",
                    "train/test disjoint",
                ],
                "out": "{n_train} train / {n_test} test",
            },
            {
                "n": "5",
                "title": "Sequencing",
                "bullets": [
                    "MinMax (fit: train)",
                    "L = 20 s, stride 1 s",
                    r"$\mathbf{x}\in\mathbb{R}^{20\times2}$, $y$ = class($t$)",
                ],
                "out": "labelled sequences",
            },
        ],
        "panel_a": "(a) Throughput over 1 s windows",
        "panel_b": "(b) L = 20 s window",
        "panel_c": "(c) 20×2 input",
        "xlabel_time": "Time (s)",
        "ylabel_thr": "Mbps",
        "legend_iat": "IAT (ms)",
        "feat_labels": ["IAT", "Thr."],
        "mat_note": "scaled to [0, 1]",
        "class_out": r"$\hat{y}$ = service",
        "model": "LSTM",
    },
}

#: Languages available for the figure labels (English only).
LANGUAGES = tuple(TEXT)


def apply_style(font: str = "sans") -> None:
    """Apply the figure style to the global Matplotlib settings.

    Args:
        font: ``"sans"`` (default) or ``"serif"`` (to match TikZ figures typeset with
            newtxtext / Times).
    """
    plt.rcParams.update(STYLE)
    if font == "serif":
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


def load_excerpt(csv_dir: str, service: str, n_samples: int, offset: int) -> pd.DataFrame:
    """Return a contiguous excerpt of the first CSV (sorted by name) of a service.

    Args:
        csv_dir: Directory with one sub-directory per service.
        service: Service sub-directory (e.g. ``"youtube"``).
        n_samples: Number of rows.
        offset: First row.

    Raises:
        FileNotFoundError: If the service directory has no CSV file.
    """
    folder = os.path.join(csv_dir, service)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    df = pd.read_csv(os.path.join(folder, files[0]))
    return df.iloc[offset : offset + n_samples].reset_index(drop=True)


def count_windows(directory: str) -> Tuple[int, int]:
    """Return the number of CSV files and of windows (rows) under a directory."""
    n_files = n_rows = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                n_files += 1
                n_rows += max(len(pd.read_csv(os.path.join(root, f), usecols=[0])), 0)
    return n_files, n_rows


def dataset_stats(train_dir: str, test_dir: str) -> Dict[str, object]:
    """Dataset figures shown in the output boxes of the block diagram.

    Args:
        train_dir: Training CSV directory.
        test_dir: Test CSV directory.

    Returns:
        ``n_traces``, ``n_train``, ``n_test`` (file counts) and the formatted strings
        ``n_windows`` and ``hours`` (one window = one second).
    """
    n_train, rows_train = count_windows(train_dir)
    n_test, rows_test = count_windows(test_dir)
    n_windows = rows_train + rows_test
    hours = n_windows / 3600.0

    return {
        "n_traces": n_train + n_test,
        "n_train": n_train,
        "n_test": n_test,
        "n_windows": f"{n_windows:,}",
        "hours": f"{hours:.1f}",
    }


def apply_scaler(x: np.ndarray, scaler_path: Optional[str]) -> Tuple[np.ndarray, bool]:
    """Scale ``x`` with the training MinMaxScaler, or fit one to ``x`` if it is missing.

    Args:
        x: Array of shape ``(LOOKBACK, n_features)``.
        scaler_path: Path to the pickled ``MinMaxScaler`` saved during training.

    Returns:
        ``(scaled, used_saved)``: the values in ``[0, 1]`` and whether the saved scaler
        was used.
    """
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            mms = pickle.load(f)
        return np.clip(mms.transform(x), 0, 1), True
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x - lo) / np.where(hi - lo == 0, 1, hi - lo), False


# ---------------------------------------------------------------------------
# Block diagram
# ---------------------------------------------------------------------------


def wrap_bullets(stage: dict) -> List[str]:
    """Wrap the bullets to the block width (bullets with mathtext are left untouched)."""
    lines: List[str] = []
    for b in stage["bullets"]:
        if "$" in b:
            lines.append(b)
        else:
            lines.extend(textwrap.wrap(b, WRAP) or [b])
    return lines


def shrink_to_fit(fig: plt.Figure, ax: plt.Axes, items: List[tuple]) -> None:
    """Reduce the font size of the texts that do not fit in their allotted width.

    Args:
        fig: Figure (drawn once to measure the texts).
        ax: Axes whose data units define the widths.
        items: ``(text_artist, max_width_in_data_units)`` pairs.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    origin = ax.transData.transform((0, 0))[0]
    for artist, max_w_units in items:
        max_px = ax.transData.transform((max_w_units, 0))[0] - origin
        w_px = artist.get_window_extent(renderer=renderer).width
        if w_px > max_px:
            artist.set_fontsize(max(4.6, artist.get_fontsize() * max_px / w_px))


def draw_stage(
    ax: plt.Axes,
    x0: float,
    y0: float,
    w: float,
    h: float,
    stage: dict,
    lines: List[str],
    geom: Dict[str, float],
    to_fit: List[tuple],
) -> None:
    """Draw a stage box: numbered header, bullets and output box.

    Texts that may overflow are appended to ``to_fit`` for :func:`shrink_to_fit`.
    """
    head_h, line_h, out_h, pad = geom["head"], geom["line"], geom["out"], geom["pad"]

    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.0",
            linewidth=0.6,
            edgecolor=EDGE,
            facecolor="white",
            zorder=2,
        )
    )
    # Header (top rectangle with rounded top corners)
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0 + h - head_h),
            w,
            head_h,
            boxstyle="round,pad=0,rounding_size=1.0",
            linewidth=0,
            facecolor=ACCENT,
            zorder=3,
        )
    )
    ax.add_patch(
        plt.Rectangle((x0, y0 + h - head_h), w, head_h * 0.55, linewidth=0, facecolor=ACCENT, zorder=3)
    )
    ax.text(
        x0 + 1.3,
        y0 + h - head_h / 2,
        stage["n"],
        ha="left",
        va="center",
        color="white",
        fontsize=FS_TITLE,
        fontweight="bold",
        zorder=4,
    )
    title = ax.text(
        x0 + 3.4,
        y0 + h - head_h / 2,
        stage["title"],
        ha="left",
        va="center",
        color="white",
        fontsize=FS_TITLE,
        fontweight="bold",
        zorder=4,
    )
    to_fit.append((title, w - 4.6))

    # Bullets
    y = y0 + h - head_h - pad - line_h / 2
    for ln in lines:
        ax.text(x0 + 1.6, y, "•", ha="left", va="center", color=EDGE, fontsize=FS_BULLET, zorder=4)
        bullet = ax.text(x0 + 3.3, y, ln, ha="left", va="center", color=INK, fontsize=FS_BULLET, zorder=4)
        to_fit.append((bullet, w - 4.5))
        y -= line_h

    # Output box
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 1.2, y0 + pad),
            w - 2.4,
            out_h,
            boxstyle="round,pad=0,rounding_size=0.8",
            linewidth=0.5,
            edgecolor=EDGE,
            facecolor=FILL_ALT,
            zorder=3,
        )
    )
    out = ax.text(
        x0 + w / 2,
        y0 + pad + out_h / 2,
        stage["out"],
        ha="center",
        va="center",
        color=ACCENT,
        fontsize=FS_OUT,
        fontweight="bold",
        zorder=4,
    )
    to_fit.append((out, w - 3.6))


def draw_arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    lw: float = 0.8,
    color: str = ACCENT,
    scale: float = 6.0,
) -> None:
    """Draw a straight arrow from ``(x0, y0)`` to ``(x1, y1)`` in data coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def build_figure(
    csv_dir: str, test_csv_dir: str, scaler_path: Optional[str], lang: str = "en"
) -> Tuple[plt.Figure, bool]:
    """Build the pipeline figure.

    Call :func:`apply_style` first to get the published look.

    Args:
        csv_dir: Training CSV directory (one sub-directory per service); the bottom-row
            excerpts are read from the first CSV of each service.
        test_csv_dir: Test CSV directory (only counted).
        scaler_path: Pickled training ``MinMaxScaler`` for panel (c); if missing, a
            MinMax fitted to the shown excerpt is used.
        lang: Label language; only ``"en"`` is available.

    Returns:
        ``(figure, used_saved_scaler)``.
    """
    t = TEXT[lang]
    stats = dataset_stats(csv_dir, test_csv_dir)
    stages = [dict(s, out=s["out"].format(**stats)) for s in t["stages"]]
    wrapped = [wrap_bullets(s) for s in stages]
    n_lines = max(len(w) for w in wrapped)

    # Diagram heights, in units (1 unit = UNIT inches)
    geom = {
        "head": 13.5 * PT / UNIT,
        "line": 8.4 * PT / UNIT,
        "out": 11.5 * PT / UNIT,
        "pad": 3.5 * PT / UNIT,
    }
    box_h = geom["head"] + 2 * geom["pad"] + n_lines * geom["line"] + geom["out"] + geom["pad"]
    diag_h_in = box_h * UNIT

    # Bottom-row heights, in inches
    panel_h = 1.50
    panel_bottom = 0.34  # room for the x axis and its label
    title_gap = 0.16  # gap between the panel title and the axes
    title_h = 0.13
    top_margin = 0.05
    gap_diag_panels = 0.16

    fig_h = panel_bottom + panel_h + title_gap + title_h + gap_diag_panels + diag_h_in + top_margin
    fig = plt.figure(figsize=(FIG_W, fig_h))

    def fy(inches: float) -> float:  # inches -> vertical figure fraction
        return inches / fig_h

    def fx(inches: float) -> float:  # inches -> horizontal figure fraction
        return inches / FIG_W

    # ---------------- Top row: block diagram ----------------
    diag_bottom = fig_h - top_margin - diag_h_in
    axd = fig.add_axes([fx(MARGIN_X), fy(diag_bottom), fx(DIAG_W), fy(diag_h_in)])
    axd.set_xlim(0, 100)
    axd.set_ylim(0, box_h)
    axd.axis("off")

    n = len(stages)
    gap = 2.2
    w = (100 - gap * (n - 1)) / n
    to_fit: List[tuple] = []
    for i, stage in enumerate(stages):
        x0 = i * (w + gap)
        draw_stage(axd, x0, 0, w, box_h, stage, wrapped[i], geom, to_fit)
        if i < n - 1:
            draw_arrow(axd, x0 + w + 0.2, box_h / 2, x0 + w + gap - 0.2, box_h / 2)
    shrink_to_fit(fig, axd, to_fit)

    # ---------------- Bottom row: example with real data ----------------
    n_show = 240  # seconds shown in panel (a)
    offsets = {"youtube": 900, "twitch": 900, "tiktok": 900}

    services = [SERVICES[k] for k in sorted(SERVICES)]
    excerpts = {s["svc"]: load_excerpt(csv_dir, s["svc"], n_show, offsets[s["svc"]]) for s in services}

    # The highlighted window is centred on a peak of the YouTube trace so that the
    # zoom shows the bursty download pattern rather than an idle interval.
    thr_ref = excerpts["youtube"]["throughput_Mbps"].values
    margin = LOOKBACK
    peak = int(np.argmax(thr_ref[margin : n_show - margin])) + margin
    win_start = int(np.clip(peak - LOOKBACK // 2, 0, n_show - LOOKBACK))

    p_bottom = fy(panel_bottom)
    p_height = fy(panel_h)
    title_y = fy(panel_bottom + panel_h + title_gap)

    # (a) three stacked mini-axes, one per service
    left_a, width_a = fx(0.42), fx(2.50)
    sub_h = p_height / 3
    axes_a = []
    for i, s in enumerate(services):
        ax = fig.add_axes([left_a, p_bottom + (2 - i) * sub_h, width_a, sub_h * 0.88])
        df = excerpts[s["svc"]]
        y = df["throughput_Mbps"].values
        x = np.arange(len(y))
        ax.fill_between(x, 0, y, color=s["color"], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=s["color"], linewidth=0.6)
        ax.axvspan(win_start, win_start + LOOKBACK, color=ACCENT, alpha=0.18, linewidth=0, zorder=0)
        ymax = float(y.max())
        ax.set_xlim(0, n_show)
        ax.set_ylim(0, ymax * 1.35)
        ax.set_yticks([0, round(ymax)])
        ax.text(
            0.99,
            0.90,
            s["label"],
            transform=ax.transAxes,
            fontsize=6,
            color=s["color"],
            fontweight="bold",
            ha="right",
            va="top",
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i < 2:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel(t["xlabel_time"], labelpad=1.5)
        axes_a.append(ax)
    axes_a[1].set_ylabel(t["ylabel_thr"], labelpad=2)
    fig.text(
        left_a + width_a / 2,
        title_y,
        t["panel_a"],
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=INK,
        fontweight="bold",
    )

    # (b) zoom on the sliding window: both features, unscaled
    left_b, width_b = fx(3.52), fx(1.20)
    axb = fig.add_axes([left_b, p_bottom, width_b, p_height * 0.96])
    df = excerpts["youtube"]
    win = df.iloc[win_start : win_start + LOOKBACK]
    xw = np.arange(LOOKBACK)
    thr_color = SERVICES[0]["color"]
    axb.step(xw, win["throughput_Mbps"].values, where="mid", color=thr_color, linewidth=0.9)
    axb.set_xlim(-0.5, LOOKBACK - 0.5)
    axb.set_xticks([0, 5, 10, 15, 19])
    axb.set_xticklabels(["1", "6", "11", "16", "20"])
    axb.set_xlabel(t["xlabel_time"], labelpad=1.5)
    axb.set_ylabel(t["ylabel_thr"], labelpad=2, color=thr_color)
    axb.tick_params(axis="y", colors=thr_color)
    axb.spines["top"].set_visible(False)
    axb.spines["left"].set_color(thr_color)

    axb2 = axb.twinx()
    axb2.step(
        xw,
        win[TIME_DELTA_FIELD].values * 1e3,
        where="mid",
        color=MUTED,
        linewidth=0.8,
        linestyle=(0, (2.5, 1.5)),
    )
    axb2.set_ylabel(t["legend_iat"], labelpad=2, color=MUTED)
    axb2.tick_params(axis="y", colors=MUTED)
    axb2.spines["top"].set_visible(False)
    axb2.spines["right"].set_color(MUTED)
    axb2.spines["left"].set_color(thr_color)
    fig.text(
        left_b + width_b / 2,
        title_y,
        t["panel_b"],
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=INK,
        fontweight="bold",
    )

    # (c) normalised 20 x 2 matrix fed to the model
    left_c, width_c = fx(5.36), fx(0.42)
    axc = fig.add_axes([left_c, p_bottom, width_c, p_height * 0.96])
    raw = win[[TIME_DELTA_FIELD, "throughput_Mbps"]].values.astype(float)
    scaled, used_saved = apply_scaler(raw, scaler_path)
    im = axc.imshow(scaled, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axc.set_xticks([0, 1])
    axc.set_xticklabels(t["feat_labels"], fontsize=5.8)
    axc.set_yticks([0, 9, 19])
    axc.set_yticklabels(["1", "10", "20"])
    axc.tick_params(length=1.5, pad=1.5)
    for side in axc.spines.values():
        side.set_linewidth(0.5)
        side.set_color(EDGE)
    fig.text(
        left_c + width_c / 2,
        title_y,
        t["panel_c"],
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=INK,
        fontweight="bold",
    )

    cax = fig.add_axes([left_c + width_c + fx(0.05), p_bottom, fx(0.05), p_height * 0.96])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(length=1.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(EDGE)

    # Final block: recurrent model and output
    left_m, width_m = fx(6.14), fx(0.96)
    axm = fig.add_axes([left_m, p_bottom, width_m, p_height * 0.96])
    axm.set_xlim(0, 1)
    axm.set_ylim(0, 1)
    axm.axis("off")
    axm.add_patch(
        FancyBboxPatch(
            (0.10, 0.42),
            0.80,
            0.28,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=0.6,
            edgecolor=ACCENT,
            facecolor=FILL_ALT,
            zorder=2,
        )
    )
    axm.text(
        0.5,
        0.56,
        t["model"],
        ha="center",
        va="center",
        fontsize=6,
        color=ACCENT,
        fontweight="bold",
        linespacing=1.3,
        zorder=3,
    )
    draw_arrow(axm, 0.5, 0.40, 0.5, 0.28)
    axm.text(0.5, 0.21, t["class_out"], ha="center", va="center", fontsize=6.2, color=INK)

    # Connecting arrows between panels, on a transparent figure-level axes
    axl = fig.add_axes([0, 0, 1, 1], zorder=8)
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.axis("off")
    axl.patch.set_alpha(0)
    ymid = p_bottom + p_height / 2
    draw_arrow(axl, left_a + width_a + fx(0.06), ymid, left_b - fx(0.36), ymid)
    draw_arrow(axl, left_b + width_b + fx(0.42), ymid, left_c - fx(0.16), ymid)
    draw_arrow(axl, left_c + width_c + fx(0.22), ymid, left_m - fx(0.02), ymid)

    return fig, used_saved


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Data-pipeline figure for the paper.")
    p.add_argument(
        "--lang",
        choices=list(LANGUAGES),
        default="en",
        help="Language of the figure labels (only 'en'; kept for compatibility).",
    )
    p.add_argument(
        "--font",
        choices=["sans", "serif"],
        default="sans",
        help="Font family; use serif to match TikZ figures typeset with newtxtext (Times).",
    )
    p.add_argument(
        "--csv-dir",
        default=os.path.join(ROOT_DIR, "data", "csv_throughput_only_ue"),
        help="Directory with the training CSV files, one sub-directory per service.",
    )
    p.add_argument(
        "--test-csv-dir",
        default=os.path.join(ROOT_DIR, "data", "csv_throughput_pruebas_ue"),
        help="Directory with the test CSV files, one sub-directory per service.",
    )
    p.add_argument(
        "--scaler",
        default=os.path.join(ROOT_DIR, "models", "scaler_1.0-0.0-0.0.pkl"),
        help="MinMaxScaler saved during training.",
    )
    p.add_argument(
        "--output",
        default=os.path.join(DOCS_FIG_DIR, "data_pipeline"),
        help="Output path without extension (a .pdf and a .png are written).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point. Writes ``<output>_<lang>[_serif].{pdf,png}``."""
    args = parse_args(argv)
    matplotlib.use("Agg")
    apply_style(args.font)
    fig, used_saved = build_figure(args.csv_dir, args.test_csv_dir, args.scaler, args.lang)

    out = f"{args.output}_{args.lang}"
    if args.font == "serif":
        out += "_serif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=600)
    plt.close(fig)

    logger.info(f"Figure saved to {out}.pdf and {out}.png")
    if not used_saved:
        logger.warning("Training scaler not found; panel (c) uses a MinMax fitted to the shown excerpt.")


if __name__ == "__main__":
    main()
