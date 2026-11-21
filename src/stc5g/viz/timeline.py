#!/usr/bin/env python3
"""Prediction timelines: throughput over time with the model's predictions on top.

One independent figure is written per evaluated capture (CSV file):
``timeline_<csv name>.{pdf,png}`` in the output directory. Optional variants colour the
samples by predicted service (``timeline_pred_*``) or zoom into a short stretch of one
capture (``timeline_zoom<N>s_*`` / ``timeline_bursts_*``).

Standalone usage::

    python -m stc5g.viz.timeline --model models/classifier_lstm.pt \\
        --scaler models/scaler_0.7-0.2-0.1.pkl --data data/csv_throughput_test

From the training entry point (``stc5g.training.train --timeline``)::

    from stc5g.viz.timeline import apply_style, evaluate_timeline, plot_timeline
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from stc5g.config import CLASS_MAP, DROPPED_COLUMNS, FIG_TIMELINE_DIR, LOOKBACK, SERVICES
from stc5g.data.loading import make_sequences
from stc5g.device import select_device
from stc5g.log import get_logger
from stc5g.models.registry import load_checkpoint

logger = get_logger(__name__)

#: Columns that are not model features (the label and the raw tshark fields).
DROP_COLS = DROPPED_COLUMNS + ["class"]

# IEEE/Elsevier double-column width: the figure goes into the paper at 1:1 scale, so the
# font size matches the body text and there is no blurry rescaling.
FIG_WIDTH_IN = 7.16
FIG_HEIGHT_IN = 2.5

# Default formats: vector PDF for the paper, PNG for a quick look.
DEFAULT_FORMATS = ("pdf", "png")

INK = "#1c1f26"  # text colour: typographic black, not pure black


def apply_style(base_fontsize: float = 11, font: str = "serif") -> None:
    """Academic matplotlib style for the timelines (same spirit as the latency figures).

    Serif by default to match the paper's TikZ figures, typeset with newtxtext (Times).
    Type 42 fonts keep the PDF text selectable.

    Args:
        base_fontsize: Base font size in points.
        font: ``"serif"`` or ``"sans-serif"``.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": font,
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stix" if font == "serif" else "dejavusans",
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize + 1,
            "axes.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize - 1,
            "xtick.labelsize": base_fontsize - 1,
            "ytick.labelsize": base_fontsize - 1,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _despine(ax, keep: Iterable[str] = ("left", "bottom")) -> None:
    """Hide every spine of ``ax`` not listed in ``keep``."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _scale(figsize: Sequence[float]) -> float:
    """Scale factor relative to the reference (double-column) width.

    Marker sizes are given in points, i.e. they are absolute: when the figure is made
    narrower the same thousands of samples fall into half the width and the dots merge into
    a blob. Scaling them with the width keeps the density of a single-column figure.
    """
    return min(figsize[0] / FIG_WIDTH_IN, 1.0)


def _legend_above(ax, ncol: int, marker_pt: float = 9):
    """Single-row legend above the axes.

    At paper size there is no free space inside the data, and the scatter markers (1-2 pt)
    would be invisible in the legend: their size is fixed here instead of using
    ``markerscale``, which makes the 'x' too large.
    """
    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=ncol,
        borderaxespad=0,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    for handle in leg.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([marker_pt])
    return leg


def _save(fig, out_base: str, formats: Sequence[str] = DEFAULT_FORMATS) -> Optional[str]:
    """Save ``fig`` in every requested format and close it.

    Scatter points are rasterised (``rasterized=True`` where they are drawn): with thousands
    of samples a fully vector PDF would be too heavy, while text and axes stay vector and
    sharp at any zoom.

    Returns:
        The path of the first format (the "main" one), or ``None`` if ``formats`` is empty.
    """
    paths = []
    for fmt in formats:
        path = f"{out_base}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths[0] if paths else None


def _load_and_evaluate_csv(model, filepath: str, scaler, device, lookback: int) -> List[dict]:
    """Load one CSV, run the model on its sliding windows and return per-sample results.

    Returns:
        One dict per window (see :func:`evaluate_timeline` for the keys); an empty list if
        the file has no ``class`` column, no timestamp column, no feature columns or not
        more than ``lookback`` valid rows.
    """
    df = pd.read_csv(filepath)

    if "class" not in df.columns:
        logger.warning(f"No 'class' column: {filepath}")
        return []

    # Pick the available timestamp column
    if "frame.time" in df.columns:
        timestamps = pd.to_datetime(df["frame.time"]).values
    elif "frame.time_epoch" in df.columns:
        timestamps = pd.to_datetime(df["frame.time_epoch"], unit="s").values
    else:
        logger.warning(f"No timestamp column: {filepath}")
        return []

    throughput = df["throughput_Mbps"].values.astype(np.float32)

    drop_cols = [c for c in DROP_COLS if c in df.columns]
    drop_cols += [c for c in df.columns if "unnamed" in c.lower()]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    if len(feature_cols) == 0:
        logger.warning(f"No feature columns left after cleaning: {filepath}")
        return []

    y_raw = df["class"].values.astype(int)
    x_raw = df[feature_cols].values.astype(np.float32)

    nan_mask = ~np.isnan(x_raw).any(axis=1)
    if not nan_mask.all():
        logger.warning(f"{filepath}: dropped {(~nan_mask).sum()} rows with NaN")
        x_raw = x_raw[nan_mask]
        y_raw = y_raw[nan_mask]
        timestamps = timestamps[nan_mask]
        throughput = throughput[nan_mask]

    if len(x_raw) <= lookback:
        logger.warning(f"{filepath}: not enough data ({len(x_raw)} rows, lookback={lookback})")
        return []

    x_scaled = scaler.transform(x_raw)

    x_seq, y_seq = make_sequences(x_scaled, y_raw, lookback)
    seq_timestamps = timestamps[lookback:]
    seq_throughput = throughput[lookback:]

    if len(x_seq) == 0:
        return []

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(device)
        outputs = model(x_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    service_name = os.path.basename(os.path.dirname(filepath))

    return [
        {
            "timestamp": seq_timestamps[i],
            "throughput_mbps": float(seq_throughput[i]),
            "y_true": int(y_seq[i]),
            "y_pred": int(preds[i]),
            "correct": int(y_seq[i]) == int(preds[i]),
            "service": service_name,
            "file": os.path.basename(filepath),
        }
        for i in range(len(preds))
    ]


def evaluate_timeline(
    model: torch.nn.Module,
    csv_dir: str,
    scaler_path: str,
    device: Union[str, torch.device],
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    """Evaluate ``model`` on every CSV under ``csv_dir``, keeping the timestamps.

    Each file is windowed on its own (windows never cross files here). The service name
    is the name of the CSV's parent directory.

    Args:
        model: Trained PyTorch model (put in eval mode).
        csv_dir: Directory with the original CSV files (they keep ``frame.time``).
        scaler_path: Path to the pickled ``MinMaxScaler``.
        device: Compute device.
        lookback: Sequence window length.

    Returns:
        DataFrame with columns ``timestamp, throughput_mbps, y_true, y_pred, correct,
        service, file`` (empty if no file could be evaluated).
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    all_rows = []

    for root, _, files in os.walk(csv_dir):
        for fname in sorted(files):
            if not fname.endswith(".csv"):
                continue

            filepath = os.path.join(root, fname)
            rows = _load_and_evaluate_csv(model, filepath, scaler, device, lookback)
            all_rows.extend(rows)

    if not all_rows:
        logger.warning(f"No CSV files could be evaluated in {csv_dir}")
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def _figure_name(svc_name: str, fname: str, used: Set[str], prefix: str = "timeline") -> str:
    """File name (WITHOUT extension) for one capture: ``<prefix>_<csv name without extension>``.

    CSV names usually start with the service already (``youtube_...``, ``twitch_...``), so
    the service is only prepended when that is not the case or when two services share a
    file name. ``used`` is updated with the returned name.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(fname)[0]).strip("_")
    svc = re.sub(r"[^A-Za-z0-9._-]+", "_", svc_name).strip("_")

    name = f"{prefix}_{stem}"
    if not stem.lower().startswith(svc.lower()) or name in used:
        name = f"{prefix}_{svc}_{stem}"
    n = 2
    while name in used:  # two CSV files with the same name within the same service
        name = f"{prefix}_{svc}_{stem}_{n}"
        n += 1

    used.add(name)
    return name


def _elapsed_axis(times) -> Tuple[np.ndarray, str]:
    """Convert a series of timestamps into elapsed time since the first sample.

    The wall-clock time at which a capture was taken says nothing about the traffic and
    prevents comparing two captures: with elapsed time, figures of different services share
    the origin and can be read side by side.

    Returns:
        ``(x, axis label)`` with ``x`` in minutes if the span exceeds 10 min, else in seconds.
    """
    t = pd.to_datetime(pd.Series(np.asarray(times))).reset_index(drop=True)
    secs = (t - t.iloc[0]).dt.total_seconds().values
    span = float(secs[-1] - secs[0]) if len(secs) > 1 else 0.0

    if span >= 600:
        return secs / 60.0, "Elapsed time (min)"
    return secs, "Elapsed time (s)"


def _label_segments(x, labels) -> List[Tuple[float, float, int]]:
    """Compress a per-sample label series into contiguous segments.

    Args:
        x: Axis positions (as returned by :func:`_elapsed_axis`), not timestamps.
        labels: Label of every sample.

    Returns:
        ``[(x_start, width, label), ...]``, ready for ``broken_barh``.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return []

    # Width of each sample: up to the next one (the last one repeats the previous step)
    edges = np.empty(len(x) + 1)
    edges[1:-1] = (x[1:] + x[:-1]) / 2.0
    step = np.median(np.diff(x)) if len(x) > 1 else 1.0
    edges[0] = x[0] - step / 2.0
    edges[-1] = x[-1] + step / 2.0

    segments = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            segments.append((edges[start_idx], edges[i] - edges[start_idx], int(labels[start_idx])))
            start_idx = i

    return segments


def plot_timeline(
    results_df: pd.DataFrame,
    output_dir: str,
    class_map: Dict[str, int] = CLASS_MAP,
    services: Dict[int, dict] = SERVICES,
    model_name: str = "LSTM",
    titles: bool = True,
    formats: Sequence[str] = DEFAULT_FORMATS,
    figsize: Tuple[float, float] = (FIG_WIDTH_IN, FIG_HEIGHT_IN),
) -> List[str]:
    """Plot one timeline per capture: throughput with correct (green) and wrong (red) predictions.

    Args:
        results_df: DataFrame returned by :func:`evaluate_timeline`.
        output_dir: Directory where the figures are saved.
        class_map: Service name -> class id.
        services: Per-class plotting metadata.
        model_name: Model name shown in the title.
        titles: ``False`` leaves the figure without title and subtitle (like ``--no-titles``
            in the latency figures): in the paper that goes into the caption.
        formats: Extensions to write (vector PDF + PNG by default).
        figsize: Size in inches; the default fits a double column at 1:1.

    Returns:
        Paths of the generated figures (first format), ordered by class and file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = list(results_df.groupby(["service", "file"], sort=False).size().index)
    files.sort(key=lambda x: (class_map.get(x[0], 99), x[1]))

    if not files:
        logger.warning("No data to plot.")
        return []

    out_paths = []
    used_names = set()

    for svc_name, fname in files:
        mask = (results_df["service"] == svc_name) & (results_df["file"] == fname)
        df = results_df[mask].sort_values("timestamp")

        if df.empty:
            continue

        cls_id = class_map.get(svc_name, 0)
        cfg = services.get(cls_id, services[0])

        correct = df[df["correct"]]
        incorrect = df[~df["correct"]]
        acc = len(correct) / max(len(df), 1) * 100

        fig, ax = plt.subplots(figsize=figsize)
        k = _scale(figsize)
        x, xlabel = _elapsed_axis(df["timestamp"].values)
        mask_ok = df["correct"].values

        ax.plot(
            x,
            df["throughput_mbps"],
            linewidth=0.4 * k,
            color="#9aa0a6",
            alpha=0.55,
            zorder=1,
            rasterized=True,
        )

        ax.scatter(
            x[mask_ok],
            correct["throughput_mbps"],
            c="#1a7f4b",
            s=1.6 * k,
            alpha=0.7,
            marker="o",
            linewidths=0,
            label=f"Correct ({len(correct)})",
            zorder=3,
            rasterized=True,
        )
        if len(incorrect) > 0:
            ax.scatter(
                x[~mask_ok],
                incorrect["throughput_mbps"],
                c="#c0392b",
                s=7 * k,
                alpha=0.95,
                marker="x",
                linewidths=0.5 * k,
                label=f"Misclassified ({len(incorrect)})",
                zorder=4,
                rasterized=True,
            )

        if titles:
            ax.set_title(
                f"{cfg['label']} — {fname}\n{model_name} | accuracy {acc:.1f}% ({len(correct)}/{len(df)})",
                color=INK,
                pad=6,
            )
        ax.set_ylabel("Throughput (Mbit/s)")
        ax.set_xlabel(xlabel)
        _legend_above(ax, ncol=2)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.set_ylim(0, max(float(df["throughput_mbps"].max()), 1e-6) * 1.08)
        ax.set_xlim(x[0], x[-1])
        _despine(ax)

        fig.tight_layout()

        out_base = os.path.join(output_dir, _figure_name(svc_name, fname, used_names))
        out_path = _save(fig, out_base, formats)
        out_paths.append(out_path)
        logger.info(f"  {cfg['label']:>8s} | Acc {acc:5.1f}% -> {os.path.basename(out_path)}")

    logger.info(f"{len(out_paths)} timelines saved to: {output_dir}")
    return out_paths


def plot_prediction_timeline(
    results_df: pd.DataFrame,
    output_dir: str,
    class_map: Dict[str, int] = CLASS_MAP,
    services: Dict[int, dict] = SERVICES,
    model_name: str = "LSTM",
    titles: bool = True,
    show_true: bool = True,
    formats: Sequence[str] = DEFAULT_FORMATS,
    figsize: Tuple[float, float] = (FIG_WIDTH_IN, FIG_HEIGHT_IN + 0.45),
) -> List[str]:
    """Like :func:`plot_timeline`, but colouring each sample by its PREDICTED service.

    One independent figure per capture, made of two parts:

    - the throughput, each sample drawn in the colour of its predicted class;
    - a lower band with the predicted segments (and, if ``show_true``, the true class below).

    Args:
        results_df: DataFrame returned by :func:`evaluate_timeline`.
        output_dir: Directory where the figures are saved.
        class_map: Service name -> class id.
        services: Per-class plotting metadata.
        model_name: Model name shown in the title.
        titles: Whether to draw the title.
        show_true: Add a band with the true class for an at-a-glance comparison.
        formats: Extensions to write (vector PDF + PNG by default).
        figsize: Size in inches; the default fits a double column at 1:1.

    Returns:
        Paths of the generated figures (first format), ordered by class and file name.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = list(results_df.groupby(["service", "file"], sort=False).size().index)
    files.sort(key=lambda x: (class_map.get(x[0], 99), x[1]))

    if not files:
        logger.warning("No data to plot.")
        return []

    class_ids = sorted(services.keys())
    out_paths = []
    used_names = set()

    for svc_name, fname in files:
        mask = (results_df["service"] == svc_name) & (results_df["file"] == fname)
        df = results_df[mask].sort_values("timestamp")

        if df.empty:
            continue

        cfg = services.get(class_map.get(svc_name, 0), services[0])
        acc = df["correct"].mean() * 100

        k = _scale(figsize)
        # Band height in inches, not as a proportion, so the bars are equally thick at any
        # size. Capped at 30% in case the figure is very short, where it would otherwise eat
        # the throughput panel.
        band_h = min((0.62 if show_true else 0.38), figsize[1] * 0.3)

        # constrained layout: tight_layout cannot place the legend outside an axes of a gridspec
        fig, (ax, ax_band) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            layout="constrained",
            height_ratios=[figsize[1] - band_h, band_h],
        )
        fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.02, wspace=0.0)

        x, xlabel = _elapsed_axis(df["timestamp"].values)

        ax.plot(
            x,
            df["throughput_mbps"],
            linewidth=0.4 * k,
            color="#9aa0a6",
            alpha=0.5,
            zorder=1,
            rasterized=True,
        )

        # Points coloured by predicted class
        for cid in class_ids:
            sel = (df["y_pred"] == cid).values
            sub = df[sel]
            if sub.empty:
                continue
            c = services[cid]
            ax.scatter(
                x[sel],
                sub["throughput_mbps"],
                c=c["color"],
                s=1.8 * k,
                alpha=0.8,
                marker="o",
                linewidths=0,
                edgecolors="none",
                zorder=3,
                rasterized=True,
                label=f"{c['label']} ({len(sub) / len(df) * 100:.1f}%)",
            )

        ax.set_ylabel("Throughput (Mbit/s)")
        _legend_above(ax, ncol=len(class_ids))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.set_ylim(0, max(float(df["throughput_mbps"].max()), 1e-6) * 1.08)
        ax.set_xlim(x[0], x[-1])
        _despine(ax, keep=("left", "bottom"))

        if titles:
            ax.set_title(
                f"{cfg['label']} — {fname}\n"
                f"{model_name} | accuracy {acc:.1f}% ({int(df['correct'].sum())}/{len(df)}) "
                f"| colour = predicted service",
                color=INK,
                pad=18,
            )

        # Segment band: prediction on top, ground truth below
        rows = [("Predicted", df["y_pred"].values, 0.55 if show_true else 0.3)]
        if show_true:
            rows.append(("True", df["y_true"].values, 0.1))

        for _, values, y0 in rows:
            for x0, width, cid in _label_segments(x, values):
                ax_band.broken_barh(
                    [(x0, width)],
                    (y0, 0.35),
                    facecolors=services.get(cid, services[0])["color"],
                    edgecolors="none",
                )

        ax_band.set_ylim(0, 1 if show_true else 0.75)
        ax_band.set_yticks([y0 + 0.175 for _, _, y0 in rows])
        ax_band.set_yticklabels([label for label, _, _ in rows])
        ax_band.set_xlabel(xlabel)
        ax_band.grid(False)
        ax_band.tick_params(axis="y", length=0)
        _despine(ax_band, keep=("bottom",))

        out_base = os.path.join(output_dir, _figure_name(svc_name, fname, used_names, prefix="timeline_pred"))
        out_path = _save(fig, out_base, formats)
        out_paths.append(out_path)
        logger.info(f"  {cfg['label']:>8s} | Acc {acc:5.1f}% -> {os.path.basename(out_path)}")

    logger.info(f"{len(out_paths)} predicted-class timelines saved to: {output_dir}")
    return out_paths


def _most_regular_window(
    values, n_samples: int, threshold: float = 1.0, valid=None, step: int = 5
) -> Tuple[int, float]:
    """Find the stretch of ``n_samples`` samples with the most regular bursts.

    Video traffic is bursty: the player requests a chunk, downloads it and waits. Each
    stretch is scored by the coefficient of variation of the periods between burst starts,
    and the most stable one wins: it shows the pattern best in a detail figure.

    Args:
        values: Throughput per sample.
        n_samples: Stretch length in samples.
        threshold: Throughput above which a sample is "on" (part of a burst).
        valid: Optional per-sample mask; only stretches where it is entirely true are
            considered. Used to request a stretch without any misclassified sample, so the
            figure shows the pattern without distracting errors.
        step: Stride between candidate stretch starts.

    Returns:
        ``(start index of the best stretch, cv)``. The index is 0 and the cv is ``inf`` if
        no valid stretch has at least 3 bursts.
    """
    on = np.asarray(values) > threshold
    valid = None if valid is None else np.asarray(valid, dtype=bool)
    best = (np.inf, 0)

    for i in range(0, max(len(on) - n_samples, 1), step):
        if valid is not None and not valid[i : i + n_samples].all():
            continue
        seg = on[i : i + n_samples]
        starts = np.flatnonzero(seg & ~np.r_[False, seg[:-1]])
        if len(starts) < 3:
            continue
        gaps = np.diff(starts)
        cv = gaps.std() / max(gaps.mean(), 1e-9)
        if cv < best[0]:
            best = (cv, i)

    return best[1], best[0]


def plot_zoom_timeline(
    results_df: pd.DataFrame,
    output_dir: str,
    file_filter: Optional[str] = None,
    start: Union[str, float, None] = None,
    seconds: Optional[float] = 300,
    class_map: Dict[str, int] = CLASS_MAP,
    services: Dict[int, dict] = SERVICES,
    model_name: str = "LSTM",
    titles: bool = True,
    formats: Sequence[str] = DEFAULT_FORMATS,
    figsize: Tuple[float, float] = (FIG_WIDTH_IN, 2.9),
    clean: bool = False,
) -> Optional[str]:
    """Zoom into a stretch of one capture, to show the pattern that is squashed at hour scale.

    Useful to explain confusions between services: YouTube alternates short bursts with
    pauses where the throughput drops to 0, and precisely those zero stretches reappear in
    captures of other services and get labelled as YouTube.

    Args:
        results_df: DataFrame returned by :func:`evaluate_timeline`.
        output_dir: Directory where the figure is saved.
        file_filter: Substring of the CSV name to zoom into (default: the first YouTube one).
        start: ``'HH:MM:SS'`` or seconds since the start of the capture. ``None`` picks the
            stretch with the most regular bursts.
        seconds: Stretch duration in seconds; ``<= 0`` or ``None`` plots the whole capture.
        class_map: Service name -> class id.
        services: Per-class plotting metadata.
        model_name: Model name shown in the title.
        titles: Whether to draw the title.
        formats: Extensions to write.
        figsize: Size in inches.
        clean: With ``start=None``, require every sample of the stretch to be correctly
            classified, so the figure shows the pattern without distracting errors.

    Returns:
        Path of the generated figure, or ``None`` if there is no data.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_all = results_df
    if file_filter:
        df_all = df_all[df_all["file"].str.contains(file_filter, case=False, regex=False)]
    else:
        df_all = df_all[df_all["service"].str.contains("youtube", case=False)]

    if df_all.empty:
        logger.warning(f"The zoom filter does not select any capture: {file_filter!r}")
        return None

    fname = sorted(df_all["file"].unique())[0]
    df_file = df_all[df_all["file"] == fname].sort_values("timestamp").reset_index(drop=True)
    svc_name = df_file["service"].iloc[0]
    cfg = services.get(class_map.get(svc_name, 0), services[0])

    times = pd.to_datetime(df_file["timestamp"])
    step_s = max(float(np.median(np.diff(times.values).astype("timedelta64[s]").astype(float))), 1e-9)

    # seconds <= 0 -> the whole capture, no stretch search
    full = seconds is None or seconds <= 0
    if full:
        n, i0, start = len(df_file), 0, 0
        seconds = n * step_s
    else:
        n = int(round(seconds / step_s))

    if start is None:
        i0, cv = _most_regular_window(
            df_file["throughput_mbps"].values,
            n,
            valid=df_file["correct"].values if clean else None,
        )
        if clean and not np.isfinite(cv):
            logger.warning(
                f"{fname}: no {seconds:.0f} s stretch is 100% correctly classified; "
                "using the most regular one even though it has errors"
            )
            i0, _ = _most_regular_window(df_file["throughput_mbps"].values, n)
    elif isinstance(start, str):
        t0 = pd.to_datetime(times.iloc[0].strftime("%Y-%m-%d ") + start)
        i0 = int(np.searchsorted(times.values, t0.to_datetime64()))
    else:
        i0 = int(round(float(start) / step_s))

    i0 = max(0, min(i0, max(len(df_file) - n, 0)))
    df = df_file.iloc[i0 : i0 + n]

    if df.empty:
        logger.warning("The zoom stretch falls outside the capture.")
        return None

    acc = df["correct"].mean() * 100

    fig, (ax, ax_band) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        layout="constrained",
        height_ratios=[figsize[1] - 0.62, 0.62],
    )
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.02, wspace=0.0)

    ymax = max(float(df["throughput_mbps"].max()), 1e-6) * 1.12
    x, xlabel = _elapsed_axis(df["timestamp"].values)

    # Samples per inch: the whole capture has 10-15x more than a 5-min stretch, and with the
    # same width THE STEMS merge into a solid block, so they are thinned with the density
    # (reference: the 300 s zoom, ~42 samples/inch). The dot is NOT changed: it sits on top
    # of the stem, does not smear, and shrinking it makes it illegible in print.
    density = len(df) / max(figsize[0], 1e-6)
    d = min(42.0 / max(density, 1e-6), 1.0)

    # Stems up to each sample: they show bursts and pauses much better than a line
    ax.vlines(x, 0, df["throughput_mbps"], color="#c3c9d2", linewidth=max(0.5 * d, 0.12), zorder=1)
    if d > 0.5:
        # When dense, the connecting line only fills the gap between bursts: it is omitted
        ax.plot(x, df["throughput_mbps"], color="#8b93a0", linewidth=0.45, alpha=0.8, zorder=2)

    for cid in sorted(services.keys()):
        sel = (df["y_pred"] == cid).values
        sub = df[sel]
        if sub.empty:
            continue
        c = services[cid]
        ax.scatter(
            x[sel],
            sub["throughput_mbps"],
            c=c["color"],
            s=5 * _scale(figsize),
            marker="o",
            linewidths=0,
            zorder=3,
            rasterized=True,
            label=f"{c['label']} ({len(sub) / len(df) * 100:.1f}%)",
        )

    ax.set_ylabel("Throughput (Mbit/s)")
    ax.set_ylim(0, ymax)
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    _despine(ax)
    _legend_above(ax, ncol=len(services), marker_pt=9)

    if titles:
        ax.set_title(
            f"{cfg['label']} — {fname}\n"
            f"{model_name} | {'full capture' if full else f'{seconds:.0f} s detail'} | "
            f"accuracy {acc:.1f}% ({int(df['correct'].sum())}/{len(df)})",
            color=INK,
            pad=18,
        )

    rows = [("Predicted", df["y_pred"].values, 0.55), ("True", df["y_true"].values, 0.1)]
    for _, values, y0 in rows:
        for x0, width, cid in _label_segments(x, values):
            ax_band.broken_barh(
                [(x0, width)],
                (y0, 0.35),
                facecolors=services.get(cid, services[0])["color"],
                edgecolors="none",
            )

    ax_band.set_ylim(0, 1)
    ax_band.set_yticks([y0 + 0.175 for _, _, y0 in rows])
    ax_band.set_yticklabels([label for label, _, _ in rows])
    ax_band.set_xlabel(xlabel)
    ax_band.grid(False)
    ax_band.tick_params(axis="y", length=0)
    _despine(ax_band, keep=("bottom",))
    # The duration goes into the name: otherwise two zooms of the same capture would overwrite each other
    prefix = "timeline_bursts" if full else f"timeline_zoom{seconds:.0f}s"
    out_base = os.path.join(output_dir, _figure_name(svc_name, fname, set(), prefix=prefix))
    out_path = _save(fig, out_base, formats)
    logger.info(
        f"  zoom {cfg['label']} | {times.iloc[i0]:%H:%M:%S} +{seconds:.0f}s "
        f"| Acc {acc:5.1f}% -> {os.path.basename(out_path)}"
    )
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line of the timeline tool."""
    parser = argparse.ArgumentParser(
        description="Plot timelines of the model's predictions on test captures."
    )
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--scaler", type=str, required=True, help="Fitted scaler (.pkl)")
    parser.add_argument("--data", type=str, required=True, help="Directory with the test CSV files")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: results/figures/timeline)"
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Sequence window length")
    parser.add_argument(
        "--no-titles",
        dest="titles",
        action="store_false",
        help="Figures without title or subtitle (the caption goes in the paper)",
    )
    parser.add_argument(
        "--by-class",
        dest="by_class",
        action="store_true",
        help="Timeline coloured by predicted service (timeline_pred_*)",
    )
    parser.add_argument(
        "--only-by-class",
        dest="only_by_class",
        action="store_true",
        help="Only plot the predicted-service timeline, not the correct/wrong one",
    )
    parser.add_argument(
        "--no-true-band",
        dest="show_true",
        action="store_false",
        help="With --by-class: omit the true-class band",
    )
    parser.add_argument(
        "--zoom",
        nargs="?",
        const="",
        default=None,
        metavar="CSV",
        help="Detail figure of a short stretch. Optionally, a substring of the CSV to zoom "
        "into (default: the first YouTube capture)",
    )
    parser.add_argument(
        "--zoom-start",
        type=str,
        default=None,
        metavar="HH:MM:SS|SECS",
        help="Start of the stretch; by default the one with the most regular bursts is chosen",
    )
    parser.add_argument(
        "--zoom-seconds",
        type=float,
        default=300,
        help="Duration of the zoomed stretch in seconds (default: 300; <= 0 = whole capture)",
    )
    parser.add_argument(
        "--zoom-clean",
        action="store_true",
        help="Pick a stretch where ALL samples are correctly classified (no effect with --zoom-start)",
    )
    parser.add_argument("--only-zoom", action="store_true", help="Only plot the detail figure")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        help="Output formats (pdf png svg...). PDF is vector: sharp at any zoom",
    )
    parser.add_argument(
        "--font",
        choices=["serif", "sans-serif"],
        default="serif",
        help="Font family; serif matches the TikZ figures typeset with newtxtext (Times)",
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=11,
        help="Base font size in pt (that of the figure at 1:1 scale in the paper)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=FIG_WIDTH_IN,
        help="Figure width in inches (default: IEEE double column)",
    )
    parser.add_argument("--height", type=float, default=FIG_HEIGHT_IN, help="Figure height in inches")
    parser.add_argument("--confusion", action="store_true", help="Also plot and save the confusion matrix")
    parser.add_argument(
        "--confusion-suffix",
        type=str,
        default="",
        help="Suffix for the confusion-matrix file name (implies --confusion; default: 'eval')",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point (``stc5g timeline``)."""
    args = parse_args(argv)

    if not os.path.isdir(args.data):
        logger.error(f"Directory not found: {args.data}")
        sys.exit(1)

    device = select_device("auto")
    logger.info(f"Device: {device}")

    model, checkpoint = load_checkpoint(args.model, device)
    model_type = checkpoint.get("model_type", "LSTM")

    logger.info(f"Evaluating CSV files in: {args.data}")
    df = evaluate_timeline(model, args.data, args.scaler, device, args.lookback)

    if df.empty:
        logger.error("No results. Check that the CSV files have timestamps and a 'class' column.")
        sys.exit(1)

    global_acc = df["correct"].mean() * 100
    logger.info(f"Overall accuracy: {global_acc:.2f}%  ({df['correct'].sum()}/{len(df)})")
    for (svc, fname), grp in df.groupby(["service", "file"], sort=False):
        acc = grp["correct"].mean() * 100
        label = SERVICES.get(CLASS_MAP.get(svc, -1), {}).get("label", svc)
        logger.info(f"  {label:>20s} | {fname:55s} | {acc:5.1f}%  ({grp['correct'].sum()}/{len(grp)})")

    output_dir = args.output or FIG_TIMELINE_DIR
    apply_style(base_fontsize=args.fontsize, font=args.font)
    only_zoom = args.only_zoom

    if not args.only_by_class and not only_zoom:
        plot_timeline(
            df,
            output_dir,
            model_name=model_type,
            titles=args.titles,
            formats=args.formats,
            figsize=(args.width, args.height),
        )
    if (args.by_class or args.only_by_class) and not only_zoom:
        plot_prediction_timeline(
            df,
            output_dir,
            model_name=model_type,
            titles=args.titles,
            show_true=args.show_true,
            formats=args.formats,
            figsize=(args.width, args.height + 0.45),
        )
    if args.zoom is not None or only_zoom:
        start = args.zoom_start
        if start is not None and ":" not in start:
            start = float(start)
        plot_zoom_timeline(
            df,
            output_dir,
            file_filter=args.zoom or None,
            start=start,
            seconds=args.zoom_seconds,
            model_name=model_type,
            titles=args.titles,
            formats=args.formats,
            figsize=(args.width, args.height + 0.85),
            clean=args.zoom_clean,
        )

    if args.confusion or args.confusion_suffix:
        from stc5g.viz.confusion import plot_confusion_matrix

        suffix = args.confusion_suffix if args.confusion_suffix else "eval"
        logger.info(f"Plotting the confusion matrix with suffix: {suffix}...")
        plot_confusion_matrix(
            y_true=df["y_true"].values,
            y_pred=df["y_pred"].values,
            class_map=CLASS_MAP,
            train_ratio=0.0,
            val_ratio=0.0,
            test_ratio=0.0,
            title_suffix=f"({suffix.upper()})",
            filename_suffix=suffix,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
