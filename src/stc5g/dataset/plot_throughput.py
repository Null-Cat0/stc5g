"""Plot the windowed throughput datasets (tshark CSV files) per service.

For a given window length, every labelled CSV under the input directory whose name
matches ``<a>_<b>_<c>_<d>_<window>s_tshark.csv`` (the fifth ``_``-separated token is
compared with ``<window>s``) is loaded and the following figures are written:

* ``<output-dir>/<file>_throughput_<window>s.png``: throughput of each file.
* ``<output-dir>/combined_throughput_<window>s.png``: all files stacked.
* ``<output-dir>/packets_and_bytes_<window>s.png``: packets and bytes per window.
* ``<output-dir>/../time_delta/<file>_time_delta_<window>s.png``: mean inter-packet time.
* ``<output-dir>/../time_delta/combined_time_delta_<window>s.png``.

Example:
    python -m stc5g.dataset.plot_throughput --window 1 \\
        --input-dir data/csv_throughput/1s --output-dir results/figures/dataset/1
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from stc5g.config import PCAP_ROOT, SERVICES, VIS_DIR
from stc5g.log import get_logger

logger = get_logger(__name__)

#: Number of samples of the centred moving average drawn over the raw series.
ROLLING_SAMPLES = 30

#: Class labels in class-index order, for the combined-figure titles ("YouTube / Twitch / TikTok").
_ALL_LABELS = " / ".join(SERVICES[c]["label"] for c in sorted(SERVICES))


def _mmss(x: float, _pos: object) -> str:
    """Tick formatter: seconds -> ``m:ss``."""
    return f"{int(x // 60)}:{int(x % 60):02d}"


def _time_delta_col(df: pd.DataFrame) -> str:
    """Name of the inter-packet time column (UE-only or all-traffic extraction)."""
    return "frame.time_delta_displayed" if "frame.time_delta_displayed" in df.columns else "frame.time_delta"


def _service_cfg(df: pd.DataFrame) -> dict:
    """Plot metadata (label, colour) for the class of a dataset."""
    cls = int(df["class"].iloc[0]) if "class" in df.columns else 0
    return SERVICES.get(cls, SERVICES[0])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tshark_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV written by :mod:`stc5g.dataset.pcap_reader` and add relative time.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        The table with an extra ``time_s`` column (seconds since the first window).
    """
    df = pd.read_csv(csv_path, parse_dates=["frame.time"])
    df["time_s"] = (df["frame.time"] - df["frame.time"].iloc[0]).dt.total_seconds()
    return df


def load_datasets(input_dir: str, window: str) -> List[pd.DataFrame]:
    """Load every labelled CSV of a given window length under ``input_dir``.

    Files are visited in ``os.walk`` order. Files without a ``class`` column or with a
    class not in :data:`stc5g.config.SERVICES` are skipped. The file name (without
    extension) is stored in ``df.attrs["file_name"]``.

    Args:
        input_dir: Root directory with the CSV files.
        window: Window length as written in the file names (e.g. ``"1"``, ``"0.5"``).

    Returns:
        The loaded tables.
    """
    datasets: List[pd.DataFrame] = []
    # Expected name: <capture>_<interface>_<x>_<y>_<window>s_tshark.csv
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv") and file.split("_")[4] == f"{window}s":
                df = load_tshark_csv(os.path.join(root, file))
                if "class" not in df.columns:
                    logger.warning(f"No 'class' column found in {file}")
                    continue
                cls = int(df["class"].iloc[0])
                if cls not in SERVICES:
                    continue

                cfg = SERVICES[cls]
                df.attrs["file_name"] = file.replace(".csv", "")
                datasets.append(df)

                logger.info(
                    f"{cfg['label']:>15s}: {len(df):>5d} samples, "
                    f"mean={df['throughput_Mbps'].mean():.2f} Mbps, "
                    f"max={df['throughput_Mbps'].max():.2f} Mbps"
                )
    return datasets


# ---------------------------------------------------------------------------
# Per-file figures
# ---------------------------------------------------------------------------
def plot_individual(df: pd.DataFrame, window_str: str, output_dir: str) -> str:
    """Plot the throughput of one file.

    Args:
        df: Table returned by :func:`load_tshark_csv`.
        window_str: Window length used in titles and file names.
        output_dir: Destination directory.

    Returns:
        Path of the PNG written.
    """
    cfg = _service_cfg(df)
    # Use the file name when available to avoid collisions between files.
    file_name = df.attrs.get("file_name", cfg["svc"])

    time_s = df["time_s"].values
    tp = df["throughput_Mbps"].values
    tp_smooth = pd.Series(tp).rolling(ROLLING_SAMPLES, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(time_s, tp, linewidth=0.6, alpha=0.45, color=cfg["color"], label="Throughput")
    ax.plot(
        time_s,
        tp_smooth,
        linewidth=2,
        color=cfg["color"],
        label=f"Moving average ({ROLLING_SAMPLES} samples)",
    )
    ax.fill_between(time_s, 0, tp_smooth, alpha=0.10, color=cfg["color"])

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Throughput (Mbps)", fontsize=12)
    ax.set_title(f"Throughput ({window_str}s) — {cfg['label']}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, time_s[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_mmss))

    stats_text = (
        f"Mean: {np.mean(tp):.2f} Mbps\n"
        f"Max: {np.max(tp):.2f} Mbps\n"
        f"Std: {np.std(tp):.2f} Mbps\n"
        f"Samples: {len(tp)}"
    )
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#ccc"),
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{file_name}_throughput_{window_str}s.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_individual_time_delta(df: pd.DataFrame, window_str: str, output_dir: str) -> str:
    """Plot the mean inter-packet time of one file.

    Args:
        df: Table returned by :func:`load_tshark_csv`.
        window_str: Window length used in titles and file names.
        output_dir: Destination directory.

    Returns:
        Path of the PNG written, or ``""`` if the table has no inter-packet time column.
    """
    cfg = _service_cfg(df)
    file_name = df.attrs.get("file_name", cfg["svc"])

    time_s = df["time_s"].values

    time_delta_col = _time_delta_col(df)
    if time_delta_col not in df.columns:
        logger.warning(f"No time-delta column found in the dataset for {file_name}")
        return ""

    td = df[time_delta_col].values
    td_smooth = pd.Series(td).rolling(ROLLING_SAMPLES, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(time_s, td, linewidth=0.6, alpha=0.45, color=cfg["color"], label="Time delta")
    ax.plot(
        time_s,
        td_smooth,
        linewidth=2,
        color=cfg["color"],
        label=f"Moving average ({ROLLING_SAMPLES} samples)",
    )
    ax.fill_between(time_s, 0, td_smooth, alpha=0.10, color=cfg["color"])

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Time delta (s)", fontsize=12)
    ax.set_title(f"Mean time delta ({window_str}s) — {cfg['label']}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, time_s[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_mmss))

    stats_text = (
        f"Mean: {np.mean(td):.4f} s\nMax: {np.max(td):.4f} s\nStd: {np.std(td):.4f} s\nSamples: {len(td)}"
    )
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#ccc"),
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{file_name}_time_delta_{window_str}s.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Combined figures
# ---------------------------------------------------------------------------
def plot_combined(datasets: List[pd.DataFrame], window_str: str, output_dir: str) -> str:
    """Plot the throughput of all files, one stacked panel each.

    Args:
        datasets: Tables returned by :func:`load_datasets`.
        window_str: Window length used in titles and file names.
        output_dir: Destination directory.

    Returns:
        Path of the PNG written.
    """
    fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 4.5 * len(datasets)), sharex=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, df in zip(axes, datasets):
        cfg = _service_cfg(df)
        time_s = df["time_s"].values
        tp = df["throughput_Mbps"].values
        tp_smooth = pd.Series(tp).rolling(ROLLING_SAMPLES, center=True, min_periods=1).mean().values

        ax.plot(time_s, tp, linewidth=0.5, alpha=0.35, color=cfg["color"])
        ax.plot(time_s, tp_smooth, linewidth=2, color=cfg["color"], label=cfg["label"])
        ax.fill_between(time_s, 0, tp_smooth, alpha=0.12, color=cfg["color"])

        ax.set_ylabel("Throughput (Mbps)", fontsize=11)
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold", color=cfg["color"])
        ax.set_xlim(0, time_s[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_mmss))

        stats_text = f"μ={np.mean(tp):.2f}  σ={np.std(tp):.2f}  max={np.max(tp):.1f} Mbps"
        ax.text(
            0.01,
            0.92,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc"),
        )

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    fig.suptitle(
        f"Throughput comparison ({window_str}s) — {_ALL_LABELS}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"combined_throughput_{window_str}s.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined_time_delta(datasets: List[pd.DataFrame], window_str: str, output_dir: str) -> str:
    """Plot the mean inter-packet time of all files, one stacked panel each.

    Args:
        datasets: Tables returned by :func:`load_datasets`.
        window_str: Window length used in titles and file names.
        output_dir: Destination directory.

    Returns:
        Path of the PNG written.
    """
    fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 4.5 * len(datasets)), sharex=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, df in zip(axes, datasets):
        cfg = _service_cfg(df)
        time_s = df["time_s"].values

        time_delta_col = _time_delta_col(df)
        if time_delta_col not in df.columns:
            continue

        td = df[time_delta_col].values
        td_smooth = pd.Series(td).rolling(ROLLING_SAMPLES, center=True, min_periods=1).mean().values

        ax.plot(time_s, td, linewidth=0.5, alpha=0.35, color=cfg["color"])
        ax.plot(time_s, td_smooth, linewidth=2, color=cfg["color"], label=cfg["label"])
        ax.fill_between(time_s, 0, td_smooth, alpha=0.12, color=cfg["color"])

        ax.set_ylabel("Time delta (s)", fontsize=11)
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold", color=cfg["color"])
        ax.set_xlim(0, time_s[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_mmss))

        stats_text = f"μ={np.mean(td):.4f}  σ={np.std(td):.4f}  max={np.max(td):.4f} s"
        ax.text(
            0.01,
            0.92,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc"),
        )

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    fig.suptitle(
        f"Time delta comparison ({window_str}s) — {_ALL_LABELS}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"combined_time_delta_{window_str}s.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_packets_and_bytes(datasets: List[pd.DataFrame], window_str: str, output_dir: str) -> str:
    """Plot the packet count and the kilobytes per window of every file.

    The packet count is read from the ``frame.time_epoch`` column (see
    :mod:`stc5g.dataset.pcap_reader`).

    Args:
        datasets: Tables returned by :func:`load_datasets`.
        window_str: Window length used in titles and file names.
        output_dir: Destination directory.

    Returns:
        Path of the PNG written.
    """
    n = len(datasets)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n), sharex=False)

    if n == 1:  # a single row: wrap so that axes[0][j] works below
        axes = [axes]

    for i, df in enumerate(datasets):
        cfg = _service_cfg(df)
        time_s = df["time_s"].values
        packets = df["frame.time_epoch"].values  # packet count per window
        total_bytes = df["frame.len"].values

        # Packets per window
        ax_pkt = axes[i][0] if n > 1 else axes[0][0]
        ax_pkt.plot(time_s, packets, linewidth=1.5, alpha=0.85, color=cfg["color"])
        ax_pkt.fill_between(time_s, 0, packets, alpha=0.10, color=cfg["color"])
        ax_pkt.set_ylabel(f"Packets / {window_str}s", fontsize=11)
        ax_pkt.set_title(f"Packets — {cfg['label']}", fontsize=13, fontweight="bold", color=cfg["color"])
        ax_pkt.set_xlim(0, time_s[-1])
        ax_pkt.set_ylim(bottom=0)
        ax_pkt.grid(True, alpha=0.25)
        ax_pkt.xaxis.set_major_formatter(ticker.FuncFormatter(_mmss))

        # Bytes per window
        ax_bytes = axes[i][1] if n > 1 else axes[0][1]
        bytes_kb = total_bytes / 1024
        ax_bytes.plot(time_s, bytes_kb, linewidth=1.5, alpha=0.85, color=cfg["color"])
        ax_bytes.fill_between(time_s, 0, bytes_kb, alpha=0.10, color=cfg["color"])
        ax_bytes.set_ylabel(f"KB / {window_str}s", fontsize=11)
        ax_bytes.set_title(f"Bytes — {cfg['label']}", fontsize=13, fontweight="bold", color=cfg["color"])
        ax_bytes.set_xlim(0, time_s[-1])
        ax_bytes.set_ylim(bottom=0)
        ax_bytes.grid(True, alpha=0.25)
        ax_bytes.xaxis.set_major_formatter(ticker.FuncFormatter(_mmss))

    fig.suptitle(f"Packets and bytes per {window_str}s window", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"packets_and_bytes_{window_str}s.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot the windowed throughput datasets (tshark) per service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--window",
        type=str,
        default="1",
        help="Window length as written in the CSV file names (e.g. '1', '0.5', '5', '10').",
    )
    parser.add_argument(
        "--input-dir",
        default=PCAP_ROOT,
        help="Input directory with the generated CSV files (default: data/pcap).",
    )
    parser.add_argument(
        "--output-dir",
        default=VIS_DIR,
        help="Output directory for the figures.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    window = args.window
    datasets = load_datasets(args.input_dir, window)

    if not datasets:
        logger.error("No datasets found. Run the pcap -> CSV conversion first.")
        sys.exit(1)

    logger.info("--- Per-file figures ---")
    for df in datasets:
        path = plot_individual(df, window, output_dir)
        logger.info(f"Saved: {path}")

    logger.info("--- Combined figure ---")
    path = plot_combined(datasets, window, output_dir)
    logger.info(f"Saved: {path}")

    logger.info("--- Packets and bytes ---")
    path = plot_packets_and_bytes(datasets, window, output_dir)
    logger.info(f"Saved: {path}")

    logger.info("--- Time delta ---")
    time_delta_dir = os.path.join(os.path.dirname(output_dir), "time_delta")
    os.makedirs(time_delta_dir, exist_ok=True)

    for df in datasets:
        path_td = plot_individual_time_delta(df, window, time_delta_dir)
        if path_td:
            logger.info(f"Saved: {path_td}")

    path_combined = plot_combined_time_delta(datasets, window, time_delta_dir)
    logger.info(f"Saved: {path_combined}")

    logger.info(f"Figures saved to: {output_dir} and {time_delta_dir}")


if __name__ == "__main__":
    main()
