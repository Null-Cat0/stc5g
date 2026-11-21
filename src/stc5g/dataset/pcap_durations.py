"""Compute the total traffic duration of the pcap captures with ``capinfos``.

All ``.pcap`` files under one or more directories are scanned, their durations are
read with ``capinfos`` and aggregated per service (``youtube``, ``twitch``, ``tiktok``,
or the first sub-directory name otherwise). A summary table and a per-file table are
printed; optionally a donut chart of the durations is saved.

Example:
    python -m stc5g.dataset.pcap_durations --dir data/pcap --plot docs/figures/durations.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

from stc5g.config import PCAP_ROOT, SERVICES
from stc5g.log import get_logger

logger = get_logger(__name__)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``Hh Mm Ss`` (leading zero units omitted)."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_val = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds_val}s")

    return " ".join(parts)


def get_pcap_files(root_dir: str) -> List[str]:
    """Return the sorted paths of all ``.pcap`` files under ``root_dir``.

    Broken symbolic links are reported and skipped.
    """
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pcap") and not fname.endswith(".corrupt.backup"):
                path = os.path.join(dirpath, fname)
                if os.path.exists(path):
                    pcap_files.append(path)
                else:
                    logger.warning(f"File/link does not exist or is broken: {path}")
    return sorted(pcap_files)


def analyze_durations(
    pcap_files: List[str], root_dirs: Optional[List[str]] = None
) -> Tuple[Dict[str, dict], List[dict]]:
    """Run capinfos on the captures and aggregate the durations per service.

    The service is detected from the keywords ``youtube``/``twitch``/``tiktok`` in the
    path; otherwise it is the first path component relative to one of ``root_dirs``,
    and as a last resort the name of the parent directory.

    Args:
        pcap_files: Captures to analyse.
        root_dirs: Root directories used for the fallback service detection.

    Returns:
        ``(summary, details)``: ``summary[service]`` has ``count``, ``duration`` (s) and
        ``size_gb``; ``details`` has one dict per file (``path``, ``filename``,
        ``service``, ``duration``, ``size_gb``, ``start``, ``end``).
    """
    if not pcap_files:
        return {}, []

    cmd = ["capinfos", "-aeu", "-T", "-m", "-r"] + pcap_files

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running capinfos: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("'capinfos' not found. Make sure Wireshark/tshark is installed.")
        sys.exit(1)

    details = []
    summary_by_service: Dict[str, dict] = {}

    # Header-less CSV: <pcap_path>,<duration_seconds>,<start_time>,<end_time>
    reader = csv.reader(result.stdout.strip().splitlines())
    for row in reader:
        if len(row) < 2:
            continue
        pcap_path = row[0]
        try:
            duration = float(row[1])
        except ValueError:
            duration = 0.0

        start_time = row[2] if len(row) > 2 else "N/A"
        end_time = row[3] if len(row) > 3 else "N/A"

        pcap_path_lower = pcap_path.lower()
        if "youtube" in pcap_path_lower:
            service = "youtube"
        elif "twitch" in pcap_path_lower:
            service = "twitch"
        elif "tiktok" in pcap_path_lower:
            service = "tiktok"
        else:
            # Fall back to the first folder of the path relative to one of the roots.
            service = "unknown"
            if root_dirs:
                for r_dir in root_dirs:
                    try:
                        rel_path = os.path.relpath(pcap_path, r_dir)
                        path_parts = rel_path.split(os.sep)
                        if len(path_parts) > 1 and path_parts[0] != "..":
                            service = path_parts[0]
                            break
                    except ValueError:
                        continue
            if service == "unknown":
                parent_dir = os.path.basename(os.path.dirname(pcap_path))
                service = parent_dir if parent_dir else "unknown"

        file_size_bytes = os.path.getsize(pcap_path)
        file_size_gb = file_size_bytes / (1024**3)

        info = {
            "path": pcap_path,
            "filename": os.path.basename(pcap_path),
            "service": service,
            "duration": duration,
            "size_gb": file_size_gb,
            "start": start_time,
            "end": end_time,
        }
        details.append(info)

        if service not in summary_by_service:
            summary_by_service[service] = {
                "count": 0,
                "duration": 0.0,
                "size_gb": 0.0,
            }

        summary_by_service[service]["count"] += 1
        summary_by_service[service]["duration"] += duration
        summary_by_service[service]["size_gb"] += file_size_gb

    return summary_by_service, details


_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#e377c2",
    "#7f7f7f",
]
_FONT = "DejaVu Serif"


def plot_durations(
    summary: Dict[str, dict],
    output_path: str,
    *,
    service_map: Optional[dict] = None,
    dpi: int = 300,
    title: str = "Dataset Duration Distribution",
    min_label_pct: float = 3.0,
) -> None:
    """Save a donut chart of the dataset durations (academic / poster style).

    Args:
        summary: ``{service: {"duration": <seconds>}}``.
        output_path: Destination path (``.pdf`` for vector, ``.png`` for raster).
        service_map: Optional ``{key: {"svc": name, "color": hex}}`` for custom colours.
        dpi: Output resolution (300 for print).
        title: Figure title.
        min_label_pct: Slices below this percentage get no callout label.
    """
    # -- data -------------------------------------------------------------------
    entries = sorted(
        ((svc, stats["duration"] / 3600.0) for svc, stats in summary.items() if stats.get("duration", 0) > 0),
        key=lambda x: x[1],
        reverse=True,
    )
    if not entries:
        logger.warning("plot_durations: no positive-duration entries — skipping.")
        return

    labels, sizes = zip(*entries)
    labels = [lb.capitalize() for lb in labels]
    sizes = list(sizes)
    smap = service_map or {}
    colors = []
    for i, lb in enumerate(labels):
        color = _PALETTE[i % len(_PALETTE)]
        for info in smap.values():
            if info.get("svc", "").lower() == lb.lower():
                color = info.get("color", color)
                break
        colors.append(color)

    total_hours = sum(sizes)
    percentages = [100 * s / total_hours for s in sizes]
    total_seconds = int(round(total_hours * 3600))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    total_str = f"{h}h {m:02d}m {s:02d}s"

    # -- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    ax.set_facecolor("white")

    # Invisible wedges, only used to get the theta angles.
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.30, "edgecolor": "none", "visible": False},
    )

    # Flat coloured wedges with thin white separators.
    for i, w in enumerate(wedges):
        ax.add_patch(
            Wedge(
                (0, 0),
                1.0,
                w.theta1,
                w.theta2,
                width=0.30,
                facecolor=colors[i],
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
        )

    # Centre annotation.
    ax.text(0, 0.13, "TOTAL", ha="center", va="center", family=_FONT, size=7.5, color="#555555")
    ax.text(
        0, -0.10, total_str, ha="center", va="center", family=_FONT, size=13, weight="bold", color="#111111"
    )

    # -- callout labels: leader line + "Service / XX.X%" ----------------------
    R_TIP = 1.04  # leader start (donut outer rim)
    R_ELBOW = 1.17  # elbow
    R_TEXT = 1.21  # text anchor

    for w, lb, pct, sz in zip(wedges, labels, percentages, sizes):
        if pct < min_label_pct:
            continue

        theta = np.radians((w.theta1 + w.theta2) / 2.0)
        ct, st = np.cos(theta), np.sin(theta)
        on_right = ct >= 0

        sz_h, sz_rem = divmod(int(round(sz * 3600)), 3600)
        sz_m, sz_s = divmod(sz_rem, 60)
        dur_str = f"{sz_h}h {sz_m:02d}m {sz_s:02d}s"

        ax.annotate(
            "",
            xy=(R_TIP * ct, R_TIP * st),
            xytext=(R_ELBOW * ct, R_ELBOW * st),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.55),
            zorder=4,
        )
        ax.text(
            R_TEXT * ct + (0.03 if on_right else -0.03),
            R_TEXT * st,
            f"{lb}\n{pct:.1f}%  ·  {dur_str}",
            ha="left" if on_right else "right",
            va="center",
            family=_FONT,
            size=7,
            color="#222222",
            linespacing=1.3,
            zorder=4,
        )

    # Title and clean-up.
    ax.set_title(title, pad=10, family=_FONT, size=10, weight="bold", color="#111111")
    ax.axis("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.50, 1.50)
    ax.axis("off")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.10)
    plt.close(fig)
    logger.info("Duration chart saved → %s", output_path)


def print_report(summary: Dict[str, dict], details: List[dict]) -> None:
    """Print the per-service summary table and the per-file detail table."""
    print("\n" + "=" * 80)
    print("                      TRAFFIC SUMMARY BY SERVICE")
    print("=" * 80)
    print(f"{'Service':<15} | {'No. files':<12} | {'Total size':<13} | {'Duration (s)':<18} | {'Duration'}")
    print("-" * 80)

    tot_files = 0
    tot_size = 0.0
    tot_duration = 0.0

    for service, stats in sorted(summary.items()):
        formatted = format_duration(stats["duration"])
        print(
            f"{service:<15} | {stats['count']:<12} | {stats['size_gb']:>10.2f} GB | {stats['duration']:>17.2f}s | {formatted}"
        )
        tot_files += stats["count"]
        tot_size += stats["size_gb"]
        tot_duration += stats["duration"]

    print("-" * 80)
    formatted_tot = format_duration(tot_duration)
    print(
        f"{'TOTAL':<15} | {tot_files:<12} | {tot_size:>10.2f} GB | {tot_duration:>17.2f}s | {formatted_tot}"
    )
    print("=" * 80)

    print("\n" + "=" * 80)
    print("                      DETAIL PER PCAP FILE")
    print("=" * 80)
    for service in sorted(summary.keys()):
        print(f"\nService: {service.upper()}")
        print("-" * 80)
        print(f"{'File':<50} | {'Size':<10} | {'Duration':<12} | {'Time range'}")
        print("-" * 80)
        for item in details:
            if item["service"] == service:
                dur_str = format_duration(item["duration"])
                time_range = (
                    f"{item['start'].split(' ')[1]} -> {item['end'].split(' ')[1]}"
                    if " " in item["start"]
                    else "N/A"
                )
                print(
                    f"{item['filename'][:50]:<50} | {item['size_gb']:>7.2f} GB | {dur_str:<12} | {time_range}"
                )
    print("=" * 80)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute the hours of traffic per service from the .pcap files."
    )
    parser.add_argument(
        "--dir",
        type=str,
        nargs="+",
        default=[PCAP_ROOT],
        help="pcap directories to scan (one or more).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="Path of the donut chart to save. If empty, no chart is produced.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)

    pcap_files = []
    for d in args.dir:
        if not os.path.isdir(d):
            logger.error(f"Directory does not exist: {d}")
            sys.exit(1)
        logger.info(f"Looking for .pcap files in {d}...")
        pcap_files.extend(get_pcap_files(d))

    if not pcap_files:
        logger.warning("No .pcap files found.")
        return

    logger.info(f"Processing {len(pcap_files)} files with capinfos...")
    summary, details = analyze_durations(pcap_files, args.dir)

    print_report(summary, details)

    if args.plot:
        plot_durations(summary, args.plot, service_map=SERVICES)


if __name__ == "__main__":
    main()
