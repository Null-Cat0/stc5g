"""Convert pcap captures into windowed throughput CSV files with tshark.

Every ``.pcap`` file found (recursively) under the input directory is read with
``tshark`` and the per-packet records are aggregated into fixed time windows of
``--secs`` seconds. The sub-directory layout of the input (one folder per service,
e.g. ``youtube/``, ``twitch/``) is replicated under the output directory, and each
capture ``name.pcap`` becomes ``name_<secs>s_tshark.csv``. Captures whose CSV already
exists are skipped.

Columns of the output CSV (one row per window, indexed by the window start):

* ``frame.time``: window start (timestamp).
* ``frame.len``: total bytes in the window.
* ``frame.time_delta_displayed`` (``--ue_only 1``) or ``frame.time_delta``
  (``--ue_only 0``): mean inter-packet time in the window, ``0`` for empty windows.
* ``frame.time_epoch``: **number of packets in the window**. After resampling this
  column holds the packet count (aggregation ``count``), not a timestamp; the original
  tshark field name is kept because downstream code depends on it.
* ``throughput_Mbps``: ``frame.len * 8 / (secs * 1e6)``.

Example:
    python -m stc5g.dataset.pcap_reader --secs 1 --input-dir data/pcap \\
        --output-dir data/csv_throughput/1s --ue-only 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List, Optional, Sequence

import pandas as pd

from stc5g.config import PCAP_ROOT, TIME_DELTA_FIELD, UE_SUBNET
from stc5g.log import get_logger

logger = get_logger(__name__)

#: Display filter that keeps only packets to/from the UE subnet.
UE_FILTER = f"ip.addr == {UE_SUBNET}"


class PCAPReader:
    """Extract per-packet fields from a pcap file with tshark.

    Args:
        pcap_path: Path to the input pcap file.
        feature_vector: tshark fields to extract (one column each).
        ue_only: If True, keep only packets to/from the UE subnet ``10.45.0.0/16``.
    """

    def __init__(self, pcap_path: str, feature_vector: Sequence[str], ue_only: bool = False) -> None:
        self.pcap_path = pcap_path
        self.feature_vector = feature_vector
        self.ue_only = ue_only
        self.dataframe: Optional[pd.DataFrame] = None

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        """Run tshark and return the extracted fields as a DataFrame of strings.

        The tshark header line is part of the parsed output (it becomes the first data
        row and is dropped later by the numeric conversion).

        Returns:
            The extracted fields, or ``None`` if tshark failed.
        """
        fields: List[str] = []
        for feature in self.feature_vector:
            fields += ["-e", feature]

        tshark_command = [
            "tshark",
            "-n",  # no name resolution (faster)
            "-r",
            self.pcap_path,
            "-T",
            "fields",
            *fields,
            "-E",
            "header=y",
            "-E",
            "separator=\t",
            "-E",
            "occurrence=l",  # last occurrence: inner (GTP-U encapsulated) headers
        ]

        if self.ue_only:
            tshark_command.extend(["-Y", UE_FILTER])

        try:
            result = subprocess.run(tshark_command, capture_output=True, text=True, check=True)
            data = [line.split("\t") for line in result.stdout.strip().split("\n")]
            self.dataframe = pd.DataFrame(data, columns=self.feature_vector)
            return self.dataframe

        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing tshark: {e}")
            if e.stderr:
                logger.error(f"Error details: {e.stderr.strip()}")
            logger.error("Ensure that tshark is correctly installed and accessible from the specified path.")
            return None


def time_delta_column(ue_only: bool) -> str:
    """Return the tshark inter-packet time field used for a given extraction mode."""
    return TIME_DELTA_FIELD if ue_only else "frame.time_delta"


def window_suffix(secs: float) -> str:
    """Format a window length for file names (``1.0`` -> ``"1"``, ``0.5`` -> ``"0.5"``)."""
    secs = float(secs)
    return f"{int(secs)}" if secs.is_integer() else f"{secs}"


def resample_windows(df: pd.DataFrame, secs: float, time_delta_col: str) -> pd.DataFrame:
    """Aggregate per-packet records into windows of ``secs`` seconds.

    Args:
        df: Raw tshark output with columns ``frame.time_epoch``, ``frame.len`` and
            ``time_delta_col`` (strings or numbers).
        secs: Window length in seconds.
        time_delta_col: Name of the inter-packet time column.

    Returns:
        One row per window with the summed bytes, the mean inter-packet time (``0`` for
        empty windows), the packet count (in ``frame.time_epoch``) and the throughput in
        Mbps.
    """
    secs = float(secs)
    df["frame.time_epoch"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce")
    df[time_delta_col] = pd.to_numeric(df[time_delta_col], errors="coerce")

    df = df.dropna(subset=["frame.time_epoch"]).copy()

    df["frame.time"] = pd.to_datetime(df["frame.time_epoch"], unit="s")
    df = df.set_index("frame.time")
    df_resampled = df.resample(f"{secs}s").agg(
        {
            "frame.len": "sum",
            time_delta_col: "mean",
            "frame.time_epoch": "count",
        }
    )

    # Windows without packets: mean() yields NaN -> 0 (frequent in video on demand).
    df_resampled[time_delta_col] = df_resampled[time_delta_col].fillna(0)

    # Throughput in Mbps: bytes * 8 / (window_seconds * 1e6).
    df_resampled["throughput_Mbps"] = (df_resampled["frame.len"] * 8) / (secs * 1_000_000)
    return df_resampled


def extract_directory(input_dir: str, output_dir: str, secs: float, ue_only: bool) -> List[str]:
    """Convert every pcap under ``input_dir`` into a windowed CSV under ``output_dir``.

    Files are visited in ``os.walk`` order. Existing output files are not overwritten.

    Args:
        input_dir: Root directory with the ``.pcap`` files.
        output_dir: Root directory for the CSV files (sub-directories are replicated).
        secs: Window length in seconds.
        ue_only: If True, keep only UE traffic and use ``frame.time_delta_displayed``.

    Returns:
        Paths of the CSV files written.
    """
    secs = float(secs)
    secs_str = window_suffix(secs)
    time_delta_col = time_delta_column(ue_only)
    written: List[str] = []

    for root, _dirs, files in os.walk(input_dir):
        # Replicate the folder structure (youtube, twitch, ...) under output_dir.
        rel_path = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel_path) if rel_path != "." else output_dir
        os.makedirs(out_root, exist_ok=True)
        for file in files:
            output_file = os.path.join(out_root, file.replace(".pcap", f"_{secs_str}s_tshark.csv"))
            if file.endswith(".pcap") and ".corrupt.backup" not in file and not os.path.exists(output_file):
                pcap_path = os.path.join(root, file)

                feature_vector = ["frame.time_epoch", "frame.len", time_delta_col]
                df = PCAPReader(pcap_path, feature_vector, ue_only=bool(ue_only)).to_dataframe()

                if df is None or df.empty:
                    logger.warning(f"No data extracted from {pcap_path}")
                    continue

                df_resampled = resample_windows(df, secs, time_delta_col)
                df_resampled.to_csv(output_file)
                written.append(output_file)
                logger.info(f"Saved {output_file} ({len(df_resampled)} windows of {secs_str}s)")

    return written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (snake_case flags kept, kebab-case aliases added)."""
    parser = argparse.ArgumentParser(description="Convert .pcap captures into windowed throughput CSV files.")
    parser.add_argument("--secs", type=float, default=1.0, help="Window length in seconds.")
    parser.add_argument(
        "--input_dir",
        "--input-dir",
        dest="input_dir",
        type=str,
        default=PCAP_ROOT,
        help="Input directory with the pcap files.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory for the CSV files.",
    )
    parser.add_argument(
        "--ue_only",
        "--ue-only",
        dest="ue_only",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 to extract only the UE traffic (10.45.0.0/16).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    extract_directory(args.input_dir, args.output_dir, args.secs, args.ue_only == 1)


if __name__ == "__main__":
    main()
