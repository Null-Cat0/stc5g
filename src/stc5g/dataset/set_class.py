"""Add the ``class`` label column to the windowed CSV files.

The root directory is walked recursively; every CSV inside a directory whose name is a
key of :data:`stc5g.config.CLASS_MAP` gets a constant ``class`` column with the
corresponding label. The files are modified **in place**; files that already have a
``class`` column are left untouched.

Class mapping (``CLASS_MAP``, 3 classes)::

    youtube -> 0
    twitch  -> 1
    tiktok  -> 2

Directory names are lower-cased and spaces removed before the lookup.

Example:
    python -m stc5g.dataset.set_class --root data/csv_throughput/1s
    python -m stc5g.dataset.set_class --root data/csv_throughput/1s --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

from stc5g.config import CLASS_MAP, PCAP_ROOT
from stc5g.log import get_logger

logger = get_logger(__name__)

DEFAULT_ROOT = PCAP_ROOT


def classify_csvs(root: str, dry_run: bool = False) -> None:
    """Add the ``class`` column to every CSV in a mapped directory under ``root``.

    Args:
        root: Root directory to walk.
        dry_run: If True, only log what would be done without modifying any file.
    """
    modified = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(root):
        # The class is given by the name of the directory.
        dir_name = os.path.basename(dirpath).lower().replace(" ", "")
        class_label = CLASS_MAP.get(dir_name)

        if class_label is None:
            logger.info(f"Unmapped directory: {dirpath}")
            continue

        csv_files = [f for f in filenames if f.endswith(".csv")]
        if not csv_files:
            logger.info(f"No CSV files in directory: {dirpath}")
            continue

        for csv_file in csv_files:
            csv_path = os.path.join(dirpath, csv_file)

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.warning(f"Could not read {csv_path}: {exc}")
                skipped += 1
                continue

            if "class" in df.columns:
                logger.info(f"Already has a 'class' column: {csv_path}")
                skipped += 1
                continue

            df["class"] = class_label

            if dry_run:
                logger.info(f"(DRY-RUN) {csv_path} -> class {class_label} ({dir_name}), {len(df)} rows")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"Labelled {csv_path} -> class {class_label} ({dir_name}), {len(df)} rows")

            modified += 1

    logger.info(f"Modified: {modified} | Skipped: {skipped}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Add a 'class' column to CSV files according to the directory that contains them.",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Root directory to walk (default: data/pcap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, without modifying any file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)

    if not os.path.isdir(args.root):
        logger.error(f"Directory not found: {args.root}")
        sys.exit(1)

    logger.info(f"Root: {os.path.abspath(args.root)}")
    logger.info(f"Mapping: {CLASS_MAP}")

    classify_csvs(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
