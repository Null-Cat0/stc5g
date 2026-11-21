"""Repair truncated or corrupt pcap files with ``editcap`` (shipped with Wireshark).

Every ``.pcap`` under the root directory is rewritten with ``editcap``. When a capture
is truncated, editcap copies all valid packets up to the truncation point; if the
rewritten file differs in size from the original, the original is renamed to
``<name>.pcap.corrupt.backup`` and replaced by the rewritten file.

Example:
    python -m stc5g.dataset.repair_pcaps                    # repair everything in data/pcap
    python -m stc5g.dataset.repair_pcaps --dir data/pcap    # custom directory
    python -m stc5g.dataset.repair_pcaps --dry-run          # inspect only, do not modify
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

from stc5g.config import PCAP_ROOT
from stc5g.log import get_logger

logger = get_logger(__name__)

BACKUP_SUFFIX = ".corrupt.backup"
EDITCAP_TIMEOUT = 300  # seconds; large captures take a while


def repair_if_corrupt(pcap_path: str, dry_run: bool = False) -> str:
    """Rewrite a capture with editcap and replace it if the result differs in size.

    Args:
        pcap_path: Path to the capture.
        dry_run: If True, do nothing and report ``"ok"``.

    Returns:
        ``"ok"`` (unchanged), ``"repaired"`` (replaced; original kept with
        :data:`BACKUP_SUFFIX`) or ``"failed"``.
    """
    orig_size = os.path.getsize(pcap_path)

    if dry_run:
        logger.info(f"  [DRY-RUN] Would be processed with editcap ({orig_size} bytes)")
        return "ok"

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["editcap", pcap_path, tmp_path],
            capture_output=True,
            text=True,
            timeout=EDITCAP_TIMEOUT,
        )

        if result.returncode != 0:
            logger.warning(f"editcap returned code {result.returncode}: {result.stderr.strip()}")
            os.unlink(tmp_path)
            return "failed"

        fixed_size = os.path.getsize(tmp_path)

        if fixed_size == 0:
            logger.warning(f"editcap produced an empty file for {pcap_path}")
            os.unlink(tmp_path)
            return "failed"

        if fixed_size == orig_size:
            os.unlink(tmp_path)
            return "ok"

        lost = orig_size - fixed_size
        backup_path = pcap_path + BACKUP_SUFFIX

        os.rename(pcap_path, backup_path)
        os.rename(tmp_path, pcap_path)

        logger.info(
            f"Repaired: {os.path.basename(pcap_path)} "
            f"({orig_size:_} -> {fixed_size:_} bytes, "
            f"{lost:_} lost from the truncated packet)"
        )
        return "repaired"

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout while processing {pcap_path} ({orig_size:_} bytes)")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return "failed"
    except Exception as exc:
        logger.error(f"Error while processing {pcap_path}: {exc}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return "failed"


def fix_all_pcaps(root_dir: str, dry_run: bool = False) -> Dict[str, int]:
    """Run :func:`repair_if_corrupt` on every ``.pcap`` under ``root_dir``.

    Args:
        root_dir: Root directory to scan.
        dry_run: If True, inspect only.

    Returns:
        Counters ``total``, ``ok``, ``repaired`` and ``failed``.
    """
    stats = {"total": 0, "ok": 0, "repaired": 0, "failed": 0}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".pcap") or fname.endswith(BACKUP_SUFFIX):
                continue

            pcap_path = os.path.join(dirpath, fname)
            stats["total"] += 1

            size_mb = os.path.getsize(pcap_path) / (1024 * 1024)
            logger.info(f"Checking: {pcap_path} ({size_mb:.0f} MB)")

            status = repair_if_corrupt(pcap_path, dry_run=dry_run)

            if status == "ok":
                logger.info("  [OK]")
                stats["ok"] += 1
            elif status == "repaired":
                stats["repaired"] += 1
            elif status == "failed":
                stats["failed"] += 1

    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan and repair corrupt/truncated pcap files with editcap.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=PCAP_ROOT,
        help="Root directory to scan (default: data/pcap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect only, do not modify any file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)

    if not os.path.isdir(args.dir):
        logger.error(f"Directory not found: {args.dir}")
        sys.exit(1)

    logger.info(f"Scanning: {os.path.abspath(args.dir)}")
    logger.info(f"Mode: {'dry-run' if args.dry_run else 'repair'}")
    logger.info("-" * 60)

    stats = fix_all_pcaps(args.dir, dry_run=args.dry_run)

    logger.info("-" * 60)
    logger.info(f"Summary: {stats['total']} pcaps | {stats['repaired']} repaired | {stats['failed']} failed")

    if stats["failed"]:
        logger.warning("Some files could not be repaired. Try pcapfix or scapy.")


if __name__ == "__main__":
    main()
