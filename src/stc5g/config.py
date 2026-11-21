"""Project-wide constants: directory layout, classification setup and plotting metadata.

This module is the single source of truth for paths, classes and sequence length. Every
other module imports from here instead of redefining values.

Paths are resolved relative to the project root (the directory that contains ``src/``).
Set the ``STC5G_ROOT`` environment variable to use a different root, e.g. when the package
is installed outside a clone of the repository.
"""

import os

# =============================================================================
# 1. Project paths
# =============================================================================

#: Directory that contains the package sources (``src/stc5g``).
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Repository root. ``src/stc5g/config.py`` -> two levels up from ``src``.
ROOT_DIR = os.environ.get("STC5G_ROOT") or os.path.dirname(os.path.dirname(PACKAGE_DIR))

#: Main data directory.
DATA_DIR = os.path.join(ROOT_DIR, "data")

#: Raw captures (``.pcap``), one sub-directory per class (``youtube/``, ``twitch/``, ...).
PCAP_ROOT = os.path.join(DATA_DIR, "pcap")

#: Windowed throughput CSV files produced from the captures with tshark.
CSV_THROUGHPUT_DIR = os.path.join(DATA_DIR, "csv_throughput")

#: Trained checkpoints (``classifier_<model>.pt``) and fitted scalers (``scaler_*.pkl``).
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# -----------------------------------------------------------------------------
# Generated outputs: everything lives under results/
# -----------------------------------------------------------------------------

RESULTS_DIR = os.path.join(ROOT_DIR, "results")

#: Classification reports and run logs.
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

#: Regenerable figures.
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
FIG_CONFUSION_DIR = os.path.join(FIGURES_DIR, "confusion_matrix")
FIG_TIMELINE_DIR = os.path.join(FIGURES_DIR, "timeline")
FIG_DATASET_DIR = os.path.join(FIGURES_DIR, "dataset")

#: Figures intended for the paper.
DOCS_FIG_DIR = os.path.join(ROOT_DIR, "docs", "figures")

#: Output directory for the dataset throughput plots.
VIS_DIR = os.path.join(FIG_DATASET_DIR, "throughput")

# =============================================================================
# 2. Model and classification constants
# =============================================================================

#: Input sequence length (number of consecutive windows fed to the model).
LOOKBACK = 20

#: UE address pool of the 5G core, used by the offline extractor and the Ryu controller.
UE_SUBNET = "10.45.0.0/16"

#: tshark field holding the inter-packet time when only UE traffic is extracted.
TIME_DELTA_FIELD = "frame.time_delta_displayed"

#: Total number of classes.
NUM_CLASSES = 3

#: Mapping from class directory name to numeric label.
CLASS_MAP = {
    "youtube": 0,
    "twitch": 1,
    "tiktok": 2,
}

#: Trailing rows (one per second) dropped from every capture before training.
TAIL_ROWS_DROPPED = 30

#: Columns removed before training; the remaining features are
#: ``frame.time_delta_displayed`` and ``throughput_Mbps``.
DROPPED_COLUMNS = ["frame.time_epoch", "frame.len", "frame.time"]

#: Mini-batch size of every DataLoader.
BATCH_SIZE = 32

# =============================================================================
# 3. Visualisation metadata
# =============================================================================

#: Per-class metadata (plot labels, colours).
SERVICES = {
    0: {
        "svc": "youtube",
        "label": "YouTube",
        "color": "#FF0000",
        "color_fill": "#FF000022",
    },
    1: {
        "svc": "twitch",
        "label": "Twitch",
        "color": "#9146FF",
        "color_fill": "#9146FF22",
    },
    2: {
        "svc": "tiktok",
        "label": "TikTok",
        "color": "#000000",
        "color_fill": "#00000022",
    },
}
