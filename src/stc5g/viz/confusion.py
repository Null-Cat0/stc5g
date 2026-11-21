"""Confusion-matrix figure."""

from __future__ import annotations

import os
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from stc5g.config import FIG_CONFUSION_DIR


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_map: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    model_name: str = "LSTM",
    cmap: str = "Blues",
    title_suffix: str = "",
    filename_suffix: str = "",
    output_dir: str = FIG_CONFUSION_DIR,
) -> None:
    """Draw and save a row-normalised confusion matrix.

    Each cell shows the count and, in brackets, the recall of the true class. The figure
    is written to ``{output_dir}/{model_name}/confusion_matrix_{slug}_{suffix}.png`` where
    ``slug`` is the lower-cased model name and ``suffix`` is ``filename_suffix`` or, if
    empty, ``{train_ratio}-{val_ratio}-{test_ratio}``.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_map: Class name -> label; tick labels are the capitalised names ordered by label.
        train_ratio: Train split ratio (used in the default file name).
        val_ratio: Validation split ratio (used in the default file name).
        test_ratio: Test split ratio (used in the default file name).
        model_name: Model name, used for the sub-directory and the file name.
        cmap: Matplotlib colour map.
        title_suffix: Unused (the figure has no title); kept for API compatibility.
        filename_suffix: Optional file-name suffix replacing the split ratios.
        output_dir: Root output directory.
    """
    classes = [name.capitalize() for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Recall per true class"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 11},
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True", fontsize=12, labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    fig.tight_layout()

    output_path = os.path.join(output_dir, model_name)
    os.makedirs(output_path, exist_ok=True)
    slug = model_name.lower().replace(" ", "_")
    if filename_suffix:
        fname = f"confusion_matrix_{slug}_{filename_suffix}.png"
    else:
        fname = f"confusion_matrix_{slug}_{train_ratio}-{val_ratio}-{test_ratio}.png"
    fig.savefig(os.path.join(output_path, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
