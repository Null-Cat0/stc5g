"""Training and evaluation pipeline (``stc5g train``).

Steps: load and split the throughput CSV files, build sequences and DataLoaders, train the
selected model (checkpoint in ``models/classifier_<model>.pt``), evaluate it on the test
split (classification report in ``results/logs/`` and confusion matrix in
``results/figures/confusion_matrix/<ModelClass>/``) and optionally plot a prediction
timeline.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report

from stc5g.config import (
    CLASS_MAP,
    CSV_THROUGHPUT_DIR,
    DATA_DIR,
    FIG_CONFUSION_DIR,
    FIG_TIMELINE_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
)
from stc5g.data.loading import get_data, make_sequences, transform_data_loader
from stc5g.device import select_device
from stc5g.log import get_logger
from stc5g.models.registry import MODEL_CHOICES, build_model
from stc5g.training.engine import evaluate, plot_confusion_matrix_lstm, train_model

logger = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments (snake_case flags are kept as aliases)."""
    parser = argparse.ArgumentParser(description="Training and evaluation pipeline")
    parser.add_argument("--model", type=str, default="lstm", choices=MODEL_CHOICES, help="Model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--train-ratio",
        "--train_ratio",
        dest="train_ratio",
        type=float,
        default=0.7,
        help="Fraction of every capture used for training",
    )
    parser.add_argument(
        "--val-ratio",
        "--val_ratio",
        dest="val_ratio",
        type=float,
        default=0.2,
        help="Fraction of every capture used for validation",
    )
    parser.add_argument(
        "--test-ratio",
        "--test_ratio",
        dest="test_ratio",
        type=float,
        default=0.1,
        help="Fraction of every capture used for testing",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=str,
        default=CSV_THROUGHPUT_DIR,
        help="Directory with the throughput CSV files",
    )
    parser.add_argument(
        "--test-dir",
        "--test_dir",
        dest="test_dir",
        type=str,
        default="",
        help="Independent test directory (the data directory is then split into train/val only)",
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default="",
        help="Directory of captures on which to plot a prediction timeline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Python, NumPy and PyTorch (default: unseeded, as in the original runs)",
    )
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the training and evaluation pipeline."""
    args = parse_args(argv)
    if args.seed is not None:
        _seed_everything(args.seed)

    device = select_device("auto")
    logger.info(f"Using device: {device}")

    # 1. Load data
    test_dir_arg = args.test_dir if args.test_dir else None
    if test_dir_arg:
        logger.info(f"Loading train/val from {args.data_dir} and test from {test_dir_arg}...")
    else:
        logger.info(f"Loading and splitting data from {args.data_dir}...")

    x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = get_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        norm=1,
        throughput_dir=args.data_dir,
        test_dir=test_dir_arg,
        splits_dir=DATA_DIR,
        models_dir=MODELS_DIR,
    )

    # 2. Build sequences
    logger.info("Building time sequences...")
    x_train, y_train = make_sequences(x_train_raw, y_train_raw)
    x_val, y_val = make_sequences(x_val_raw, y_val_raw) if len(x_val_raw) > 0 else ([], [])
    x_test, y_test = make_sequences(x_test_raw, y_test_raw) if len(x_test_raw) > 0 else ([], [])

    logger.info(f"Train shape: {x_train.shape}, {y_train.shape}")
    if len(x_val) > 0:
        logger.info(f"Val shape: {x_val.shape}, {y_val.shape}")
    else:
        logger.info("Val: empty (train_ratio=1.0)")
    if len(x_test) > 0:
        logger.info(f"Test shape: {x_test.shape}, {y_test.shape}")
    else:
        logger.info("Test: empty (an independent test_dir will be used)")

    # 3. DataLoaders
    logger.info("Building DataLoaders...")
    train_loader, val_loader, test_loader = transform_data_loader(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    # 4. Model
    logger.info(f"Instantiating model: {args.model}...")
    input_size = x_train.shape[2] if len(x_train) > 0 else 2
    model = build_model(args.model, input_size=input_size, num_classes=NUM_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training
    logger.info("Starting training...")
    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        path=os.path.join(MODELS_DIR, f"classifier_{args.model}.pt"),
        device=device,
        input_size=input_size,
    )

    # 6. Test evaluation and confusion matrix
    if test_loader is not None:
        logger.info("Evaluating the model on the test set...")
        test_results = evaluate(model, test_loader, device)
        logger.info(f"Test accuracy: {test_results['accuracy']:.2f}%")

        y_true = test_results["y_true"]
        y_pred = test_results["y_pred"]
        class_names = [name.capitalize() for name, _ in sorted(CLASS_MAP.items(), key=lambda x: x[1])]
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        logger.info(f"\nClassification Report:\n{report}")

        os.makedirs(LOGS_DIR, exist_ok=True)
        report_path = os.path.join(LOGS_DIR, f"classification_report_{args.model}_test.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Classification report saved to: {report_path}")

        logger.info("Plotting the confusion matrix...")
        plot_confusion_matrix_lstm(
            model=model,
            data_loader=test_loader,
            device=device,
            class_map=CLASS_MAP,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            title_suffix="(Test Set)",
            filename_suffix="test",
            output_dir=FIG_CONFUSION_DIR,
        )
    else:
        logger.warning("No test set: skipping evaluation and the confusion matrix.")

    # 7. Prediction timeline (optional)
    if args.timeline:
        from stc5g.viz.timeline import apply_style, evaluate_timeline, plot_timeline

        timeline_dir = args.timeline
        scaler_path = os.path.join(
            MODELS_DIR, f"scaler_{args.train_ratio}-{args.val_ratio}-{args.test_ratio}.pkl"
        )
        logger.info(f"Building timeline from: {timeline_dir}")
        df_tl = evaluate_timeline(model, timeline_dir, scaler_path, device)
        if not df_tl.empty:
            acc = df_tl["correct"].mean() * 100
            logger.info(f"Timeline accuracy: {acc:.2f}%")
            apply_style()
            plot_timeline(df_tl, FIG_TIMELINE_DIR, model_name=model.__class__.__name__)
        else:
            logger.warning("Timeline: no usable data found.")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
