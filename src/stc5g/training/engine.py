"""Training loop, evaluation and confusion-matrix helper."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stc5g.config import FIG_CONFUSION_DIR, LOOKBACK, MODELS_DIR
from stc5g.log import get_logger
from stc5g.models.registry import model_input_size
from stc5g.viz.confusion import plot_confusion_matrix

logger = get_logger(__name__)

Device = Union[str, torch.device, None]


def evaluate(
    model: nn.Module, data_loader: DataLoader, device: Device, criterion: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """Run inference over a DataLoader and compute metrics.

    The model is switched to ``eval`` mode.

    Args:
        model: PyTorch model.
        data_loader: DataLoader yielding ``(x, y)`` batches.
        device: Compute device.
        criterion: Optional loss function; if given, the mean loss is computed.

    Returns:
        A dict with ``accuracy`` (percentage), ``loss`` (mean loss, or ``None`` without
        ``criterion``), ``y_true`` and ``y_pred`` (lists of labels).
    """
    model.eval()
    correct, total = 0, 0
    running_loss, total_loss_samples = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for xi, yi in data_loader:
            xi = xi.to(device)
            yi = yi.to(device).long()

            outputs = model(xi)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == yi).sum().item()
            total += yi.size(0)

            y_true.extend(yi.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            if criterion is not None:
                loss = criterion(outputs, yi)
                running_loss += float(loss.item()) * yi.size(0)
                total_loss_samples += yi.size(0)

    return {
        "accuracy": 100 * correct / max(total, 1),
        "loss": running_loss / max(total_loss_samples, 1) if criterion else None,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _checkpoint(model: nn.Module, input_size: int) -> Dict[str, Any]:
    """Build the checkpoint dict.

    ``model_state_dict`` holds the live state-dict tensors (not copies), exactly as the
    original training loop did.
    """
    return {
        "model_state_dict": model.state_dict(),
        "lookback": LOOKBACK,
        "input_size": input_size,
        "model_type": model.__class__.__name__,
    }


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 100,
    path: Optional[str] = None,
    device: Device = None,
    input_size: Optional[int] = None,
) -> None:
    """Train a classifier and checkpoint it on validation-accuracy improvement.

    Cross-entropy loss, ``StepLR(step_size=5, gamma=0.75)`` stepped once per epoch and
    gradient-norm clipping at 1.0. With a validation loader, the checkpoint is written each
    time the validation accuracy strictly improves; without one, it is written after every
    epoch.

    Note:
        The checkpoint dict references the live parameter tensors, so the final
        ``model.load_state_dict`` re-loads the *current* weights: after this function
        returns, ``model`` holds the weights of the **last** epoch while the file on disk
        holds those of the best epoch. This is the original behaviour and is kept.

    Args:
        model: Model to train (already on ``device``).
        optimizer: Optimizer over ``model.parameters()``.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader, or ``None``.
        epochs: Number of epochs.
        path: Checkpoint file. Defaults to ``{MODELS_DIR}/classifier_lstm.pt``.
        device: Compute device.
        input_size: Features per time step, stored in the checkpoint. ``None`` reads it
            from the model (:func:`~stc5g.models.registry.model_input_size`).

    Raises:
        ValueError: If ``input_size`` does not match the input width of ``model``.
    """
    model_width = model_input_size(model)
    if input_size is None:
        input_size = model_width
    elif input_size != model_width:
        raise ValueError(f"input_size={input_size} does not match the model input width ({model_width})")

    if path is None:
        path = os.path.join(MODELS_DIR, "classifier_lstm.pt")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss, total_samples = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).long()
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            total_samples += xb.size(0)
        scheduler.step()
        epoch_loss = running_loss / max(total_samples, 1)

        if val_loader is not None:
            val = evaluate(model, val_loader, device, criterion=criterion)
            val_acc = val["accuracy"]
            epoch_val_loss = val["loss"]

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_state = _checkpoint(model, input_size)
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                torch.save(best_state, path)
            flag = " <- best" if improved else ""
            logger.info(
                f"Epoch {epoch + 1:3d}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {epoch_val_loss:.4f}"
                f" - val_acc: {val_acc:.2f}%{flag}"
            )
        else:
            # No validation set: always keep the latest model.
            best_state = _checkpoint(model, input_size)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(best_state, path)
            logger.info(f"Epoch {epoch + 1:3d}/{epochs} - loss: {epoch_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        if val_loader is not None:
            logger.info(f"Model restored from best epoch with Val Acc: {best_val_acc:.2f}%. Saved in {path}.")
        else:
            logger.info(f"Model saved (no validation). Saved in {path}.")


def multiclass_accuracy_lstm(model: nn.Module, data_loader: DataLoader, device: Device) -> float:
    """Return the accuracy (percentage) of ``model`` on ``data_loader``."""
    return evaluate(model, data_loader, device)["accuracy"]


def plot_confusion_matrix_lstm(
    model: nn.Module,
    data_loader: DataLoader,
    device: Device,
    class_map: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    title_suffix: str = "",
    filename_suffix: str = "",
    output_dir: str = FIG_CONFUSION_DIR,
) -> None:
    """Evaluate ``model`` on ``data_loader`` and save its confusion matrix.

    See :func:`stc5g.viz.confusion.plot_confusion_matrix` for the output location.
    """
    result = evaluate(model, data_loader, device)
    model_name = model.__class__.__name__
    plot_confusion_matrix(
        result["y_true"],
        result["y_pred"],
        class_map,
        train_ratio,
        val_ratio,
        test_ratio,
        model_name=model_name,
        cmap="Blues",
        title_suffix=title_suffix,
        filename_suffix=filename_suffix,
        output_dir=output_dir,
    )
