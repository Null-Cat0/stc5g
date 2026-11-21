"""Model factory and checkpoint loader.

This is the single place that knows the hyper-parameters of the published LSTM
classifier (those used to train the published model).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from stc5g.config import NUM_CLASSES
from stc5g.models.lstm import LSTM

#: Model names accepted on the command line.
MODEL_CHOICES = ("lstm",)

#: Model class name as stored in checkpoints under ``model_type``.
MODEL_TYPE = "LSTM"


def build_model(name: str, input_size: int, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Instantiate the LSTM classifier with its original hyper-parameters.

    Args:
        name: ``"lstm"`` (command-line name) or ``"LSTM"`` (class name).
        input_size: Number of features per time step.
        num_classes: Number of output classes.

    Returns:
        The model on the CPU (move it with ``.to(device)``).

    Raises:
        ValueError: If ``name`` is not ``"lstm"``/``"LSTM"``.
    """
    if name not in ("lstm", MODEL_TYPE):
        raise ValueError(f"Unsupported model: {name} (only 'lstm' is available)")
    return LSTM(hidden1=100, hidden2=50, input_size=input_size, output_size=num_classes)


def model_input_size(model: nn.Module) -> int:
    """Return the number of features per time step that ``model`` was built for.

    The width is read from the first recurrent layer (``nn.LSTM``/``nn.GRU``/``nn.RNN``)
    or, if a linear layer comes first, from its ``in_features``.

    Raises:
        ValueError: If ``model`` contains no such layer.
    """
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            return module.input_size
        if isinstance(module, nn.Linear):
            return module.in_features
    raise ValueError(f"Cannot infer the input size of {type(model).__name__}")


def load_checkpoint(path: str, device: Union[str, torch.device]) -> Tuple[nn.Module, dict]:
    """Rebuild the LSTM classifier from a checkpoint written by the training loop.

    ``model_type`` must be ``"LSTM"`` (a missing or empty value is read as ``"LSTM"``, as
    in the original loaders); the input size comes from ``input_size`` (2 if missing).
    The model is returned in its default (training) mode; call ``.eval()`` before
    inference.

    Args:
        path: Checkpoint file (``.pt``).
        device: Device to map the tensors to and to place the model on.

    Returns:
        ``(model, checkpoint)``: the model on ``device`` and the raw checkpoint dict.

    Raises:
        ValueError: If the checkpoint holds another model type.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type") or MODEL_TYPE
    if model_type != MODEL_TYPE:
        raise ValueError(
            f"Unsupported model type {model_type!r} in {path}: only LSTM checkpoints are supported"
        )
    input_size = checkpoint.get("input_size", 2)
    model = build_model(model_type, input_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
