"""PyTorch device selection."""

import torch


def select_device(preference: str = "auto") -> torch.device:
    """Return the compute device.

    ``"auto"`` picks CUDA, then Apple MPS, then CPU (the original selection order).
    Any other value (``"cpu"``, ``"cuda"``, ``"mps"``, ``"cuda:1"``) is used as given.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
