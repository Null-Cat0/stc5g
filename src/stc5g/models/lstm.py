"""Stacked three-layer LSTM classifier."""

import torch
import torch.nn as nn

from stc5g.config import NUM_CLASSES


class LSTM(nn.Module):
    """Stacked LSTM for throughput-sequence classification.

    Three ``batch_first`` LSTM layers with dropout, the last time step of the third layer
    feeding a linear output layer.

    Args:
        hidden1: Hidden size of the first LSTM layer.
        hidden2: Hidden size of the second and third LSTM layers.
        input_size: Number of features per time step.
        output_size: Number of classes.
    """

    def __init__(
        self, hidden1: int = 100, hidden2: int = 50, input_size: int = 2, output_size: int = NUM_CLASSES
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.drop1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.drop2 = nn.Dropout(p=0.2)
        self.lstm3 = nn.LSTM(input_size=hidden2, hidden_size=hidden2, batch_first=True)
        self.drop3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(hidden2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, output_size)`` for ``x`` of shape ``(batch, seq, feat)``."""
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.drop3(x)
        return self.out(x)
