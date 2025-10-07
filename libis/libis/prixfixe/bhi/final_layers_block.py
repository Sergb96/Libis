from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch import nn, Generator

from ..prixfixe import FinalLayersBlock


class BHIFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int = 320,
        seqsize: int = 110,
        hidden_dim: int = 64,
    ):
        super().__init__(in_channels, seqsize)
        self.pooler = nn.AdaptiveAvgPool1d(1, )
        self.main = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = self.pooler(x).squeeze(-1)  # (batch_size, in_channels)
        x = self.main(x)  # (batch_size, output_dim)
        return x.squeeze(-1)  # (batch_size,)
