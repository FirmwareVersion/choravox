"""Optional CPU-friendly trainable models for synthetic speaker and CTC tasks."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpeakerEncoder(nn.Module):
    """Small convolutional/GRU encoder returning unit speaker embeddings."""

    def __init__(
        self,
        feature_dim: int,
        *,
        embedding_dim: int = 16,
        channels: int = 16,
        hidden_size: int = 16,
    ) -> None:
        super().__init__()
        if min(feature_dim, embedding_dim, channels, hidden_size) < 1:
            raise ValueError("speaker encoder dimensions must be positive")
        self.feature_dim = feature_dim
        self.convolution = nn.Conv1d(feature_dim, channels, kernel_size=3, padding=1)
        self.recurrent = nn.GRU(channels, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, embedding_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] == 0 or features.shape[2] != self.feature_dim:
            raise ValueError("speaker features must be batch-by-frame-by-feature")
        convolved: Tensor = torch.relu(self.convolution(features.transpose(1, 2)))
        sequence = convolved.transpose(1, 2)
        _output, hidden = self.recurrent(sequence)
        projected: Tensor = self.projection(hidden[-1])
        normalized: Tensor = F.normalize(projected, dim=-1)
        return normalized


class CtcAcousticModel(nn.Module):
    """Compact bidirectional GRU producing frame-level CTC log probabilities."""

    def __init__(self, feature_dim: int, token_count: int, *, hidden_size: int = 24) -> None:
        super().__init__()
        if feature_dim < 1 or token_count < 2 or hidden_size < 1:
            raise ValueError("CTC model dimensions are invalid")
        self.feature_dim = feature_dim
        self.recurrent = nn.GRU(
            feature_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_size * 2, token_count)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] == 0 or features.shape[2] != self.feature_dim:
            raise ValueError("CTC features must be batch-by-frame-by-feature")
        encoded, _hidden = self.recurrent(features)
        logits: Tensor = self.output(encoded)
        probabilities: Tensor = F.log_softmax(logits, dim=-1)
        return probabilities
