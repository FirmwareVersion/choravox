"""Optional CPU-friendly trainable models for synthetic speaker and CTC tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class TrainingTrace:
    """Measured optimization losses from one deterministic synthetic run."""

    task: str
    losses: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.task
            or len(self.losses) < 2
            or not all(math.isfinite(loss) and loss >= 0.0 for loss in self.losses)
        ):
            raise ValueError("training trace requires finite nonnegative losses")

    @property
    def decrease(self) -> float:
        return round(self.losses[0] - self.losses[-1], 9)


def synthetic_speaker_training_trace(*, steps: int = 20, seed: int = 0) -> TrainingTrace:
    """Train the speaker encoder on two separable synthetic voice patterns."""
    if steps < 2 or seed < 0:
        raise ValueError("speaker training controls are invalid")
    torch.manual_seed(seed)
    labels = torch.arange(8, dtype=torch.long) % 2
    features = torch.zeros(8, 10, 4)
    features[:, :, 0] = torch.where(labels[:, None] == 0, 1.0, -1.0)
    features[:, :, 1] = torch.where(labels[:, None] == 0, -0.4, 0.4)
    features += torch.randn_like(features) * 0.03
    encoder = SpeakerEncoder(4, embedding_dim=4, channels=6, hidden_size=6)
    classifier = nn.Linear(4, 2)
    optimizer = torch.optim.Adam((*encoder.parameters(), *classifier.parameters()), lr=0.03)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss: Tensor = F.cross_entropy(classifier(encoder(features)), labels)
        losses.append(round(float(loss.detach()), 9))
        torch.autograd.backward(loss)
        optimizer.step()
    return TrainingTrace("speaker-classification", tuple(losses))


def synthetic_ctc_training_trace(*, steps: int = 24, seed: int = 0) -> TrainingTrace:
    """Train the CTC GRU on two deterministic single-token synthetic sequences."""
    if steps < 2 or seed < 0:
        raise ValueError("CTC training controls are invalid")
    torch.manual_seed(seed)
    features = torch.zeros(2, 7, 3)
    features[0, :, 0] = 1.0
    features[1, :, 1] = 1.0
    features += torch.randn_like(features) * 0.02
    targets = torch.tensor([1, 2], dtype=torch.long)
    input_lengths = torch.full((2,), features.shape[1], dtype=torch.long)
    target_lengths = torch.ones(2, dtype=torch.long)
    model = CtcAcousticModel(3, 3, hidden_size=6)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        frame_scores = model(features).transpose(0, 1)
        loss: Tensor = criterion(frame_scores, targets, input_lengths, target_lengths)
        losses.append(round(float(loss.detach()), 9))
        torch.autograd.backward(loss)
        optimizer.step()
    return TrainingTrace("ctc", tuple(losses))
