from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
models = pytest.importorskip("choravox.torch_models")
CtcAcousticModel = models.CtcAcousticModel
SpeakerEncoder = models.SpeakerEncoder

pytestmark = pytest.mark.model


def test_speaker_encoder_returns_unit_embeddings() -> None:
    torch.manual_seed(3)
    model = SpeakerEncoder(6, embedding_dim=5, channels=4, hidden_size=7)
    embeddings = model(torch.randn(3, 12, 6))
    assert embeddings.shape == (3, 5)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3), atol=1e-6)


def test_ctc_model_returns_normalized_frame_distributions() -> None:
    torch.manual_seed(5)
    model = CtcAcousticModel(4, 7, hidden_size=6)
    log_probabilities = model(torch.randn(2, 9, 4))
    assert log_probabilities.shape == (2, 9, 7)
    totals = torch.exp(log_probabilities).sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)


def test_torch_models_validate_feature_geometry() -> None:
    with pytest.raises(ValueError, match="batch-by-frame"):
        SpeakerEncoder(4)(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="batch-by-frame"):
        CtcAcousticModel(4, 3)(torch.zeros(2, 5, 3))
