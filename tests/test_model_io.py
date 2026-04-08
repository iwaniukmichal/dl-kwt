from __future__ import annotations

import torch

from speech_kws.models.wrappers import build_model


def test_bcresnet_single_head_output_shape() -> None:
    model = build_model(
        model_config={"name": "bcresnet", "channels": 32, "depth": 2, "embedding_dim": 64},
        strategy="b",
        frontend_feature_dim=40,
    )
    outputs = model(torch.randn(2, 40, 98))
    assert outputs["logits"].shape == (2, 12)


def test_kwt_two_head_output_shape() -> None:
    model = build_model(
        model_config={"name": "kwt", "d_model": 64, "num_heads": 4, "depth": 2, "max_frames": 128},
        strategy="c",
        frontend_feature_dim=40,
    )
    outputs = model(torch.randn(2, 40, 98))
    assert outputs["head1_logits"].shape == (2, 3)
    assert outputs["head2_logits"].shape == (2, 10)


def test_matchboxnet_known_class_output_shape() -> None:
    model = build_model(
        model_config={"name": "matchboxnet", "channels": 32, "repeats": 2, "embedding_dim": 64},
        strategy="d",
        frontend_feature_dim=64,
    )
    outputs = model(torch.randn(2, 64, 128))
    assert outputs["logits"].shape == (2, 11)
