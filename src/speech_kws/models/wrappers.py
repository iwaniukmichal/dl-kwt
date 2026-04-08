from __future__ import annotations

import torch
import torch.nn as nn

from speech_kws.data.labels import EVAL_LABELS, KNOWN_LABELS_STRATEGY_D, RAW_WORD_LABELS, TARGET_KEYWORDS
from speech_kws.models.bcresnet import BCResNetBackbone
from speech_kws.models.kwt import KWTBackbone
from speech_kws.models.matchboxnet import MatchboxNetBackbone


BACKBONES = {
    "bcresnet": BCResNetBackbone,
    "kwt": KWTBackbone,
    "matchboxnet": MatchboxNetBackbone,
}


def build_backbone(name: str, input_dim: int, config: dict) -> nn.Module:
    normalized = name.lower()
    if normalized not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {name}")
    return BACKBONES[normalized](input_dim=input_dim, config=config)


class SingleHeadModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, features: torch.Tensor) -> dict:
        embeddings = self.backbone(features)
        logits = self.classifier(embeddings)
        return {"embeddings": embeddings, "logits": logits}


class TwoHeadModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_head1_classes: int, num_head2_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head1 = nn.Linear(backbone.output_dim, num_head1_classes)
        self.head2 = nn.Linear(backbone.output_dim, num_head2_classes)

    def forward(self, features: torch.Tensor) -> dict:
        embeddings = self.backbone(features)
        return {
            "embeddings": embeddings,
            "head1_logits": self.head1(embeddings),
            "head2_logits": self.head2(embeddings),
        }


def build_model(model_config: dict, strategy: str, frontend_feature_dim: int) -> nn.Module:
    backbone = build_backbone(
        name=model_config["name"],
        input_dim=frontend_feature_dim,
        config=model_config,
    )
    normalized_strategy = strategy.lower()
    if normalized_strategy == "a":
        return SingleHeadModel(backbone, num_classes=len(RAW_WORD_LABELS) + 1)
    if normalized_strategy == "b":
        return SingleHeadModel(backbone, num_classes=len(EVAL_LABELS))
    if normalized_strategy == "c":
        return TwoHeadModel(backbone, num_head1_classes=3, num_head2_classes=len(TARGET_KEYWORDS))
    if normalized_strategy == "d":
        return SingleHeadModel(backbone, num_classes=len(KNOWN_LABELS_STRATEGY_D))
    raise ValueError(f"Unsupported strategy: {strategy}")
