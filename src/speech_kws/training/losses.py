from __future__ import annotations

import torch
import torch.nn.functional as F


def uniform_target_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs.mean(dim=-1).mean()


def compute_strategy_loss(strategy: str, outputs: dict, batch: dict, strategy_config: dict | None = None) -> dict:
    strategy_config = strategy_config or {}
    normalized = strategy.lower()

    if normalized in {"a", "b"}:
        target = batch["train_target"]
        loss = F.cross_entropy(outputs["logits"], target)
        return {"loss": loss, "classification_loss": float(loss.detach().item())}

    if normalized == "c":
        lambda_head2 = float(strategy_config.get("lambda_head2", 1.0))
        head1_loss = F.cross_entropy(outputs["head1_logits"], batch["head1_target"])
        valid_keyword_mask = batch["head2_target"] >= 0
        if valid_keyword_mask.any():
            head2_loss = F.cross_entropy(
                outputs["head2_logits"][valid_keyword_mask],
                batch["head2_target"][valid_keyword_mask],
            )
        else:
            head2_loss = outputs["head2_logits"].new_tensor(0.0)
        total = head1_loss + (lambda_head2 * head2_loss)
        return {
            "loss": total,
            "head1_loss": float(head1_loss.detach().item()),
            "head2_loss": float(head2_loss.detach().item()),
        }

    if normalized == "d":
        oe_weight = float(strategy_config.get("oe_weight", 1.0))
        known_mask = batch["known_target"] >= 0
        unknown_mask = batch["is_outlier"].bool()

        if known_mask.any():
            known_loss = F.cross_entropy(outputs["logits"][known_mask], batch["known_target"][known_mask])
        else:
            known_loss = outputs["logits"].new_tensor(0.0)

        if unknown_mask.any():
            oe_loss = uniform_target_loss(outputs["logits"][unknown_mask])
        else:
            oe_loss = outputs["logits"].new_tensor(0.0)

        total = known_loss + (oe_weight * oe_loss)
        return {
            "loss": total,
            "known_loss": float(known_loss.detach().item()),
            "oe_loss": float(oe_loss.detach().item()),
        }

    raise ValueError(f"Unsupported strategy: {strategy}")
