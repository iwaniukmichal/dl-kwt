from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class NovoGrad(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.5),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("NovoGrad does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros((), device=parameter.device)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                grad_norm = grad.pow(2).sum()
                exp_avg_sq.mul_(beta2).add_(grad_norm, alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                grad_normalized = grad / denom
                if weight_decay > 0:
                    grad_normalized = grad_normalized + (weight_decay * parameter)

                exp_avg.mul_(beta1).add_(grad_normalized, alpha=1.0 - beta1)
                parameter.add_(exp_avg, alpha=-lr)

        return loss


def build_optimizer(parameters, optimizer_config: dict) -> Optimizer:
    name = optimizer_config["name"].lower()
    params = [parameter for parameter in parameters if parameter.requires_grad]
    lr = float(optimizer_config.get("lr", 1e-3))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(optimizer_config.get("momentum", 0.0)),
            weight_decay=weight_decay,
            nesterov=bool(optimizer_config.get("nesterov", False)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(
                float(optimizer_config.get("beta1", 0.9)),
                float(optimizer_config.get("beta2", 0.999)),
            ),
            eps=float(optimizer_config.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    if name == "novograd":
        return NovoGrad(
            params,
            lr=lr,
            betas=(
                float(optimizer_config.get("beta1", 0.95)),
                float(optimizer_config.get("beta2", 0.5)),
            ),
            eps=float(optimizer_config.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")


def _resolve_step_count(config: dict, steps_per_epoch: int, step_key: str, epoch_key: str) -> int:
    if step_key in config:
        return int(config[step_key])
    if epoch_key in config:
        return int(config[epoch_key] * steps_per_epoch)
    return 0


def build_scheduler(optimizer: Optimizer, scheduler_config: dict | None, total_steps: int, steps_per_epoch: int):
    if not scheduler_config or scheduler_config.get("name", "none").lower() == "none":
        return None

    name = scheduler_config["name"].lower()
    warmup_steps = _resolve_step_count(scheduler_config, steps_per_epoch, "warmup_steps", "warmup_epochs")

    if name == "cosine":
        min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.0))

        def lr_lambda(current_step: int) -> float:
            step = current_step + 1
            if warmup_steps > 0 and step <= warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + ((1.0 - min_lr_ratio) * cosine)

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if name == "warmup_hold_decay":
        hold_steps = _resolve_step_count(scheduler_config, steps_per_epoch, "hold_steps", "hold_epochs")
        min_lr = float(scheduler_config.get("min_lr", 0.0))
        base_lr = optimizer.param_groups[0]["lr"]
        min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(current_step: int) -> float:
            step = current_step + 1
            if warmup_steps > 0 and step <= warmup_steps:
                return step / warmup_steps
            if step <= warmup_steps + hold_steps:
                return 1.0
            progress = (step - warmup_steps - hold_steps) / max(1, total_steps - warmup_steps - hold_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + ((1.0 - min_lr_ratio) * cosine)

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
