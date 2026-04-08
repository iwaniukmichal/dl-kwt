from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, _ = x.shape
        reshaped = x.view(batch_size, tokens, self.num_heads, self.head_dim)
        return reshaped.transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(inputs))
        key = self._split_heads(self.key(inputs))
        value = self._split_heads(self.value(inputs))
        # Keep attention score and softmax math in fp32 to avoid AMP-related overflows.
        query_fp32 = query.float()
        key_fp32 = key.float()
        value_fp32 = value.float()
        attention_scores = torch.matmul(query_fp32, key_fp32.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = attention_scores.softmax(dim=-1)
        attended = torch.matmul(attention_weights, value_fp32).to(dtype=inputs.dtype)
        attended = attended.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(1), self.embed_dim)
        return self.combine(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        prenorm: bool,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.prenorm = prenorm

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            normalized = self.norm1(inputs)
            attention = self.dropout1(self.attention(normalized))
            hidden = inputs + attention
            mlp_hidden = self.dropout2(self.mlp(self.norm2(hidden)))
            return hidden + mlp_hidden

        attention = self.dropout1(self.attention(inputs))
        hidden = self.norm1(inputs + attention)
        mlp_hidden = self.dropout2(self.mlp(hidden))
        return self.norm2(hidden + mlp_hidden)


class _TransformerBranch(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        num_patches: int,
        d_model: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        prenorm: bool,
    ) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    prenorm=prenorm,
                )
                for _ in range(depth)
            ]
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_projection(sequence)
        batch_size, patch_count, _ = tokens.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : patch_count + 1, :]
        for block in self.blocks:
            tokens = block(tokens)
        return tokens[:, 0]


class KWTBackbone(nn.Module):
    def __init__(self, input_dim: int, config: dict) -> None:
        super().__init__()
        d_model = int(config.get("d_model", 64))
        num_heads = int(config.get("num_heads", 1))
        depth = int(config.get("depth", 12))
        mlp_dim = int(config.get("mlp_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        attention_type = str(config.get("attention_type", "time")).lower()
        max_frames = int(config.get("max_frames", 98))
        prenorm = bool(config.get("prenorm", False))

        if attention_type not in {"time", "freq", "both"}:
            raise ValueError(f"Unsupported KWT attention_type: {attention_type}")

        self.attention_type = attention_type
        self.time_branch = None
        self.freq_branch = None
        if attention_type in {"time", "both"}:
            self.time_branch = _TransformerBranch(
                patch_dim=input_dim,
                num_patches=max_frames,
                d_model=d_model,
                depth=depth,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                prenorm=prenorm,
            )
        if attention_type in {"freq", "both"}:
            self.freq_branch = _TransformerBranch(
                patch_dim=max_frames,
                num_patches=input_dim,
                d_model=d_model,
                depth=depth,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                prenorm=prenorm,
            )

        self.output_dim = d_model if attention_type != "both" else d_model * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        if self.time_branch is not None:
            outputs.append(self.time_branch(features.transpose(1, 2)))
        if self.freq_branch is not None:
            outputs.append(self.freq_branch(features))
        if not outputs:
            raise RuntimeError("KWT backbone has no active transformer branch.")
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=-1)
