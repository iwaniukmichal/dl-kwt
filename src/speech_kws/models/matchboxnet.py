from __future__ import annotations

import torch
import torch.nn as nn


class SeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MatchboxResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        repeats: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(repeats):
            layers.append(
                SeparableConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(x) + self.residual(x))


class MatchboxNetBackbone(nn.Module):
    def __init__(self, input_dim: int, config: dict) -> None:
        super().__init__()
        block_channels = int(config.get("block_channels", 64))
        prologue_channels = int(config.get("prologue_channels", 128))
        epilogue_channels = int(config.get("epilogue_channels", 128))
        repeats = int(config.get("subblocks_per_block", 1))
        dropout = float(config.get("dropout", 0.1))
        kernels = [int(value) for value in config.get("block_kernels", [13, 15, 17])]
        conv1_kernel = int(config.get("conv1_kernel", 11))
        conv2_kernel = int(config.get("conv2_kernel", 29))
        conv2_dilation = int(config.get("conv2_dilation", 2))

        self.prologue = SeparableConvBlock(
            in_channels=input_dim,
            out_channels=prologue_channels,
            kernel_size=conv1_kernel,
            dropout=dropout,
        )
        blocks: list[nn.Module] = []
        in_channels = prologue_channels
        for kernel in kernels:
            blocks.append(
                MatchboxResidualBlock(
                    in_channels=in_channels,
                    out_channels=block_channels,
                    kernel_size=kernel,
                    repeats=repeats,
                    dropout=dropout,
                )
            )
            in_channels = block_channels
        self.blocks = nn.Sequential(*blocks)
        self.epilogue = nn.Sequential(
            SeparableConvBlock(
                in_channels=block_channels,
                out_channels=epilogue_channels,
                kernel_size=conv2_kernel,
                dilation=conv2_dilation,
                dropout=dropout,
            ),
            SeparableConvBlock(
                in_channels=epilogue_channels,
                out_channels=epilogue_channels,
                kernel_size=1,
                dropout=dropout,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.output_dim = epilogue_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.prologue(features)
        x = self.blocks(x)
        return self.epilogue(x)
