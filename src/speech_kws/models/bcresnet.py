from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubSpectralNorm(nn.Module):
    def __init__(self, num_features: int, spec_groups: int = 5) -> None:
        super().__init__()
        self.spec_groups = spec_groups
        self.norm = nn.BatchNorm2d(num_features * spec_groups, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, freq_bins, time_steps = x.shape
        if freq_bins % self.spec_groups != 0:
            raise ValueError(
                f"Frequency dimension {freq_bins} must be divisible by spec_groups={self.spec_groups}."
            )
        reshaped = x.view(batch_size, channels * self.spec_groups, freq_bins // self.spec_groups, time_steps)
        normalized = self.norm(reshaped)
        return normalized.view(batch_size, channels, freq_bins, time_steps)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_index: int,
        kernel_size=3,
        stride=1,
        groups: int = 1,
        dilation_by_stage: bool = False,
        activation: str | None = "relu",
        use_batch_norm: bool = True,
        use_subspectral_norm: bool = False,
        spec_groups: int = 5,
    ) -> None:
        super().__init__()
        dilation = 1
        if dilation_by_stage:
            dilation = int(2**stage_index)

        if isinstance(kernel_size, (tuple, list)):
            padding = []
            dilation_tuple = []
            for kernel in kernel_size:
                padding.append(((kernel - 1) // 2) * dilation if kernel > 1 else 0)
                dilation_tuple.append(dilation if kernel > 1 else 1)
            padding = tuple(padding)
            dilation_tuple = tuple(dilation_tuple)
        else:
            padding = ((kernel_size - 1) // 2) * dilation
            dilation_tuple = dilation

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation_tuple,
                groups=groups,
                bias=False,
            )
        ]
        if use_subspectral_norm:
            layers.append(SubSpectralNorm(out_channels, spec_groups=spec_groups))
        elif use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "silu":
            layers.append(nn.SiLU(inplace=True))
        elif activation is None:
            pass
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_index: int,
        stride: tuple[int, int],
        dropout: float,
        spec_groups: int,
    ) -> None:
        super().__init__()
        self.transition_block = in_channels != out_channels

        f2_layers: list[nn.Module] = []
        if self.transition_block:
            f2_layers.append(
                ConvNormAct(
                    in_channels,
                    out_channels,
                    stage_index,
                    kernel_size=1,
                    stride=1,
                    activation="relu",
                )
            )
            in_channels = out_channels
        f2_layers.append(
            ConvNormAct(
                in_channels,
                out_channels,
                stage_index,
                kernel_size=(3, 1),
                stride=(stride[0], 1),
                groups=in_channels,
                activation=None,
                use_batch_norm=False,
                use_subspectral_norm=True,
                spec_groups=spec_groups,
            )
        )
        self.f2 = nn.Sequential(*f2_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.f1 = nn.Sequential(
            ConvNormAct(
                out_channels,
                out_channels,
                stage_index,
                kernel_size=(1, 3),
                stride=(1, stride[1]),
                groups=out_channels,
                dilation_by_stage=True,
                activation="silu",
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.f2(x)
        aux_2d_residual = x
        x = self.avg_pool(x)
        x = self.f1(x)
        x = x + aux_2d_residual
        if not self.transition_block:
            x = x + shortcut
        return F.relu(x, inplace=True)


class BCResNetBackbone(nn.Module):
    def __init__(self, input_dim: int, config: dict) -> None:
        super().__init__()
        variant = str(config.get("variant", "bcresnet-3")).lower()
        base_channels = int(config.get("base_channels", 16 if variant == "bcresnet-3" else 8))
        stage_blocks = [int(value) for value in config.get("stage_blocks", [2, 2, 4, 4])]
        stride_stages = {int(value) for value in config.get("stride_stages", [1, 2])}
        dropout = float(config.get("dropout", 0.1))
        spec_groups = int(config.get("spec_groups", 5))

        channel_plan = [
            base_channels * 2,
            base_channels,
            int(base_channels * 1.5),
            base_channels * 2,
            int(base_channels * 2.5),
            base_channels * 4,
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(1, channel_plan[0], kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(channel_plan[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        for stage_index, num_blocks in enumerate(stage_blocks):
            blocks: list[nn.Module] = []
            channels = [channel_plan[stage_index]] + [channel_plan[stage_index + 1]] * num_blocks
            for block_index in range(num_blocks):
                stride = (2, 1) if stage_index in stride_stages and block_index == 0 else (1, 1)
                blocks.append(
                    BCResBlock(
                        in_channels=channels[block_index],
                        out_channels=channels[block_index + 1],
                        stage_index=stage_index,
                        stride=stride,
                        dropout=dropout,
                        spec_groups=spec_groups,
                    )
                )
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.embedding_head = nn.Sequential(
            nn.Conv2d(
                channel_plan[-2],
                channel_plan[-2],
                kernel_size=(5, 5),
                padding=(0, 2),
                groups=channel_plan[-2],
                bias=False,
            ),
            nn.Conv2d(channel_plan[-2], channel_plan[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channel_plan[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.output_dim = channel_plan[-1]
        self.input_dim = input_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(1)
        x = self.stem(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return self.embedding_head(x)
