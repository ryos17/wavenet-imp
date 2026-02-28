from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn

from .util import get_activation, load_json


@dataclass(frozen=True)
class WaveNetStackConfig:
    """One residual stack in the full WaveNet."""

    input_size: int
    channels: int
    head_size: int
    kernel_size: int
    dilations: List[int]
    activation: str
    gated: bool

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WaveNetStackConfig":
        return WaveNetStackConfig(
            input_size=int(data["input_size"]),
            channels=int(data["channels"]),
            head_size=int(data["head_size"]),
            kernel_size=int(data["kernel_size"]),
            dilations=[int(d) for d in data["dilations"]],
            activation=str(data["activation"]),
            gated=bool(data["gated"]),
        )


class CausalConv1d(nn.Conv1d):
    """
    1D convolution with causal padding.
    Output at time t depends only on inputs <= t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        self._causal_crop = (kernel_size - 1) * dilation
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self._causal_crop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._causal_crop > 0:
            y = y[..., :-self._causal_crop]
        return y


class ResidualBlock(nn.Module):
    """Dilated residual block with skip connection."""

    def __init__(self, channels: int, head_size: int, kernel_size: int, dilation: int, activation: str, gated: bool):
        super().__init__()
        hidden_channels = 2 * channels if gated else channels
        self.dilated = CausalConv1d(channels, hidden_channels, kernel_size, dilation)
        self.activation = get_activation(activation)
        self.residual_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(channels, head_size, kernel_size=1)
        self.gated = gated

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dilated(x)
        if self.gated:
            filter_term, gate_term = torch.chunk(h, chunks=2, dim=1)
            h = self.activation(filter_term) * torch.sigmoid(gate_term)
        else:
            h = self.activation(h)
        residual = x + self.residual_proj(h)
        skip = self.skip_proj(h)
        return residual, skip


class WaveNetStack(nn.Module):
    """A stack of residual blocks over a dilation schedule."""

    def __init__(self, cfg: WaveNetStackConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Conv1d(cfg.input_size, cfg.channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=cfg.channels,
                    head_size=cfg.head_size,
                    kernel_size=cfg.kernel_size,
                    dilation=d,
                    activation=cfg.activation,
                    gated=cfg.gated,
                )
                for d in cfg.dilations
            ]
        )

    @property
    def receptive_field(self) -> int:
        return 1 + (self.cfg.kernel_size - 1) * sum(self.cfg.dilations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        skip_total = None
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip if skip_total is None else skip_total + skip
        return x, skip_total


class WaveNet(nn.Module):
    """
    Readable WaveNet implementation following the core paper design:
    causal dilated convolutions + gated residual blocks + skip outputs.

    The model is built from `layers_configs`, matching `cfg/model/example.json`.
    """

    def __init__(self, layers_configs: Iterable[Dict[str, Any]]):
        super().__init__()
        cfgs = [WaveNetStackConfig.from_dict(c) for c in layers_configs]
        if not cfgs:
            raise ValueError("layers_configs must contain at least one stack config.")

        for i in range(1, len(cfgs)):
            prev_channels = cfgs[i - 1].channels
            if cfgs[i].input_size != prev_channels:
                raise ValueError(
                    f"Stack {i} input_size={cfgs[i].input_size} does not match "
                    f"previous stack channels={prev_channels}."
                )

        self.stacks = nn.ModuleList([WaveNetStack(cfg) for cfg in cfgs])
        self._stack_cfgs = cfgs

    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> "WaveNet":
        if "layers_configs" not in config:
            raise ValueError("Config must include 'layers_configs'.")
        return cls(layers_configs=config["layers_configs"])

    @classmethod
    def from_config_json(cls, config_path: str | Path) -> "WaveNet":
        return cls.from_config_dict(load_json(config_path))

    @property
    def receptive_field(self) -> int:
        # Total context size is additive across stacks in this implementation.
        return 1 + sum(stack.receptive_field - 1 for stack in self.stacks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) or (B, C, T)
        Returns:
            (B, T) if final head has 1 channel and input was 2D.
            Otherwise (B, C_out, T).
        """
        squeeze_output = False
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, T)
            squeeze_output = True
        if x.ndim != 3:
            raise ValueError(f"Expected input with 2 or 3 dims, got shape {tuple(x.shape)}")

        stage_input = x
        stage_head = None
        for stack in self.stacks:
            stage_input, stage_head = stack(stage_input)

        if stage_head is None:
            raise RuntimeError("WaveNet produced no output head.")

        if squeeze_output and stage_head.shape[1] == 1:
            return stage_head[:, 0, :]
        return stage_head
