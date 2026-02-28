from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_activation(name: str) -> nn.Module:
    """Return a torch activation module from a readable name."""
    key = name.strip().lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key in {"leakyrelu", "leaky_relu"}:
        return nn.LeakyReLU(negative_slope=0.2)
    if key in {"gelu"}:
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_audio_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(Path(path)), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def pre_emphasis(x: torch.Tensor, coeff: float=0.95) -> torch.Tensor:
    """
    Pre-emphasis filter: y[t] = x[t] - coeff * x[t-1].
    Applied along the time dimension.
    """
    if coeff <= 0.0:
        return x
    y = torch.empty_like(x)
    y[..., 0] = x[..., 0]
    y[..., 1:] = x[..., 1:] - coeff * x[..., :-1]
    return y


class RandomChunkDataset(Dataset):
    """
    Draw random aligned chunks from paired input/target waveforms.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, chunk_size: int, samples_per_epoch: int):
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Input and target audio must be 1-D mono signals.")
        if len(x) != len(y):
            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]
        if len(x) < chunk_size:
            raise ValueError(
                f"Audio length ({len(x)}) is shorter than chunk_size ({chunk_size})."
            )

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.chunk_size = int(chunk_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_start = len(x) - self.chunk_size

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = random.randint(0, self.max_start)
        end = start + self.chunk_size
        return self.x[start:end], self.y[start:end]


def build_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    chunk_size: int,
    steps_per_epoch: int,
    num_workers: int,
) -> DataLoader:
    ds = RandomChunkDataset(
        x=x,
        y=y,
        chunk_size=chunk_size,
        samples_per_epoch=steps_per_epoch * batch_size,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
