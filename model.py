from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


@dataclass(slots=True)
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2


class LandmarkMLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, config.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_hidden_dims(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        dims = tuple(int(chunk.strip()) for chunk in value.split(",") if chunk.strip())
    else:
        dims = tuple(int(chunk) for chunk in value)
    if not dims:
        raise ValueError("hidden_dims must contain at least one layer size.")
    return dims


def save_checkpoint(
    path: Path,
    model: LandmarkMLP,
    config: MLPConfig,
    class_names: Sequence[str],
    feature_spec_version: str,
    validation_indices: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": config.input_dim,
        "num_classes": config.num_classes,
        "hidden_dims": list(config.hidden_dims),
        "dropout": config.dropout,
        "label_names": list(class_names),
        "feature_spec_version": feature_spec_version,
        "validation_indices": [int(index) for index in validation_indices],
    }
    torch.save(payload, path)


def load_checkpoint_payload(path: Path, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} is invalid.")
    return payload


def load_model_from_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[LandmarkMLP, dict]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    config = MLPConfig(
        input_dim=int(payload["input_dim"]),
        num_classes=int(payload["num_classes"]),
        hidden_dims=parse_hidden_dims(payload["hidden_dims"]),
        dropout=float(payload.get("dropout", 0.2)),
    )
    model = LandmarkMLP(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
