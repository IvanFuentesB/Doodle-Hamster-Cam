from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import LandmarkMLP, MLPConfig, parse_hidden_dims, save_checkpoint
from utils.augment import AugmentConfig, augment_feature_vector
from utils.feature_utils import FEATURE_SPEC_VERSION, load_class_names, save_class_map


class LandmarkDataset(Dataset):
    def __init__(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        seed: int = 0,
        augment_config: AugmentConfig | None = None,
    ) -> None:
        self.samples = samples.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.augment_config = augment_config or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.samples[index]
        if self.augment:
            features = augment_feature_vector(features, self.rng, self.augment_config)
        target = self.labels[index]
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Doodle Hamster Cam landmark MLP.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", type=str, default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    samples_path = data_dir / "samples.npy"
    labels_path = data_dir / "labels.npy"
    class_map_path = data_dir / "class_map.json"
    if not samples_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Expected data/samples.npy and data/labels.npy. Run collect_data.py first.")

    samples = np.load(samples_path, allow_pickle=False).astype(np.float32)
    labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples to be 2D, got shape {samples.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels to be 1D, got shape {labels.shape}.")
    if samples.shape[0] != labels.shape[0]:
        raise ValueError("samples.npy and labels.npy must contain the same number of rows.")

    class_names = load_class_names(class_map_path if class_map_path.exists() else None)
    return samples, labels, class_names


def split_indices(num_samples: int, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_samples < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split.")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)

    val_count = int(round(num_samples * val_split))
    val_count = max(1, min(num_samples - 1, val_count))
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return train_indices, val_indices


def run_epoch(
    model: LandmarkMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * features.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == targets).sum().item())
            total_examples += int(features.size(0))

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples, labels, class_names = load_dataset(args.data_dir)
    input_dim = int(samples.shape[1])
    num_classes = len(class_names)

    train_indices, val_indices = split_indices(samples.shape[0], args.val_split, args.seed)
    train_samples = samples[train_indices]
    train_labels = labels[train_indices]
    val_samples = samples[val_indices]
    val_labels = labels[val_indices]

    train_dataset = LandmarkDataset(train_samples, train_labels, augment=True, seed=args.seed)
    val_dataset = LandmarkDataset(val_samples, val_labels, augment=False, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    config = MLPConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
    )
    model = LandmarkMLP(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = args.checkpoint_dir / "best_model.pt"
    best_val_accuracy = -1.0

    print(f"Training on {train_samples.shape[0]} samples, validating on {val_samples.shape[0]} samples.")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                class_names,
                FEATURE_SPEC_VERSION,
                val_indices,
            )

    save_class_map(args.data_dir / "class_map.json", class_names)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
