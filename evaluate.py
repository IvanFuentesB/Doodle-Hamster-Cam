from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import load_model_from_checkpoint
from utils.feature_utils import load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained Doodle Hamster Cam checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / "best_model.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def batched_predictions(
    model: torch.nn.Module,
    samples: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, samples.shape[0], batch_size):
            batch = torch.from_numpy(samples[start : start + batch_size]).float()
            logits = model(batch)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def print_confusion_matrix(confusion: np.ndarray, class_names: list[str]) -> None:
    width = max(10, max(len(name) for name in class_names) + 2)
    header = "true\\pred".ljust(width) + "".join(name[: width - 1].rjust(width) for name in class_names)
    print(header)
    for row_index, class_name in enumerate(class_names):
        row = class_name.ljust(width) + "".join(str(value).rjust(width) for value in confusion[row_index])
        print(row)


def main() -> None:
    args = parse_args()
    samples = np.load(args.data_dir / "samples.npy", allow_pickle=False).astype(np.float32)
    labels = np.load(args.data_dir / "labels.npy", allow_pickle=False).astype(np.int64)

    model, payload = load_model_from_checkpoint(args.checkpoint, map_location="cpu")
    class_names = payload.get("label_names") or load_class_names(args.data_dir / "class_map.json")
    validation_indices = np.asarray(payload.get("validation_indices", []), dtype=np.int64)
    if validation_indices.size == 0:
        raise ValueError("Checkpoint does not contain validation_indices.")

    val_samples = samples[validation_indices]
    val_labels = labels[validation_indices]
    predictions = batched_predictions(model, val_samples, args.batch_size)

    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(val_labels, predictions):
        confusion[int(actual), int(predicted)] += 1

    accuracy = float(np.trace(confusion)) / max(int(confusion.sum()), 1)
    print(f"Validation accuracy: {accuracy:.4f}")
    print()
    print("Per-class metrics")
    for class_index, class_name in enumerate(class_names):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)
        support = int(confusion[class_index, :].sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        print(
            f"{class_name:>12}  precision={precision:.3f} recall={recall:.3f} "
            f"f1={f1:.3f} support={support}"
        )

    print()
    print("Confusion matrix")
    print_confusion_matrix(confusion, list(class_names))


if __name__ == "__main__":
    main()
