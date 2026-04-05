from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from utils.display import WINDOW_CLOSED, draw_overlay_lines, prepare_window, read_window_key
from utils.feature_utils import (
    FEATURE_SPEC_VERSION,
    FEATURE_VECTOR_LENGTH,
    extract_feature_bundle,
    load_class_names,
    parse_class_names,
    save_class_map,
)
from utils.mediapipe_utils import annotate_frame, detect_landmarks, ensure_default_task_models, open_landmarkers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect webcam landmark samples for Doodle Hamster Cam.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--hand-model", type=Path, default=None)
    parser.add_argument("--burst-count", type=int, default=5)
    parser.add_argument("--burst-delay", type=float, default=1.0)
    parser.add_argument("--window-width", type=int, default=1080)
    parser.add_argument("--window-height", type=int, default=660)
    parser.add_argument("--selfie", dest="selfie", action="store_true", default=True)
    parser.add_argument("--no-selfie", dest="selfie", action="store_false")
    return parser.parse_args()


def select_label_from_key(key: int, current_index: int, class_count: int) -> int:
    if ord("1") <= key <= ord("9"):
        candidate = key - ord("1")
        if candidate < class_count:
            return candidate
    if key == ord("0") and class_count > 9:
        return 9
    if key in (ord("["), ord(",")):
        return (current_index - 1) % class_count
    if key in (ord("]"), ord(".")):
        return (current_index + 1) % class_count
    return current_index


def load_existing_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    samples_path = data_dir / "samples.npy"
    labels_path = data_dir / "labels.npy"
    if samples_path.exists() and labels_path.exists():
        samples = np.load(samples_path, allow_pickle=False).astype(np.float32)
        labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
        return samples, labels
    return (
        np.empty((0, FEATURE_VECTOR_LENGTH), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )


def save_dataset(data_dir: Path, samples: np.ndarray, labels: np.ndarray) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "samples.npy", samples.astype(np.float32))
    np.save(data_dir / "labels.npy", labels.astype(np.int64))


def save_metadata(
    data_dir: Path,
    class_names: list[str],
    samples: np.ndarray,
    labels: np.ndarray,
    selfie: bool,
    burst_count: int,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: int((labels == index).sum()) for index, label in enumerate(class_names)}
    payload = {
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "input_dim": FEATURE_VECTOR_LENGTH,
        "sample_count": int(samples.shape[0]),
        "class_names": class_names,
        "counts": counts,
        "selfie_mode": bool(selfie),
        "burst_count": int(burst_count),
    }
    import json

    with (data_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_class_names(data_dir: Path, requested: str | None, append_label: str | None = None) -> list[str]:
    class_map_path = data_dir / "class_map.json"
    if class_map_path.exists():
        existing = load_class_names(class_map_path)
        if requested:
            desired = parse_class_names(requested)
            if desired != existing:
                raise ValueError(
                    "Existing class_map.json does not match --classes. "
                    "Keep the label order stable or start with a fresh data directory."
                )
        if append_label and append_label not in existing:
            existing.append(append_label)
            save_class_map(class_map_path, existing)
        return existing

    class_names = parse_class_names(requested)
    if append_label and append_label not in class_names:
        class_names.append(append_label)
    save_class_map(class_map_path, class_names)
    return class_names


def append_sample(
    samples: np.ndarray,
    labels: np.ndarray,
    feature_vector: np.ndarray,
    label_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_row = feature_vector.reshape(1, -1).astype(np.float32)
    updated_samples = np.vstack([samples, sample_row])
    updated_labels = np.append(labels, np.int64(label_index))
    return updated_samples, updated_labels


def main() -> None:
    args = parse_args()
    window_name = "Doodle Hamster Cam - Data Collection"
    data_dir = args.data_dir.resolve()
    class_names = resolve_class_names(data_dir, args.classes)
    current_label_index = 0
    samples, labels = load_existing_arrays(data_dir)

    face_model_path, hand_model_path = ensure_default_task_models(
        args.face_model,
        args.hand_model,
        models_dir=Path("models"),
    )

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")
    prepare_window(window_name, width=args.window_width, height=args.window_height)

    pending_burst = 0
    burst_starts_at: float | None = None
    message = "Ready to capture."
    message_frames = 0
    start_time = time.perf_counter()

    try:
        with open_landmarkers(face_model_path, hand_model_path) as (face_landmarker, hand_landmarker):
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")

                if args.selfie:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                now = time.perf_counter()
                face_result, hand_result = detect_landmarks(face_landmarker, hand_landmarker, frame, timestamp_ms)
                bundle = extract_feature_bundle(face_result, hand_result)
                annotated = annotate_frame(frame, face_result, hand_result)

                if pending_burst > 0:
                    if burst_starts_at is not None and now < burst_starts_at:
                        countdown = max(0.0, burst_starts_at - now)
                        message = f"Burst starts in {countdown:.1f}s for {class_names[current_label_index]}"
                        message_frames = 2
                    elif bundle is not None:
                        samples, labels = append_sample(samples, labels, bundle.vector, current_label_index)
                        save_dataset(data_dir, samples, labels)
                        save_metadata(data_dir, class_names, samples, labels, args.selfie, args.burst_count)
                        pending_burst -= 1
                        message = (
                            f"Burst capture: saved {class_names[current_label_index]} "
                            f"({args.burst_count - pending_burst}/{args.burst_count})"
                        )
                        message_frames = 20
                    else:
                        message = "Burst is armed. Waiting for a clear face."
                        message_frames = 2

                lines = [
                    f"Current label: {class_names[current_label_index]}",
                    f"Saved samples: {samples.shape[0]}",
                    f"Face detected: {'yes' if bundle is not None else 'no'}",
                    "Keys: 1-9 / 0 choose label, [ ] cycle, space save, b burst, q quit",
                ]
                if message_frames > 0:
                    lines.append(message)
                    message_frames -= 1
                elif bundle is None:
                    lines.append("A visible face is required before capture.")
                else:
                    lines.append("Landmarks look valid. Press space to save a sample.")

                draw_overlay_lines(annotated, lines)
                cv2.imshow(window_name, annotated)

                key = read_window_key(window_name)
                if key in (WINDOW_CLOSED, 27, ord("q")):
                    break

                new_index = select_label_from_key(key, current_label_index, len(class_names))
                if new_index != current_label_index:
                    current_label_index = new_index
                    message = f"Selected label: {class_names[current_label_index]}"
                    message_frames = 25
                    continue

                if key == ord(" "):
                    if bundle is None:
                        message = "No face detected. Sample not saved."
                        message_frames = 30
                        continue
                    samples, labels = append_sample(samples, labels, bundle.vector, current_label_index)
                    save_dataset(data_dir, samples, labels)
                    save_metadata(data_dir, class_names, samples, labels, args.selfie, args.burst_count)
                    message = f"Saved sample for {class_names[current_label_index]}"
                    message_frames = 30
                elif key == ord("b"):
                    if bundle is None:
                        message = "No face detected. Burst capture not started."
                        message_frames = 30
                        continue
                    pending_burst = max(1, args.burst_count)
                    burst_starts_at = time.perf_counter() + max(0.0, float(args.burst_delay))
                    message = f"Burst armed for {class_names[current_label_index]}. Hold the pose."
                    message_frames = 30
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
