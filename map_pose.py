from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import cv2

from collect_data import append_sample, load_existing_arrays, resolve_class_names, save_dataset, save_metadata
from utils.display import (
    WINDOW_CLOSED,
    compose_pose_mapping_frame,
    prepare_window,
    read_window_key,
    register_custom_asset,
)
from utils.feature_utils import extract_feature_bundle, sanitize_label_name
from utils.mediapipe_utils import annotate_frame, detect_landmarks, ensure_default_task_models, open_landmarkers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map a custom reaction image to a pose and record samples for it.")
    parser.add_argument("--label", default=None, help="Class label to map the image to. Defaults to the image filename.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the custom reaction image.")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--hand-model", type=Path, default=None)
    parser.add_argument("--burst-count", type=int, default=5)
    parser.add_argument("--burst-delay", type=float, default=1.0)
    parser.add_argument("--window-width", type=int, default=1020)
    parser.add_argument("--window-height", type=int, default=640)
    parser.add_argument("--train-after", action="store_true", help="Start train.py after you finish capture.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=str, default="256,128,64")
    parser.add_argument("--selfie", dest="selfie", action="store_true", default=True)
    parser.add_argument("--no-selfie", dest="selfie", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_name = "Doodle Hamster Cam - Pose Mapper"
    data_dir = args.data_dir.resolve()
    assets_dir = args.assets_dir.resolve()
    mapped_label = sanitize_label_name(args.label or args.image.stem)
    class_names = resolve_class_names(data_dir, args.classes, append_label=mapped_label)
    label_index = class_names.index(mapped_label)
    samples, labels = load_existing_arrays(data_dir)
    save_metadata(data_dir, class_names, samples, labels, args.selfie, args.burst_count)
    label_sample_count = int((labels == label_index).sum())

    asset_path = register_custom_asset(args.image.resolve(), assets_dir, mapped_label)
    reference_image = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise RuntimeError(f"Could not read the reference image: {asset_path}")

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
    message = f"Mapped {asset_path.name} to {mapped_label}. Match the pose."
    should_train = False
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
                        message = f"Burst starts in {countdown:.1f}s for {mapped_label}."
                    elif bundle is not None:
                        samples, labels = append_sample(samples, labels, bundle.vector, label_index)
                        label_sample_count += 1
                        save_dataset(data_dir, samples, labels)
                        save_metadata(data_dir, class_names, samples, labels, args.selfie, args.burst_count)
                        pending_burst -= 1
                        saved = args.burst_count - pending_burst
                        message = f"Saved burst sample {saved}/{args.burst_count} for {mapped_label}."
                    else:
                        message = "Burst is armed. Waiting for a clear face."
                elif bundle is None:
                    message = "Face not detected clearly enough. Adjust pose and lighting."

                composed = compose_pose_mapping_frame(
                    annotated,
                    reference_image,
                    mapped_label,
                    label_sample_count,
                    message,
                )
                cv2.imshow(window_name, composed)

                key = read_window_key(window_name)
                if key in (WINDOW_CLOSED, 27, ord("q")):
                    break
                if key == ord("t"):
                    should_train = True
                    break
                if key == ord(" "):
                    if bundle is None:
                        message = "No valid face detected. Sample not saved."
                        continue
                    samples, labels = append_sample(samples, labels, bundle.vector, label_index)
                    label_sample_count += 1
                    save_dataset(data_dir, samples, labels)
                    save_metadata(data_dir, class_names, samples, labels, args.selfie, args.burst_count)
                    message = f"Saved sample {label_sample_count} for {mapped_label}."
                elif key == ord("b"):
                    if bundle is None:
                        message = "No valid face detected. Burst capture not started."
                        continue
                    pending_burst = max(1, args.burst_count)
                    burst_starts_at = time.perf_counter() + max(0.0, float(args.burst_delay))
                    message = f"Burst armed for {mapped_label}. Hold the pose."
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if args.train_after or should_train:
        command = [
            str(Path(sys.executable)),
            str(Path("train.py").resolve()),
            "--data-dir",
            str(data_dir),
            "--checkpoint-dir",
            str(args.checkpoint_dir.resolve()),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--hidden-dims",
            args.hidden_dims,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
