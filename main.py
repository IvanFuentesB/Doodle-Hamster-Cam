from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from model import load_model_from_checkpoint
from utils.display import (
    WINDOW_CLOSED,
    compose_app_frame,
    load_reaction_images,
    prepare_window,
    read_window_key,
    save_screenshot,
)
from utils.feature_utils import (
    DEFAULT_CLASSES,
    FEATURE_SPEC_VERSION,
    NEUTRAL_LABEL,
    extract_feature_bundle,
    load_class_names,
    load_prototype_bank,
    prototype_probabilities,
    rule_based_probabilities,
)
from utils.mediapipe_utils import annotate_frame, detect_landmarks, ensure_default_task_models, open_landmarkers
from utils.smoothing import PredictionSmoother


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live Doodle Hamster Cam webcam app.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / "best_model.pt")
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--hand-model", type=Path, default=None)
    parser.add_argument("--mode", choices=("auto", "ml", "rule"), default="auto")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--confirm-seconds", type=float, default=0.17)
    parser.add_argument("--window-width", type=int, default=1020)
    parser.add_argument("--window-height", type=int, default=640)
    parser.add_argument("--selfie", dest="selfie", action="store_true", default=True)
    parser.add_argument("--no-selfie", dest="selfie", action="store_false")
    return parser.parse_args()


def softmax_logits(logits: torch.Tensor) -> np.ndarray:
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probabilities.astype(np.float32)


def load_runtime_model(checkpoint_path: Path) -> tuple[torch.nn.Module | None, dict | None]:
    if not checkpoint_path.exists():
        return None, None
    model, payload = load_model_from_checkpoint(checkpoint_path, map_location="cpu")
    feature_version = payload.get("feature_spec_version")
    if feature_version not in (None, FEATURE_SPEC_VERSION):
        raise ValueError(
            f"Checkpoint feature spec {feature_version!r} does not match {FEATURE_SPEC_VERSION!r}."
        )
    return model, payload


def resolve_class_names(payload: dict | None, data_dir: Path) -> list[str]:
    class_map_path = data_dir / "class_map.json"
    if class_map_path.exists():
        return load_class_names(class_map_path)
    if payload and payload.get("label_names"):
        return list(payload["label_names"])
    return list(DEFAULT_CLASSES)


def blend_probabilities(primary: np.ndarray, secondary: np.ndarray | None, primary_weight: float) -> np.ndarray:
    if secondary is None:
        return primary.astype(np.float32)
    blended = primary_weight * primary + (1.0 - primary_weight) * secondary
    blended = np.clip(blended, 1e-6, None)
    blended /= blended.sum()
    return blended.astype(np.float32)


def count_observed_labels(prototype_bank: object | None) -> int:
    counts = getattr(prototype_bank, "counts", None)
    if counts is None:
        return 0
    return int(np.sum(np.asarray(counts) > 0))


def update_confirmed_prediction(
    target_index: int,
    displayed_index: int,
    pending_index: int | None,
    pending_since: float | None,
    now: float,
    confirm_seconds: float,
) -> tuple[int, int | None, float | None]:
    if target_index == displayed_index:
        return displayed_index, None, None
    if pending_index != target_index:
        return displayed_index, target_index, now
    if pending_since is None:
        return displayed_index, target_index, now
    if now - pending_since >= confirm_seconds:
        return target_index, None, None
    return displayed_index, pending_index, pending_since


def main() -> None:
    args = parse_args()
    window_name = "Doodle Hamster Cam"
    checkpoint_path = args.checkpoint.resolve()
    model, payload = load_runtime_model(checkpoint_path)
    data_dir = args.data_dir.resolve()
    assets_dir = args.assets_dir.resolve()
    class_names = resolve_class_names(payload, data_dir)
    neutral_index = class_names.index(NEUTRAL_LABEL) if NEUTRAL_LABEL in class_names else 0
    reaction_images = load_reaction_images(assets_dir, class_names)
    prototype_bank = load_prototype_bank(data_dir, class_names)
    checkpoint_labels = list(payload["label_names"]) if payload and payload.get("label_names") else None
    ml_compatible = model is not None and (checkpoint_labels is None or checkpoint_labels == class_names)
    observed_label_count = count_observed_labels(prototype_bank)
    ml_ready = ml_compatible and (prototype_bank is None or observed_label_count >= 2)

    face_model_path, hand_model_path = ensure_default_task_models(
        args.face_model,
        args.hand_model,
        models_dir=Path("models"),
    )

    requested_mode = args.mode
    current_mode = "ml" if requested_mode == "auto" and ml_ready else "rule"
    if requested_mode == "ml" and ml_ready:
        current_mode = "ml"

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")
    prepare_window(window_name, width=args.window_width, height=args.window_height)

    smoother = PredictionSmoother(window_size=args.smooth_window, missing_threshold=5)
    device = torch.device("cpu")
    last_tick = time.perf_counter()
    fps = 0.0
    start_time = time.perf_counter()
    transient_message = ""
    transient_frames = 0
    displayed_index = neutral_index
    pending_index: int | None = None
    pending_since: float | None = None
    if model is not None and not ml_compatible:
        transient_message = "Checkpoint labels do not match. Retrain for ML."
        transient_frames = 75
    elif model is not None and ml_compatible and not ml_ready:
        transient_message = "Need at least 2 labeled reactions before ML is useful."
        transient_frames = 90

    try:
        with open_landmarkers(face_model_path, hand_model_path) as (face_landmarker, hand_landmarker):
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")

                if args.selfie:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                face_result, hand_result = detect_landmarks(face_landmarker, hand_landmarker, frame, timestamp_ms)
                bundle = extract_feature_bundle(face_result, hand_result)
                annotated = annotate_frame(frame, face_result, hand_result)

                if bundle is None:
                    probabilities = None
                    status_line = "No face. Neutral."
                else:
                    prototype_scores = prototype_probabilities(bundle.vector, prototype_bank)
                    if current_mode == "ml" and model is not None and ml_ready:
                        with torch.no_grad():
                            features = torch.from_numpy(bundle.vector).float().unsqueeze(0).to(device)
                            ml_probabilities = softmax_logits(model(features))
                        probabilities = blend_probabilities(ml_probabilities, prototype_scores, primary_weight=0.78)
                        status_line = (
                            "ML + mapped memory."
                            if prototype_scores is not None
                            else "ML classifier."
                        )
                    else:
                        rule_probabilities = rule_based_probabilities(bundle, class_names)
                        probabilities = blend_probabilities(rule_probabilities, prototype_scores, primary_weight=0.38)
                        status_line = (
                            "Mapped memory + rules."
                            if prototype_scores is not None
                            else "Rule fallback."
                        )

                smoothed = smoother.update(probabilities, neutral_index, len(class_names))
                prediction_index = int(np.argmax(smoothed))
                confidence = float(smoothed[prediction_index])

                confidence_gate = 0.42 if current_mode == "ml" else 0.16
                if prediction_index != neutral_index and confidence < confidence_gate:
                    prediction_index = neutral_index
                    confidence = float(smoothed[neutral_index])
                    status_line = (
                        "Low confidence. Neutral."
                        if current_mode == "ml"
                        else "Weak confidence. Neutral."
                    )

                displayed_index, pending_index, pending_since = update_confirmed_prediction(
                    prediction_index,
                    displayed_index,
                    pending_index,
                    pending_since,
                    now=time.perf_counter(),
                    confirm_seconds=max(0.0, float(args.confirm_seconds)),
                )
                label = class_names[displayed_index]
                confidence = float(smoothed[displayed_index])
                if pending_index is not None and pending_index != displayed_index:
                    status_line = f"{status_line} Confirming..."
                reaction_image = reaction_images.get(label)
                if reaction_image is None:
                    reaction_image = reaction_images.get(NEUTRAL_LABEL)

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - last_tick, 1e-6)
                fps = instantaneous_fps if fps == 0.0 else 0.9 * fps + 0.1 * instantaneous_fps
                last_tick = now

                if transient_frames > 0:
                    status_line = f"{status_line} {transient_message}"
                    transient_frames -= 1

                composite = compose_app_frame(
                    annotated,
                    reaction_image,
                    label,
                    confidence,
                    current_mode.upper(),
                    fps,
                    status_line,
                )
                cv2.imshow(window_name, composite)

                key = read_window_key(window_name)
                if key in (WINDOW_CLOSED, 27, ord("q")):
                    break
                if key == ord("m"):
                    if model is None or not ml_ready:
                        current_mode = "rule"
                        transient_message = (
                            "Need a compatible checkpoint and at least 2 labeled reactions."
                        )
                    else:
                        current_mode = "rule" if current_mode == "ml" else "ml"
                        transient_message = f"{current_mode.upper()} mode."
                    transient_frames = 45
                elif key == ord("r"):
                    smoother.reset()
                    displayed_index = neutral_index
                    pending_index = None
                    pending_since = None
                    class_names = resolve_class_names(payload, data_dir)
                    neutral_index = class_names.index(NEUTRAL_LABEL) if NEUTRAL_LABEL in class_names else 0
                    displayed_index = neutral_index
                    checkpoint_labels = list(payload["label_names"]) if payload and payload.get("label_names") else None
                    ml_compatible = model is not None and (checkpoint_labels is None or checkpoint_labels == class_names)
                    observed_label_count = count_observed_labels(prototype_bank)
                    ml_ready = ml_compatible and (prototype_bank is None or observed_label_count >= 2)
                    if current_mode == "ml" and not ml_ready:
                        current_mode = "rule"
                    reaction_images = load_reaction_images(assets_dir, class_names)
                    prototype_bank = load_prototype_bank(data_dir, class_names)
                    observed_label_count = count_observed_labels(prototype_bank)
                    ml_ready = ml_compatible and (prototype_bank is None or observed_label_count >= 2)
                    transient_message = "Reloaded PNGs and mapped poses."
                    transient_frames = 45
                elif key == ord("s"):
                    saved_path = save_screenshot(composite, Path("captures").resolve(), label)
                    transient_message = f"Saved {saved_path.name}."
                    transient_frames = 60
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
