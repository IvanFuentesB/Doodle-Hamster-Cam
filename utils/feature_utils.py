from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

DEFAULT_CLASSES = [
    "neutral",
    "happy",
    "shocked",
    "smug",
    "crying",
    "angry",
    "suspicious",
    "peace_pose",
    "phone_pose",
    "blush",
    "meltdown",
    "pointing",
    "batman",
    "lip_bite",
    "hand_up",
    "pout",
]

FACE_LANDMARK_COUNT = 478
HAND_LANDMARK_COUNT = 21
FACE_CENTER_INDICES = (234, 454, 10, 152)
FACE_SCALE_INDICES = (234, 454)
FEATURE_SPEC_VERSION = "v1_face478_hand21_xy_mask2"
FACE_BLOCK_SIZE = FACE_LANDMARK_COUNT * 2
HAND_BLOCK_SIZE = HAND_LANDMARK_COUNT * 2
FEATURE_VECTOR_LENGTH = FACE_BLOCK_SIZE + HAND_BLOCK_SIZE * 2 + 2
NEUTRAL_LABEL = "neutral"


@dataclass(slots=True)
class FeatureBundle:
    vector: np.ndarray
    face_points: np.ndarray
    left_hand_points: np.ndarray
    right_hand_points: np.ndarray
    hand_presence: np.ndarray
    blendshapes: dict[str, float]
    metrics: dict[str, float]


@dataclass(slots=True)
class PrototypeBank:
    class_names: list[str]
    prototypes: np.ndarray
    counts: np.ndarray


def parse_class_names(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_CLASSES)
    class_names = [item.strip() for item in value.split(",") if item.strip()]
    if not class_names:
        raise ValueError("At least one class label is required.")
    return class_names


def sanitize_label_name(value: str) -> str:
    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Label name must contain letters or numbers.")
    return cleaned


def class_map_from_names(class_names: Sequence[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(class_names)}


def ordered_labels_from_class_map(class_map: Mapping[str, int]) -> list[str]:
    return [label for label, _ in sorted(class_map.items(), key=lambda item: item[1])]


def save_class_map(path: Path, class_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(class_map_from_names(class_names), handle, indent=2)


def load_class_map(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid class map in {path}.")
    return {str(key): int(value) for key, value in payload.items()}


def load_class_names(path: Path | None, fallback: Sequence[str] | None = None) -> list[str]:
    if path and path.exists():
        return ordered_labels_from_class_map(load_class_map(path))
    return list(fallback or DEFAULT_CLASSES)


def neutral_distribution(class_names: Sequence[str]) -> np.ndarray:
    probabilities = np.zeros(len(class_names), dtype=np.float32)
    neutral_index = class_names.index(NEUTRAL_LABEL) if NEUTRAL_LABEL in class_names else 0
    probabilities[neutral_index] = 1.0
    return probabilities


def load_prototype_bank(data_dir: Path, class_names: Sequence[str]) -> PrototypeBank | None:
    samples_path = data_dir / "samples.npy"
    labels_path = data_dir / "labels.npy"
    if not samples_path.exists() or not labels_path.exists():
        return None

    samples = np.load(samples_path, allow_pickle=False).astype(np.float32)
    labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
    if samples.ndim != 2 or labels.ndim != 1 or samples.shape[0] != labels.shape[0]:
        raise ValueError("Saved samples and labels are malformed. Rebuild the dataset files.")
    if samples.shape[0] == 0:
        return None

    stored_class_names = load_class_names(data_dir / "class_map.json", fallback=class_names)
    stored_map = class_map_from_names(stored_class_names)
    prototypes = np.zeros((len(class_names), samples.shape[1]), dtype=np.float32)
    counts = np.zeros(len(class_names), dtype=np.int64)

    for runtime_index, label in enumerate(class_names):
        stored_index = stored_map.get(label)
        if stored_index is None:
            continue
        class_mask = labels == stored_index
        if not np.any(class_mask):
            continue
        prototypes[runtime_index] = samples[class_mask].mean(axis=0, dtype=np.float32)
        counts[runtime_index] = int(class_mask.sum())

    if not np.any(counts > 0):
        return None
    return PrototypeBank(class_names=list(class_names), prototypes=prototypes, counts=counts)


def prototype_probabilities(
    feature_vector: np.ndarray,
    prototype_bank: PrototypeBank | None,
    temperature: float = 0.14,
) -> np.ndarray | None:
    if prototype_bank is None or prototype_bank.prototypes.size == 0:
        return None

    valid_mask = prototype_bank.counts > 0
    if not np.any(valid_mask):
        return None

    vector = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    vector_norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector_norm) or vector_norm < 1e-6:
        return None

    prototypes = prototype_bank.prototypes[valid_mask]
    prototype_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototype_norms = np.clip(prototype_norms, 1e-6, None)
    normalized_prototypes = prototypes / prototype_norms
    normalized_vector = vector / vector_norm

    similarities = normalized_prototypes @ normalized_vector
    class_weights = 1.0 + 0.1 * np.log1p(prototype_bank.counts[valid_mask].astype(np.float32))
    raw_scores = np.exp((similarities - similarities.max()) / temperature) * class_weights

    probabilities = np.full(len(prototype_bank.class_names), 1e-5, dtype=np.float32)
    probabilities[valid_mask] = raw_scores.astype(np.float32)
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float32)


def _extract_xy_array(landmarks: Sequence[object], expected_count: int, kind: str) -> np.ndarray:
    points = np.asarray([(landmark.x, landmark.y) for landmark in landmarks], dtype=np.float32)
    if points.shape != (expected_count, 2):
        raise ValueError(
            f"Expected {expected_count} {kind} landmarks but received shape {points.shape}."
        )
    return points


def compute_face_center_and_scale(face_points: np.ndarray) -> tuple[np.ndarray, float]:
    center = face_points[list(FACE_CENTER_INDICES)].mean(axis=0)
    scale = float(np.linalg.norm(face_points[FACE_SCALE_INDICES[0]] - face_points[FACE_SCALE_INDICES[1]]))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("Face scale is invalid. Move closer to the camera and keep the face visible.")
    return center.astype(np.float32), scale


def normalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    normalized = (points - center) / scale
    return normalized.astype(np.float32)


def pack_feature_vector(
    face_points: np.ndarray,
    left_hand_points: np.ndarray,
    right_hand_points: np.ndarray,
    hand_presence: np.ndarray,
) -> np.ndarray:
    vector = np.concatenate(
        [
            face_points.reshape(-1),
            left_hand_points.reshape(-1),
            right_hand_points.reshape(-1),
            hand_presence.astype(np.float32),
        ]
    )
    if vector.shape[0] != FEATURE_VECTOR_LENGTH:
        raise ValueError(f"Expected feature length {FEATURE_VECTOR_LENGTH}, got {vector.shape[0]}.")
    return vector.astype(np.float32)


def unpack_feature_vector(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_vector = np.asarray(vector, dtype=np.float32)
    if feature_vector.shape != (FEATURE_VECTOR_LENGTH,):
        raise ValueError(
            f"Expected feature vector of shape ({FEATURE_VECTOR_LENGTH},), got {feature_vector.shape}."
        )
    cursor = 0
    face_points = feature_vector[cursor : cursor + FACE_BLOCK_SIZE].reshape(FACE_LANDMARK_COUNT, 2).copy()
    cursor += FACE_BLOCK_SIZE
    left_hand_points = feature_vector[cursor : cursor + HAND_BLOCK_SIZE].reshape(HAND_LANDMARK_COUNT, 2).copy()
    cursor += HAND_BLOCK_SIZE
    right_hand_points = feature_vector[cursor : cursor + HAND_BLOCK_SIZE].reshape(HAND_LANDMARK_COUNT, 2).copy()
    cursor += HAND_BLOCK_SIZE
    hand_presence = feature_vector[cursor : cursor + 2].copy()
    return face_points, left_hand_points, right_hand_points, hand_presence


def mirror_feature_vector(vector: np.ndarray) -> np.ndarray:
    face_points, left_hand_points, right_hand_points, hand_presence = unpack_feature_vector(vector)
    face_points[:, 0] *= -1.0
    left_hand_points[:, 0] *= -1.0
    right_hand_points[:, 0] *= -1.0
    return pack_feature_vector(face_points, right_hand_points, left_hand_points, hand_presence[::-1])


def scale_feature_vector(vector: np.ndarray, factor: float) -> np.ndarray:
    face_points, left_hand_points, right_hand_points, hand_presence = unpack_feature_vector(vector)
    face_points *= factor
    left_hand_points *= factor
    right_hand_points *= factor
    return pack_feature_vector(face_points, left_hand_points, right_hand_points, hand_presence)


def add_coordinate_noise(vector: np.ndarray, noise: np.ndarray) -> np.ndarray:
    face_points, left_hand_points, right_hand_points, hand_presence = unpack_feature_vector(vector)
    if noise.shape != (FEATURE_VECTOR_LENGTH,):
        raise ValueError("Noise vector shape must match the feature vector.")
    face_points += noise[:FACE_BLOCK_SIZE].reshape(FACE_LANDMARK_COUNT, 2)
    left_hand_points += noise[FACE_BLOCK_SIZE : FACE_BLOCK_SIZE + HAND_BLOCK_SIZE].reshape(HAND_LANDMARK_COUNT, 2)
    right_hand_points += noise[
        FACE_BLOCK_SIZE + HAND_BLOCK_SIZE : FACE_BLOCK_SIZE + HAND_BLOCK_SIZE * 2
    ].reshape(HAND_LANDMARK_COUNT, 2)
    return pack_feature_vector(face_points, left_hand_points, right_hand_points, hand_presence)


def _extract_handedness_label(entry: object) -> str | None:
    if entry is None:
        return None
    if isinstance(entry, (list, tuple)) and entry:
        entry = entry[0]
    label = getattr(entry, "category_name", None)
    return str(label).lower() if label else None


def _extract_blendshape_scores(face_result: object) -> dict[str, float]:
    rows = getattr(face_result, "face_blendshapes", None) or []
    if not rows:
        return {}
    scores: dict[str, float] = {}
    for category in rows[0]:
        name = getattr(category, "category_name", None)
        score = getattr(category, "score", None)
        if name is None or score is None:
            continue
        scores[str(name)] = float(score)
    return scores


def _distance(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def _bounded(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _finger_extension_score(hand_points: np.ndarray, tip_index: int, pip_index: int) -> float:
    wrist = hand_points[0]
    tip_distance = float(np.linalg.norm(hand_points[tip_index] - wrist))
    pip_distance = float(np.linalg.norm(hand_points[pip_index] - wrist))
    return _bounded(tip_distance - pip_distance, 0.02, 0.18)


def _peace_pose_score(hand_points: np.ndarray) -> float:
    index_score = _finger_extension_score(hand_points, 8, 6)
    middle_score = _finger_extension_score(hand_points, 12, 10)
    ring_score = _finger_extension_score(hand_points, 16, 14)
    pinky_score = _finger_extension_score(hand_points, 20, 18)
    folded_score = min(1.0 - ring_score, 1.0 - pinky_score)
    return min(index_score, middle_score, folded_score)


def _phone_pose_score(face_points: np.ndarray, hand_points: np.ndarray) -> float:
    centroid = hand_points.mean(axis=0)
    wrist = hand_points[0]
    left_ear = face_points[234]
    right_ear = face_points[454]
    centroid_distance = min(
        float(np.linalg.norm(centroid - left_ear)),
        float(np.linalg.norm(centroid - right_ear)),
    )
    wrist_distance = min(
        float(np.linalg.norm(wrist - left_ear)),
        float(np.linalg.norm(wrist - right_ear)),
    )
    thumb_score = _finger_extension_score(hand_points, 4, 2)
    pinky_score = _finger_extension_score(hand_points, 20, 18)
    return 0.65 * _bounded(0.75 - min(centroid_distance, wrist_distance), 0.0, 0.45) + 0.35 * min(
        1.0, thumb_score + pinky_score
    )


def compute_expression_metrics(
    face_points: np.ndarray,
    left_hand_points: np.ndarray,
    right_hand_points: np.ndarray,
    hand_presence: np.ndarray,
    blendshapes: Mapping[str, float] | None = None,
) -> dict[str, float]:
    blend = dict(blendshapes or {})
    mouth_open = _distance(face_points, 13, 14)
    mouth_width = _distance(face_points, 61, 291)
    eye_open_left = _distance(face_points, 159, 145)
    eye_open_right = _distance(face_points, 386, 374)
    eye_open = (eye_open_left + eye_open_right) * 0.5
    brow_gap_left = _distance(face_points, 105, 159)
    brow_gap_right = _distance(face_points, 334, 386)
    brow_gap = (brow_gap_left + brow_gap_right) * 0.5
    brow_asymmetry = abs(brow_gap_left - brow_gap_right)
    mouth_tilt = abs(float(face_points[61, 1] - face_points[291, 1]))

    peace_pose = 0.0
    phone_pose = 0.0
    if hand_presence[0] > 0.5:
        peace_pose = max(peace_pose, _peace_pose_score(left_hand_points))
        phone_pose = max(phone_pose, _phone_pose_score(face_points, left_hand_points))
    if hand_presence[1] > 0.5:
        peace_pose = max(peace_pose, _peace_pose_score(right_hand_points))
        phone_pose = max(phone_pose, _phone_pose_score(face_points, right_hand_points))

    return {
        "mouth_open": mouth_open,
        "mouth_width": mouth_width,
        "eye_open": eye_open,
        "brow_gap": brow_gap,
        "brow_asymmetry": brow_asymmetry,
        "mouth_tilt": mouth_tilt,
        "peace_pose": peace_pose,
        "phone_pose": phone_pose,
        "jaw_open_blendshape": blend.get("jawOpen", 0.0),
        "smile_blendshape": max(blend.get("mouthSmileLeft", 0.0), blend.get("mouthSmileRight", 0.0)),
        "frown_blendshape": max(blend.get("mouthFrownLeft", 0.0), blend.get("mouthFrownRight", 0.0)),
        "brow_down_blendshape": max(blend.get("browDownLeft", 0.0), blend.get("browDownRight", 0.0)),
        "brow_up_blendshape": max(
            blend.get("browOuterUpLeft", 0.0),
            blend.get("browOuterUpRight", 0.0),
            blend.get("browInnerUp", 0.0),
        ),
        "eye_wide_blendshape": max(blend.get("eyeWideLeft", 0.0), blend.get("eyeWideRight", 0.0)),
        "eye_squint_blendshape": max(blend.get("eyeSquintLeft", 0.0), blend.get("eyeSquintRight", 0.0)),
    }


def extract_feature_bundle(face_result: object, hand_result: object) -> FeatureBundle | None:
    face_landmarks = getattr(face_result, "face_landmarks", None) or []
    if not face_landmarks:
        return None

    face_points_raw = _extract_xy_array(face_landmarks[0], FACE_LANDMARK_COUNT, "face")
    center, scale = compute_face_center_and_scale(face_points_raw)
    face_points = normalize_points(face_points_raw, center, scale)

    left_hand_points = np.zeros((HAND_LANDMARK_COUNT, 2), dtype=np.float32)
    right_hand_points = np.zeros((HAND_LANDMARK_COUNT, 2), dtype=np.float32)
    hand_presence = np.zeros(2, dtype=np.float32)

    hand_landmarks = getattr(hand_result, "hand_landmarks", None) or []
    handedness = getattr(hand_result, "handedness", None) or []
    for hand_index, landmarks in enumerate(hand_landmarks[:2]):
        hand_points_raw = _extract_xy_array(landmarks, HAND_LANDMARK_COUNT, "hand")
        hand_points = normalize_points(hand_points_raw, center, scale)
        label = _extract_handedness_label(handedness[hand_index] if hand_index < len(handedness) else None)

        if label == "left" and hand_presence[0] < 0.5:
            left_hand_points = hand_points
            hand_presence[0] = 1.0
        elif label == "right" and hand_presence[1] < 0.5:
            right_hand_points = hand_points
            hand_presence[1] = 1.0
        elif hand_presence[0] < 0.5:
            left_hand_points = hand_points
            hand_presence[0] = 1.0
        elif hand_presence[1] < 0.5:
            right_hand_points = hand_points
            hand_presence[1] = 1.0

    blendshapes = _extract_blendshape_scores(face_result)
    metrics = compute_expression_metrics(face_points, left_hand_points, right_hand_points, hand_presence, blendshapes)
    vector = pack_feature_vector(face_points, left_hand_points, right_hand_points, hand_presence)
    return FeatureBundle(
        vector=vector,
        face_points=face_points,
        left_hand_points=left_hand_points,
        right_hand_points=right_hand_points,
        hand_presence=hand_presence,
        blendshapes=blendshapes,
        metrics=metrics,
    )


def rule_based_probabilities(bundle: FeatureBundle | None, class_names: Sequence[str]) -> np.ndarray:
    if bundle is None:
        return neutral_distribution(class_names)

    scores = np.full(len(class_names), 0.03, dtype=np.float32)
    label_to_index = class_map_from_names(class_names)
    metrics = bundle.metrics

    happy_strength = max(
        0.55 * _bounded(metrics["mouth_width"], 0.23, 0.37)
        + 0.45 * _bounded(metrics["smile_blendshape"], 0.2, 0.85),
        0.45 * _bounded(metrics["mouth_open"], 0.02, 0.09) + 0.55 * _bounded(metrics["mouth_width"], 0.22, 0.38),
    )
    shocked_strength = max(
        0.55 * _bounded(metrics["mouth_open"], 0.06, 0.18)
        + 0.45 * _bounded(metrics["eye_open"], 0.035, 0.08),
        0.65 * _bounded(metrics["jaw_open_blendshape"], 0.2, 0.95)
        + 0.35 * _bounded(metrics["eye_wide_blendshape"], 0.1, 0.8),
    )
    angry_strength = max(
        0.55 * _bounded(0.12 - metrics["brow_gap"], 0.0, 0.05)
        + 0.45 * _bounded(0.04 - metrics["eye_open"], 0.0, 0.02),
        0.6 * _bounded(metrics["brow_down_blendshape"], 0.1, 0.8)
        + 0.4 * _bounded(metrics["frown_blendshape"], 0.1, 0.75),
    )
    suspicious_strength = max(
        0.6 * _bounded(metrics["brow_asymmetry"], 0.006, 0.04)
        + 0.4 * _bounded(metrics["mouth_tilt"], 0.003, 0.03),
        0.55 * _bounded(metrics["eye_squint_blendshape"], 0.1, 0.8)
        + 0.45 * _bounded(metrics["brow_up_blendshape"], 0.1, 0.8),
    )
    smug_strength = 0.55 * _bounded(metrics["mouth_tilt"], 0.004, 0.035) + 0.45 * _bounded(
        metrics["smile_blendshape"], 0.1, 0.6
    )
    crying_strength = 0.55 * _bounded(metrics["eye_squint_blendshape"], 0.1, 0.75) + 0.45 * _bounded(
        metrics["frown_blendshape"], 0.15, 0.8
    )
    tongue_out_strength = 0.6 * _bounded(metrics["jaw_open_blendshape"], 0.25, 0.95) + 0.4 * _bounded(
        metrics["mouth_open"], 0.07, 0.18
    )
    meltdown_strength = max(
        0.55 * tongue_out_strength + 0.45 * crying_strength,
        0.65 * _bounded(metrics["jaw_open_blendshape"], 0.25, 0.98)
        + 0.35 * _bounded(metrics["mouth_open"], 0.08, 0.20),
    )
    pout_strength = max(
        0.6 * _bounded(metrics["mouth_width"], 0.18, 0.28) * (1.0 - _bounded(metrics["mouth_open"], 0.05, 0.18))
        + 0.4 * _bounded(metrics["brow_up_blendshape"], 0.08, 0.7),
        0.55 * suspicious_strength + 0.45 * smug_strength,
    )
    lip_bite_strength = max(
        0.5 * _bounded(metrics["mouth_tilt"], 0.004, 0.03)
        + 0.5 * _bounded(metrics["mouth_open"], 0.01, 0.06),
        0.7 * suspicious_strength + 0.3 * pout_strength,
    )
    peace_strength = metrics["peace_pose"]
    phone_strength = metrics["phone_pose"]
    hand_up_strength = 0.45 if float(bundle.hand_presence.max()) > 0.5 else 0.0

    emotion_strength = max(
        happy_strength,
        shocked_strength,
        angry_strength,
        suspicious_strength,
        smug_strength,
        crying_strength,
        meltdown_strength,
        pout_strength,
        lip_bite_strength,
        peace_strength,
        phone_strength,
        hand_up_strength,
    )
    neutral_strength = 0.3 + 0.95 * (1.0 - emotion_strength)

    strength_by_label = {
        NEUTRAL_LABEL: neutral_strength,
        "happy": happy_strength,
        "shocked": shocked_strength,
        "smug": smug_strength,
        "crying": crying_strength,
        "angry": angry_strength,
        "suspicious": suspicious_strength,
        "peace_pose": peace_strength,
        "phone_pose": phone_strength,
        "blush": happy_strength * 0.85,
        "meltdown": meltdown_strength,
        "pointing": hand_up_strength * 0.85,
        "batman": suspicious_strength * 0.75,
        "lip_bite": lip_bite_strength,
        "hand_up": hand_up_strength,
        "pout": pout_strength,
    }
    for label, strength in strength_by_label.items():
        index = label_to_index.get(label)
        if index is None:
            continue
        scores[index] = max(scores[index], 0.08 + 1.35 * float(np.clip(strength, 0.0, 1.0)))

    scores = np.clip(scores, 1e-6, None)
    scores /= scores.sum()
    return scores.astype(np.float32)
