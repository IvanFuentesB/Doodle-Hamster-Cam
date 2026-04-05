from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import shutil
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


FACE_MODEL_NAME = "face_landmarker.task"
HAND_MODEL_NAME = "hand_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
OVERLAY_SHADOW = (70, 70, 70)
FACE_REGION_STYLES = (
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW],
        (166, 196, 150),
        (46, 52, 53, 55, 65),
    ),
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW],
        (152, 184, 214),
        (276, 282, 283, 285, 295),
    ),
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE],
        (188, 208, 176),
        (33, 133, 159, 145),
    ),
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE],
        (174, 204, 224),
        (263, 362, 374, 386),
    ),
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS],
        (184, 216, 228),
        (468,),
    ),
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS],
        (184, 216, 228),
        (473,),
    ),
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS],
        (196, 160, 196),
        (0, 13, 14, 17, 61, 78, 308, 291),
    ),
    (
        [(item.start, item.end) for item in vision.FaceLandmarksConnections.FACE_LANDMARKS_NOSE],
        (182, 206, 214),
        (1, 4, 5, 195),
    ),
)
HAND_REGION_STYLES = (
    (
        [(item.start, item.end) for item in vision.HandLandmarksConnections.HAND_PALM_CONNECTIONS],
        (210, 210, 210),
        (0, 1, 2, 5, 9, 13, 17),
    ),
    (
        [(item.start, item.end) for item in vision.HandLandmarksConnections.HAND_THUMB_CONNECTIONS],
        (190, 176, 214),
        (1, 2, 3, 4),
    ),
    (
        [(item.start, item.end) for item in vision.HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS],
        (168, 206, 182),
        (5, 6, 7, 8),
    ),
    (
        [(item.start, item.end) for item in vision.HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS],
        (166, 194, 220),
        (9, 10, 11, 12),
    ),
    (
        [(item.start, item.end) for item in vision.HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS],
        (178, 188, 220),
        (13, 14, 15, 16),
    ),
    (
        [(item.start, item.end) for item in vision.HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS],
        (214, 184, 158),
        (17, 18, 19, 20),
    ),
)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DoodleHamsterCam/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary_path.replace(destination)


def resolve_task_model_path(
    explicit_path: str | Path | None,
    default_name: str,
    url: str,
    models_dir: Path,
) -> Path:
    if explicit_path:
        model_path = Path(explicit_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / default_name
    if not model_path.exists():
        _download_file(url, model_path)
    return model_path


def ensure_default_task_models(
    face_model: str | Path | None = None,
    hand_model: str | Path | None = None,
    models_dir: Path | None = None,
) -> tuple[Path, Path]:
    target_dir = Path(models_dir or "models").resolve()
    face_model_path = resolve_task_model_path(face_model, FACE_MODEL_NAME, FACE_MODEL_URL, target_dir)
    hand_model_path = resolve_task_model_path(hand_model, HAND_MODEL_NAME, HAND_MODEL_URL, target_dir)
    return face_model_path, hand_model_path


def create_face_landmarker(model_path: Path, num_faces: int = 1) -> vision.FaceLandmarker:
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker(model_path: Path, num_hands: int = 2) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


@contextmanager
def open_landmarkers(
    face_model_path: Path,
    hand_model_path: Path,
    num_faces: int = 1,
    num_hands: int = 2,
):
    with ExitStack() as stack:
        face_landmarker = stack.enter_context(create_face_landmarker(face_model_path, num_faces=num_faces))
        hand_landmarker = stack.enter_context(create_hand_landmarker(hand_model_path, num_hands=num_hands))
        yield face_landmarker, hand_landmarker


def bgr_frame_to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def detect_landmarks(
    face_landmarker: vision.FaceLandmarker,
    hand_landmarker: vision.HandLandmarker,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
) -> tuple[object, object]:
    image = bgr_frame_to_mp_image(frame_bgr)
    face_result = face_landmarker.detect_for_video(image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)
    return face_result, hand_result


def _draw_connections(
    frame: np.ndarray,
    landmarks: object,
    connections: list[tuple[int, int]],
    color: tuple[int, int, int],
    width_px: int,
    height_px: int,
    shadow_thickness: int,
    line_thickness: int,
) -> None:
    for start_index, end_index in connections:
        start = landmarks[start_index]
        end = landmarks[end_index]
        start_xy = (int(start.x * width_px), int(start.y * height_px))
        end_xy = (int(end.x * width_px), int(end.y * height_px))
        cv2.line(frame, start_xy, end_xy, OVERLAY_SHADOW, shadow_thickness, cv2.LINE_AA)
        cv2.line(frame, start_xy, end_xy, color, line_thickness, cv2.LINE_AA)


def _draw_points(
    frame: np.ndarray,
    landmarks: object,
    point_indices: tuple[int, ...],
    color: tuple[int, int, int],
    width_px: int,
    height_px: int,
    outer_radius: int,
    inner_radius: int,
) -> None:
    for point_index in point_indices:
        landmark = landmarks[point_index]
        x = int(landmark.x * width_px)
        y = int(landmark.y * height_px)
        cv2.circle(frame, (x, y), outer_radius, OVERLAY_SHADOW, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), inner_radius, color, thickness=-1, lineType=cv2.LINE_AA)


def annotate_frame(frame_bgr: np.ndarray, face_result: object, hand_result: object) -> np.ndarray:
    annotated = frame_bgr.copy()
    overlay = annotated.copy()
    height, width = annotated.shape[:2]

    face_landmarks = getattr(face_result, "face_landmarks", None) or []
    for face_landmark_list in face_landmarks:
        for connections, color, point_indices in FACE_REGION_STYLES:
            _draw_connections(
                overlay,
                face_landmark_list,
                connections,
                color,
                width,
                height,
                shadow_thickness=2,
                line_thickness=1,
            )
            _draw_points(
                overlay,
                face_landmark_list,
                point_indices,
                color,
                width,
                height,
                outer_radius=2,
                inner_radius=1,
            )

    hand_landmarks = getattr(hand_result, "hand_landmarks", None) or []
    for hand_landmark_list in hand_landmarks:
        for connections, color, point_indices in HAND_REGION_STYLES:
            _draw_connections(
                overlay,
                hand_landmark_list,
                connections,
                color,
                width,
                height,
                shadow_thickness=2,
                line_thickness=1,
            )
            _draw_points(
                overlay,
                hand_landmark_list,
                point_indices,
                color,
                width,
                height,
                outer_radius=3,
                inner_radius=1,
            )

    return cv2.addWeighted(overlay, 0.66, annotated, 0.34, 0.0)
