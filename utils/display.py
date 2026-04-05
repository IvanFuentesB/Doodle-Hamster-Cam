from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from textwrap import wrap
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HUD_YELLOW = (0, 235, 255)
HUD_WHITE = (252, 252, 252)
HUD_BLACK = (18, 18, 18)
PANEL_BG = (248, 248, 248)
SEPARATOR = (218, 218, 218)
ASSET_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

PLACEHOLDER_COLORS = {
    "neutral": (248, 248, 248),
    "happy": (248, 248, 248),
    "shocked": (248, 248, 248),
    "smug": (248, 248, 248),
    "crying": (248, 248, 248),
    "angry": (248, 248, 248),
    "tongue_out": (248, 248, 248),
    "suspicious": (248, 248, 248),
    "peace_pose": (248, 248, 248),
    "phone_pose": (248, 248, 248),
}


def _default_asset_color(label: str) -> tuple[int, int, int]:
    return PLACEHOLDER_COLORS.get(label, (248, 248, 248))


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _shift_points(points: list[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in points]


def _stroke(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width: int = 6, fill: tuple[int, int, int] = (0, 0, 0)) -> None:
    offsets = [(0, 0), (1, -1), (-1, 1)]
    for dx, dy in offsets:
        draw.line(_shift_points(points, dx, dy), fill=fill, width=width, joint="curve")


def _ellipse_outline(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    width: int = 5,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    for offset in (0, 1):
        left, top, right, bottom = bbox
        draw.ellipse((left - offset, top, right + offset, bottom), outline=fill, width=width)


def _draw_base_hamster(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> dict[str, tuple[float, float]]:
    width, height = size
    draw.rectangle((0, 0, width, height), fill=_default_asset_color("neutral"))

    outline = (12, 12, 12)
    blush = (245, 184, 203)
    tears = (84, 206, 255)

    body_path = [
        (width * 0.50, height * 0.05),
        (width * 0.40, height * 0.075),
        (width * 0.33, height * 0.15),
        (width * 0.27, height * 0.28),
        (width * 0.23, height * 0.46),
        (width * 0.23, height * 0.71),
        (width * 0.27, height * 0.90),
        (width * 0.33, height * 0.98),
        (width * 0.67, height * 0.98),
        (width * 0.73, height * 0.90),
        (width * 0.77, height * 0.71),
        (width * 0.77, height * 0.46),
        (width * 0.73, height * 0.28),
        (width * 0.67, height * 0.15),
        (width * 0.60, height * 0.075),
        (width * 0.50, height * 0.05),
    ]
    _stroke(draw, body_path, width=6, fill=outline)
    _stroke(draw, [(width * 0.34, height * 0.68), (width * 0.33, height * 0.73), (width * 0.34, height * 0.77)], width=3, fill=outline)
    _stroke(draw, [(width * 0.66, height * 0.68), (width * 0.67, height * 0.73), (width * 0.66, height * 0.77)], width=3, fill=outline)

    left_eye = (width * 0.40, height * 0.27, width * 0.46, height * 0.34)
    right_eye = (width * 0.54, height * 0.27, width * 0.60, height * 0.34)
    draw.ellipse(left_eye, fill=outline)
    draw.ellipse(right_eye, fill=outline)
    draw.ellipse((width * 0.43, height * 0.29, width * 0.445, height * 0.305), fill=HUD_WHITE)
    draw.ellipse((width * 0.57, height * 0.29, width * 0.585, height * 0.305), fill=HUD_WHITE)

    draw.polygon(
        [
            (width * 0.50, height * 0.355),
            (width * 0.47, height * 0.392),
            (width * 0.53, height * 0.392),
        ],
        fill=(255, 186, 205),
        outline=outline,
    )

    return {
        "outline": outline,
        "blush": blush,
        "tears": tears,
        "mouth_center": (width * 0.50, height * 0.53),
        "left_eye": (width * 0.43, height * 0.305),
        "right_eye": (width * 0.57, height * 0.305),
        "width": width,
        "height": height,
    }


def _draw_neutral(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    mouth_x, mouth_y = refs["mouth_center"]
    _stroke(draw, [(mouth_x - 34, mouth_y), (mouth_x + 34, mouth_y)], width=4, fill=outline)


def _draw_happy(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    blush = refs["blush"]
    mouth_x, mouth_y = refs["mouth_center"]
    draw.ellipse((mouth_x - 110, mouth_y - 40, mouth_x - 55, mouth_y + 15), fill=blush)
    draw.ellipse((mouth_x + 55, mouth_y - 40, mouth_x + 110, mouth_y + 15), fill=blush)
    _stroke(
        draw,
        [(mouth_x - 82, mouth_y - 6), (mouth_x - 38, mouth_y + 26), (mouth_x, mouth_y + 34), (mouth_x + 38, mouth_y + 26), (mouth_x + 82, mouth_y - 6)],
        width=5,
        fill=outline,
    )


def _draw_shocked(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    mouth_x, mouth_y = refs["mouth_center"]
    _ellipse_outline(draw, (mouth_x - 72, mouth_y - 18, mouth_x + 72, mouth_y + 145), width=6, fill=outline)
    draw.ellipse((mouth_x - 60, mouth_y - 6, mouth_x + 60, mouth_y + 133), fill=(8, 8, 8))
    draw.rectangle((mouth_x - 20, mouth_y + 18, mouth_x + 20, mouth_y + 55), fill=HUD_WHITE)


def _draw_smug(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    width = refs["width"]
    height = refs["height"]
    _stroke(draw, [(width * 0.37, height * 0.24), (width * 0.45, height * 0.235)], width=5, fill=outline)
    _stroke(draw, [(width * 0.55, height * 0.235), (width * 0.63, height * 0.25)], width=5, fill=outline)
    _stroke(draw, [(width * 0.45, height * 0.56), (width * 0.52, height * 0.545), (width * 0.59, height * 0.57)], width=5, fill=outline)
    _stroke(draw, [(width * 0.44, height * 0.60), (width * 0.48, height * 0.615)], width=3, fill=outline)


def _draw_crying(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    tears = refs["tears"]
    mouth_x, mouth_y = refs["mouth_center"]
    left_eye_x, left_eye_y = refs["left_eye"]
    right_eye_x, right_eye_y = refs["right_eye"]
    draw.line((left_eye_x - 8, left_eye_y + 20, left_eye_x - 4, left_eye_y + 150), fill=tears, width=10)
    draw.line((right_eye_x + 8, right_eye_y + 20, right_eye_x + 4, right_eye_y + 150), fill=tears, width=10)
    draw.ellipse((mouth_x - 62, mouth_y - 8, mouth_x + 62, mouth_y + 120), fill=(8, 8, 8))
    draw.rectangle((mouth_x - 14, mouth_y + 14, mouth_x + 14, mouth_y + 42), fill=HUD_WHITE)


def _draw_angry(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    width = refs["width"]
    height = refs["height"]
    mouth_x, mouth_y = refs["mouth_center"]
    _stroke(draw, [(width * 0.35, height * 0.22), (width * 0.45, height * 0.18)], width=6, fill=outline)
    _stroke(draw, [(width * 0.55, height * 0.18), (width * 0.65, height * 0.22)], width=6, fill=outline)
    draw.ellipse((mouth_x - 90, mouth_y - 2, mouth_x + 90, mouth_y + 84), fill=(8, 8, 8))
    draw.ellipse((mouth_x - 30, mouth_y + 18, mouth_x + 30, mouth_y + 56), fill=(255, 255, 255))


def _draw_tongue_out(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    mouth_x, mouth_y = refs["mouth_center"]
    _stroke(draw, [(mouth_x - 78, mouth_y - 8), (mouth_x - 30, mouth_y + 12), (mouth_x + 30, mouth_y + 12), (mouth_x + 78, mouth_y - 8)], width=5, fill=outline)
    draw.rounded_rectangle((mouth_x - 52, mouth_y + 6, mouth_x + 52, mouth_y + 142), radius=28, fill=(255, 181, 197), outline=outline, width=4)
    draw.line((mouth_x, mouth_y + 12, mouth_x, mouth_y + 136), fill=(222, 111, 144), width=4)


def _draw_suspicious(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    width = refs["width"]
    height = refs["height"]
    _stroke(draw, [(width * 0.35, height * 0.21), (width * 0.46, height * 0.205)], width=4, fill=outline)
    _stroke(draw, [(width * 0.56, height * 0.19), (width * 0.65, height * 0.23)], width=4, fill=outline)
    _stroke(draw, [(width * 0.45, height * 0.57), (width * 0.60, height * 0.55)], width=5, fill=outline)
    _stroke(draw, [(width * 0.47, height * 0.60), (width * 0.50, height * 0.615)], width=3, fill=outline)


def _draw_peace_pose(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    _draw_neutral(draw, refs)
    width = refs["width"]
    height = refs["height"]
    _stroke(draw, [(width * 0.68, height * 0.60), (width * 0.76, height * 0.54), (width * 0.79, height * 0.50)], width=8, fill=outline)
    _stroke(draw, [(width * 0.79, height * 0.50), (width * 0.82, height * 0.38)], width=7, fill=outline)
    _stroke(draw, [(width * 0.79, height * 0.50), (width * 0.88, height * 0.40)], width=7, fill=outline)
    _stroke(draw, [(width * 0.76, height * 0.54), (width * 0.84, height * 0.60)], width=7, fill=outline)


def _draw_phone_pose(draw: ImageDraw.ImageDraw, refs: dict[str, tuple[float, float]]) -> None:
    outline = refs["outline"]
    _draw_neutral(draw, refs)
    width = refs["width"]
    height = refs["height"]
    draw.rounded_rectangle((width * 0.75, height * 0.34, width * 0.84, height * 0.56), radius=16, fill=(126, 142, 223), outline=outline, width=4)
    draw.rectangle((width * 0.785, height * 0.37, width * 0.805, height * 0.39), fill=HUD_WHITE)
    _stroke(draw, [(width * 0.67, height * 0.54), (width * 0.74, height * 0.49), (width * 0.76, height * 0.46)], width=7, fill=outline)


def create_placeholder_asset(path: Path, label: str, size: tuple[int, int] = (900, 900)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, _default_asset_color(label))
    draw = ImageDraw.Draw(image)
    refs = _draw_base_hamster(draw, size)

    if label == "happy":
        _draw_happy(draw, refs)
    elif label == "shocked":
        _draw_shocked(draw, refs)
    elif label == "smug":
        _draw_smug(draw, refs)
    elif label == "crying":
        _draw_crying(draw, refs)
    elif label == "angry":
        _draw_angry(draw, refs)
    elif label == "tongue_out":
        _draw_tongue_out(draw, refs)
    elif label == "suspicious":
        _draw_suspicious(draw, refs)
    elif label == "peace_pose":
        _draw_peace_pose(draw, refs)
    elif label == "phone_pose":
        _draw_phone_pose(draw, refs)
    else:
        _draw_neutral(draw, refs)

    image.save(path)


def _ignore_marker_path(assets_dir: Path, label: str) -> Path:
    return assets_dir / "custom" / f"{label}.ignore"


def _find_existing_asset_path(assets_dir: Path, label: str) -> Path | None:
    if _ignore_marker_path(assets_dir, label).exists():
        return None

    search_roots = [assets_dir / "custom", assets_dir]
    for root in search_roots:
        for suffix in ASSET_SUFFIXES:
            candidate = root / f"{label}{suffix}"
            if candidate.exists():
                return candidate
    return None


def ensure_placeholder_assets(assets_dir: Path, class_names: Sequence[str]) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "custom").mkdir(parents=True, exist_ok=True)
    for label in class_names:
        asset_path = _find_existing_asset_path(assets_dir, label)
        if asset_path is None:
            create_placeholder_asset(assets_dir / f"{label}.png", label)


def load_reaction_images(assets_dir: Path, class_names: Sequence[str]) -> dict[str, np.ndarray]:
    ensure_placeholder_assets(assets_dir, class_names)
    images: dict[str, np.ndarray] = {}
    for label in class_names:
        asset_path = _find_existing_asset_path(assets_dir, label)
        if asset_path is None:
            continue
        image = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
        if image is not None:
            images[label] = image
    return images


def register_custom_asset(source_path: Path, assets_dir: Path, label: str) -> Path:
    placeholder_hint = str(source_path).lower()
    if not source_path.exists():
        if placeholder_hint.startswith("c:\\path\\to\\") or "the_exact_" in placeholder_hint:
            raise FileNotFoundError(
                "That path is still the placeholder example. Replace it with the real PNG/JPG path on your PC."
            )
        raise FileNotFoundError(f"Image file not found: {source_path}")
    source_path = source_path.resolve()
    synthetic_pack_root = (assets_dir.resolve() / "reaction_packs" / "user_sent_20260403").resolve()
    if source_path.is_relative_to(synthetic_pack_root):
        raise ValueError(
            "The user_sent_20260403 pack contains temporary stand-ins, not the exact original images. "
            "Map the real PNG or JPG file instead."
        )
    target_dir = assets_dir / "custom"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{label}{source_path.suffix.lower()}"
    shutil.copy2(source_path, target_path)
    for suffix in ASSET_SUFFIXES:
        sibling = target_dir / f"{label}{suffix}"
        if sibling != target_path and sibling.exists():
            sibling.unlink()
    ignore_marker = _ignore_marker_path(assets_dir, label)
    if ignore_marker.exists():
        ignore_marker.unlink()
    return target_path


def _fit_image_to_panel(image: np.ndarray | None, panel_size: tuple[int, int]) -> np.ndarray:
    panel_width, panel_height = panel_size
    panel = np.full((panel_height, panel_width, 3), PANEL_BG, dtype=np.uint8)
    if image is None:
        return panel

    height, width = image.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    x_offset = (panel_width - resized.shape[1]) // 2
    y_offset = (panel_height - resized.shape[0]) // 2
    panel[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized
    return panel


def _fill_image_to_panel(image: np.ndarray, panel_size: tuple[int, int]) -> np.ndarray:
    panel_width, panel_height = panel_size
    height, width = image.shape[:2]
    scale = max(panel_width / width, panel_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    x_offset = max((resized.shape[1] - panel_width) // 2, 0)
    y_offset = max((resized.shape[0] - panel_height) // 2, 0)
    return resized[y_offset : y_offset + panel_height, x_offset : x_offset + panel_width].copy()


def _draw_text_with_shadow(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, HUD_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _wrap_status_line(status_line: str, width: int = 34) -> list[str]:
    wrapped = wrap(status_line, width=width)
    return wrapped[:2] if wrapped else [status_line]


def draw_overlay_lines(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int] = (16, 30),
    line_height: int = 28,
    text_color: tuple[int, int, int] = (34, 30, 28),
    box_color: tuple[int, int, int] = (245, 244, 239),
) -> np.ndarray:
    if not lines:
        return frame
    box_width = min(frame.shape[1] - origin[0] * 2, 760)
    box_height = 18 + len(lines) * line_height
    x1, y1 = origin
    x2, y2 = x1 + box_width, y1 + box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 10, y1 - 22), (x2, y2), box_color, thickness=-1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0.0, dst=frame)
    cv2.rectangle(frame, (x1 - 10, y1 - 22), (x2, y2), (64, 56, 52), thickness=2)

    for index, line in enumerate(lines):
        y = y1 + index * line_height
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, text_color, 2, cv2.LINE_AA)
    return frame


def compose_app_frame(
    webcam_frame: np.ndarray,
    reaction_image: np.ndarray | None,
    label: str,
    confidence: float,
    mode: str,
    fps: float,
    status_line: str,
) -> np.ndarray:
    target_height = max(560, webcam_frame.shape[0])
    target_width = max(1000, int(target_height * 1.50))
    left_panel_width = int(target_width * 0.61)
    right_panel_width = target_width - left_panel_width

    camera_panel = _fill_image_to_panel(webcam_frame, (left_panel_width, target_height))
    _draw_text_with_shadow(camera_panel, f"Prediction: {label}", (18, 40), 0.76, HUD_YELLOW, 2)
    _draw_text_with_shadow(camera_panel, f"{mode}  |  {fps:.1f} FPS  |  {confidence:.2f}", (20, 68), 0.42, HUD_WHITE, 1)

    right_panel = np.full((target_height, right_panel_width, 3), PANEL_BG, dtype=np.uint8)
    image_area_top = 8
    image_area_height = int(target_height * 0.81)
    fitted = _fit_image_to_panel(reaction_image, (right_panel_width - 24, image_area_height))
    right_panel[image_area_top : image_area_top + image_area_height, 12 : right_panel_width - 12] = fitted

    cv2.putText(right_panel, label.replace("_", " ").title(), (16, image_area_top + image_area_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, HUD_BLACK, 2, cv2.LINE_AA)
    cv2.putText(right_panel, f"Confidence {confidence:.2f}", (16, image_area_top + image_area_height + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1, cv2.LINE_AA)

    status_lines = _wrap_status_line(status_line, width=32)
    status_y = image_area_top + image_area_height + 82
    for index, line in enumerate(status_lines):
        cv2.putText(right_panel, line, (16, status_y + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (92, 92, 92), 1, cv2.LINE_AA)

    app_frame = np.hstack([camera_panel, right_panel])
    cv2.line(app_frame, (left_panel_width, 0), (left_panel_width, target_height), SEPARATOR, 2)
    return app_frame


def compose_pose_mapping_frame(
    webcam_frame: np.ndarray,
    reference_image: np.ndarray | None,
    label: str,
    sample_count: int,
    message: str,
) -> np.ndarray:
    target_height = max(580, webcam_frame.shape[0])
    target_width = max(1020, int(target_height * 1.52))
    left_panel_width = int(target_width * 0.60)
    right_panel_width = target_width - left_panel_width

    camera_panel = _fill_image_to_panel(webcam_frame, (left_panel_width, target_height))
    _draw_text_with_shadow(camera_panel, f"Map Pose: {label}", (18, 40), 0.76, HUD_YELLOW, 2)
    _draw_text_with_shadow(camera_panel, f"Samples {sample_count}", (20, 68), 0.42, HUD_WHITE, 1)

    right_panel = np.full((target_height, right_panel_width, 3), PANEL_BG, dtype=np.uint8)
    fitted = _fit_image_to_panel(reference_image, (right_panel_width - 24, int(target_height * 0.82)))
    right_panel[12 : 12 + fitted.shape[0], 12 : 12 + fitted.shape[1]] = fitted
    cv2.putText(right_panel, "Match This Pose", (16, target_height - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.66, HUD_BLACK, 2, cv2.LINE_AA)
    cv2.putText(right_panel, "space save  |  b burst  |  t train  |  q quit", (16, target_height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (84, 84, 84), 1, cv2.LINE_AA)
    wrapped = _wrap_status_line(message, width=34)
    for index, line in enumerate(wrapped[:2]):
        cv2.putText(right_panel, line, (16, target_height - 18 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (94, 94, 94), 1, cv2.LINE_AA)

    frame = np.hstack([camera_panel, right_panel])
    cv2.line(frame, (left_panel_width, 0), (left_panel_width, target_height), SEPARATOR, 2)
    return frame


def prepare_window(window_name: str, width: int = 1020, height: int = 640) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)


WINDOW_CLOSED = -2


def read_window_key(window_name: str, delay_ms: int = 10) -> int:
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return WINDOW_CLOSED
    except cv2.error:
        return WINDOW_CLOSED

    key = cv2.waitKeyEx(delay_ms)
    if key < 0:
        return key
    if key == 27:
        return key

    try:
        key_char = chr(key)
    except (TypeError, ValueError):
        return key

    if key_char.isalpha():
        return ord(key_char.lower())
    return key


def save_screenshot(frame: np.ndarray, captures_dir: Path, label: str) -> Path:
    captures_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = captures_dir / f"{timestamp}_{label.replace(' ', '_')}.png"
    cv2.imwrite(str(path), frame)
    return path
