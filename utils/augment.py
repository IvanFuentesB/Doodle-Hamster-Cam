from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.feature_utils import (
    FEATURE_VECTOR_LENGTH,
    add_coordinate_noise,
    mirror_feature_vector,
    scale_feature_vector,
)


@dataclass(slots=True)
class AugmentConfig:
    mirror_prob: float = 0.5
    scale_min: float = 0.95
    scale_max: float = 1.05
    jitter_sigma: float = 0.01
    jitter_clip: float = 0.03


def augment_feature_vector(
    vector: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig | None = None,
) -> np.ndarray:
    cfg = config or AugmentConfig()
    augmented = np.asarray(vector, dtype=np.float32).copy()

    if rng.random() < cfg.mirror_prob:
        augmented = mirror_feature_vector(augmented)

    scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
    augmented = scale_feature_vector(augmented, scale)

    noise = rng.normal(0.0, cfg.jitter_sigma, size=FEATURE_VECTOR_LENGTH).astype(np.float32)
    noise = np.clip(noise, -cfg.jitter_clip, cfg.jitter_clip)
    noise[-2:] = 0.0
    augmented = add_coordinate_noise(augmented, noise)
    augmented[-2:] = np.clip(augmented[-2:], 0.0, 1.0)
    return augmented.astype(np.float32)
