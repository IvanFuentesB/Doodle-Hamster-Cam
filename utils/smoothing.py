from __future__ import annotations

from collections import deque

import numpy as np


class PredictionSmoother:
    def __init__(self, window_size: int = 5, missing_threshold: int = 5) -> None:
        self.window_size = max(1, int(window_size))
        self.missing_threshold = max(1, int(missing_threshold))
        self.history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.missing_frames = 0

    def reset(self) -> None:
        self.history.clear()
        self.missing_frames = 0

    def _neutral(self, num_classes: int, neutral_index: int) -> np.ndarray:
        probabilities = np.zeros(num_classes, dtype=np.float32)
        probabilities[neutral_index] = 1.0
        return probabilities

    def update(
        self,
        probabilities: np.ndarray | None,
        neutral_index: int,
        num_classes: int | None = None,
    ) -> np.ndarray:
        if probabilities is None:
            self.missing_frames += 1
            if self.missing_frames >= self.missing_threshold:
                self.reset()
                if num_classes is None:
                    raise ValueError("num_classes is required when no prediction is available.")
                neutral = self._neutral(num_classes, neutral_index)
                self.history.append(neutral)
                return neutral
            if self.history:
                averaged = np.stack(tuple(self.history)).mean(axis=0)
                averaged /= max(float(averaged.sum()), 1e-6)
                return averaged.astype(np.float32)
            if num_classes is None:
                raise ValueError("num_classes is required before the smoother has any history.")
            return self._neutral(num_classes, neutral_index)

        current = np.asarray(probabilities, dtype=np.float32)
        self.missing_frames = 0
        self.history.append(current)
        averaged = np.stack(tuple(self.history)).mean(axis=0)
        averaged /= max(float(averaged.sum()), 1e-6)
        return averaged.astype(np.float32)
