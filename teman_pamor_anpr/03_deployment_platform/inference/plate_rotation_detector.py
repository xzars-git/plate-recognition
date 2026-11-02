"""
Plate rotation detector used by deployment inference scripts.
Provides lightweight orientation correction (0/90/180/270 degrees).
"""

from __future__ import annotations

import cv2
import numpy as np


class PlateRotationDetector:
    """Detect and correct coarse plate rotation angles."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def detect_rotation_angle(self, image: np.ndarray) -> tuple[int, float]:
        h, w = image.shape[:2]
        aspect_ratio = w / h if h > 0 else 0.0

        if self.debug:
            print(f"Aspect ratio: {aspect_ratio:.2f} (w={w}, h={h})")

        if 1.5 < aspect_ratio < 4.0:
            return self._check_upside_down(image)
        if 0.25 < aspect_ratio < 0.67:
            return self._check_90_or_270(image)
        return 0, 0.3

    def _check_upside_down(self, image: np.ndarray) -> tuple[int, float]:
        h, _ = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)

        top_edges = int(np.sum(edges[: h // 2, :] > 0))
        bottom_edges = int(np.sum(edges[h // 2 :, :] > 0))
        total_edges = top_edges + bottom_edges

        if self.debug:
            print(f"Edges top/bottom: {top_edges}/{bottom_edges}")

        if total_edges == 0:
            return 0, 0.5

        return (180, 0.7) if (bottom_edges / total_edges) > 0.6 else (0, 0.7)

    def _check_90_or_270(self, image: np.ndarray) -> tuple[int, float]:
        _, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)

        left_edges = int(np.sum(edges[:, : w // 2] > 0))
        right_edges = int(np.sum(edges[:, w // 2 :] > 0))
        total_edges = left_edges + right_edges

        if self.debug:
            print(f"Edges left/right: {left_edges}/{right_edges}")

        if total_edges == 0:
            return 90, 0.5

        return (270, 0.7) if (right_edges / total_edges) > 0.55 else (90, 0.7)

    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, float]:
        angle, confidence = self.detect_rotation_angle(image)
        if angle != 0 and confidence > 0.5:
            corrected = self.rotate_image(image, angle)
            return corrected, angle, confidence
        return image, 0, confidence
