"""
Signify — Vision Agent
Wraps MediaPipe hand detection and bounding-box cropping into a clean,
reusable class ready to be called by the agent pipeline.

Usage:
    agent = VisionAgent()
    result = agent.process_frame(bgr_frame)
    if result:
        crop, landmarks, bbox = result.crop, result.landmarks, result.bbox
        # crop      -> 128x128 RGB PIL Image, ready for val_transform -> model
        # landmarks -> mediapipe NormalizedLandmarkList
        # bbox      -> (x, y, w, h) in pixel coords (padded)
    agent.release()
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple


# --------------------------------------------------
# Return type
# --------------------------------------------------

@dataclass
class VisionResult:
    """Everything downstream agents need from a single frame."""
    crop: Image.Image                        # 128x128 RGB PIL image
    landmarks: object                        # mediapipe NormalizedLandmarkList
    bbox: Tuple[int, int, int, int]          # (x, y, w, h) padded, clipped


# --------------------------------------------------
# Agent
# --------------------------------------------------

class VisionAgent:
    """
    Detects a single hand in a BGR frame (OpenCV format), crops it with
    padding, and returns a 128x128 RGB PIL image plus raw landmarks.

    Parameters
    ----------
    padding : int
        Pixels added to each side of the tight hand bounding box.
        Must match the value used during camera_test (default 40).
    target_size : tuple[int, int]
        (width, height) to resize the crop to.
        Must match model input (default (128, 128)).
    detection_confidence : float
        MediaPipe min_detection_confidence (default 0.70).
    tracking_confidence : float
        MediaPipe min_tracking_confidence (default 0.70).
    max_num_hands : int
        Maximum hands to detect; pipeline uses 1 (default 1).
    """

    def __init__(
        self,
        padding: int = 40,
        target_size: Tuple[int, int] = (128, 128),
        detection_confidence: float = 0.70,
        tracking_confidence: float = 0.70,
        max_num_hands: int = 1,
    ) -> None:
        self.padding = padding
        self.target_size = target_size

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # -- public API -------------------------------------

    def process_frame(self, bgr_frame: np.ndarray) -> Optional[VisionResult]:
        """
        Run hand detection on one BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            Raw frame from cv2.VideoCapture.read().

        Returns
        -------
        VisionResult | None
            None if no hand detected; otherwise crop, landmarks, bbox.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return None

        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        bbox = self._compute_bbox(hand_landmarks, frame_w=w, frame_h=h)
        crop = self._crop_and_resize(rgb_frame, bbox)

        return VisionResult(crop=crop, landmarks=hand_landmarks, bbox=bbox)

    def release(self) -> None:
        """Close the MediaPipe Hands instance and free resources."""
        self._hands.close()

    # -- context manager --------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    # -- internal helpers --------------------------------

    def _compute_bbox(
        self,
        hand_landmarks,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        """
        Convert normalised MediaPipe landmarks to a padded pixel bounding box.
        Returns (x, y, w, h) clipped to frame boundaries.
        """
        xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
        ys = [lm.y * frame_h for lm in hand_landmarks.landmark]

        x_min = max(0, int(min(xs)) - self.padding)
        y_min = max(0, int(min(ys)) - self.padding)
        x_max = min(frame_w, int(max(xs)) + self.padding)
        y_max = min(frame_h, int(max(ys)) + self.padding)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def _crop_and_resize(
        self,
        rgb_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """
        Crop the padded region and resize to target_size.
        Returns a PIL Image in RGB mode ready for val_transform.
        """
        x, y, bw, bh = bbox
        crop_np = rgb_frame[y : y + bh, x : x + bw]
        pil_img = Image.fromarray(crop_np)
        return pil_img.resize(self.target_size, Image.BILINEAR)