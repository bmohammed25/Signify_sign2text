"""
landmark_agent.py
Signify

Quality gate that validates MediaPipe landmark data before passing
a frame to RecognitionAgent. Also extracts the 63-float landmark
vector for potential future use (landmark-based classifier, motion
detection for J/Z, etc.).

Usage:
    from agents.landmark_agent import LandmarkAgent, LandmarkResult
    from agents.vision_agent   import VisionAgent, VisionResult

    vision_agent   = VisionAgent()
    landmark_agent = LandmarkAgent()

    result = vision_agent.process_frame(frame)
    if result:
        lm_result = landmark_agent.process(result)
        if lm_result.passed:
            # send result.crop to RecognitionAgent
            pass
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── MediaPipe landmark indices (wrist + all finger tips/knuckles) ─────────────
# These 16 points carry the most pose information.
# If the majority are occluded the crop is likely not worth inferring on.
KEY_LANDMARK_INDICES = [
    0,   # WRIST
    4,   # THUMB_TIP
    5,   # INDEX_FINGER_MCP
    8,   # INDEX_FINGER_TIP
    9,   # MIDDLE_FINGER_MCP
    12,  # MIDDLE_FINGER_TIP
    13,  # RING_FINGER_MCP
    16,  # RING_FINGER_TIP
    17,  # PINKY_MCP
    20,  # PINKY_TIP
    # Mid-knuckles for better fist/open discrimination
    6,   # INDEX_FINGER_PIP
    10,  # MIDDLE_FINGER_PIP
    14,  # RING_FINGER_PIP
    18,  # PINKY_PIP
    2,   # THUMB_MCP
    3,   # THUMB_IP
]


@dataclass
class LandmarkResult:
    """
    Output of LandmarkAgent.process().

    Attributes
    ----------
    landmarks_flat : list[float]
        63 normalised values — x, y, z for each of the 21 MediaPipe
        hand landmarks, in landmark index order.
        Range: x/y in [0, 1] (normalised to frame), z relative depth.

    visibility : list[float]
        Per-landmark visibility score (0.0–1.0) for all 21 points.
        MediaPipe does not always populate this field; defaults to 1.0
        when absent so the gate does not falsely reject valid frames.

    passed : bool
        True  → frame passed quality gate, safe to send to RecognitionAgent.
        False → frame rejected; RecognitionAgent should be skipped.

    visible_key_count : int
        Number of KEY_LANDMARK_INDICES whose visibility >= min_visibility.
        Useful for debugging / logging.

    span_ok : bool
        Whether the wrist-to-middle-fingertip span exceeded the minimum
        threshold. False frames are usually motion-blurred or too close.

    pose_hint : str
        Lightweight pose classification: "open", "fist", "pinch", or "".
        Not used by RecognitionAgent (which uses the raw image crop), but
        helpful for live debugging and future landmark-based classifiers.

    reject_reason : str
        Human-readable reason for rejection, empty string when passed=True.
    """
    landmarks_flat:    list[float]
    visibility:        list[float]
    passed:            bool
    visible_key_count: int         = 0
    span_ok:           bool        = True
    pose_hint:         str         = ""
    reject_reason:     str         = ""


class LandmarkAgent:
    """
    Validates MediaPipe landmarks from a VisionResult and returns a
    LandmarkResult with a pass/fail decision.

    Parameters
    ----------
    min_visibility : float
        Visibility threshold per landmark (0.0–1.0).
        Landmarks below this are treated as occluded.
    min_visible_key_count : int
        Minimum number of KEY_LANDMARK_INDICES that must meet
        min_visibility for the frame to pass.
        Default 12 out of 16 key landmarks.
    min_span_fraction : float
        Minimum wrist-to-middle-fingertip distance as a fraction of the
        frame diagonal (normalised coords, so frame diagonal = √2 ≈ 1.414).
        Rejects frames where the hand is too small / partially in frame.
        Default 0.05 (hand must span ≥ 5% of frame diagonal).
    """

    def __init__(
        self,
        min_visibility:        float = 0.6,
        min_visible_key_count: int   = 12,
        min_span_fraction:     float = 0.05,
    ) -> None:
        self.min_visibility        = min_visibility
        self.min_visible_key_count = min_visible_key_count
        self.min_span_fraction     = min_span_fraction

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, vision_result) -> LandmarkResult:
        """
        Validate landmarks from a VisionResult.

        Parameters
        ----------
        vision_result : VisionResult
            Output of VisionAgent.process_frame().  Must have a
            .landmarks attribute (MediaPipe NormalizedLandmarkList).

        Returns
        -------
        LandmarkResult
            .passed=True  → safe to pass vision_result.crop to RecognitionAgent
            .passed=False → skip this frame
        """
        lm_list = vision_result.landmarks.landmark  # list of NormalizedLandmark

        # ── 1. Extract flat vector and visibility scores ───────────────────
        landmarks_flat: list[float] = []
        visibility:     list[float] = []

        for pt in lm_list:
            landmarks_flat.extend([pt.x, pt.y, pt.z])
            # MediaPipe only sets visibility reliably for some models;
            # fall back to 1.0 so we don't penalise frames from older pipelines.
            vis = getattr(pt, "visibility", 1.0)
            visibility.append(float(vis) if (vis is not None and vis > 0.0) else 1.0)

        # ── 2. Key-landmark visibility check ──────────────────────────────
        visible_key_count = sum(
            visibility[i] >= self.min_visibility
            for i in KEY_LANDMARK_INDICES
            if i < len(visibility)
        )
        visibility_ok = visible_key_count >= self.min_visible_key_count

        # ── 3. Span check (wrist → middle fingertip) ──────────────────────
        span_ok, span_val = self._check_span(lm_list)

        # ── 4. Pose hint (lightweight, no extra cost) ─────────────────────
        pose_hint = self._estimate_pose(lm_list)

        # ── 5. Gate decision ──────────────────────────────────────────────
        if not visibility_ok:
            return LandmarkResult(
                landmarks_flat    = landmarks_flat,
                visibility        = visibility,
                passed            = False,
                visible_key_count = visible_key_count,
                span_ok           = span_ok,
                pose_hint         = pose_hint,
                reject_reason     = (
                    f"only {visible_key_count}/{self.min_visible_key_count} "
                    f"key landmarks visible (threshold {self.min_visibility:.2f})"
                ),
            )

        if not span_ok:
            return LandmarkResult(
                landmarks_flat    = landmarks_flat,
                visibility        = visibility,
                passed            = False,
                visible_key_count = visible_key_count,
                span_ok           = False,
                pose_hint         = pose_hint,
                reject_reason     = (
                    f"hand span {span_val:.3f} < min {self.min_span_fraction:.3f} "
                    "(hand too small / partially in frame)"
                ),
            )

        return LandmarkResult(
            landmarks_flat    = landmarks_flat,
            visibility        = visibility,
            passed            = True,
            visible_key_count = visible_key_count,
            span_ok           = True,
            pose_hint         = pose_hint,
            reject_reason     = "",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_span(self, lm_list) -> tuple[bool, float]:
        """
        Compute wrist → middle fingertip Euclidean distance in normalised
        coords. Returns (passed, distance).
        """
        if len(lm_list) < 13:
            return False, 0.0
        wrist  = lm_list[0]   # WRIST
        tip    = lm_list[12]  # MIDDLE_FINGER_TIP
        dx = tip.x - wrist.x
        dy = tip.y - wrist.y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        return dist >= self.min_span_fraction, dist

    def _estimate_pose(self, lm_list) -> str:
        """
        Lightweight pose hint based on finger extension.
        Returns one of: "open", "fist", "pinch", or "" on error.

        Logic:
          - For each finger (index/middle/ring/pinky), compare the tip y
            coordinate to the MCP y coordinate. In normalised coords,
            y increases downward, so a raised fingertip has LOWER y than MCP.
          - Count extended fingers (tip.y < mcp.y - threshold).
          - open  → 4 fingers extended
          - fist  → 0–1 fingers extended
          - pinch → thumb tip close to index tip
        """
        try:
            # Finger tip / MCP pairs  (tip_idx, mcp_idx)
            fingers = [
                (8,  5),   # index
                (12, 9),   # middle
                (16, 13),  # ring
                (20, 17),  # pinky
            ]
            extended = sum(
                lm_list[tip].y < lm_list[mcp].y - 0.04
                for tip, mcp in fingers
            )

            # Pinch: thumb tip (4) close to index tip (8)
            t4 = lm_list[4]
            t8 = lm_list[8]
            pinch_dist = ((t4.x - t8.x) ** 2 + (t4.y - t8.y) ** 2) ** 0.5
            if pinch_dist < 0.06:
                return "pinch"

            if extended >= 4:
                return "open"
            if extended <= 1:
                return "fist"
            return ""
        except Exception:
            return ""
