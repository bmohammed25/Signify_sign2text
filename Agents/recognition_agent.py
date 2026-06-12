"""
RecognitionAgent
Signify

Loads best_model_mobilenet.pth, applies val_transform to a 128×128 PIL crop,
runs GPU inference, and returns a RecognitionResult with per-frame smoothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import sys, os
import config
sys.path.insert(0, os.path.abspath(".."))


# ── Class labels (must match training order exactly) ─────────────────────────
CLASSES: list[str] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'space', 'nothing',
]


# ── Return dataclass ──────────────────────────────────────────────────────────
@dataclass
class RecognitionResult:
    letter:     str    # predicted class label, e.g. "A"
    confidence: float  # softmax probability 0–1
    stable:     bool   # True when smoothing window is unanimous (5/5 frames)


# ── Agent ─────────────────────────────────────────────────────────────────────
class RecognitionAgent:
    """
    Wraps MobileNetV2 inference with a 5-frame smoothing window.

    Parameters
    ----------
    model_path : str | Path
        Path to best_model_mobilenet.pth
    device : str | None
        'cuda', 'cpu', or None (auto-detect)
    window_size : int
        Number of consecutive frames that must agree for stable=True (default 5)
    """

    # val_transform — must match training exactly
    _transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __init__(
        self,
        model_path: str | Path = Path(config.BASE_DIR) / "models" / "saved" / "best_model_mobilenet.pth",
        device: str | None = None,
        window_size: int = 5,
    ) -> None:
        # ── Device ──────────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Model ───────────────────────────────────────────────────────────
        self._model = self._load_model(Path(model_path))

        # ── Smoothing buffer ─────────────────────────────────────────────────
        self._window_size = window_size
        self._buffer: deque[str] = deque(maxlen=window_size)

        print(f"[RecognitionAgent] device={self.device}  window={window_size}")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, crop: Image.Image) -> RecognitionResult:
        """
        Run inference on a 128×128 RGB PIL Image.

        Parameters
        ----------
        crop : PIL.Image.Image
            The crop from VisionResult.crop — already 128×128 RGB.

        Returns
        -------
        RecognitionResult
        """
        letter, confidence = self._infer(crop)
        self._buffer.append(letter)
        stable = (len(self._buffer) == self._window_size
                  and len(set(self._buffer)) == 1)
        return RecognitionResult(letter=letter, confidence=confidence, stable=stable)

    def reset_buffer(self) -> None:
        """Clear the smoothing window (call when a letter has been emitted)."""
        self._buffer.clear()

    @property
    def buffer_snapshot(self) -> list[str]:
        """Current contents of the smoothing window (most recent last)."""
        return list(self._buffer)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self, path: Path) -> nn.Module:
        if not path.exists():
            raise FileNotFoundError(
                f"[RecognitionAgent] Model not found: {path}\n"
                "Make sure models/saved/best_model_mobilenet.pth is present."
            )
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),        # [0]
            nn.Linear(1280, 512),   # [1]
            nn.ReLU(),              # [2]
            nn.Dropout(0.2),        # [3]
            nn.Linear(512, 29),     # [4]
        )
        state = torch.load(str(path), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        print(f"[RecognitionAgent] Loaded model from {path}")
        return model

    @torch.no_grad()
    def _infer(self, crop: Image.Image) -> tuple[str, float]:
        """Apply transform, run forward pass, return (letter, confidence)."""
        # crop should already be 128×128 RGB, but Resize handles any size
        tensor = self._transform(crop).unsqueeze(0).to(self.device)  # [1,3,128,128]
        logits = self._model(tensor)                                   # [1,29]
        probs  = torch.softmax(logits, dim=1)                          # [1,29]
        conf, idx = probs.max(dim=1)
        return CLASSES[idx.item()], round(conf.item(), 4)
