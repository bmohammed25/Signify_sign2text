"""
main.py — FastAPI Server
Signify

Model A pipeline: every frame is processed by MobileNetV2 for
ASL letter recognition. Finger-spelled letters are accumulated and
converted to natural sentences via phi3:mini (local, no internet).

Run with:
    uvicorn API.main:app --host 0.0.0.0 --port 8000 --reload

Then open the app at:
    http://localhost:8000/app
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Agents.vision_agent      import VisionAgent
from Agents.landmark_agent    import LandmarkAgent
from Agents.recognition_agent import RecognitionAgent
from Agents.language_agent    import LanguageAgent
from Agents.speech_agent      import SpeechAgent

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
log = logging.getLogger("signify.api")

# ── Project root ──────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent


# ════════════════════════════════════════════════════════════════════════════
#  Singleton agent container
# ════════════════════════════════════════════════════════════════════════════

class _Agents:
    vision:      VisionAgent
    landmark:    LandmarkAgent
    recognition: RecognitionAgent
    language:    LanguageAgent
    speech:      SpeechAgent
    device:      torch.device


agents = _Agents()


# ════════════════════════════════════════════════════════════════════════════
#  Lifespan
# ════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — initialising agents …")

    agents.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents.vision      = VisionAgent()
    agents.landmark    = LandmarkAgent()
    agents.recognition = RecognitionAgent()
    agents.language    = LanguageAgent()
    agents.speech      = SpeechAgent(rate=160, volume=1.0, non_blocking=True)

    log.info(f"Device: {agents.device}")
    log.info("All agents ready.")
    yield

    log.info("Shutting down …")
    agents.vision.release()
    agents.speech.shutdown()
    log.info("Goodbye.")


# ════════════════════════════════════════════════════════════════════════════
#  App
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Signify API",
    version="1.0.0",
    description="ASL hand-sign → text → voice  |  Model A (MobileNetV2)  |  Group 7",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
#  Response schemas
# ════════════════════════════════════════════════════════════════════════════

class FrameResponse(BaseModel):
    # VisionAgent
    hand_detected:   bool
    bbox:            list[int] | None

    # LandmarkAgent
    landmark_passed: bool
    pose_hint:       str
    reject_reason:   str

    # RecognitionAgent (Model A)
    letter:          str
    confidence:      float
    stable:          bool
    buffer_snapshot: list[str]
    letter_emitted:  bool

    # LanguageAgent accumulated state
    raw_text:        str
    current_word:    str
    word_count:      int


class GenerateResponse(BaseModel):
    raw_text:  str
    sentence:  str
    success:   bool


class SpeakResponse(BaseModel):
    text:   str
    spoken: bool
    error:  str


class ResetResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status:   str
    device:   str
    pipeline: list[str]


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _decode_upload(data: bytes) -> np.ndarray:
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image — send JPEG or PNG.")
    return frame


# ════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ════════════════════════════════════════════════════════════════════════════

@app.get("/app", tags=["frontend"])
def serve_app():
    """Serve the Signify web app."""
    html_path = PROJECT_ROOT / "front-end" / "signify_app_auto_with_video.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(html_path))


@app.get("/", response_model=StatusResponse, tags=["health"])
def root():
    return StatusResponse(
        status="ok",
        device=str(agents.device),
        pipeline=[
            "VisionAgent      — hand detection & crop",
            "LandmarkAgent    — quality gate",
            "RecognitionAgent — Model A letters (MobileNetV2)",
            "LanguageAgent    — phi3:mini sentence assembly",
            "SpeechAgent      — text to speech",
        ],
    )


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/frame", response_model=FrameResponse, tags=["pipeline"])
async def process_frame(file: UploadFile = File(...)):
    """
    Submit a single video frame (JPEG or PNG).

    The pipeline runs VisionAgent → LandmarkAgent → RecognitionAgent.
    Stable letters are accumulated by LanguageAgent.
    Call POST /generate when ready to convert letters to a sentence.
    Call POST /speak to speak the sentence aloud.
    Call at webcam frame-rate (15–30 fps).
    """
    data  = await file.read()
    frame = _decode_upload(data)

    # ── VisionAgent ───────────────────────────────────────────────────────
    vision_result = agents.vision.process_frame(frame)

    if vision_result is None:
        agents.recognition.reset_buffer()
        return FrameResponse(
            hand_detected=False, bbox=None,
            landmark_passed=False, pose_hint="", reject_reason="no hand detected",
            letter="nothing", confidence=0.0, stable=False,
            buffer_snapshot=[], letter_emitted=False,
            raw_text=agents.language.raw_text,
            current_word=agents.language.current_word,
            word_count=agents.language.word_count,
        )

    # ── LandmarkAgent ─────────────────────────────────────────────────────
    landmark_result = agents.landmark.process(vision_result)
    bbox = list(vision_result.bbox) if vision_result.bbox else None

    # ── RecognitionAgent (Model A) ────────────────────────────────────────
    recognition_result = agents.recognition.process(vision_result.crop)
    letter     = recognition_result.letter
    confidence = recognition_result.confidence
    stable     = recognition_result.stable
    letter_emitted = False

    if recognition_result.stable and landmark_result.passed:
        agents.language.push_letter(recognition_result.letter)
        agents.recognition.reset_buffer()
        letter_emitted = True
        log.info("Letter emitted: %s  raw=%r", letter, agents.language.raw_text)

    return FrameResponse(
        hand_detected=True,
        bbox=bbox,
        landmark_passed=landmark_result.passed,
        pose_hint=landmark_result.pose_hint,
        reject_reason=landmark_result.reject_reason,
        letter=letter,
        confidence=confidence,
        stable=stable,
        buffer_snapshot=agents.recognition.buffer_snapshot,
        letter_emitted=letter_emitted,
        raw_text=agents.language.raw_text,
        current_word=agents.language.current_word,
        word_count=agents.language.word_count,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["pipeline"])
def generate_sentence():
    """
    Flush the letter buffer through phi3:mini → natural sentence.
    Call this when the user finishes signing.
    """
    if not agents.language.raw_text.strip():
        raise HTTPException(status_code=400, detail="No letters accumulated yet.")

    result = agents.language.generate()
    log.info("generate() → success=%s  sentence=%r", result.success, result.sentence)
    return GenerateResponse(
        raw_text=result.raw_text,
        sentence=result.sentence,
        success=result.success,
    )


@app.post("/speak", response_model=SpeakResponse, tags=["pipeline"])
def speak_sentence(text: str | None = None):
    """
    Speak the given text, or auto-select the best available text.
    Pass text as query param to override: POST /speak?text=Hello+world
    """
    speak_text = text or agents.language.raw_text
    if not speak_text or not speak_text.strip():
        raise HTTPException(status_code=400, detail="Nothing to speak.")

    result = agents.speech.speak(speak_text)
    log.info("speak() → spoken=%s  error=%r", result.spoken, result.error)
    return SpeakResponse(text=result.text, spoken=result.spoken, error=result.error)


@app.post("/reset", response_model=ResetResponse, tags=["pipeline"])
def reset_pipeline():
    """Clear all buffers — call after a sentence has been spoken."""
    agents.language.reset()
    agents.recognition.reset_buffer()
    log.info("Pipeline reset.")
    return ResetResponse(message="All buffers cleared.")


@app.get("/state", tags=["pipeline"])
def get_state():
    """Return current accumulated text state without triggering inference."""
    return {
        "raw_text":        agents.language.raw_text,
        "current_word":    agents.language.current_word,
        "word_count":      agents.language.word_count,
        "buffer_snapshot": agents.recognition.buffer_snapshot,
    }
