"""
SpeechAgent
Signify

Wraps pyttsx3 TTS. Reinitialises the engine on every call to fix
the Windows one-shot exhaustion bug.
"""

from __future__ import annotations
import threading
from dataclasses import dataclass
from typing import Union

import pyttsx3

try:
    from agents.language_agent import LanguageResult
except ImportError:
    LanguageResult = None  # type: ignore


@dataclass
class SpeechResult:
    text:    str
    spoken:  bool
    error:   str


class SpeechAgent:
    def __init__(self, rate: int = 160, volume: float = 1.0, non_blocking: bool = True) -> None:
        self._rate = rate
        self._volume = volume
        self._non_blocking = non_blocking
        self._lock = threading.Lock()
        print(f"[SpeechAgent] rate={rate} wpm  volume={volume}  non_blocking={non_blocking}")

    def speak(self, input: Union[str, "LanguageResult"], wait: bool = False) -> SpeechResult:
        if LanguageResult is not None and isinstance(input, LanguageResult):
            if not input.success:
                return SpeechResult(text=input.sentence, spoken=False, error="LanguageResult.success is False")
            text = input.sentence
        else:
            text = str(input).strip()

        if not text:
            return SpeechResult(text="", spoken=False, error="Empty string")

        if self._non_blocking:
            t = threading.Thread(target=self._run, args=(text,), daemon=True)
            t.start()
            if wait:
                t.join(timeout=15)
            return SpeechResult(text=text, spoken=True, error="")
        else:
            return self._run(text)

    def _run(self, text: str) -> SpeechResult:
        with self._lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                return SpeechResult(text=text, spoken=True, error="")
            except Exception as exc:
                return SpeechResult(text=text, spoken=False, error=str(exc))

    def set_rate(self, rate: int) -> None:
        self._rate = rate

    def set_volume(self, volume: float) -> None:
        self._volume = volume

    def shutdown(self) -> None:
        pass
