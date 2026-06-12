"""
LanguageAgent
Signify

Handles two input modes:
  - Model A (letters): accumulates finger-spelled letters, sends to phi3:mini
    to produce a natural sentence (same as Step 15)
  - Model B (words): accepts top-5 word candidates from SignifyLSTM,
    uses phi3:mini to pick the most contextually correct word given the
    conversation so far, then appends it to the sentence buffer

phi3:mini is always local — no internet required.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass, field


# ── Return dataclasses ────────────────────────────────────────────────────────

@dataclass
class LanguageResult:
    raw_text: str    # accumulated letters / words as typed
    sentence: str    # phi3:mini output
    success:  bool   # False if Ollama unreachable or returned an error


@dataclass
class WordResult:
    chosen_word:  str         # word chosen by phi3:mini from top-5
    top5:         list[str]   # original top-5 candidates from Model B
    sentence:     str         # full sentence buffer so far
    success:      bool        # False if Ollama unreachable


# ── Agent ─────────────────────────────────────────────────────────────────────

class LanguageAgent:
    """
    Dual-mode language agent for Signify.

    Model A mode  — push_letter() / generate()
        Accumulates finger-spelled letters and converts them to a natural
        sentence via phi3:mini. Identical to Step 15 behaviour.

    Model B mode  — push_word_candidates() / get_sentence()
        Accepts top-5 word candidates from SignifyLSTM each time a word
        sign is recognised. phi3:mini picks the most contextually correct
        word given the sentence built so far, then appends it.

    Parameters
    ----------
    ollama_url : str
        Base URL for the Ollama API (default: http://localhost:11434).
    model : str
        Ollama model name (default: phi3:mini).
    timeout : int
        Seconds to wait for Ollama (default: 15 — phi3:mini can be slow).
    max_context_words : int
        How many previously chosen words to include as context when
        disambiguating the next word (default: 6).
    """

    _OLLAMA_ENDPOINT = "/api/generate"

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "phi3:mini",
        timeout: int = 15,
        max_context_words: int = 6,
    ) -> None:
        self._ollama_url        = ollama_url.rstrip("/")
        self._model             = model
        self._timeout           = timeout
        self._max_context_words = max_context_words

        # ── Model A state (letters) ───────────────────────────────────────
        self._words: list[str] = [""]   # list of words; last = current word

        # ── Model B state (word sequences) ────────────────────────────────
        self._sentence_words: list[str] = []   # chosen words so far
        self.last_sentence:   str       = ""   # last phi3:mini output

        print(
            f"[LanguageAgent] model={model}  "
            f"endpoint={self._ollama_url}{self._OLLAMA_ENDPOINT}"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Model A  — letter-by-letter (finger spelling)
    # ═════════════════════════════════════════════════════════════════════

    def push_letter(self, letter: str) -> None:
        """
        Accept one stable letter from RecognitionAgent and update the buffer.

        Special classes
        ---------------
        'space'   → commit the current word, start a new one
        'del'     → backspace
        'nothing' → ignored
        """
        if letter == "nothing":
            return

        if letter == "space":
            if self._words[-1]:
                self._words.append("")
            return

        if letter == "del":
            if self._words[-1]:
                self._words[-1] = self._words[-1][:-1]
            elif len(self._words) > 1:
                self._words.pop()
            return

        self._words[-1] += letter

    def generate(self) -> LanguageResult:
        """
        Send the buffered finger-spelled text to phi3:mini and return a
        natural sentence. Buffer is NOT cleared — call reset() after use.
        """
        raw_text = self.raw_text

        if not raw_text.strip():
            return LanguageResult(raw_text=raw_text, sentence="", success=True)

        prompt = (
            "Convert these ASL finger-spelled letters into a natural, "
            f"grammatically correct English sentence: {raw_text}\n"
            "Reply with the sentence only, no explanation."
        )

        sentence = self._call_ollama(prompt, fallback=raw_text)
        if sentence is None:
            return LanguageResult(raw_text=raw_text, sentence=raw_text, success=False)

        self.last_sentence = sentence
        return LanguageResult(raw_text=raw_text, sentence=sentence, success=True)

    def reset(self) -> None:
        """Clear the Model A letter buffer."""
        self._words = [""]

    # ═════════════════════════════════════════════════════════════════════
    #  Model B  — word-level signing
    # ═════════════════════════════════════════════════════════════════════

    def push_word_candidates(self, top5: list[str]) -> WordResult:
        """
        Accept top-5 word candidates from SignifyLSTM.

        phi3:mini picks the most contextually appropriate word given the
        sentence built so far, then appends it to the sentence buffer.

        Parameters
        ----------
        top5 : list[str]
            Top-5 predicted words from Model B, highest confidence first.

        Returns
        -------
        WordResult with the chosen word and updated sentence.
        """
        if not top5:
            return WordResult(chosen_word="", top5=[], sentence=self.sentence, success=False)

        # Build context from the last N chosen words
        context = " ".join(self._sentence_words[-self._max_context_words:])

        if context:
            prompt = (
                f"You are helping build an ASL sentence word by word.\n"
                f"Sentence so far: \"{context}\"\n"
                f"The next word is one of these (in order of likelihood): "
                f"{', '.join(top5)}\n"
                f"Choose the single most contextually appropriate word from the list above.\n"
                f"Reply with ONLY that one word, nothing else."
            )
        else:
            # No context yet — pick the highest-confidence word
            prompt = (
                f"You are helping build an ASL sentence word by word.\n"
                f"The first word is one of these (in order of likelihood): "
                f"{', '.join(top5)}\n"
                f"Choose the single most appropriate word to start a sentence.\n"
                f"Reply with ONLY that one word, nothing else."
            )

        chosen = self._call_ollama(prompt, fallback=top5[0])
        if chosen is None:
            # Ollama unreachable — fall back to top-1
            chosen = top5[0]
            success = False
        else:
            # Clean up — model sometimes adds punctuation or extra words
            chosen = chosen.strip().strip(".,!?\"'").split()[0] if chosen.strip() else top5[0]
            success = True

        self._sentence_words.append(chosen)
        sentence = self.sentence
        self.last_sentence = sentence

        print(f"[LanguageAgent] Word chosen: {chosen}  (from {top5})  sentence: \"{sentence}\"")

        return WordResult(
            chosen_word=chosen,
            top5=top5,
            sentence=sentence,
            success=success,
        )

    def reset_sentence(self) -> None:
        """Clear the Model B word buffer."""
        self._sentence_words = []

    def reset_all(self) -> None:
        """Clear both Model A and Model B buffers."""
        self.reset()
        self.reset_sentence()

    # ═════════════════════════════════════════════════════════════════════
    #  Shared
    # ═════════════════════════════════════════════════════════════════════

    def speak_ready(self) -> str:
        """
        Return the best available text for SpeechAgent.
        Prefers the Model B sentence if words have been signed,
        otherwise falls back to the Model A raw_text.
        """
        if self._sentence_words:
            return self.sentence
        return self.raw_text

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def raw_text(self) -> str:
        """Model A: current buffer as space-joined string, e.g. 'HEL LO'."""
        return " ".join(self._words).strip()

    @property
    def current_word(self) -> str:
        """Model A: the word currently being spelled."""
        return self._words[-1]

    @property
    def word_count(self) -> int:
        """Model A: number of completed words."""
        return sum(1 for w in self._words if w)

    @property
    def sentence(self) -> str:
        """Model B: sentence built from chosen words so far."""
        return " ".join(self._sentence_words)

    @property
    def sentence_word_count(self) -> int:
        """Model B: number of words chosen so far."""
        return len(self._sentence_words)

    # ── Private ───────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str, fallback: str) -> str | None:
        """
        Send prompt to Ollama and return the response text.
        Returns None on connection error / timeout (caller handles fallback).
        """
        try:
            response = requests.post(
                f"{self._ollama_url}{self._OLLAMA_ENDPOINT}",
                json={
                    "model":  self._model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except requests.exceptions.ConnectionError:
            print("[LanguageAgent] ⚠  Cannot reach Ollama — is it running?")
            return None

        except requests.exceptions.Timeout:
            print(f"[LanguageAgent] ⚠  Ollama timed out after {self._timeout}s")
            return None

        except Exception as exc:  # noqa: BLE001
            print(f"[LanguageAgent] ⚠  Unexpected error: {exc}")
            return None
