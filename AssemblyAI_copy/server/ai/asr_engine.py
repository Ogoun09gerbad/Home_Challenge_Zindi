"""
Streaming ASR engine based on faster-whisper, exposing a synchronous
inference method (blocking, CPU/GPU-bound) meant to be called from a
ThreadPoolExecutor by the asynchronous session manager (see
session_manager.py) — never directly from the asyncio loop, otherwise the
whole server would freeze during inference.

The model is loaded ONLY ONCE at process startup (weights of ~500 MB to
1.5 GB depending on the chosen size, cached locally by huggingface_hub after
the first download, then reusable 100% offline) and shared by all sessions.
Audio is always passed as an in-memory numpy float32 array: no temporary
file is ever written to disk.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel

from server.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger("asr_server.asr")


@dataclass
class WordResult:
    text: str
    start: float  # seconds, relative to the start of the provided audio buffer
    end: float
    probability: float


class ASREngine:
    """Wrapper around faster-whisper for streaming inference."""

    def __init__(self) -> None:
        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)...",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
        )
        # ctranslate2 (the faster-whisper backend) is thread-safe for
        # concurrent calls to `generate`/`transcribe` on the same model
        # instance; we nevertheless guard it with a lightweight Python-side
        # lock to avoid any race on internal structures when testing
        # against older backend versions in the future.
        self._lock = threading.Lock()
        self._warmup()
        logger.info("Whisper model ready.")

    def _warmup(self) -> None:
        """Runs a dummy inference at startup to force CUDA kernel loading /
        JIT compilation and avoid a latency spike on the very first real
        user request."""
        silence = np.zeros(16_000, dtype=np.float32)
        list(self._model.transcribe(silence, language=WHISPER_LANGUAGE, beam_size=1)[0])

    def transcribe_buffer(
        self,
        audio_float32: np.ndarray,
        initial_prompt: Optional[str] = None,
    ) -> List[WordResult]:
        """Transcribes a complete audio buffer (mono, 16 kHz, float32 in
        [-1, 1]) and returns the list of words with timestamps relative to
        the START OF THIS BUFFER.

        Parameters are tuned for latency rather than maximum quality
        (beam_size=1, greedy decoding): suited to the very frequent
        re-inference required by streaming. `condition_on_previous_text=False`
        avoids the repetition/hallucination loops typical of continuously
        re-inferring a truncated buffer.
        """
        if audio_float32.size == 0:
            return []

        with self._lock:
            segments, _info = self._model.transcribe(
                audio_float32,
                language=WHISPER_LANGUAGE,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
                vad_filter=False,  # VAD is already handled upstream by Silero
            )

            words: List[WordResult] = []
            for segment in segments:
                if not segment.words:
                    continue
                for word in segment.words:
                    cleaned = word.word.strip()
                    if not cleaned:
                        continue
                    words.append(
                        WordResult(
                            text=cleaned,
                            start=word.start,
                            end=word.end,
                            probability=word.probability,
                        )
                    )

        return words


# Single instance shared by the whole process, initialized at application
# startup (see main.py -> lifespan) rather than at module import, to keep
# explicit control over when loading happens (and to report it cleanly in
# the server startup logs).
_engine_singleton: Optional[ASREngine] = None
_engine_lock = threading.Lock()


def get_asr_engine() -> ASREngine:
    global _engine_singleton
    if _engine_singleton is None:
        with _engine_lock:
            if _engine_singleton is None:
                _engine_singleton = ASREngine()
    return _engine_singleton
