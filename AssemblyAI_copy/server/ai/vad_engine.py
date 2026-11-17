"""
Streaming voice activity detection (VAD) with Silero VAD.

Silero VAD requires windows of exactly 512 samples for a 16 kHz stream.
Audio packets received from the client (50-100 ms, i.e. 800 to 1600
samples) are not multiples of 512: this module absorbs the difference with
a small internal buffer that accumulates incoming samples and only consumes
complete 512-sample windows, keeping the remainder for the next call.

The model is loaded only once per process (weights bundled in the
`silero-vad` pip package, so no network access is needed after the first
installation) and shared read-only across all sessions; only the streaming
state (`VADIterator`) is specific to each session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from server.config import (
    SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
    VAD_WINDOW_SAMPLES,
)

logger = logging.getLogger("asr_server.vad")

# Silero VAD model loaded only once for the whole process (thread-safe for
# pure inference since each call is stateless with respect to the weights;
# the streaming state lives in VADIterator, instantiated per session below).
_VAD_MODEL = load_silero_vad(onnx=False)


@dataclass
class VadEvent:
    """Result of the VAD analysis of an audio packet."""

    speech_started: bool = False
    # True as soon as the VAD confirms continuous silence >= VAD_MIN_SILENCE_MS
    # after a speech segment: this is the "Endpoint" signal.
    endpoint_reached: bool = False


@dataclass
class VadSessionState:
    """VAD streaming state specific to ONE user session."""

    iterator: VADIterator = field(
        default_factory=lambda: VADIterator(
            _VAD_MODEL,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )
    )
    # Leftover samples (float32, [-1, 1]) waiting to reach
    # VAD_WINDOW_SAMPLES before they can be fed to the model.
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    in_speech: bool = False

    def reset(self) -> None:
        """Resets the internal state (to be called after each Endpoint)."""
        self.iterator.reset_states()
        self._pending = np.zeros(0, dtype=np.float32)
        self.in_speech = False

    def process(self, audio_float32: np.ndarray) -> VadEvent:
        """Analyzes a new audio packet and returns the detected events.

        `audio_float32` must be mono, 16 kHz, normalized to [-1, 1].
        """
        event = VadEvent()

        buffer = np.concatenate([self._pending, audio_float32])
        n_complete_windows = len(buffer) // VAD_WINDOW_SAMPLES

        for i in range(n_complete_windows):
            start = i * VAD_WINDOW_SAMPLES
            window = buffer[start : start + VAD_WINDOW_SAMPLES]
            tensor_window = torch.from_numpy(window)

            result: Optional[dict] = self.iterator(tensor_window, return_seconds=False)
            if result is None:
                continue

            if "start" in result:
                self.in_speech = True
                event.speech_started = True
            elif "end" in result:
                self.in_speech = False
                event.endpoint_reached = True

        # Keep the remainder that is not a multiple of 512 for the next call.
        remainder_start = n_complete_windows * VAD_WINDOW_SAMPLES
        self._pending = buffer[remainder_start:]

        return event
