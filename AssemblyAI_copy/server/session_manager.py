"""
Transcription session manager: in-memory queue, audio buffers, and the
asynchronous loop orchestrating VAD + ASR (Local Agreement) for ONE
WebSocket connection.

Overall flow per session:

    WebSocket.receive_bytes()  ->  asyncio.Queue  ->  processing loop
                                                          |-> VAD (endpoint?)
                                                          |-> ASR (Local Agreement)
                                                          |-> WebSocket.send_json()

All audio data flows exclusively through RAM (numpy.ndarray / Python
bytes): no disk write is performed at any stage. Buffers are explicitly
released (`del` + `gc.collect()`) as soon as an utterance is finalized or
the session ends, in accordance with the strict confidentiality
requirement.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from server.ai.asr_engine import ASREngine, WordResult, get_asr_engine
from server.ai.local_agreement import LocalAgreementBuffer
from server.ai.vad_engine import VadSessionState
from server.config import (
    ASR_MIN_NEW_AUDIO_MS,
    MAX_QUEUE_SIZE_PER_SESSION,
    SAMPLE_RATE,
    WHISPER_MAX_CONCURRENT_INFERENCES,
)

logger = logging.getLogger("asr_server.session")

# Thread pool shared by all sessions to run blocking faster-whisper calls
# without freezing the asyncio loop. Its size is bounded by
# WHISPER_MAX_CONCURRENT_INFERENCES (see config.py) to avoid GPU/CPU
# contention when many sessions are active simultaneously.
_inference_executor = ThreadPoolExecutor(
    max_workers=WHISPER_MAX_CONCURRENT_INFERENCES,
    thread_name_prefix="asr-inference",
)
# Semaphore mirroring the pool size: guarantees we never submit more
# in-flight tasks than available workers, avoiding a pile-up of pending
# futures that would make latency drift.
_inference_semaphore = asyncio.Semaphore(WHISPER_MAX_CONCURRENT_INFERENCES)

# One "utterance" instance == 1 final message. speaker_id is fixed to 1
# because this system does not perform multi-speaker diarization (possible
# extension via pyannote.audio, outside the scope of this implementation).
DEFAULT_SPEAKER_ID = 1


def _pcm16_bytes_to_float32(chunk: bytes) -> np.ndarray:
    """Converts a raw little-endian 16-bit PCM packet to float32 normalized
    to [-1, 1], without ever touching the disk."""
    int16_array = np.frombuffer(chunk, dtype="<i2")
    return int16_array.astype(np.float32) / 32768.0


@dataclass
class UtteranceState:
    """State of the utterance currently being transcribed.

    Reset at each Endpoint (end of utterance detected by the VAD).
    """

    # Audio not yet confirmed by Local Agreement (float32, 16 kHz).
    pending_audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Words already confirmed for the current utterance (will become the
    # final text once the Endpoint is reached).
    committed_words: List[str] = field(default_factory=list)
    agreement: LocalAgreementBuffer = field(default_factory=LocalAgreementBuffer)
    # Number of samples added since the last inference, used to throttle
    # calls to Whisper.
    samples_since_last_inference: int = 0

    def purge(self) -> None:
        """Explicitly releases the buffers of the current utterance."""
        del self.pending_audio
        del self.committed_words
        self.pending_audio = np.zeros(0, dtype=np.float32)
        self.committed_words = []
        self.agreement.reset()
        self.samples_since_last_inference = 0


class TranscriptionSession:
    """Manages the full lifecycle of a transcription WebSocket connection:
    ingestion, queueing, processing, sending results, and strict memory
    cleanup on disconnect.
    """

    def __init__(self, websocket: WebSocket, user_claims: dict):
        self.session_id = str(uuid.uuid4())
        self.websocket = websocket
        self.user_claims = user_claims

        # In-memory queue of incoming binary packets. Bounded to apply
        # explicit backpressure rather than letting memory grow
        # indefinitely if the producer (network) is faster than the
        # consumer (VAD+ASR).
        self.input_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(
            maxsize=MAX_QUEUE_SIZE_PER_SESSION
        )

        self.vad_state = VadSessionState()
        self.utterance = UtteranceState()
        self.asr_engine: ASREngine = get_asr_engine()

        self._closed = False

    # ------------------------------------------------------------------
    # Producer: network reception -> queue
    # ------------------------------------------------------------------
    async def receive_loop(self) -> None:
        """Continuously reads binary packets from the WebSocket and pushes
        them into the queue. Stops cleanly on disconnect."""
        try:
            while True:
                message = await self.websocket.receive_bytes()
                try:
                    self.input_queue.put_nowait(message)
                except asyncio.QueueFull:
                    # Backpressure: drop the oldest packet rather than the
                    # newest, favoring freshness (real time) at the cost of
                    # a small audio gap under exceptional overload.
                    logger.warning(
                        "[%s] Queue full, oldest packet dropped.",
                        self.session_id,
                    )
                    try:
                        self.input_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await self.input_queue.put(message)
        except WebSocketDisconnect:
            logger.info("[%s] Client disconnected.", self.session_id)
        finally:
            # End-of-stream sentinel to wake up the processing loop.
            await self.input_queue.put(None)

    # ------------------------------------------------------------------
    # Consumer: queue -> VAD -> ASR -> send results
    # ------------------------------------------------------------------
    async def processing_loop(self) -> None:
        try:
            while True:
                chunk = await self.input_queue.get()
                if chunk is None:  # end-of-stream sentinel
                    break

                await self._handle_chunk(chunk)
        finally:
            await self._flush_remaining_and_cleanup()

    async def _handle_chunk(self, chunk: bytes) -> None:
        audio_float32 = _pcm16_bytes_to_float32(chunk)

        # 1) Endpointing: the VAD consumes raw audio independently of the
        #    ASR inference buffer (the two have different granularities).
        vad_event = self.vad_state.process(audio_float32)

        # 2) Accumulate into the buffer of the current utterance.
        self.utterance.pending_audio = np.concatenate(
            [self.utterance.pending_audio, audio_float32]
        )
        self.utterance.samples_since_last_inference += len(audio_float32)

        if vad_event.endpoint_reached:
            await self._finalize_utterance()
            return

        # 3) Throttled partial inference: Whisper is only re-run if enough
        #    new audio has arrived since the last time.
        new_audio_ms = (self.utterance.samples_since_last_inference / SAMPLE_RATE) * 1000
        if new_audio_ms >= ASR_MIN_NEW_AUDIO_MS and self.utterance.pending_audio.size > 0:
            await self._run_partial_inference()

    # ------------------------------------------------------------------
    # Partial inference ("partial" mode) with Local Agreement
    # ------------------------------------------------------------------
    async def _run_partial_inference(self) -> None:
        prompt = " ".join(self.utterance.committed_words) or None
        audio_snapshot = self.utterance.pending_audio

        words = await self._transcribe_async(audio_snapshot, prompt)
        self.utterance.samples_since_last_inference = 0

        if not words:
            return

        word_texts = [w.text for w in words]
        result = self.utterance.agreement.update(word_texts)

        if result.confirmed_words:
            k = len(result.confirmed_words)
            cut_time_s = words[k - 1].end
            cut_sample = min(
                int(cut_time_s * SAMPLE_RATE), self.utterance.pending_audio.size
            )

            self.utterance.committed_words.extend(result.confirmed_words)
            # Keep only the audio not yet consumed by the confirmed words:
            # reduces the cost of subsequent inferences and avoids
            # re-confirming the same words indefinitely.
            trimmed = self.utterance.pending_audio[cut_sample:].copy()
            del self.utterance.pending_audio
            self.utterance.pending_audio = trimmed
            # The audio buffer has changed: the previous hypothesis is no
            # longer directly comparable, so start a fresh comparison.
            self.utterance.agreement.reset()

        partial_text = " ".join(self.utterance.committed_words + result.partial_words)
        if partial_text:
            await self._send_json(
                {"status": "partial", "text": partial_text}
            )

    # ------------------------------------------------------------------
    # Finalization ("final" mode) triggered by the VAD Endpoint
    # ------------------------------------------------------------------
    async def _finalize_utterance(self) -> None:
        prompt = " ".join(self.utterance.committed_words) or None

        if self.utterance.pending_audio.size > 0:
            words = await self._transcribe_async(self.utterance.pending_audio, prompt)
            self.utterance.committed_words.extend(w.text for w in words)

        final_text = " ".join(self.utterance.committed_words).strip()

        if final_text:
            await self._send_json(
                {
                    "status": "final",
                    "text": final_text,
                    "speaker_id": DEFAULT_SPEAKER_ID,
                }
            )
            logger.info("[%s] Utterance finalized: %r", self.session_id, final_text)

        # --- Strict memory cleanup: the utterance is complete, no trace of
        # the raw audio or the text must remain beyond this point (strict
        # confidentiality requirement). ---
        self.utterance.purge()
        self.vad_state.reset()
        gc.collect()

    async def _flush_remaining_and_cleanup(self) -> None:
        """Called at the end of processing_loop (disconnect or error):
        finalizes any utterance in progress, then purges all buffers.
        """
        if self.utterance.pending_audio.size > 0 or self.utterance.committed_words:
            await self._finalize_utterance()

        # Explicit final purge of all large session objects before the
        # garbage collector reclaims them.
        del self.utterance
        del self.vad_state
        self.utterance = UtteranceState()
        self.vad_state = VadSessionState()
        gc.collect()
        self._closed = True
        logger.info("[%s] Session cleaned up, memory purged.", self.session_id)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    async def _transcribe_async(
        self, audio: np.ndarray, prompt: Optional[str]
    ) -> List[WordResult]:
        """Runs the (blocking) Whisper inference in the dedicated thread
        pool, without ever blocking the main asyncio loop."""
        loop = asyncio.get_running_loop()
        async with _inference_semaphore:
            start = time.monotonic()
            words = await loop.run_in_executor(
                _inference_executor, self.asr_engine.transcribe_buffer, audio, prompt
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.debug(
                "[%s] ASR inference took %.1f ms on %.0f ms of audio.",
                self.session_id,
                elapsed_ms,
                (audio.size / SAMPLE_RATE) * 1000,
            )
        return words

    async def _send_json(self, payload: dict) -> None:
        if self._closed:
            return
        try:
            await self.websocket.send_json(payload)
        except Exception:  # connection closed in the meantime
            logger.debug("[%s] Send failed, connection probably closed.", self.session_id)

    async def run(self) -> None:
        """Entry point: starts the producer and consumer in parallel and
        waits for both to finish."""
        receiver = asyncio.create_task(self.receive_loop(), name=f"recv-{self.session_id}")
        processor = asyncio.create_task(self.processing_loop(), name=f"proc-{self.session_id}")

        try:
            await asyncio.gather(receiver, processor)
        finally:
            for task in (receiver, processor):
                if not task.done():
                    task.cancel()
