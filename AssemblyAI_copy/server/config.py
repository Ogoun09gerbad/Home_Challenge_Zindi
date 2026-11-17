"""
Centralized configuration for the streaming transcription infrastructure.

All sensitive values (secrets, model paths) are read from environment
variables so that secrets are NEVER hard-coded and committed, and to allow
an air-gapped deployment where only the configuration changes from one
environment to another (model weights are downloaded once and then cached
locally by huggingface_hub / torch, so no network access is required
afterwards).
"""

from __future__ import annotations

import os

import torch

# --------------------------------------------------------------------------
# Audio: format mandated by the specification
# --------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000          # Hz
CHANNELS: int = 1                  # mono
SAMPLE_WIDTH_BYTES: int = 2        # 16-bit linear PCM (int16)

# Window size required by Silero VAD for a 16 kHz stream: 512 samples
# (i.e. 32 ms). Client chunks (50-100 ms) are internally re-split into
# windows of this size (see ai/vad_engine.py).
VAD_WINDOW_SAMPLES: int = 512

# --------------------------------------------------------------------------
# VAD (Silero) — endpointing
# --------------------------------------------------------------------------
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
# Continuous silence that triggers end of utterance ("Endpoint").
VAD_MIN_SILENCE_MS: int = int(os.getenv("VAD_MIN_SILENCE_MS", "600"))
# Safety margin added before/after a detected speech segment, to avoid
# clipping the very beginning or very end of a word.
VAD_SPEECH_PAD_MS: int = int(os.getenv("VAD_SPEECH_PAD_MS", "200"))

# --------------------------------------------------------------------------
# ASR (faster-whisper)
# --------------------------------------------------------------------------
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "fr")

_CUDA_AVAILABLE = torch.cuda.is_available()
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda" if _CUDA_AVAILABLE else "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv(
    "WHISPER_COMPUTE_TYPE", "float16" if WHISPER_DEVICE == "cuda" else "int8"
)
# Number of CPU threads allocated to CTranslate2 per inference instance
# (ignored on GPU). 0 = let CTranslate2 decide automatically.
WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "0"))

# Number of Whisper inferences that may run in parallel. On GPU this is
# limited to 1 to avoid memory/kernel contention; on CPU several concurrent
# threads are allowed, bounded by the number of cores.
WHISPER_MAX_CONCURRENT_INFERENCES: int = int(
    os.getenv(
        "WHISPER_MAX_CONCURRENT_INFERENCES",
        "1" if WHISPER_DEVICE == "cuda" else str(max(1, (os.cpu_count() or 4) // 2)),
    )
)

# Minimum amount of new audio (ms) that must have accumulated since the
# last inference before Whisper is run again. This avoids re-running
# inference on every 50 ms packet received, which would saturate the
# CPU/GPU with no perceptible latency gain.
ASR_MIN_NEW_AUDIO_MS: int = int(os.getenv("ASR_MIN_NEW_AUDIO_MS", "280"))

# --------------------------------------------------------------------------
# Security — JWT validation during the WebSocket handshake
# --------------------------------------------------------------------------
# WARNING: this is a demonstration implementation, but a FUNCTIONAL one
# (the token is actually decoded and its signature verified — it is not a
# no-op). In production, replace HS256 + shared secret with RS256/ES256
# verification against your IdP's public keys (JWKS) (Keycloak, Auth0,
# Azure AD, etc.), and serve this secret from a vault (Vault, KMS) rather
# than a plaintext environment variable.
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_DEV_ONLY_SECRET")
JWT_ALGORITHM: str = "HS256"

# --------------------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------------------
# Maximum number of audio packets waiting in the per-session queue before
# a backpressure mechanism kicks in (dropping the oldest packets) so that
# latency can never drift indefinitely.
MAX_QUEUE_SIZE_PER_SESSION: int = int(os.getenv("MAX_QUEUE_SIZE_PER_SESSION", "50"))
