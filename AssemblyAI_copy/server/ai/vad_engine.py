"""
Détection d'activité vocale (VAD) streaming avec Silero VAD.

Silero VAD exige des fenêtres d'exactement 512 échantillons pour un flux à
16 kHz. Les paquets audio reçus du client (50-100 ms, soit 800 à 1600
échantillons) ne sont pas des multiples de 512 : ce module absorbe cette
différence via un petit buffer interne qui accumule les échantillons
entrants et ne consomme que des fenêtres complètes de 512, en conservant le
reliquat pour l'appel suivant.

Le modèle est chargé une seule fois au niveau du process (poids embarqués
dans le package pip `silero-vad`, donc aucun accès réseau après la première
installation) et partagé en lecture seule entre toutes les sessions ; seul
l'état de streaming (`VADIterator`) est propre à chaque session.
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

# Modèle Silero VAD chargé une seule fois pour tout le process (thread-safe
# en inférence pure car chaque appel est stateless côté poids ; l'état de
# streaming vit dans VADIterator, instancié par session plus bas).
_VAD_MODEL = load_silero_vad(onnx=False)


@dataclass
class VadEvent:
    """Résultat de l'analyse d'un paquet audio par le VAD."""

    speech_started: bool = False
    # True dès que le VAD confirme un silence continu >= VAD_MIN_SILENCE_MS
    # après un segment de parole : c'est le signal d' "Endpoint".
    endpoint_reached: bool = False


@dataclass
class VadSessionState:
    """État de streaming du VAD propre à UNE session utilisateur."""

    iterator: VADIterator = field(
        default_factory=lambda: VADIterator(
            _VAD_MODEL,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )
    )
    # Reliquat d'échantillons (float32, [-1, 1]) en attente d'atteindre
    # VAD_WINDOW_SAMPLES pour pouvoir être soumis au modèle.
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    in_speech: bool = False

    def reset(self) -> None:
        """Réinitialise l'état interne (à appeler après chaque Endpoint)."""
        self.iterator.reset_states()
        self._pending = np.zeros(0, dtype=np.float32)
        self.in_speech = False

    def process(self, audio_float32: np.ndarray) -> VadEvent:
        """Analyse un nouveau paquet audio et retourne les événements détectés.

        `audio_float32` doit être mono, 16 kHz, normalisé dans [-1, 1].
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

        # On conserve le reliquat non multiple de 512 pour le prochain appel.
        remainder_start = n_complete_windows * VAD_WINDOW_SAMPLES
        self._pending = buffer[remainder_start:]

        return event
