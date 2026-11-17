"""
Moteur ASR streaming basé sur faster-whisper, exposant une méthode
d'inférence synchrone (bloquante, CPU/GPU-bound) destinée à être appelée
depuis un ThreadPoolExecutor par le gestionnaire de session asynchrone
(cf. session_manager.py) — jamais directement depuis la boucle asyncio, sous
peine de geler tout le serveur pendant l'inférence.

Le modèle est chargé UNE SEULE FOIS au démarrage du process (poids ~500 Mo à
1.5 Go selon la taille choisie, mis en cache localement par huggingface_hub
dès le premier téléchargement, puis réutilisables 100% hors-ligne) et
partagé par toutes les sessions. L'audio est toujours passé sous forme de
tableau numpy float32 directement en mémoire : à aucun moment un fichier
temporaire n'est écrit sur le disque.
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
    start: float  # secondes, relatif au début du buffer audio fourni
    end: float
    probability: float


class ASREngine:
    """Enveloppe autour de faster-whisper pour l'inférence streaming."""

    def __init__(self) -> None:
        logger.info(
            "Chargement du modèle Whisper '%s' (device=%s, compute_type=%s)...",
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
        # ctranslate2 (backend de faster-whisper) est thread-safe pour des
        # appels concurrents à `generate`/`transcribe` sur une même
        # instance de modèle ; on protège malgré tout par un verrou léger
        # côté Python pour éviter toute course sur des structures internes
        # lors de tests futurs sur des versions plus anciennes du backend.
        self._lock = threading.Lock()
        self._warmup()
        logger.info("Modèle Whisper prêt.")

    def _warmup(self) -> None:
        """Exécute une inférence factice au démarrage pour forcer le
        chargement des kernels CUDA / la compilation JIT et éviter un pic de
        latence sur la toute première requête utilisateur réelle."""
        silence = np.zeros(16_000, dtype=np.float32)
        list(self._model.transcribe(silence, language=WHISPER_LANGUAGE, beam_size=1)[0])

    def transcribe_buffer(
        self,
        audio_float32: np.ndarray,
        initial_prompt: Optional[str] = None,
    ) -> List[WordResult]:
        """Transcrit un buffer audio complet (mono, 16 kHz, float32 in
        [-1, 1]) et retourne la liste des mots avec leurs horodatages
        relatifs au DÉBUT DE CE BUFFER.

        Paramètres choisis pour la latence plutôt que la qualité maximale
        (beam_size=1, greedy decoding) : adapté au ré-échantillonnage très
        fréquent imposé par le streaming. `condition_on_previous_text=False`
        évite les boucles de répétition/hallucination typiques quand on
        ré-infère un buffer tronqué en continu.
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
                vad_filter=False,  # le VAD est déjà géré en amont par Silero
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


# Instance unique partagée par tout le process, initialisée au démarrage de
# l'application (cf. main.py -> lifespan) plutôt qu'à l'import du module,
# pour garder un contrôle explicite sur le moment du chargement (et pouvoir
# l'exposer proprement dans les logs de démarrage du serveur).
_engine_singleton: Optional[ASREngine] = None
_engine_lock = threading.Lock()


def get_asr_engine() -> ASREngine:
    global _engine_singleton
    if _engine_singleton is None:
        with _engine_lock:
            if _engine_singleton is None:
                _engine_singleton = ASREngine()
    return _engine_singleton
