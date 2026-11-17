"""
Gestionnaire de session de transcription : file d'attente en mémoire vive,
buffers audio, et boucle asynchrone orchestrant VAD + ASR (Local Agreement)
pour UNE connexion WebSocket.

Flux général par session :

    WebSocket.receive_bytes()  ->  asyncio.Queue  ->  boucle de traitement
                                                          |-> VAD (endpoint ?)
                                                          |-> ASR (Local Agreement)
                                                          |-> WebSocket.send_json()

Toutes les données audio transitent exclusivement en mémoire vive
(numpy.ndarray / bytes Python) : aucune écriture disque n'est effectuée à
quelque étape que ce soit. Les buffers sont explicitement libérés (`del` +
`gc.collect()`) dès qu'une phrase est finalisée ou que la session se
termine, conformément à la contrainte de confidentialité absolue.
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

# Pool de threads partagé par toutes les sessions pour exécuter les appels
# bloquants à faster-whisper sans geler la boucle asyncio. La taille est
# bornée par WHISPER_MAX_CONCURRENT_INFERENCES (cf. config.py) pour éviter
# la contention GPU/CPU en cas de nombreuses sessions simultanées.
_inference_executor = ThreadPoolExecutor(
    max_workers=WHISPER_MAX_CONCURRENT_INFERENCES,
    thread_name_prefix="asr-inference",
)
# Sémaphore miroir de la taille du pool : garantit qu'on ne soumet jamais
# plus de tâches en vol que de workers disponibles, évitant un
# empilement de futures en attente qui ferait dériver la latence.
_inference_semaphore = asyncio.Semaphore(WHISPER_MAX_CONCURRENT_INFERENCES)

# Une seule instance de "phrase" == 1 message final. speaker_id est fixé à 1
# car ce système ne fait pas de diarisation multi-locuteur (extension
# possible via pyannote.audio, hors périmètre de cette implémentation).
DEFAULT_SPEAKER_ID = 1


def _pcm16_bytes_to_float32(chunk: bytes) -> np.ndarray:
    """Convertit un paquet PCM 16 bits little-endian brut en float32
    normalisé dans [-1, 1], sans jamais toucher le disque."""
    int16_array = np.frombuffer(chunk, dtype="<i2")
    return int16_array.astype(np.float32) / 32768.0


@dataclass
class UtteranceState:
    """État de la phrase actuellement en cours de transcription.

    Réinitialisé à chaque Endpoint (fin de phrase détectée par le VAD).
    """

    # Audio non encore confirmé par le Local Agreement (float32, 16 kHz).
    pending_audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Mots déjà confirmés pour la phrase en cours (deviendront le texte
    # final une fois l'Endpoint atteint).
    committed_words: List[str] = field(default_factory=list)
    agreement: LocalAgreementBuffer = field(default_factory=LocalAgreementBuffer)
    # Horodatage (monotonic) du dernier échantillon ajouté depuis la
    # dernière inférence, pour throttler les appels à Whisper.
    samples_since_last_inference: int = 0

    def purge(self) -> None:
        """Libère explicitement les buffers de la phrase courante."""
        del self.pending_audio
        del self.committed_words
        self.pending_audio = np.zeros(0, dtype=np.float32)
        self.committed_words = []
        self.agreement.reset()
        self.samples_since_last_inference = 0


class TranscriptionSession:
    """Gère le cycle de vie complet d'une connexion WebSocket de
    transcription : ingestion, file d'attente, traitement, envoi des
    résultats, et nettoyage mémoire strict à la déconnexion.
    """

    def __init__(self, websocket: WebSocket, user_claims: dict):
        self.session_id = str(uuid.uuid4())
        self.websocket = websocket
        self.user_claims = user_claims

        # File d'attente en mémoire vive des paquets binaires entrants.
        # Bornée pour appliquer une backpressure explicite plutôt que de
        # laisser la mémoire croître indéfiniment si le producteur (réseau)
        # est plus rapide que le consommateur (VAD+ASR).
        self.input_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(
            maxsize=MAX_QUEUE_SIZE_PER_SESSION
        )

        self.vad_state = VadSessionState()
        self.utterance = UtteranceState()
        self.asr_engine: ASREngine = get_asr_engine()

        self._closed = False

    # ------------------------------------------------------------------
    # Producteur : réception réseau -> file d'attente
    # ------------------------------------------------------------------
    async def receive_loop(self) -> None:
        """Lit en continu les paquets binaires du WebSocket et les pousse
        dans la file d'attente. S'arrête proprement sur déconnexion."""
        try:
            while True:
                message = await self.websocket.receive_bytes()
                try:
                    self.input_queue.put_nowait(message)
                except asyncio.QueueFull:
                    # Backpressure : on supprime le paquet le plus ancien
                    # plutôt que le plus récent, pour privilégier la
                    # fraîcheur (temps réel) au prix d'un léger trou audio
                    # en cas de surcharge exceptionnelle.
                    logger.warning(
                        "[%s] File d'attente saturée, paquet le plus ancien supprimé.",
                        self.session_id,
                    )
                    try:
                        self.input_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await self.input_queue.put(message)
        except WebSocketDisconnect:
            logger.info("[%s] Client déconnecté.", self.session_id)
        finally:
            # Sentinelle de fin pour réveiller la boucle de traitement.
            await self.input_queue.put(None)

    # ------------------------------------------------------------------
    # Consommateur : file d'attente -> VAD -> ASR -> envoi résultats
    # ------------------------------------------------------------------
    async def processing_loop(self) -> None:
        try:
            while True:
                chunk = await self.input_queue.get()
                if chunk is None:  # sentinelle de fin de flux
                    break

                await self._handle_chunk(chunk)
        finally:
            await self._flush_remaining_and_cleanup()

    async def _handle_chunk(self, chunk: bytes) -> None:
        audio_float32 = _pcm16_bytes_to_float32(chunk)

        # 1) Endpointing : le VAD consomme l'audio brut indépendamment du
        #    buffer d'inférence ASR (les deux ont des granularités différentes).
        vad_event = self.vad_state.process(audio_float32)

        # 2) Accumulation dans le buffer de la phrase en cours.
        self.utterance.pending_audio = np.concatenate(
            [self.utterance.pending_audio, audio_float32]
        )
        self.utterance.samples_since_last_inference += len(audio_float32)

        if vad_event.endpoint_reached:
            await self._finalize_utterance()
            return

        # 3) Inférence partielle throttlée : on ne relance Whisper que si
        #    suffisamment de nouvel audio est arrivé depuis la dernière fois.
        new_audio_ms = (self.utterance.samples_since_last_inference / SAMPLE_RATE) * 1000
        if new_audio_ms >= ASR_MIN_NEW_AUDIO_MS and self.utterance.pending_audio.size > 0:
            await self._run_partial_inference()

    # ------------------------------------------------------------------
    # Inférence partielle (mode "partial") avec Local Agreement
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
            # On ne conserve que l'audio pas encore consommé par les mots
            # confirmés : réduit le coût des inférences suivantes et évite
            # de re-confirmer indéfiniment les mêmes mots.
            trimmed = self.utterance.pending_audio[cut_sample:].copy()
            del self.utterance.pending_audio
            self.utterance.pending_audio = trimmed
            # Le buffer audio a changé : l'hypothèse précédente n'est plus
            # comparable telle quelle, on repart sur une comparaison fraîche.
            self.utterance.agreement.reset()

        partial_text = " ".join(self.utterance.committed_words + result.partial_words)
        if partial_text:
            await self._send_json(
                {"status": "partial", "text": partial_text}
            )

    # ------------------------------------------------------------------
    # Finalisation (mode "final") déclenchée par l'Endpoint du VAD
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
            logger.info("[%s] Phrase finalisée : %r", self.session_id, final_text)

        # --- Nettoyage mémoire strict : la phrase est terminée, plus aucune
        # trace de l'audio brut ni du texte ne doit subsister au-delà de ce
        # point (contrainte de confidentialité absolue). ---
        self.utterance.purge()
        self.vad_state.reset()
        gc.collect()

    async def _flush_remaining_and_cleanup(self) -> None:
        """Appelé à la fin de processing_loop (déconnexion ou erreur) :
        finalise une éventuelle phrase en cours puis purge tous les buffers.
        """
        if self.utterance.pending_audio.size > 0 or self.utterance.committed_words:
            await self._finalize_utterance()

        # Purge finale explicite de tous les objets volumineux de la
        # session avant que le garbage collector ne les récupère.
        del self.utterance
        del self.vad_state
        self.utterance = UtteranceState()
        self.vad_state = VadSessionState()
        gc.collect()
        self._closed = True
        logger.info("[%s] Session nettoyée, mémoire purgée.", self.session_id)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------
    async def _transcribe_async(
        self, audio: np.ndarray, prompt: Optional[str]
    ) -> List[WordResult]:
        """Exécute l'inférence Whisper (bloquante) dans le pool de threads
        dédié, sans jamais bloquer la boucle asyncio principale."""
        loop = asyncio.get_running_loop()
        async with _inference_semaphore:
            start = time.monotonic()
            words = await loop.run_in_executor(
                _inference_executor, self.asr_engine.transcribe_buffer, audio, prompt
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.debug(
                "[%s] Inférence ASR en %.1f ms sur %.0f ms d'audio.",
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
        except Exception:  # connexion fermée entre-temps
            logger.debug("[%s] Envoi impossible, connexion probablement fermée.", self.session_id)

    async def run(self) -> None:
        """Point d'entrée : démarre producteur et consommateur en parallèle
        et attend leur terminaison conjointe."""
        receiver = asyncio.create_task(self.receive_loop(), name=f"recv-{self.session_id}")
        processor = asyncio.create_task(self.processing_loop(), name=f"proc-{self.session_id}")

        try:
            await asyncio.gather(receiver, processor)
        finally:
            for task in (receiver, processor):
                if not task.done():
                    task.cancel()
