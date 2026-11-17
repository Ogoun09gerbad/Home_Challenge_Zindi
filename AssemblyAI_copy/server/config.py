"""
Configuration centralisée de l'infrastructure de transcription streaming.

Toutes les valeurs sensibles (secrets, chemins de modèles) sont lues depuis
des variables d'environnement afin de ne JAMAIS committer de secret en dur,
et de permettre un déploiement air-gapped où seule la configuration change
d'un environnement à l'autre (les poids de modèles, eux, sont pré-téléchargés
une fois puis mis en cache localement par huggingface_hub / torch, donc plus
aucun accès réseau n'est requis ensuite).
"""

from __future__ import annotations

import os

import torch

# --------------------------------------------------------------------------
# Audio : format imposé par le cahier des charges
# --------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000          # Hz
CHANNELS: int = 1                  # mono
SAMPLE_WIDTH_BYTES: int = 2        # PCM 16 bits linéaire (int16)

# Taille de fenêtre imposée par Silero VAD pour un flux à 16 kHz : 512
# échantillons (soit 32 ms). Les chunks du client (50-100 ms) sont
# redécoupés en interne en fenêtres de cette taille (cf. ai/vad_engine.py).
VAD_WINDOW_SAMPLES: int = 512

# --------------------------------------------------------------------------
# VAD (Silero) — endpointing
# --------------------------------------------------------------------------
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
# Silence continu déclenchant la fin de phrase ("Endpoint").
VAD_MIN_SILENCE_MS: int = int(os.getenv("VAD_MIN_SILENCE_MS", "600"))
# Marge de sécurité ajoutée avant/après un segment de parole détecté, pour
# éviter de couper le tout début ou la toute fin d'un mot.
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
# Nombre de threads CPU alloués à CTranslate2 par instance d'inférence
# (ignoré sur GPU). 0 = laisse CTranslate2 déterminer automatiquement.
WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "0"))

# Nombre d'inférences Whisper pouvant s'exécuter en parallèle. Sur GPU on
# limite à 1 pour éviter la contention mémoire/kernels ; sur CPU on autorise
# plusieurs threads concurrents bornés par le nombre de coeurs.
WHISPER_MAX_CONCURRENT_INFERENCES: int = int(
    os.getenv(
        "WHISPER_MAX_CONCURRENT_INFERENCES",
        "1" if WHISPER_DEVICE == "cuda" else str(max(1, (os.cpu_count() or 4) // 2)),
    )
)

# Quantité minimale de nouvel audio (ms) devant s'être accumulée depuis la
# dernière inférence avant de relancer Whisper. Cela évite de ré-exécuter
# l'inférence sur chaque paquet de 50 ms reçu, ce qui saturerait le CPU/GPU
# sans gain de latence perceptible.
ASR_MIN_NEW_AUDIO_MS: int = int(os.getenv("ASR_MIN_NEW_AUDIO_MS", "280"))

# --------------------------------------------------------------------------
# Sécurité — validation du jeton JWT à la poignée de main WebSocket
# --------------------------------------------------------------------------
# ATTENTION : ceci est une implémentation de démonstration mais FONCTIONNELLE
# (le jeton est réellement décodé, signé et vérifié — ce n'est pas un
# no-op). En production, remplacez HS256 + secret partagé par une
# vérification RS256/ES256 contre les clés publiques (JWKS) de votre IdP
# (Keycloak, Auth0, Azure AD, etc.), et servez ce secret via un coffre-fort
# (Vault, KMS) plutôt qu'une variable d'environnement en clair.
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_DEV_ONLY_SECRET")
JWT_ALGORITHM: str = "HS256"

# --------------------------------------------------------------------------
# Divers
# --------------------------------------------------------------------------
# Nombre maximal de paquets audio en attente dans la file par session avant
# application d'un mécanisme de backpressure (rejet des paquets les plus
# anciens) afin de ne jamais laisser la latence dériver indéfiniment.
MAX_QUEUE_SIZE_PER_SESSION: int = int(os.getenv("MAX_QUEUE_SIZE_PER_SESSION", "50"))
