"""
Point d'entrée de l'infrastructure de transcription vocale streaming.

Démarrage (développement) :
    uvicorn server.main:app --host 0.0.0.0 --port 8000

Démarrage (production, HTTPS/WSS via reverse proxy TLS-terminating type
nginx/traefik en amont, ou --ssl-keyfile/--ssl-certfile directement) :
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1 \
        --ssl-keyfile key.pem --ssl-certfile cert.pem

NOTE IMPORTANTE : --workers doit rester à 1 par GPU disponible. Le modèle
Whisper est chargé en mémoire par process ; plusieurs workers dupliqueraient
le modèle en VRAM/RAM. Pour scaler horizontalement, déployez plusieurs
instances (une par GPU) derrière un load-balancer WebSocket-aware.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from server.ai.asr_engine import get_asr_engine
from server.security import AuthenticationError, authenticate_websocket
from server.session_manager import TranscriptionSession

CLIENT_HTML_PATH = Path(__file__).resolve().parent.parent / "client" / "index.html"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("asr_server.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle Whisper (+ warm-up) une seule fois au démarrage
    # du process, avant d'accepter la moindre connexion WebSocket, pour que
    # la toute première session utilisateur bénéficie déjà de la latence
    # nominale (< 300 ms) sans pic de "cold start".
    logger.info("Initialisation du moteur ASR...")
    get_asr_engine()
    logger.info("Serveur prêt à accepter des connexions.")
    yield
    logger.info("Arrêt du serveur.")


app = FastAPI(
    title="Streaming ASR — Infrastructure souveraine on-premise",
    lifespan=lifespan,
)

# CORS : à restreindre explicitement aux origines de vos applications
# clientes en production (ne jamais garder "*" en environnement sensible).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Endpoint de supervision (liveness/readiness probe)."""
    return {"status": "ok"}


@app.get("/")
async def serve_test_client() -> FileResponse:
    """Sert le client de test HTML/JS (cf. client/index.html) directement
    depuis le serveur, par simple confort de développement : évite d'avoir
    à ouvrir le fichier local manuellement et place le client sur la même
    origine que l'API (pas d'enjeu CORS). À retirer en production, où le
    client sera plutôt servi par un CDN/reverse proxy dédié."""
    return FileResponse(CLIENT_HTML_PATH)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    """Point d'entrée WebSocket du flux de transcription.

    Protocole :
      1. Le client se connecte à `wss://host/ws/transcribe?token=<JWT>`.
      2. Le jeton est validé AVANT l'acceptation de la connexion
         (cf. server/security.py). En cas d'échec, la connexion est fermée
         avec le code 1008 (policy violation) sans jamais être "acceptée".
      3. Une fois acceptée, le client envoie des frames BINAIRES contenant
         du PCM 16 bits linéaire, mono, 16 kHz, par paquets de 50-100 ms.
      4. Le serveur répond en continu par des frames TEXTE JSON :
             {"status": "partial", "text": "..."}
             {"status": "final",   "text": "...", "speaker_id": 1}
    """
    try:
        user_claims = await authenticate_websocket(websocket)
    except AuthenticationError:
        return  # la connexion a déjà été fermée par authenticate_websocket

    await websocket.accept()

    session = TranscriptionSession(websocket, user_claims)
    logger.info(
        "[%s] Session démarrée pour sub=%s", session.session_id, user_claims.get("sub")
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] Erreur non gérée durant la session.", session.session_id)
    finally:
        logger.info("[%s] Session terminée.", session.session_id)
