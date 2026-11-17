"""
Entry point of the streaming speech transcription infrastructure.

Startup (development):
    uvicorn server.main:app --host 0.0.0.0 --port 8000

Startup (production, HTTPS/WSS via an upstream TLS-terminating reverse proxy
such as nginx/traefik, or directly with --ssl-keyfile/--ssl-certfile):
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1 \
        --ssl-keyfile key.pem --ssl-certfile cert.pem

IMPORTANT NOTE: --workers must stay at 1 per available GPU. The Whisper
model is loaded into memory per process; multiple workers would duplicate
the model in VRAM/RAM. To scale horizontally, deploy several instances
(one per GPU) behind a WebSocket-aware load balancer.
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
    # Load the Whisper model (+ warm-up) only once at process startup,
    # before accepting any WebSocket connection, so that the very first
    # user session already benefits from nominal latency (< 300 ms) with
    # no "cold start" spike.
    logger.info("Initializing ASR engine...")
    get_asr_engine()
    logger.info("Server ready to accept connections.")
    yield
    logger.info("Shutting down server.")


app = FastAPI(
    title="Streaming ASR — Sovereign on-premise infrastructure",
    lifespan=lifespan,
)

# CORS: restrict explicitly to the origins of your client applications in
# production (never keep "*" in a sensitive environment).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Monitoring endpoint (liveness/readiness probe)."""
    return {"status": "ok"}


@app.get("/")
async def serve_test_client() -> FileResponse:
    """Serves the HTML/JS test client (see client/index.html) directly from
    the server, purely for development convenience: avoids having to open
    the local file manually and puts the client on the same origin as the
    API (no CORS concerns). Remove in production, where the client would
    rather be served by a dedicated CDN/reverse proxy."""
    return FileResponse(CLIENT_HTML_PATH)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    """WebSocket entry point for the transcription stream.

    Protocol:
      1. The client connects to `wss://host/ws/transcribe?token=<JWT>`.
      2. The token is validated BEFORE the connection is accepted
         (see server/security.py). On failure, the connection is closed
         with code 1008 (policy violation) without ever being "accepted".
      3. Once accepted, the client sends BINARY frames containing 16-bit
         linear PCM, mono, 16 kHz, in packets of 50-100 ms.
      4. The server continuously replies with JSON TEXT frames:
             {"status": "partial", "text": "..."}
             {"status": "final",   "text": "...", "speaker_id": 1}
    """
    try:
        user_claims = await authenticate_websocket(websocket)
    except AuthenticationError:
        return  # the connection has already been closed by authenticate_websocket

    await websocket.accept()

    session = TranscriptionSession(websocket, user_claims)
    logger.info(
        "[%s] Session started for sub=%s", session.session_id, user_claims.get("sub")
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] Unhandled error during session.", session.session_id)
    finally:
        logger.info("[%s] Session ended.", session.session_id)
