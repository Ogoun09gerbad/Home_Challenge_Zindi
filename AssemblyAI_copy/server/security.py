"""
Interception et validation du jeton d'authentification (JWT) lors de la
poignée de main WebSocket.

Le protocole WebSocket ne permet pas de renvoyer un header 401 arbitraire
comme le ferait une route HTTP classique : la seule option correcte côté
client/serveur est soit de rejeter la connexion en amont (avant `accept()`)
en fermant avec un code applicatif dédié, soit d'exiger le jeton dans le
premier message une fois la connexion établie. Ici, on choisit la première
approche (jeton transmis en query string `?token=...`) qui est la pratique
la plus répandue, tout en documentant l'alternative "premier message".
"""

from __future__ import annotations

import logging

import jwt
from fastapi import WebSocket, status

from server.config import JWT_ALGORITHM, JWT_SECRET_KEY

logger = logging.getLogger("asr_server.security")


class AuthenticationError(Exception):
    """Levée quand le jeton JWT est absent, invalide, expiré ou malformé."""


def _decode_token(token: str) -> dict:
    """Décode et vérifie la signature + l'expiration d'un JWT.

    Lève AuthenticationError si le jeton n'est pas valide. Ne fait AUCUNE
    hypothèse sur le contenu du payload au-delà des claims standard
    (exp, sub) : à adapter selon votre IdP (rôles, scopes, tenant_id...).
    """
    if not token:
        raise AuthenticationError("Jeton JWT manquant.")

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Jeton JWT expiré.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(f"Jeton JWT invalide : {exc}") from exc

    return payload


async def authenticate_websocket(websocket: WebSocket) -> dict:
    """Valide le jeton JWT porté par la requête de handshake WebSocket.

    À appeler AVANT `websocket.accept()`. En cas d'échec, ferme proprement
    la connexion avec le code de policy violation (1008) et lève
    AuthenticationError pour interrompre le traitement de la route.

    Retourne le payload décodé (claims) en cas de succès, afin que
    l'appelant puisse l'associer à la session (identité de l'utilisateur,
    tenant, permissions, etc.).
    """
    token = websocket.query_params.get("token")

    try:
        payload = _decode_token(token)
    except AuthenticationError as exc:
        logger.warning("Handshake WebSocket refusé : %s", exc)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
        raise

    logger.info("Handshake WebSocket authentifié pour sub=%s", payload.get("sub"))
    return payload
