"""
Interception and validation of the authentication token (JWT) during the
WebSocket handshake.

The WebSocket protocol does not allow returning an arbitrary 401 header the
way a regular HTTP route would: the only correct options on the
client/server side are either to reject the connection upfront (before
`accept()`) by closing it with a dedicated application code, or to require
the token in the first message once the connection is established. Here we
choose the first approach (token passed in the query string `?token=...`),
which is the most widespread practice, while documenting the "first message"
alternative.
"""

from __future__ import annotations

import logging

import jwt
from fastapi import WebSocket, status

from server.config import JWT_ALGORITHM, JWT_SECRET_KEY

logger = logging.getLogger("asr_server.security")


class AuthenticationError(Exception):
    """Raised when the JWT is missing, invalid, expired or malformed."""


def _decode_token(token: str) -> dict:
    """Decodes a JWT and verifies its signature + expiration.

    Raises AuthenticationError if the token is not valid. Makes NO
    assumption about the payload content beyond the standard claims
    (exp, sub): adapt to your IdP as needed (roles, scopes, tenant_id...).
    """
    if not token:
        raise AuthenticationError("Missing JWT.")

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Expired JWT.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(f"Invalid JWT: {exc}") from exc

    return payload


async def authenticate_websocket(websocket: WebSocket) -> dict:
    """Validates the JWT carried by the WebSocket handshake request.

    Must be called BEFORE `websocket.accept()`. On failure, cleanly closes
    the connection with the policy violation code (1008) and raises
    AuthenticationError to abort processing of the route.

    Returns the decoded payload (claims) on success, so that the caller can
    associate it with the session (user identity, tenant, permissions,
    etc.).
    """
    token = websocket.query_params.get("token")

    try:
        payload = _decode_token(token)
    except AuthenticationError as exc:
        logger.warning("WebSocket handshake rejected: %s", exc)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
        raise

    logger.info("WebSocket handshake authenticated for sub=%s", payload.get("sub"))
    return payload
