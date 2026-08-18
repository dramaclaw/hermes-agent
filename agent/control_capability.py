"""Per-turn egress capability propagation.

A host that embeds Hermes over ACP may need each outbound model request to
carry a short-lived, host-issued capability identifying the logical unit of
work the turn belongs to. Hermes cannot derive that identity: it is a property
of the host's product, not of the conversation.

The capability is carried in a ContextVar rather than on the agent, the client,
the session or the process environment. Those are all shared across turns, and
the ACP adapter runs concurrent turns on one ThreadPoolExecutor, so any shared
location would cross-contaminate: turn A's request would carry turn B's
capability. ``acp_adapter.server`` already wraps each turn in
``contextvars.copy_context()`` for exactly this reason, which is what makes a
ContextVar correct here and everything else wrong.

Hermes treats the value as opaque. It does not parse it, does not persist it,
does not log it, and never lets it influence agent behaviour. It is attached to
outbound requests aimed at the configured model endpoint and to nothing else.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import Any, Optional
from urllib.parse import urlsplit

#: ACP ``_meta`` key. Namespaced so it cannot collide with other extensions.
CAPABILITY_META_KEY = "dramaclaw.control_context_capability"

#: The only header this module will ever set. The name is a constant: ``_meta``
#: supplies a value, never a header name, so a malicious host cannot use this
#: path to inject arbitrary headers into provider traffic.
CAPABILITY_HEADER = "X-DramaClaw-Control-Capability"

#: Values above this are refused outright rather than truncated. A truncated
#: capability would be a subtly invalid one, which is harder to diagnose than
#: an absent one.
MAX_CAPABILITY_LENGTH = 4096

_capability: ContextVar[Optional[str]] = ContextVar(
    "hermes_egress_control_capability", default=None
)


def parse_capability(meta: Any) -> Optional[str]:
    """Extract a capability from an ACP ``_meta`` mapping, or return None.

    Every rejection is silent and returns None. The value may be a credential,
    so neither it nor a fragment of it is ever placed in an exception message
    or a log line. An unusable capability degrades the turn to "no capability",
    which the host is expected to treat as unattested — it never fails the turn.
    """
    if not isinstance(meta, dict):
        return None
    value = meta.get(CAPABILITY_META_KEY)
    # Single string only. A list would be an ambiguous multi-value claim, and
    # anything else is not a capability.
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_CAPABILITY_LENGTH:
        return None
    # ASCII and printable: this becomes an HTTP header value, so control
    # characters — CR and LF above all — must never reach it.
    if not value.isascii() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        return None
    return value


def bind_capability(value: Optional[str]) -> Token:
    """Bind the capability for the current context. Always pair with a reset."""
    return _capability.set(value)


def clear_capability(token: Token) -> None:
    _capability.reset(token)


def current_capability() -> Optional[str]:
    return _capability.get()


def _origin(url: str) -> Optional[tuple[str, str]]:
    parts = urlsplit(url or "")
    if not parts.scheme or not parts.netloc:
        return None
    return parts.scheme.lower(), parts.netloc.lower()


def _configured_model_origin() -> Optional[tuple[str, str]]:
    return _origin(os.environ.get("NEWAPI_BASE_URL", "").strip())


def should_attach(request_url: str) -> bool:
    """True only for requests aimed at the configured model endpoint.

    Hermes talks to many hosts — providers, search, MCP servers, telemetry. A
    capability leaked to any of them is a credential disclosure, so the default
    is not to attach. Without a configured endpoint nothing is ever attached.
    """
    configured = _configured_model_origin()
    if configured is None:
        return False
    return _origin(request_url) == configured


def attach_to_headers(headers: Any, request_url: str) -> bool:
    """Set the capability header on a request, if there is one to set.

    Returns whether it was attached, for tests. Callers must not log the value.
    """
    capability = current_capability()
    if not capability or not should_attach(request_url):
        return False
    headers[CAPABILITY_HEADER] = capability
    return True


def httpx_request_hook():
    """Build an httpx ``request`` event hook that attaches the capability.

    A hook rather than a client-level default header: one client is reused
    across turns, so a default header would pin the first turn's capability to
    every later request. The hook runs inside the caller's context, so it sees
    that turn's value and no other's.
    """

    def _hook(request: Any) -> None:
        try:
            attach_to_headers(request.headers, str(request.url))
        except Exception:
            # Never fail a model call over egress metadata.
            return

    return _hook


def async_httpx_request_hook():
    """Async variant of :func:`httpx_request_hook`."""

    async def _hook(request: Any) -> None:
        try:
            attach_to_headers(request.headers, str(request.url))
        except Exception:
            return

    return _hook
