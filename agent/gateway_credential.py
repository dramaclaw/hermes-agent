"""Per-turn gateway credential.

Deliberately a separate module from ``agent.control_capability``, because the
two carry opposite failure semantics and merging them would make it easy to
inherit the wrong one:

    control capability   evidence identity    fail-open: no header, request proceeds
    gateway credential   authentication       fail-closed: no key, request must not go out

A capability that goes missing costs a data point. A credential that goes
missing must not silently fall back to whatever the process environment holds,
because that would bill an organisation's traffic to the platform account and
grant it the platform's model permissions — a failure that looks like success
from every angle except the invoice.

Like the capability, the value lives only in a ContextVar for the current turn.
One worker is pooled per user and serves many turns concurrently on a shared
executor, so anything longer-lived — the agent, the client, the session, the
process environment — would attach one turn's credential to another turn's
request.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit

#: ACP ``_meta`` key. Namespaced, and distinct from the capability's.
CREDENTIAL_META_KEY = "dramaclaw.gateway_api_key"

#: Set to true in ``_meta`` when the turn must not fall back to the environment.
CREDENTIAL_REQUIRED_META_KEY = "dramaclaw.gateway_api_key_required"

MAX_CREDENTIAL_LENGTH = 4096

#: Process-level latch. A host that multiplexes tenants sets this on the workers
#: it starts; plain Hermes leaves it unset and keeps its existing behaviour.
CREDENTIAL_MODE_ENV = "DRAMACLAW_GATEWAY_CREDENTIAL_MODE"
PER_TURN_REQUIRED = "per_turn_required"

_credential: ContextVar[Optional[str]] = ContextVar(
    "hermes_gateway_credential", default=None)
_required: ContextVar[bool] = ContextVar(
    "hermes_gateway_credential_required", default=False)


class MissingGatewayCredential(RuntimeError):
    """Raised instead of sending a request the host did not authorise.

    Deliberately loud. The alternative is an outbound request carrying the
    platform's key, which succeeds and bills the wrong account.
    """


def parse_credential(meta: Any) -> tuple[Optional[str], bool]:
    """Extract ``(credential, required)`` from an ACP ``_meta`` mapping.

    A malformed credential yields ``(None, required)`` — never a partially
    accepted one — so an unusable value cannot become a usable-looking one.
    ``required`` survives that rejection on purpose: a host that declared the
    turn credentialed must not be downgraded to the environment key just
    because the value it sent was broken.
    """
    if not isinstance(meta, dict):
        return None, False
    required = bool(meta.get(CREDENTIAL_REQUIRED_META_KEY))
    value = meta.get(CREDENTIAL_META_KEY)
    if not isinstance(value, str):
        return None, required
    # Not stripped: silently repairing a malformed credential turns an illegal
    # input into a legal-looking one, and " sk-abc" is not the key the host
    # meant to send. Refuse it and let the latch below decide what that costs.
    if value != value.strip():
        return None, required
    if not value or len(value) > MAX_CREDENTIAL_LENGTH:
        return None, required
    if not value.isascii():
        return None, required
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        # This becomes an HTTP header value; CR and LF above all must not reach it.
        return None, required
    return value, required


@contextmanager
def bound_credential(credential: Optional[str], required: bool) -> Iterator[None]:
    """Bind for one turn and release unconditionally."""
    credential_token: Token = _credential.set(credential)
    required_token: Token = _required.set(required)
    try:
        yield
    finally:
        _credential.reset(credential_token)
        _required.reset(required_token)


def current_credential() -> Optional[str]:
    return _credential.get()


def credential_is_required() -> bool:
    return _required.get()


def _origin(url: str) -> Optional[tuple[str, str]]:
    parts = urlsplit(url or "")
    if not parts.scheme or not parts.netloc:
        return None
    return parts.scheme.lower(), parts.netloc.lower()


def per_turn_credential_required() -> bool:
    """Whether this process refuses to authenticate from its environment.

    The per-turn ``required`` flag only covers "the field was there and the value
    was broken". It cannot cover the case that actually matters: a caller,
    retry path or older client that omits ``_meta`` entirely. Without a latch
    that request falls through to whatever key the worker was started with,
    which is precisely the cross-tenant billing and permission failure the
    per-turn credential exists to remove.

    So the decision is made once, for the process, by whoever started it.
    """
    return os.environ.get(CREDENTIAL_MODE_ENV, "").strip().lower() == PER_TURN_REQUIRED


def targets_model_endpoint(request_url: str) -> bool:
    """True only for the configured model endpoint.

    Hermes talks to providers, search, MCP servers and telemetry. A credential
    sent to any of those is a disclosure, so the default is not to send it.
    """
    configured = _origin(os.environ.get("NEWAPI_BASE_URL", "").strip())
    return configured is not None and _origin(request_url) == configured


def apply_to_headers(headers: Any, request_url: str) -> bool:
    """Authorise one outbound request, or refuse to let it leave.

    Returns whether the header was replaced. Raises
    :class:`MissingGatewayCredential` when the turn declared a credential is
    required and none is usable — the request is then never sent, which is the
    entire point of the distinction from the capability path.
    """
    if not targets_model_endpoint(request_url):
        return False
    credential = current_credential()
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
        return True
    if credential_is_required() or per_turn_credential_required():
        # Whatever the client already carries is a placeholder that exists to
        # let the SDK initialise. It must never authenticate a request.
        raise MissingGatewayCredential(
            "this worker authenticates per turn and no usable credential is bound; "
            "refusing to fall back to the process credential")
    return False


def httpx_request_hook():
    """Build an httpx ``request`` event hook.

    A hook rather than a client-level header: one client serves many turns, so a
    default header would pin the first turn's credential to every later request.
    Unlike the capability hook this one does not swallow exceptions — a refusal
    has to reach the caller.
    """

    def _hook(request: Any) -> None:
        apply_to_headers(request.headers, str(request.url))

    return _hook


def async_httpx_request_hook():
    async def _hook(request: Any) -> None:
        apply_to_headers(request.headers, str(request.url))

    return _hook
