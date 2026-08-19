"""Per-turn egress capability: isolation, validation and blast radius.

Scope note: the concurrency tests here cover the ACP executor layer only. They
show that two ACP turns do not see each other's ContextVars; they do NOT show
that the thread issuing the model call sees anything, because that call is
handed to a separate thread inside `chat_completion_helpers`. Whole-chain
propagation is covered by tests/test_egress_thread_sees_turn_identity.py.
"""

from __future__ import annotations

import contextvars
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from agent import control_capability as cc


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWAPI_BASE_URL", "https://models.example/v1")
    yield


def _send(url: str) -> httpx.Headers:
    """Run one request through the hook and return what would go on the wire."""
    seen: dict[str, httpx.Headers] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        return httpx.Response(200, json={})

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [cc.httpx_request_hook()]},
    ) as client:
        client.post(url, json={"model": "m"})
    return seen["headers"]


def _turn(capability: str | None, url: str) -> str | None:
    """Simulate one ACP turn: bind, send, unconditionally release."""
    token = cc.bind_capability(capability)
    try:
        return _send(url).get(cc.CAPABILITY_HEADER)
    finally:
        cc.clear_capability(token)


URL = "https://models.example/v1/chat/completions"


# The load-bearing property: the ACP adapter runs concurrent turns on one
# ThreadPoolExecutor, each inside contextvars.copy_context(). Two interleaved
# turns must never see each other's capability.
def test_concurrent_turns_do_not_cross_contaminate() -> None:
    def run_turn(capability: str) -> str | None:
        # Exactly what acp_adapter.server does per turn.
        return contextvars.copy_context().run(_turn, capability, URL)

    with ThreadPoolExecutor(max_workers=4) as executor:
        capabilities = [f"cap-{index}" for index in range(40)]
        results = list(executor.map(run_turn, capabilities))

    assert results == capabilities


def test_a_later_turn_never_inherits_a_previous_capability() -> None:
    """Executor threads are reused; the finally-reset is what makes that safe."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(contextvars.copy_context().run, _turn, "cap-A", URL).result()
        # A turn with no capability, scheduled onto the very same thread.
        second = executor.submit(contextvars.copy_context().run, _turn, None, URL).result()
    assert first == "cap-A"
    assert second is None
    # And nothing leaked into this context either.
    assert cc.current_capability() is None


def test_an_exception_still_releases_the_capability() -> None:
    token = cc.bind_capability("cap-boom")
    try:
        raise RuntimeError("turn failed")
    except RuntimeError:
        pass
    finally:
        cc.clear_capability(token)
    assert cc.current_capability() is None


def test_every_model_call_in_one_turn_carries_it() -> None:
    """An agent loop makes many calls per turn; all of them must be covered."""
    token = cc.bind_capability("cap-multi")
    try:
        assert [_send(URL).get(cc.CAPABILITY_HEADER) for _ in range(5)] == ["cap-multi"] * 5
    finally:
        cc.clear_capability(token)


# Blast radius: Hermes talks to providers, search, MCP servers and telemetry.
# A capability reaching any of them is a credential disclosure.
@pytest.mark.parametrize(
    "url",
    [
        "https://provider.example/v1/chat/completions",   # different host
        "http://models.example/v1/chat/completions",      # different scheme
        "https://models.example:8443/v1/chat/completions",  # different port
        "https://evil.example/?next=https://models.example/",
    ],
)
def test_the_capability_goes_nowhere_but_the_configured_endpoint(url: str) -> None:
    assert _turn("cap-secret", url) is None


def test_without_a_configured_endpoint_nothing_is_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEWAPI_BASE_URL")
    assert _turn("cap-secret", URL) is None


# _meta validation. Every rejection is silent and yields None: an unusable
# capability degrades the turn to unattested, it never fails it.
@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        "not-a-dict",
        {cc.CAPABILITY_META_KEY: 42},
        {cc.CAPABILITY_META_KEY: ["a", "b"]},          # multi-value claim
        {cc.CAPABILITY_META_KEY: ""},
        {cc.CAPABILITY_META_KEY: "   "},
        {cc.CAPABILITY_META_KEY: "x" * (cc.MAX_CAPABILITY_LENGTH + 1)},
        {cc.CAPABILITY_META_KEY: "capability-é"},       # non-ASCII
        {cc.CAPABILITY_META_KEY: "cap\r\nX-Injected: 1"},  # header injection
        {cc.CAPABILITY_META_KEY: "cap\x00null"},
        {"some.other.extension": "cap-valid"},          # wrong namespace
    ],
)
def test_unusable_meta_yields_no_capability(meta: object) -> None:
    assert cc.parse_capability(meta) is None


def test_a_well_formed_capability_is_accepted_verbatim() -> None:
    assert cc.parse_capability({cc.CAPABILITY_META_KEY: "  cap.v1.abc  "}) == "cap.v1.abc"
    longest = "x" * cc.MAX_CAPABILITY_LENGTH
    assert cc.parse_capability({cc.CAPABILITY_META_KEY: longest}) == longest


def test_meta_cannot_choose_the_header_name() -> None:
    """Only the value is host-supplied; the name is a constant."""
    token = cc.bind_capability("cap-x")
    try:
        headers = _send(URL)
    finally:
        cc.clear_capability(token)
    assert headers.get(cc.CAPABILITY_HEADER) == "cap-x"
    assert cc.CAPABILITY_HEADER == "X-DramaClaw-Control-Capability"


def test_the_value_never_appears_in_an_error_message() -> None:
    """It may be a credential, so no rejection path may echo it."""
    secret = "cap-é-secret"  # non-ASCII, so it is rejected
    try:
        assert cc.parse_capability({cc.CAPABILITY_META_KEY: secret}) is None
    except Exception as error:  # pragma: no cover - defensive
        assert secret not in str(error)
