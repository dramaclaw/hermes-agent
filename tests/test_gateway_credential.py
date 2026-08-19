"""Per-turn gateway credential: isolation, fail-closed refusal, blast radius."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from agent import control_capability as capability_module
from agent import gateway_credential as credential_module

URL = "https://models.example/v1/chat/completions"


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWAPI_BASE_URL", "https://models.example/v1")
    yield


def _send(url: str = URL) -> httpx.Headers:
    seen: dict[str, httpx.Headers] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        return httpx.Response(200, json={})

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [credential_module.httpx_request_hook()]},
        headers={"Authorization": "Bearer PLATFORM-ENV-KEY"},
    ) as client:
        client.post(url, json={"model": "m"})
    return seen["headers"]


def _turn(credential: str | None, required: bool, url: str = URL) -> str | None:
    with credential_module.bound_credential(credential, required):
        return _send(url).get("Authorization")


# The property everything else rests on: one pooled worker, many turns.
def test_concurrent_turns_never_use_each_others_key() -> None:
    def run(index: int) -> str | None:
        key = f"sk-org-{index % 3}"
        return contextvars.copy_context().run(_turn, key, True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(48)))
    assert results == [f"Bearer sk-org-{i % 3}" for i in range(48)]


def test_a_platform_turn_and_org_turns_share_one_worker() -> None:
    """P, A and B interleaved on one executor, each authenticated as itself."""
    plan = [(None, False), ("sk-org-A", True), ("sk-org-B", True)] * 12

    def run(item):
        credential, required = item
        return contextvars.copy_context().run(_turn, credential, required)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run, plan))

    for (credential, _), authorization in zip(plan, results):
        if credential is None:
            # The platform turn keeps the client's own environment credential.
            assert authorization == "Bearer PLATFORM-ENV-KEY"
        else:
            assert authorization == f"Bearer {credential}"


def test_a_later_turn_never_inherits_a_key() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(contextvars.copy_context().run, _turn, "sk-org-A", True).result()
        second = executor.submit(contextvars.copy_context().run, _turn, None, False).result()
    assert first == "Bearer sk-org-A"
    assert second == "Bearer PLATFORM-ENV-KEY"
    assert credential_module.current_credential() is None


def test_an_exception_still_releases_the_credential() -> None:
    with pytest.raises(RuntimeError):
        with credential_module.bound_credential("sk-org-A", True):
            raise RuntimeError("turn failed")
    assert credential_module.current_credential() is None
    assert credential_module.credential_is_required() is False


# Fail-closed: the whole reason this is a separate module from the capability.
def test_a_required_turn_with_no_usable_key_refuses_to_send() -> None:
    """Falling back would bill an organisation's traffic to the platform."""
    with pytest.raises(credential_module.MissingGatewayCredential):
        _turn(None, True)


@pytest.mark.parametrize(
    "meta",
    [
        {credential_module.CREDENTIAL_META_KEY: 42,
         credential_module.CREDENTIAL_REQUIRED_META_KEY: True},
        {credential_module.CREDENTIAL_META_KEY: "",
         credential_module.CREDENTIAL_REQUIRED_META_KEY: True},
        {credential_module.CREDENTIAL_META_KEY: "sk-\r\nX-Injected: 1",
         credential_module.CREDENTIAL_REQUIRED_META_KEY: True},
        {credential_module.CREDENTIAL_META_KEY: "sk-é",
         credential_module.CREDENTIAL_REQUIRED_META_KEY: True},
        {credential_module.CREDENTIAL_META_KEY: "x" * 5000,
         credential_module.CREDENTIAL_REQUIRED_META_KEY: True},
    ],
)
def test_a_broken_credential_is_refused_not_downgraded(meta: dict) -> None:
    """A host that declared the turn credentialed must not silently fall back."""
    credential, required = credential_module.parse_credential(meta)
    assert credential is None
    assert required is True, "required must survive a rejected value"
    with pytest.raises(credential_module.MissingGatewayCredential):
        _turn(credential, required)


def test_an_optional_turn_may_use_the_environment() -> None:
    credential, required = credential_module.parse_credential({})
    assert (credential, required) == (None, False)
    assert _turn(credential, required) == "Bearer PLATFORM-ENV-KEY"


@pytest.mark.parametrize(
    "url",
    [
        "https://provider.example/v1/chat/completions",
        "http://models.example/v1/chat/completions",
        "https://models.example:8443/v1/chat/completions",
    ],
)
def test_the_key_goes_nowhere_but_the_configured_endpoint(url: str) -> None:
    """Search, MCP and telemetry hosts must never see it."""
    assert _turn("sk-org-A", True, url) == "Bearer PLATFORM-ENV-KEY"


# The two ContextVars must not interfere: opposite failure semantics, one turn.
def test_the_capability_and_the_credential_are_independent() -> None:
    capability_token = capability_module.bind_capability("cap-value")
    try:
        with credential_module.bound_credential("sk-org-A", True):
            assert capability_module.current_capability() == "cap-value"
            assert credential_module.current_credential() == "sk-org-A"
        # Releasing one must not release the other.
        assert capability_module.current_capability() == "cap-value"
        assert credential_module.current_credential() is None
    finally:
        capability_module.clear_capability(capability_token)

    # And a missing capability stays fail-open while a missing credential does not.
    with credential_module.bound_credential("sk-org-B", True):
        headers = _send()
    assert headers.get("Authorization") == "Bearer sk-org-B"
    with pytest.raises(credential_module.MissingGatewayCredential):
        _turn(None, True)
