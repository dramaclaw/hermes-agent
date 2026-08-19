"""A per-turn worker may reach its gateway and nothing else.

An unauthenticated call to a vendor is not harmless: the request body has
already left the machine. The canary caught the auxiliary title generator
falling through to openrouter and nous, which returned payment errors — the
absence of a bill is not the absence of a disclosure.
"""
from __future__ import annotations

import pytest

from agent.gateway_credential import (
    ForeignModelEndpoint, GatewayConfigurationError, async_httpx_request_hook,
    httpx_request_hook, refuse_foreign_endpoint,
)

GATEWAY = "https://gateway.example"


class _Request:
    def __init__(self, url: str) -> None:
        self.url = url
        self.headers: dict[str, str] = {}


@pytest.fixture
def latched(monkeypatch):
    monkeypatch.setenv("DRAMACLAW_GATEWAY_CREDENTIAL_MODE", "per_turn_required")
    monkeypatch.setenv("NEWAPI_BASE_URL", GATEWAY)


@pytest.mark.parametrize("url", [
    "https://openrouter.ai/api/v1/chat/completions",
    "https://inference-api.nousresearch.com/v1/chat/completions",
    "https://api.openai.com/v1/chat/completions",
    "https://api.anthropic.com/v1/messages",
])
def test_a_vendor_endpoint_is_refused(latched, url):
    with pytest.raises(ForeignModelEndpoint):
        refuse_foreign_endpoint(url)


def test_the_configured_gateway_is_allowed(latched):
    refuse_foreign_endpoint(f"{GATEWAY}/v1/chat/completions")


@pytest.mark.parametrize("host", ["127.0.0.1:19052", "localhost:8080", "[::1]:9000"])
def test_a_loopback_host_that_is_not_the_gateway_is_refused(latched, host):
    """Loopback is not a licence. Another local listener is still not ours.

    A canary gateway is loopback, but so is a proxy, a second service, or a
    developer's tunnel. Same-origin already admits the canary, so a blanket
    loopback exception gave away the rule and bought nothing.
    """
    with pytest.raises(ForeignModelEndpoint):
        refuse_foreign_endpoint(f"http://{host}/v1/chat/completions")


def test_a_loopback_gateway_is_allowed_when_it_is_the_configured_one(monkeypatch):
    """Which is how the canary itself passes."""
    monkeypatch.setenv("DRAMACLAW_GATEWAY_CREDENTIAL_MODE", "per_turn_required")
    monkeypatch.setenv("NEWAPI_BASE_URL", "http://127.0.0.1:19052")
    refuse_foreign_endpoint("http://127.0.0.1:19052/v1/chat/completions")


@pytest.mark.parametrize("value", ["", "   ", "not a url", "://broken"])
def test_a_missing_or_unusable_gateway_refuses_everything(monkeypatch, value):
    """An unset gateway is not permission to reach anywhere.

    This was inverted: a worker that did not know where its gateway was passed
    every destination, so the rule vanished exactly when it was needed.
    """
    monkeypatch.setenv("DRAMACLAW_GATEWAY_CREDENTIAL_MODE", "per_turn_required")
    monkeypatch.setenv("NEWAPI_BASE_URL", value)
    with pytest.raises(GatewayConfigurationError):
        refuse_foreign_endpoint("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(GatewayConfigurationError):
        refuse_foreign_endpoint("http://127.0.0.1:19052/v1/chat/completions")


def test_without_the_latch_nothing_is_refused(monkeypatch):
    """A deployment that has not moved to per-turn credentials is unaffected."""
    monkeypatch.delenv("DRAMACLAW_GATEWAY_CREDENTIAL_MODE", raising=False)
    monkeypatch.setenv("NEWAPI_BASE_URL", GATEWAY)
    refuse_foreign_endpoint("https://openrouter.ai/api/v1/chat/completions")


def test_the_refusal_names_no_secret(latched):
    with pytest.raises(ForeignModelEndpoint) as caught:
        refuse_foreign_endpoint("https://openrouter.ai/api/v1/chat/completions")
    message = str(caught.value)
    assert "openrouter.ai" in message
    assert "sk-" not in message and "Bearer" not in message


def test_the_request_hook_refuses_before_the_body_is_written(latched):
    hook = httpx_request_hook()
    request = _Request("https://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(ForeignModelEndpoint):
        hook(request)
    assert "Authorization" not in request.headers


def test_the_async_hook_refuses_too(latched):
    """Driven with asyncio.run so the suite needs no async plugin."""
    import asyncio

    hook = async_httpx_request_hook()
    with pytest.raises(ForeignModelEndpoint):
        asyncio.run(hook(_Request("https://openrouter.ai/api/v1/chat/completions")))


def test_the_auxiliary_chain_keeps_only_the_gateway_under_the_latch(latched):
    from agent import auxiliary_client

    labels = [label for label, _ in auxiliary_client._get_provider_chain()]
    assert labels == ["local/custom"], (
        "an auxiliary task must reach the gateway or fail, never a vendor")


def test_the_auxiliary_chain_is_unchanged_without_the_latch(monkeypatch):
    monkeypatch.delenv("DRAMACLAW_GATEWAY_CREDENTIAL_MODE", raising=False)
    from agent import auxiliary_client

    labels = [label for label, _ in auxiliary_client._get_provider_chain()]
    assert "openrouter" in labels and "nous" in labels
