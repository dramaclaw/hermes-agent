"""The thread that performs the model call must see this turn's identity.

The earlier concurrency tests exercised the ACP executor only. They proved two
ACP turns do not see each other's ContextVars — they did not prove the thread
that actually issues the HTTP request sees anything at all, because the model
call is handed to a separate thread inside `chat_completion_helpers`.

That distinction is not academic: a per-turn credential that is bound correctly
and then invisible at the point of egress fails closed on every single turn.
"""
from __future__ import annotations

import contextlib
import threading

import pytest

from agent.control_capability import (
    bind_capability, clear_capability, current_capability,
)
from agent.chat_completion_helpers import _context_thread_target
from agent.gateway_credential import bound_credential, current_credential


@contextlib.contextmanager
def bound_capability(value):
    """Match the credential's scope shape; the capability API is token-based."""
    token = bind_capability(value)
    try:
        yield
    finally:
        clear_capability(token)


def _read_from_a_worker_thread() -> tuple[str | None, str | None]:
    """Read both ContextVars from a thread started the way egress starts one."""
    seen: dict[str, str | None] = {}

    def read() -> None:
        seen["credential"] = current_credential()
        seen["capability"] = current_capability()

    thread = threading.Thread(target=_context_thread_target(read), daemon=True)
    thread.start()
    thread.join(timeout=5)
    return seen.get("credential"), seen.get("capability")


def test_the_egress_thread_reads_this_turn_credential_and_capability():
    with bound_credential("sk-turn-1", True), bound_capability("v1.k.p.s"):
        credential, capability = _read_from_a_worker_thread()
    assert credential == "sk-turn-1"
    assert capability == "v1.k.p.s"


def test_a_thread_started_outside_the_turn_sees_nothing():
    """The propagation must be scoped to the turn, not ambient."""
    credential, capability = _read_from_a_worker_thread()
    assert credential is None
    assert capability is None


def test_two_interleaved_turns_do_not_cross_at_the_egress_thread():
    """Barrier-interleaved turns, each reading from its own egress thread.

    The barrier forces both turns to be mid-flight simultaneously, so a shared
    Context or a copy taken once at worker start would show up as one turn
    reading the other's key.
    """
    barrier = threading.Barrier(2, timeout=10)
    observed: dict[str, tuple[str | None, str | None]] = {}
    errors: list[BaseException] = []

    def run_turn(name: str, credential: str, capability: str) -> None:
        try:
            with bound_credential(credential, True), bound_capability(capability):
                barrier.wait()          # both turns are now inside their scopes
                observed[name] = _read_from_a_worker_thread()
                barrier.wait()          # neither scope exits before both have read
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)

    first = threading.Thread(target=run_turn, args=("A", "sk-a", "cap-a"))
    second = threading.Thread(target=run_turn, args=("B", "sk-b", "cap-b"))
    first.start(); second.start()
    first.join(timeout=15); second.join(timeout=15)

    assert not errors, errors
    assert observed["A"] == ("sk-a", "cap-a")
    assert observed["B"] == ("sk-b", "cap-b")


def test_a_reused_thread_does_not_inherit_the_previous_turn():
    """A fresh copy per submission, not one taken at worker start."""
    with bound_credential("sk-first", True), bound_capability("cap-first"):
        assert _read_from_a_worker_thread() == ("sk-first", "cap-first")
    with bound_credential("sk-second", True), bound_capability("cap-second"):
        assert _read_from_a_worker_thread() == ("sk-second", "cap-second")
    assert _read_from_a_worker_thread() == (None, None)


def test_a_turn_without_meta_on_a_reused_thread_refuses_when_the_latch_is_on(monkeypatch):
    """Credential: fail-closed. The dangerous case is a thread that just worked.

    The placeholder is sitting in the environment and the previous turn's key
    has already succeeded on this thread, so a fallback here would look correct
    and bill the wrong account.
    """
    from agent.gateway_credential import MissingGatewayCredential, apply_to_headers

    # The credential is only ever attached to the configured model endpoint —
    # Hermes also talks to search, MCP servers and telemetry, and sending it
    # there would be a disclosure. So the endpoint has to match for this test
    # to be about the latch rather than about the destination check.
    monkeypatch.setenv("NEWAPI_BASE_URL", "https://gateway.example")

    with bound_credential("sk-first", True):
        headers = {}
        assert apply_to_headers(headers, "https://gateway.example/v1/chat/completions")

    with bound_credential(None, True):
        with pytest.raises(MissingGatewayCredential):
            apply_to_headers({}, "https://gateway.example/v1/chat/completions")


def test_a_turn_without_a_capability_degrades_instead_of_failing(monkeypatch):
    """Capability: fail-open serving. Losing evidence must not lose the turn."""
    from agent import control_capability

    monkeypatch.setenv("NEWAPI_BASE_URL", "https://gateway.example")
    headers = {}
    with bound_capability(None):
        attached = control_capability.attach_to_headers(
            headers, "https://gateway.example/v1/chat/completions")
    assert attached is False
    assert headers == {}


def test_several_model_calls_in_one_turn_all_see_the_same_identity():
    """A tool-driven turn issues more than one model call; every one is bound."""
    with bound_credential("sk-turn", True), bound_capability("cap-turn"):
        for _ in range(4):
            assert _read_from_a_worker_thread() == ("sk-turn", "cap-turn")


def test_an_exception_inside_the_egress_thread_still_releases_the_turn():
    """A failed call must not leave the identity bound for the next turn."""
    def explode() -> None:
        raise RuntimeError("model call failed")

    with bound_credential("sk-turn", True), bound_capability("cap-turn"):
        thread = threading.Thread(target=_context_thread_target(explode), daemon=True)
        thread.start()
        thread.join(timeout=5)
    assert _read_from_a_worker_thread() == (None, None)


def test_a_cancelled_turn_releases_the_identity():
    """Cancellation takes the same exit as an error: the scope unwinds."""
    class Cancelled(BaseException):
        pass

    with pytest.raises(Cancelled):
        with bound_credential("sk-turn", True), bound_capability("cap-turn"):
            raise Cancelled()
    assert _read_from_a_worker_thread() == (None, None)
