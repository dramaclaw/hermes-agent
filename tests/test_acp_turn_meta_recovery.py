"""The ACP router splats _meta into kwargs; the handler must still see it.

This is the gap that made the per-turn credential fail in every real turn while
28 unit tests passed: those tests bound the ContextVar directly, so none of them
crossed the router boundary where the mapping is taken apart.

The first test asserts the router's actual behaviour rather than our belief
about it — if a future acp release passes `_meta` as a mapping again, that test
tells us, instead of the recovery silently becoming dead code.
"""
from __future__ import annotations

import pytest

from acp_adapter.server import _recover_turn_meta

CREDENTIAL = "dramaclaw.gateway_api_key"
REQUIRED = "dramaclaw.gateway_api_key_required"
CAPABILITY = "dramaclaw.control_context_capability"


def test_the_router_splats_meta_keys_rather_than_passing_a_mapping():
    """Pin the upstream behaviour this recovery exists for."""
    import inspect
    from acp import router

    source = inspect.getsource(router)
    assert "params.update(meta)" in source, (
        "the acp router no longer splats _meta into kwargs — re-check whether "
        "_recover_turn_meta is still needed, and whether it is still correct")


def test_meta_is_recovered_from_splatted_keys():
    recovered = _recover_turn_meta({
        "session_id": "s-1",
        CREDENTIAL: "sk-example",
        REQUIRED: True,
        CAPABILITY: "v1.k.p.s",
    })
    assert recovered == {CREDENTIAL: "sk-example", REQUIRED: True,
                         CAPABILITY: "v1.k.p.s"}


def test_a_mapping_is_used_directly_when_the_router_provides_one():
    """Version tolerance: a router that passes _meta must keep working."""
    meta = {CREDENTIAL: "sk-example"}
    assert _recover_turn_meta({"_meta": meta, "session_id": "s-1"}) is meta


def test_ordinary_keyword_arguments_are_never_mistaken_for_meta():
    assert _recover_turn_meta({"session_id": "s-1", "cwd": "/tmp"}) == {}


def test_a_turn_without_meta_recovers_nothing_rather_than_guessing():
    assert _recover_turn_meta({}) == {}


@pytest.mark.parametrize("value", [None, "", 0, []])
def test_a_falsy_direct_meta_falls_through_to_the_splatted_keys(value):
    """`kwargs.get("_meta") or ...` must not swallow a real splatted turn."""
    recovered = _recover_turn_meta({"_meta": value, CREDENTIAL: "sk-example"})
    assert recovered == {CREDENTIAL: "sk-example"}
