"""Unit tests for the HttpCacheMiddleware."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.http_cache import (
    HttpCacheMiddleware,
    _build_key,
    _resource_segment,
    _should_bypass,
)


class _FakeRedis:
    """Minimal in-process Redis stand-in."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    def delete(self, *keys: str) -> int:
        removed = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                removed += 1
        return removed

    def scan_iter(self, match: str | None = None, count: int = 100):
        pattern = (match or "").replace("*", "")
        for k in list(self.store.keys()):
            if not match or pattern in k or k.startswith(pattern):
                yield k


@pytest.fixture
def fake_cache():
    fake = _FakeRedis()
    # Force non-testing env so middleware doesn't short-circuit.
    settings_mock = MagicMock()
    settings_mock.env = "development"
    # Dirty-scope markers live in the real Redis; a marker left by another test
    # (an integration write, say) would turn every read here into a BYPASS.
    with (
        patch("core.http_cache.cache_db", fake),
        patch("core.http_cache.get_settings", return_value=settings_mock),
        patch("core.http_cache.is_scope_dirty", return_value=False),
    ):
        yield fake


def _build_app() -> FastAPI:
    app = FastAPI()
    app.state.counter = 0
    app.state.checkin_counter = 0
    app.state.checkout_counter = 0
    app.add_middleware(HttpCacheMiddleware)

    @app.get("/v1/visitors")
    async def list_visitors() -> dict:
        app.state.counter += 1
        return {"count": app.state.counter}

    @app.post("/v1/visitors")
    async def create_visitor() -> dict:
        return {"created": True}

    @app.get("/v1/plans")
    async def list_plans() -> dict:
        return {"plans": []}

    @app.get("/v1/tenants/{tenant_id}/checkins")
    async def list_tenant_checkins(tenant_id: str, state: str = "pending") -> dict:
        app.state.checkin_counter += 1
        return {
            "tenant_id": tenant_id,
            "state": state,
            "count": app.state.checkin_counter,
        }

    @app.post("/v1/checkins/{checkin_id}/confirm")
    async def confirm_checkin(checkin_id: str) -> dict:
        return {"id": checkin_id, "state": "approved"}

    @app.get("/v1/checkout/sessions/by-reference/{reference}")
    async def checkout_by_reference(reference: str) -> dict:
        app.state.checkout_counter += 1
        return {
            "reference": reference,
            "status": "pending",
            "count": app.state.checkout_counter,
        }

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True}

    return app


@pytest.fixture
def app() -> FastAPI:
    return _build_app()


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


def test_resource_segment_v1_path() -> None:
    assert _resource_segment("/v1/visitors/abc/checkout") == "v1-visitors"
    assert _resource_segment("/v1/plans") == "v1-plans"
    assert _resource_segment("/v1/tenants/t1/checkins") == "v1-checkins"
    assert _resource_segment("/v1/tenants/t1/checkins/analytics") == "v1-checkins"
    assert _resource_segment("/health") == "health"
    assert _resource_segment("/") == "_root"


def test_should_bypass_health_and_auth_paths() -> None:
    assert _should_bypass("/health/ready")
    assert _should_bypass("/docs")
    assert _should_bypass("/v1/admins/login")
    assert _should_bypass("/v1/system-users/2fa/setup")
    assert _should_bypass("/v1/admins/verify-otp")
    assert not _should_bypass("/v1/visitors")


def test_build_key_varies_with_auth_and_query() -> None:
    class _StubHeaders:
        def __init__(self, headers: dict[str, str]) -> None:
            self._h = {k.lower(): v for k, v in headers.items()}

        def get(self, key: str, default: str = "") -> str:
            return self._h.get(key.lower(), default)

    class _StubURL:
        def __init__(self, path: str, query: str) -> None:
            self.path = path
            self.query = query

    class _StubRequest:
        def __init__(self, path: str, query: str, auth: str, case: str) -> None:
            self.url = _StubURL(path, query)
            self.headers = _StubHeaders(
                {"Authorization": auth, "X-Response-Case": case}
            )
            self.cookies: dict[str, str] = {}

    r1 = _StubRequest("/v1/visitors", "", "Bearer aaa", "camel")
    r2 = _StubRequest("/v1/visitors", "", "Bearer bbb", "camel")
    r3 = _StubRequest("/v1/visitors", "limit=10", "Bearer aaa", "camel")
    r4 = _StubRequest("/v1/visitors", "", "Bearer aaa", "snake")

    k1 = _build_key("t:tenantA", "v1-visitors", cast(Any, r1))
    k2 = _build_key("t:tenantA", "v1-visitors", cast(Any, r2))
    k3 = _build_key("t:tenantA", "v1-visitors", cast(Any, r3))
    k4 = _build_key("t:tenantA", "v1-visitors", cast(Any, r4))

    assert k1 != k2, "different auth tokens must produce different keys"
    assert k1 != k3, "different query strings must produce different keys"
    assert k1 != k4, "different response case must produce different keys"
    assert k1.startswith("httpcache:t:tenantA:v1-visitors:")


# ---------------------------------------------------------------------------
# Middleware behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bypass_path_is_not_cached(app, fake_cache) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r1 = await client.get("/health")
    assert r1.status_code == 200
    assert fake_cache.store == {}


@pytest.mark.asyncio
async def test_get_is_cached_and_second_call_is_served_from_cache(
    app, fake_cache
) -> None:
    with patch(
        "core.http_cache.get_access_token_allow_expired", new_callable=AsyncMock
    ) as m:
        m.return_value = None  # anonymous caller
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/v1/visitors")
            r2 = await client.get("/v1/visitors")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.headers.get("x-cache") == "MISS"
    assert r2.headers.get("x-cache") == "HIT"
    assert json.loads(r1.content) == json.loads(r2.content), "cached body must match"
    assert app.state.counter == 1, "handler should run once"
    assert any(k.startswith("httpcache:anon:v1-visitors:") for k in fake_cache.store)


@pytest.mark.asyncio
async def test_post_invalidates_related_cache(app, fake_cache) -> None:
    with patch(
        "core.http_cache.get_access_token_allow_expired", new_callable=AsyncMock
    ) as m:
        m.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/v1/visitors")
            assert any(
                k.startswith("httpcache:anon:v1-visitors:") for k in fake_cache.store
            )
            await client.post("/v1/visitors", json={})
            assert not any(
                k.startswith("httpcache:anon:v1-visitors:") for k in fake_cache.store
            )
            # Subsequent GET now misses and re-populates
            r = await client.get("/v1/visitors")
    assert r.headers.get("x-cache") == "MISS"
    assert app.state.counter == 2


@pytest.mark.asyncio
async def test_post_does_not_invalidate_unrelated_resource(app, fake_cache) -> None:
    with patch(
        "core.http_cache.get_access_token_allow_expired", new_callable=AsyncMock
    ) as m:
        m.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/v1/plans")
            plan_keys_before = [k for k in fake_cache.store if "v1-plans" in k]
            await client.post("/v1/visitors", json={})
            plan_keys_after = [k for k in fake_cache.store if "v1-plans" in k]

    assert plan_keys_before, "expected /v1/plans to be cached"
    assert plan_keys_before == plan_keys_after, (
        "writes on /v1/visitors must not wipe /v1/plans cache"
    )


@pytest.mark.asyncio
async def test_confirm_checkin_invalidates_tenant_scoped_checkin_list(
    app, fake_cache
) -> None:
    with patch(
        "core.http_cache.get_access_token_allow_expired", new_callable=AsyncMock
    ) as m:
        m.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/v1/tenants/t1/checkins?state=pending_approval")
            assert any(
                k.startswith("httpcache:anon:v1-checkins:") for k in fake_cache.store
            )
            await client.post("/v1/checkins/c1/confirm", json={"action": "approve"})
            assert not any(
                k.startswith("httpcache:anon:v1-checkins:") for k in fake_cache.store
            )


@pytest.mark.asyncio
async def test_checkout_paths_bypass_cache(app, fake_cache) -> None:
    """Payment status reads must never be served stale — /v1/checkout/*
    bypasses the response cache entirely."""
    with patch(
        "core.http_cache.get_access_token_allow_expired", new_callable=AsyncMock
    ) as m:
        m.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/v1/checkout/sessions/by-reference/chk_1")
            r2 = await client.get("/v1/checkout/sessions/by-reference/chk_1")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert app.state.checkout_counter == 2, "handler must run on every request"
    assert not any("v1-checkout" in k for k in fake_cache.store), (
        "checkout responses must not be written to the cache"
    )


@pytest.mark.asyncio
async def test_no_cache_header_skips_lookup(app, fake_cache) -> None:
    with patch(
        "core.http_cache.get_access_token_allow_expired", new_callable=AsyncMock
    ) as m:
        m.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/v1/visitors")
            assert app.state.counter == 1
            r = await client.get("/v1/visitors", headers={"Cache-Control": "no-cache"})
    assert r.status_code == 200
    assert r.headers.get("x-cache") == "MISS"
    assert app.state.counter == 2, "no-cache must force the handler to run"


@pytest.mark.asyncio
async def test_cookie_session_never_shares_the_anonymous_cache(app, fake_cache) -> None:
    """A cookie-authenticated read must not be served, or stored, under the anon scope."""
    tokens = {
        "tok-a": MagicMock(role="super_admin", userId="user-a", tenant_id="tenant-a"),
        "tok-b": MagicMock(role="super_admin", userId="user-b", tenant_id="tenant-b"),
    }

    async def lookup(accessToken: str):
        return tokens.get(accessToken)

    with patch("core.http_cache.get_access_token_allow_expired", side_effect=lookup):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            a = await client.get("/v1/visitors", cookies={"access_token": "tok-a"})
            anon = await client.get("/v1/visitors")
            b = await client.get("/v1/visitors", cookies={"access_token": "tok-b"})

    assert a.headers.get("x-cache") == "MISS"
    assert anon.headers.get("x-cache") == "MISS", (
        "anonymous caller must not get tenant A's response"
    )
    assert b.headers.get("x-cache") == "MISS", (
        "tenant B must not get tenant A's response"
    )
    assert app.state.counter == 3
    assert any(k.startswith("httpcache:t:tenant-a:") for k in fake_cache.store)
    assert any(k.startswith("httpcache:t:tenant-b:") for k in fake_cache.store)


@pytest.mark.asyncio
async def test_unknown_credential_is_not_cached(app, fake_cache) -> None:
    """A credential that resolves to no token skips the cache instead of falling back to anon."""
    with patch(
        "core.http_cache.get_access_token_allow_expired", new_callable=AsyncMock
    ) as m:
        m.return_value = None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/v1/visitors", cookies={"access_token": "stale"})

    assert r.status_code == 200
    assert "x-cache" not in r.headers
    assert fake_cache.store == {}
