"""HttpCacheMiddleware — global response cache keyed by principal + request shape."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Awaitable, Callable, Iterable, Optional, cast

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from core.queue.precompute import is_scope_dirty
from core.redis_cache import cache_db
from core.settings import get_settings
from repositories.tokens_repo import get_access_token_allow_expired
from security.principal import TENANT_USER_ROLES

CACHE_PREFIX = "httpcache"
DEFAULT_TTL_SECONDS = 60

CACHEABLE_METHODS = frozenset({"GET"})
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Full-path prefixes that bypass caching entirely.
_BYPASS_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)

# Path segments that always bypass caching.
#
# - ``v1-jobs`` reflects state from every queued write; the middleware only
#   invalidates entries whose first path segment matches the write's URL,
#   so a POST to ``/v1/notifications/...`` never touches a cached
#   ``/v1/jobs`` entry. Serving stale (often empty) job lists for up to 60s
#   is worse than the per-request DB lookup.
# - ``v1-notifications`` is per-user and dispatched by many synchronous
#   ``send_notification`` callers (incidents, check-ins, support cases) that
#   write directly to MongoDB and don't pass through the middleware. With
#   tenant-scoped scope keys, a stale cache would mask new badge counts and
#   freshly delivered notifications for up to 60s. The collection is
#   per-user with a tight {user_id, user_type, [read]} filter, so direct
#   reads are cheap.
# - ``v1-dashboard`` already uses the precompute layer for heavy payloads.
#   A second full-response cache can keep live check-in counters stale after
#   direct synchronous writes, so dashboard routes bypass this middleware.
# - ``v1-checkout`` is payment status: it flips PENDING → SUCCEEDED via the
#   Paystack webhook (anon scope, ``/v1/payments/...`` path) or the scheduler
#   poll — neither of which invalidates the tenant's cached ``v1-checkout``
#   entries, so a cached response can show a stale "pending" for up to 60s
#   after the payment already cleared. The return page polls this; never
#   cache it.
_BYPASS_RESOURCE_SEGMENTS = frozenset(
    {"v1-jobs", "v1-notifications", "v1-dashboard", "v1-checkout"}
)

# Substrings that mark a path as auth-related (never cacheable).
_BYPASS_SUBSTRINGS = (
    "/login",
    "/logout",
    "/verify-otp",
    "/2fa/",
    "/change-password",
    "/backup-codes",
    "/signup",
    "/register",
    "/bootstrap",
)

# Response headers that must not be preserved in the cache — they are
# per-request and would mislead downstream clients if replayed.
_EXCLUDED_HEADERS = frozenset(
    {
        "x-request-id",
        "x-process-time",
        "date",
        "server",
        "content-length",
    }
)


def _should_bypass(path: str) -> bool:
    for prefix in _BYPASS_PREFIXES:
        if path.startswith(prefix):
            return True
    for needle in _BYPASS_SUBSTRINGS:
        if needle in path:
            return True
    return False


def _resource_segment(path: str) -> str:
    """Take the first two path components after the leading slash, joined with '-'.

    /v1/visitors/abc/checkout                  -> 'v1-visitors'
    /v1/plans                                  -> 'v1-plans'
    /v1/tenants/{tid}/checkins                 -> 'v1-checkins'
    /v1/tenants/{tid}/pending-approvals        -> 'v1-checkins'
    /health                                    -> 'health'

    The tenant-scoped check-in list endpoints are routed under
    ``/v1/tenants/{tid}/...`` but conceptually belong to the check-in
    domain. We bucket them under ``v1-checkins`` so a write to any
    ``/v1/checkins/*`` endpoint invalidates both lists in one shot
    (otherwise the pending-approvals GET stays stale for up to TTL
    after an approval write).
    """
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 4 and parts[0] == "v1" and parts[1] == "tenants":
        if parts[3] in ("checkins", "pending-approvals"):
            return "v1-checkins"
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return "_root"


# Cross-resource invalidation graph. A write to the key invalidates the
# value-set in addition to itself.
#
# The visitor lifecycle straddles three URL surfaces:
#   * ``/v1/visitors/...``     — visit_sessions writer surface
#   * ``/v1/checkins/...``     — receptionist approval queue
#   * ``/v1/tenants/{tid}/checkins`` / ``/pending-approvals`` (also
#     bucketed under ``v1-checkins`` via ``_resource_segment`` above)
#
# A successful approve / deny / confirm / check-out / host-approve on
# any of these MUST clear cached GETs on the other side so the UI
# doesn't display a row in two states at once for up to the cache TTL.
_LINKED_INVALIDATIONS: dict[str, tuple[str, ...]] = {
    "v1-visitors": ("v1-checkins",),
    "v1-checkins": ("v1-visitors",),
    "v1-appointments": ("v1-checkins", "v1-visitors"),
}


def _credential(request: Request) -> Optional[str]:
    """The access token the auth layer will use: a Bearer header, else the auth cookie."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", maxsplit=1)[1]
    return request.cookies.get("access_token") or None


class _UnresolvedCredential(Exception):
    """A credential was sent but maps to no known token; such a request must not share the anon cache."""


async def _resolve_scope(request: Request) -> str:
    """Determine the cache scope key from the auth context."""
    token = _credential(request)
    if not token:
        return "anon"
    access_token = await get_access_token_allow_expired(accessToken=token)
    if not access_token:
        raise _UnresolvedCredential()

    role = (access_token.role or "").lower()
    user_id = getattr(access_token, "userId", None) or "unknown"

    if role in TENANT_USER_ROLES:
        tenant_id = getattr(access_token, "tenant_id", None)
        if tenant_id:
            return f"t:{tenant_id}"
        return f"usr:{user_id}"
    if role == "admin":
        return f"adm:{user_id}"
    if role == "user":
        return f"usr:{user_id}"
    raise _UnresolvedCredential()


def _build_key(scope: str, resource: str, request: Request) -> str:
    case = request.headers.get("X-Response-Case", "camel").strip().lower()
    fingerprint = hashlib.sha256(
        "|".join(
            [request.url.path, request.url.query, case, _credential(request) or ""]
        ).encode("utf-8")
    ).hexdigest()
    return f"{CACHE_PREFIX}:{scope}:{resource}:{fingerprint}"


def _invalidation_pattern(scope: str, resource: str) -> str:
    return f"{CACHE_PREFIX}:{scope}:{resource}:*"


def _serialize(status_code: int, body: bytes, headers: dict[str, str]) -> str:
    return json.dumps(
        {
            "status_code": status_code,
            "body": body.decode("utf-8", errors="replace"),
            "headers": {
                k: v for k, v in headers.items() if k.lower() not in _EXCLUDED_HEADERS
            },
        }
    )


def _deserialize(raw: str) -> Response:
    data = json.loads(raw)
    body = data["body"].encode("utf-8")
    response = Response(
        content=body,
        status_code=data["status_code"],
        headers=data.get("headers") or {},
    )
    response.headers["X-Cache"] = "HIT"
    return response


async def _read_body(response: Response) -> bytes:
    """Drain a streaming response body and return the full bytes."""
    chunks: list[bytes] = []
    body_iterator = cast(Any, getattr(response, "body_iterator", None))
    if body_iterator is None:
        body = getattr(response, "body", b"") or b""
        return body if isinstance(body, bytes) else body.encode("utf-8")
    async for chunk in body_iterator:
        if isinstance(chunk, str):
            chunks.append(chunk.encode("utf-8"))
        else:
            chunks.append(chunk)
    return b"".join(chunks)


def _rehydrate(response: Response, body: bytes) -> Response:
    """Rebuild a Response after its body_iterator was consumed."""
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return Response(
        content=body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
    )


class HttpCacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        super().__init__(app)
        self.ttl_seconds = ttl_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if get_settings().env == "testing":
            return await call_next(request)

        method = request.method.upper()
        path = request.url.path

        if method == "OPTIONS" or _should_bypass(path):
            return await call_next(request)

        is_cacheable = method in CACHEABLE_METHODS
        is_write = method in WRITE_METHODS
        if not (is_cacheable or is_write):
            return await call_next(request)

        client_cc = request.headers.get("Cache-Control", "").lower()
        skip_cache = "no-cache" in client_cc or "no-store" in client_cc

        try:
            scope = await _resolve_scope(request)
        except Exception:
            # Any auth-resolution failure means we cannot safely key the cache.
            return await call_next(request)

        resource = _resource_segment(path)

        if resource in _BYPASS_RESOURCE_SEGMENTS:
            return await call_next(request)

        if is_cacheable:
            return await self._handle_get(
                request, call_next, scope, resource, skip_cache
            )

        # Write path: run the handler first, then invalidate on success.
        response = await call_next(request)
        if 200 <= response.status_code < 400:
            self._invalidate(scope, resource)
        return response

    async def _handle_get(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
        scope: str,
        resource: str,
        skip_cache: bool,
    ) -> Response:
        key = _build_key(scope, resource, request)

        # A write enqueued in the last few seconds for this scope (or a
        # global write affecting everyone) means cached responses may be
        # stale. Bypass the read AND the write so we don't lock in
        # pre-commit state for ttl_seconds while the worker catches up.
        if is_scope_dirty(scope):
            response = await call_next(request)
            response.headers["X-Cache"] = "BYPASS"
            return response

        if not skip_cache:
            cached = self._safe_get(key)
            if cached:
                return _deserialize(cached)

        response = await call_next(request)

        if not (200 <= response.status_code < 300):
            return response

        resp_ct = response.headers.get("content-type", "")
        if "application/json" not in resp_ct:
            # Skip streaming file downloads and other non-JSON payloads.
            return response

        resp_cc = response.headers.get("cache-control", "").lower()
        body = await _read_body(response)
        rehydrated = _rehydrate(response, body)

        if "no-store" in resp_cc or "private" in resp_cc:
            return rehydrated

        try:
            payload = _serialize(response.status_code, body, dict(response.headers))
            cache_db.setex(key, self.ttl_seconds, payload)
            rehydrated.headers["X-Cache"] = "MISS"
        except Exception:
            # Never let a cache write break the request.
            pass

        return rehydrated

    def _safe_get(self, key: str) -> Optional[str]:
        try:
            return cast(Optional[str], cache_db.get(key))
        except Exception:
            return None

    def _invalidate(self, scope: str, resource: str) -> None:
        resources_to_clear = {resource, *_LINKED_INVALIDATIONS.get(resource, ())}
        try:
            keys: list[str] = []
            for res in resources_to_clear:
                pattern = _invalidation_pattern(scope, res)
                for key in cast(
                    Iterable[str], cache_db.scan_iter(match=pattern, count=200)
                ):
                    keys.append(key)
            if keys:
                cache_db.delete(*keys)
        except Exception:
            pass
