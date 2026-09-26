"""Ensure MongoDB indexes exist for hot query paths.

Called once at startup (from ``main.lifespan``). ``create_index`` is idempotent
in Motor — repeated calls are cheap once the index exists. Adding an index
here is almost always net-positive for read-heavy collections; if a collection
is write-heavy we pick a narrow compound rather than many single-field indexes.

Keep this list pruned. Unused indexes cost RAM and write throughput.
"""

from __future__ import annotations

import logging
from typing import Any

from pymongo import ASCENDING, DESCENDING

logger = logging.getLogger(__name__)


# Each entry: (collection_name, keys, options)
# ``keys`` is a list of (field, direction) tuples.
# ``options`` is passed straight to ``create_index``; common ones:
#   unique=True, sparse=True, name="<explicit>", expireAfterSeconds=<ttl>
_INDEX_PLAN: list[tuple[str, list[tuple[str, int]], dict[str, Any]]] = [
    # ── auth / tokens ────────────────────────────────────────────────
    # verify_*_token → look up by _id (auto-indexed), but delete_all_* scans by userId
    ("accessToken", [("userId", ASCENDING)], {}),
    ("accessToken", [("status", ASCENDING)], {"sparse": True}),
    ("refreshToken", [("userId", ASCENDING)], {}),
    # Brute-force lockout / password history
    (
        "login_attempts",
        [("identifier", ASCENDING)],
        {"unique": True},
    ),
    (
        "password_history",
        [("user_id", ASCENDING), ("role", ASCENDING), ("changed_at", DESCENDING)],
        {},
    ),
    # Self-service password reset: tokens are looked up by hash on every
    # consume; selections by hash on every step-2 send.
    (
        "password_reset_tokens",
        [("token_hash", ASCENDING)],
        {"unique": True},
    ),
    (
        "password_reset_tokens",
        [("user_id", ASCENDING), ("user_type", ASCENDING), ("used", ASCENDING)],
        {},
    ),
    (
        "password_reset_selections",
        [("selection_token_hash", ASCENDING)],
        {"unique": True},
    ),
    # ── tenants ──────────────────────────────────────────────────────
    (
        "tenant_companies",
        [("company_name", ASCENDING)],
        {"unique": True},
    ),
    # ── accounts ─────────────────────────────────────────────────────
    ("admins", [("email", ASCENDING)], {"unique": True}),
    ("users", [("email", ASCENDING)], {"unique": True}),
    # Tenant users: email is only unique WITHIN a tenant.
    (
        "system_users",
        [("tenant_id", ASCENDING), ("email", ASCENDING)],
        {"unique": True},
    ),
    ("system_users", [("tenant_id", ASCENDING), ("role", ASCENDING)], {}),
    # Main super_admin invariant: at most one row per tenant carries
    # ``is_main_super_admin=True``. Partial filter expression keeps the
    # uniqueness scoped to flagged rows so the millions of False rows
    # don't even enter the index, and a single insert that violates the
    # invariant is rejected at the DB layer regardless of which service
    # forgot to call the guard. Belt and braces with the service guards
    # in services/main_super_admin_guard.py.
    (
        "system_users",
        [("tenant_id", ASCENDING), ("is_main_super_admin", ASCENDING)],
        {
            "unique": True,
            "partialFilterExpression": {"is_main_super_admin": True},
            "name": "tenant_main_super_admin_unique",
        },
    ),
    # ── plans / subscriptions / billing ──────────────────────────────
    ("plans", [("name", ASCENDING)], {"unique": True}),
    ("plans", [("status", ASCENDING), ("is_public", ASCENDING)], {}),
    (
        "subscriptions",
        [("tenant_id", ASCENDING), ("status", ASCENDING)],
        {},
    ),
    # Dunning / renewal scan: "due subscriptions"
    (
        "subscriptions",
        [("status", ASCENDING), ("current_period_end", ASCENDING)],
        {},
    ),
    (
        "subscriptions",
        [("status", ASCENDING), ("next_retry_at", ASCENDING)],
        {"sparse": True},
    ),
    (
        "discounts",
        [("code", ASCENDING)],
        {"unique": True, "sparse": True},
    ),
    ("discounts", [("status", ASCENDING), ("valid_until", ASCENDING)], {}),
    # Trial codes — one redeemed ("used") trial per tenant is a hard
    # invariant of the billing system. The partial-unique index rejects a
    # second redemption at the DB layer even if two requests race past the
    # app-level get_tenant_redeemed_trial() check in trial_code_service.
    # Mirrors the tenant_main_super_admin_unique partial-unique pattern.
    ("trial_codes", [("code", ASCENDING)], {"unique": True}),
    ("trial_codes", [("tenant_id", ASCENDING), ("status", ASCENDING)], {}),
    (
        "trial_codes",
        [("tenant_id", ASCENDING)],
        {
            "unique": True,
            "partialFilterExpression": {"status": "used"},
            "name": "tenant_used_trial_unique",
        },
    ),
    # Usage aggregates — hot on every authenticated request (quota checks).
    (
        "usage_aggregates",
        [
            ("tenant_id", ASCENDING),
            ("collection", ASCENDING),
            ("operation", ASCENDING),
            ("period_key", ASCENDING),
        ],
        {"unique": True},
    ),
    (
        "usage_records",
        [("tenant_id", ASCENDING), ("timestamp", DESCENDING)],
        {},
    ),
    # Invoices: list by tenant newest-first, lookup by number.
    (
        "invoices",
        [("tenant_id", ASCENDING), ("date_created", DESCENDING)],
        {},
    ),
    ("invoices", [("number", ASCENDING)], {"unique": True, "sparse": True}),
    ("invoice_counters", [("year", ASCENDING)], {"unique": True}),
    # ── payments / checkout / webhooks ───────────────────────────────
    (
        "payment_transactions",
        [("reference", ASCENDING)],
        {"unique": True, "sparse": True},
    ),
    ("payment_transactions", [("owner_id", ASCENDING)], {}),
    ("payment_transactions", [("status", ASCENDING)], {}),
    (
        "webhook_events",
        [("provider", ASCENDING), ("event_id", ASCENDING)],
        {"unique": True},
    ),
    (
        "checkout_sessions",
        [("tenant_id", ASCENDING), ("date_created", DESCENDING)],
        {},
    ),
    (
        "checkout_sessions",
        [("provider_reference", ASCENDING)],
        {"unique": True, "sparse": True},
    ),
    ("checkout_sessions", [("status", ASCENDING), ("expires_at", ASCENDING)], {}),
    # ── core tenant data ─────────────────────────────────────────────
    ("branches", [("tenant_id", ASCENDING), ("status", ASCENDING)], {}),
    ("branches", [("tenant_id", ASCENDING), ("name", ASCENDING)], {}),
    ("departments", [("tenant_id", ASCENDING)], {}),
    ("departments", [("tenant_id", ASCENDING), ("name", ASCENDING)], {}),
    ("hosts", [("tenant_id", ASCENDING), ("name", ASCENDING)], {}),
    ("hosts", [("tenant_id", ASCENDING), ("department_id", ASCENDING)], {}),
    (
        "hosts",
        [("tenant_id", ASCENDING), ("source_system_user_id", ASCENDING)],
        {"sparse": True},
    ),
    # Visitor lookups at the kiosk match on phone (or email) within a tenant.
    ("visitors", [("tenant_id", ASCENDING), ("phone", ASCENDING)], {}),
    ("visitors", [("tenant_id", ASCENDING), ("email", ASCENDING)], {}),
    # Visit sessions carry check_in_time, status and host_id; these three lived
    # on ``visitors`` by mistake, where none of the fields exist.
    (
        "visit_sessions",
        [("tenant_id", ASCENDING), ("check_in_time", DESCENDING)],
        {},
    ),
    ("visit_sessions", [("tenant_id", ASCENDING), ("status", ASCENDING)], {}),
    ("visit_sessions", [("tenant_id", ASCENDING), ("host_id", ASCENDING)], {}),
    # Check-ins: the reception queue (tenant + state, newest first), the
    # awaiting-checkout list (tenant + state by approval time), date-range
    # analytics, and the in-flight duplicate guard per visitor.
    (
        "checkins",
        [("tenant_id", ASCENDING), ("state", ASCENDING), ("date_created", DESCENDING)],
        {},
    ),
    (
        "checkins",
        [("tenant_id", ASCENDING), ("state", ASCENDING), ("approved_at", DESCENDING)],
        {},
    ),
    ("checkins", [("tenant_id", ASCENDING), ("date_created", DESCENDING)], {}),
    (
        "checkins",
        [("tenant_id", ASCENDING), ("visitor_id", ASCENDING), ("state", ASCENDING)],
        {},
    ),
    (
        "consent_records",
        [("tenant_id", ASCENDING), ("consent_timestamp", DESCENDING)],
        {},
    ),
    ("consent_records", [("tenant_id", ASCENDING), ("visitor_id", ASCENDING)], {}),
    (
        "expected_appointments",
        [("tenant_id", ASCENDING), ("scheduled_datetime", DESCENDING)],
        {},
    ),
    # /v1/jobs/{task_id} is polled after every queued write.
    ("queue_job_log", [("task_id", ASCENDING)], {}),
    ("queue_job_log", [("tenant_id", ASCENDING), ("date_created", DESCENDING)], {}),
    ("queue_job_log", [("actor_id", ASCENDING), ("date_created", DESCENDING)], {}),
    (
        "visitor_profiles",
        [("tenant_id", ASCENDING), ("email_normalized", ASCENDING)],
        {"sparse": True},
    ),
    # Phone is the canonical visitor-identity key per-tenant (the kiosk
    # checks for an existing profile by phone before falling back to
    # email or id_number). Sparse-unique so legacy profiles with no
    # phone aren't rejected, but new writes can't create a duplicate
    # ``(tenant_id, phone)`` pair.
    (
        "visitor_profiles",
        [("tenant_id", ASCENDING), ("phone", ASCENDING)],
        {"sparse": True, "unique": True, "name": "tenant_phone_unique"},
    ),
    # Per-tenant enum config (purpose-of-visit, id_type, …). One row per
    # ``(tenant_id, kind)`` so the kiosk can fetch every picker in one hit.
    (
        "tenant_enums",
        [("tenant_id", ASCENDING), ("kind", ASCENDING)],
        {"unique": True},
    ),
    # KYC verification records — one row per check-in attempt, keyed by
    # provider reference_id for webhook idempotency.
    (
        "kyc_verifications",
        [("reference_id", ASCENDING)],
        {"unique": True, "sparse": True},
    ),
    (
        "kyc_verifications",
        [("checkin_id", ASCENDING)],
        {"sparse": True},
    ),
    (
        "kyc_verifications",
        [("tenant_id", ASCENDING), ("status", ASCENDING)],
        {},
    ),
    (
        "kyc_webhook_events",
        [("event_id", ASCENDING)],
        {"unique": True, "sparse": True},
    ),
    (
        "appointments",
        [("tenant_id", ASCENDING), ("scheduled_time", ASCENDING)],
        {},
    ),
    ("appointments", [("tenant_id", ASCENDING), ("status", ASCENDING)], {}),
    ("incidents", [("tenant_id", ASCENDING), ("status", ASCENDING)], {}),
    ("incidents", [("tenant_id", ASCENDING), ("detection_time", DESCENDING)], {}),
    ("documents", [("tenant_id", ASCENDING), ("date_created", DESCENDING)], {}),
    (
        "audit_trail",
        [("tenant_id", ASCENDING), ("timestamp", DESCENDING)],
        {},
    ),
    (
        "audit_trail",
        [("resource_type", ASCENDING), ("resource_id", ASCENDING)],
        {},
    ),
    (
        "data_subject_requests",
        [("tenant_id", ASCENDING), ("status", ASCENDING)],
        {},
    ),
    ("privacy_notices", [("tenant_id", ASCENDING), ("version", ASCENDING)], {}),
    # Tenant agreements (DPA + Visitor Privacy Policy): one row per
    # (tenant, agreement). Unique compound backs the upsert + the gate lookup.
    (
        "tenant_agreements",
        [("tenant_id", ASCENDING), ("agreement_key", ASCENDING)],
        {"unique": True},
    ),
    ("sub_processors", [("tenant_id", ASCENDING)], {}),
    # ── settings / sessions / notifications ──────────────────────────
    (
        "user_settings",
        [("user_id", ASCENDING), ("user_type", ASCENDING)],
        {"unique": True},
    ),
    (
        "user_preferences",
        [("user_id", ASCENDING), ("user_type", ASCENDING), ("key", ASCENDING)],
        {"unique": True},
    ),
    # Tutorial progress — one row per (user_id, tutorial_type, version) so a
    # version bump re-runs a redesigned tutorial without losing history. The
    # unique key also backs the upsert in services/tutorial_service.py.
    (
        "tutorials",
        [
            ("user_id", ASCENDING),
            ("tutorial_type", ASCENDING),
            ("version", ASCENDING),
        ],
        {"unique": True},
    ),
    # List a user's progress (the GET /v1/tutorials default read).
    (
        "tutorials",
        [("user_id", ASCENDING), ("user_type", ASCENDING)],
        {},
    ),
    ("tenant_settings", [("tenant_id", ASCENDING)], {"unique": True}),
    (
        "sessions",
        [("user_id", ASCENDING), ("is_current", ASCENDING)],
        {},
    ),
    ("sessions", [("user_id", ASCENDING), ("last_active_at", DESCENDING)], {}),
    (
        "notifications",
        [("user_id", ASCENDING), ("read", ASCENDING), ("date_created", DESCENDING)],
        {},
    ),
    # Backs the read-receipt auto-mark: flip a user's unread notifications
    # to read when they read the resource that triggered them.
    (
        "notifications",
        [
            ("user_id", ASCENDING),
            ("user_type", ASCENDING),
            ("resource_type", ASCENDING),
            ("resource_id", ASCENDING),
            ("read", ASCENDING),
        ],
        {},
    ),
    (
        "notification_preferences",
        [("user_id", ASCENDING), ("user_type", ASCENDING)],
        {"unique": True},
    ),
    # ── 2FA / OTP ────────────────────────────────────────────────────
    (
        "totp_secrets",
        [("user_id", ASCENDING), ("user_type", ASCENDING)],
        {"unique": True},
    ),
    ("backup_codes", [("user_id", ASCENDING)], {}),
    (
        "otp_challenges",
        [("challenge_id", ASCENDING)],
        {"unique": True, "sparse": True},
    ),
    # ── check-in system ─────────────────────────────────────────────
    ("checkin_configs", [("tenant_id", ASCENDING)], {}),
    ("badges", [("tenant_id", ASCENDING), ("qr_code_value", ASCENDING)], {}),
    ("id_verification_hashes", [("tenant_id", ASCENDING), ("hash", ASCENDING)], {}),
    # ── support cases (platform support threads) ─────────────────────
    (
        "support_cases",
        [
            ("tenant_id", ASCENDING),
            ("status", ASCENDING),
            ("last_message_at", DESCENDING),
        ],
        {},
    ),
    ("support_cases", [("tenant_id", ASCENDING), ("status", ASCENDING)], {}),
    ("support_cases", [("assigned_admin_id", ASCENDING), ("status", ASCENDING)], {}),
    ("support_cases", [("status", ASCENDING), ("sla_due_at", ASCENDING)], {}),
    (
        "support_case_messages",
        [("case_id", ASCENDING), ("date_created", ASCENDING)],
        {},
    ),
    # ── self-onboarding (public marketing-site lead capture) ─────────
    (
        "onboarding_submissions",
        [("submitted_at", DESCENDING)],
        {},
    ),
    (
        "onboarding_submissions",
        [("status", ASCENDING), ("submitted_at", DESCENDING)],
        {},
    ),
    (
        "onboarding_submissions",
        [("email", ASCENDING)],
        {"sparse": True},
    ),
    (
        "onboarding_submissions",
        [("super_admin_user_id", ASCENDING)],
        {"sparse": True},
    ),
    (
        "onboarding_submissions",
        [("tenant_id", ASCENDING)],
        {"sparse": True},
    ),
    # ── new-visitor first-seen ledger (WS0.3) ─────────────────────────
    # One row per (tenant, branch, visitor_profile) — the unique index IS
    # the "is this visitor new to this branch?" check: record_first_seen
    # relies on a DuplicateKeyError to detect a returning visitor.
    (
        "visitor_branch_firsts",
        [
            ("tenant_id", ASCENDING),
            ("branch_id", ASCENDING),
            ("visitor_profile_id", ASCENDING),
        ],
        {"unique": True, "name": "tenant_branch_visitor_profile_unique"},
    ),
    (
        "visitor_branch_firsts",
        [("tenant_id", ASCENDING), ("first_seen_at", ASCENDING)],
        {},
    ),
]


async def ensure_indexes(db: Any) -> dict[str, int]:
    """Create (or confirm) every index in ``_INDEX_PLAN``. Idempotent.

    Returns a summary mapping collection → count of indexes ensured on it.
    Failures are logged but never raised: the app must be able to boot with a
    Mongo that's slow to build indexes.
    """
    summary: dict[str, int] = {}
    for collection, keys, options in _INDEX_PLAN:
        try:
            name = await db[collection].create_index(keys, **options)
            summary[collection] = summary.get(collection, 0) + 1
            logger.debug("Ensured index %s on %s (%s)", name, collection, keys)
        except Exception as err:
            logger.warning(
                "Failed to ensure index on %s (%s): %s", collection, keys, err
            )
    logger.info(
        "Mongo indexes ensured: %s total across %s collections",
        sum(summary.values()),
        len(summary),
    )
    return summary
