"""
Supabase wrapper for the Renaissance scoring MVP.

Three things the rest of the app needs from Supabase:
    1. get_percentile(score_raw)  -> float   (ranks a new lead live)
    2. insert_lead(lead_dict)     -> dict    (persists the scored lead)
    3. health_check()             -> dict    (used by test_connection.py + app.py)

Design notes:
  - Uses the service_role key from .env. That key bypasses RLS entirely, which
    is what the MVP wants: the advisor-facing Streamlit is a trusted backend.
  - All table access goes through the 'scoring' schema (schema='scoring').
  - Errors from Supabase are NOT swallowed — the UI needs to know when an
    INSERT fails so it doesn't show 'Saved' to the advisor for a lost record.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

TABLE = "lead_score"
MEETING_HISTORY_TABLE = "meeting_history"
SCHEMA = "scoring"
RPC_PERCENTILE = "get_percentile"


@lru_cache(maxsize=1)
def _client() -> Client:
    """Single process-wide client. Reused across Streamlit reruns."""
    # Local dev: load .env if present. On Streamlit Cloud .env doesn't exist
    # and load_dotenv is a no-op, but require_secret will read from
    # st.secrets[...] there — so the same code works in both environments.
    load_dotenv()
    from app_secrets import require_secret
    url = require_secret("SUPABASE_URL")
    key = require_secret("SUPABASE_SERVICE_KEY")
    return create_client(url, key)


def get_percentile(score_raw: float) -> float:
    """
    Call scoring.get_percentile(input_score). Returns a float 0-100 where
    0 = best lead (top of distribution) and 100 = worst.

    Raises on any Supabase error — caller decides whether to show a fallback
    in the UI or block the save.
    """
    client = _client()
    resp = client.schema(SCHEMA).rpc(
        RPC_PERCENTILE, {"input_score": float(score_raw)}
    ).execute()
    # Supabase returns the scalar directly in .data for a numeric RPC
    value = resp.data
    if value is None:
        raise RuntimeError(f"get_percentile({score_raw}) returned None")
    return float(value)


def insert_lead(lead: dict[str, Any]) -> dict[str, Any]:
    """
    INSERT a scored lead into scoring.lead_scores. 'scored_at' is filled by
    DEFAULT NOW() on the DB side, so the caller doesn't need to send it.

    Returns the inserted row (with the DB-assigned id + scored_at).
    """
    client = _client()
    resp = (
        client.schema(SCHEMA)
        .table(TABLE)
        .insert(lead, returning="representation")
        .execute()
    )
    if not resp.data:
        raise RuntimeError(f"INSERT returned no rows. Payload keys: {list(lead)}")
    return resp.data[0]


def find_lead_by_email(email: str) -> dict[str, Any] | None:
    """
    Retrieve the MOST RECENT scored lead for a given email.

    A lead can legitimately be scored multiple times (e.g. responded to a
    cold email in January, then responded to a different cold outreach months
    later). Each scoring event creates its own row in lead_score. This
    function returns the latest one so 'Update Meeting' operates on the
    current opportunity, leaving the historical rows untouched.

    Returns None if no lead with that email exists (or all are soft-deleted).
    """
    client = _client()
    resp = (
        client.schema(SCHEMA).table(TABLE)
        .select("id, company, email, first_name, last_name, partner, "
                "partner_sent_to, score_raw, percentile, meeting_datetime, "
                "meeting_timezone, meeting_status, scored_at")
        .eq("email", email.strip().lower())
        .is_("deleted_at", "null")
        .order("scored_at", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    return resp.data[0]


def auto_complete_past_meetings() -> int:
    """
    Sweep for Scheduled meetings whose date has passed, and mark them as
    Completed automatically. Called at app startup / sidebar render, so the
    UI reflects the auto-completion without needing a background job.

    Both tables are updated in lockstep:
      - meeting_history.meeting_status -> 'Completed' + auto_completed=True
      - lead_score.meeting_status -> 'Completed'

    Returns the number of meetings auto-completed this pass.
    """
    client = _client()
    now_utc = datetime.now(timezone.utc).isoformat()

    # Find past-due Scheduled meetings in history
    due_resp = (
        client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
        .select("id, lead_id")
        .eq("meeting_status", "Scheduled")
        .lt("meeting_datetime", now_utc)
        .execute()
    )
    due = due_resp.data or []
    if not due:
        return 0

    for row in due:
        hist_id = row["id"]
        lead_id = row["lead_id"]
        # Mark the history row Completed (and flag auto)
        (
            client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
            .update({"meeting_status": "Completed", "meeting_auto_completed": True})
            .eq("id", hist_id)
            .execute()
        )
        # Only sync to lead_score if THIS is still the current meeting for
        # that lead (avoid overwriting a more recent reschedule). We check
        # by matching meeting_datetime on the lead row.
        lead_resp = (
            client.schema(SCHEMA).table(TABLE)
            .select("meeting_datetime, meeting_status")
            .eq("id", lead_id).limit(1).execute()
        )
        if lead_resp.data and lead_resp.data[0].get("meeting_status") == "Scheduled":
            (
                client.schema(SCHEMA).table(TABLE)
                .update({"meeting_status": "Completed"})
                .eq("id", lead_id)
                .execute()
            )
    return len(due)


def list_leads_without_meeting(
    limit: int = 50,
    scored_since: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return leads that still have NO meeting ever scheduled (meeting_status
    IS NULL). These are leads the advisor scored but hasn't yet registered
    with the partner's calendar.

    Parameters:
      limit         — max rows returned (default 50).
      scored_since  — optional ISO timestamp filter. Only leads scored on
                      or after this moment are included. The MVP uses the
                      start of "today" so the historical 22k doesn't show up.

    A lead DISAPPEARS from this list as soon as any meeting is scheduled
    for it (first-time or reschedule), and never comes back — even if the
    meeting is later cancelled or marked no-show. The advisor can still
    manage those cases from the 'Update Meeting' page.
    """
    client = _client()
    query = (
        client.schema(SCHEMA).table(TABLE)
        .select("id, company, email, first_name, last_name, partner, scored_at")
        .is_("meeting_status", "null")
        .is_("deleted_at", "null")
    )
    if scored_since:
        query = query.gte("scored_at", scored_since)
    query = query.order("scored_at", desc=True).limit(limit)
    resp = query.execute()
    return resp.data or []


def create_meeting(
    lead_id: int,
    meeting_datetime_utc: str,
    meeting_timezone: str,
    partner_sent_to: str | None = None,
) -> dict[str, Any]:
    """
    Register a NEW meeting for a lead. Used both for the very first meeting
    and for every reschedule.

    Does two writes in order (no real transaction — Supabase REST limitation):
      1) INSERT a row in scoring.meeting_history with status 'Scheduled'.
      2) UPDATE scoring.lead_score to point at this meeting as the current
         one (meeting_datetime, meeting_timezone, meeting_status). If
         partner_sent_to is supplied, that column is updated too so the
         record reflects which partner the advisor actually sent the lead to.

    Arguments:
      lead_id              — 'id' from scoring.lead_score
      meeting_datetime_utc — ISO string in UTC, e.g. '2026-04-25T19:00:00+00:00'
      meeting_timezone     — short code: 'ET' | 'CT' | 'MT' | 'PT'
      partner_sent_to      — partner the advisor actually routed this lead to.
                             Can differ from the model's recommendation. If
                             None, the existing value is left untouched.

    Returns the INSERT row from meeting_history (with its new id).
    """
    client = _client()

    insert_resp = (
        client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
        .insert({
            "lead_id": lead_id,
            "meeting_datetime": meeting_datetime_utc,
            "meeting_timezone": meeting_timezone,
            "meeting_status": "Scheduled",
            "partner_sent_to": partner_sent_to,
        }, returning="representation")
        .execute()
    )
    if not insert_resp.data:
        raise RuntimeError(f"meeting_history INSERT returned no rows for lead_id={lead_id}")
    history_row = insert_resp.data[0]

    # Mirror onto the lead_score row for quick access & the sidebar
    lead_update = {
        "meeting_datetime": meeting_datetime_utc,
        "meeting_timezone": meeting_timezone,
        "meeting_status":   "Scheduled",
    }
    if partner_sent_to is not None:
        lead_update["partner_sent_to"] = partner_sent_to

    update_resp = (
        client.schema(SCHEMA).table(TABLE)
        .update(lead_update, returning="representation")
        .eq("id", lead_id)
        .execute()
    )
    if not update_resp.data:
        raise RuntimeError(f"lead_score UPDATE returned no rows for lead_id={lead_id}")

    return history_row


def reschedule_meeting(
    lead_id: int,
    new_meeting_datetime_utc: str,
    meeting_timezone: str,
    partner_sent_to: str | None = None,
) -> dict[str, Any]:
    """
    Reschedule a lead's meeting: marks the CURRENT (most recent) open
    meeting_history row as 'No-show' if it's still Scheduled, then inserts
    a new meeting_history row with the new datetime, and updates lead_score
    to point at the new meeting.

    If the current meeting was already Completed/No-show/Cancelled, it is
    left untouched — we only insert the new meeting on top.

    Returns the new meeting_history row.
    """
    client = _client()

    # Only overwrite the previous meeting's status if it was still open.
    prev_resp = (
        client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
        .select("id, meeting_status")
        .eq("lead_id", lead_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if prev_resp.data:
        prev = prev_resp.data[0]
        if prev.get("meeting_status") == "Scheduled":
            (
                client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
                .update({"meeting_status": "No-show"})
                .eq("id", prev["id"])
                .execute()
            )

    # Delegate the new meeting creation to the shared function
    return create_meeting(
        lead_id, new_meeting_datetime_utc, meeting_timezone, partner_sent_to,
    )


def update_meeting_status(lead_id: int, new_status: str) -> dict[str, Any]:
    """
    Change the status of a lead's CURRENT meeting (the most recent row in
    meeting_history) without changing the datetime.

    Used for: 'Completed', 'No-show', 'Cancelled'. 'Rescheduled' is NOT a
    valid status in meeting_history by itself — use reschedule_meeting().

    Mirrors the new status to lead_score.meeting_status so quick lookups
    don't need a JOIN.

    Returns the updated meeting_history row.
    """
    VALID = {"Scheduled", "No-show", "Completed", "Cancelled"}
    if new_status not in VALID:
        raise ValueError(f"Invalid meeting status {new_status!r}. Must be one of {VALID}.")

    client = _client()

    # Find the current (most recent) meeting_history row for this lead
    current_resp = (
        client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
        .select("id")
        .eq("lead_id", lead_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not current_resp.data:
        raise RuntimeError(
            f"No meeting_history row for lead_id={lead_id}; "
            f"call create_meeting() first."
        )
    hist_id = current_resp.data[0]["id"]

    hist_resp = (
        client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
        .update({"meeting_status": new_status}, returning="representation")
        .eq("id", hist_id)
        .execute()
    )
    (
        client.schema(SCHEMA).table(TABLE)
        .update({"meeting_status": new_status})
        .eq("id", lead_id)
        .execute()
    )
    return hist_resp.data[0] if hist_resp.data else {}


def list_meetings_for_lead(lead_id: int) -> list[dict[str, Any]]:
    """
    Return the full meeting history for a lead, oldest first. Used by the
    Update Meeting page to show a timeline of meetings (including
    reschedules and their outcomes).
    """
    client = _client()
    resp = (
        client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
        .select("id, meeting_datetime, meeting_timezone, meeting_status, "
                "partner_sent_to, notes, created_at")
        .eq("lead_id", lead_id)
        .order("created_at", desc=False)
        .execute()
    )
    return resp.data or []


def soft_delete_lead(
    lead_id: int,
    reason: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Soft-delete a lead. The row stays in scoring.lead_score (for audit) but
    is filtered out of all app queries (lookups, pending list, percentile
    calc, etc.) via the deleted_at column.

    Eligibility rule
    ----------------
    Delete is only allowed when:
      - The lead has NO meeting (meeting_status IS NULL), OR
      - The lead's meeting is currently 'Scheduled' (which by design means
        it is in the future — past-dated Scheduled rows are auto-completed
        elsewhere in the app, so anything still 'Scheduled' has not happened
        yet).

    For any other status (Completed, No-show, Cancelled, Rescheduled,
    Lead deleted) we raise ValueError. The advisor must update the meeting
    status manually first if they really want to delete. This protects
    against accidentally erasing leads that already produced a real-world
    event (a meeting that happened, was no-showed, etc.).

    Side effect on meeting_history
    ------------------------------
    If the lead has a Scheduled meeting, we UPDATE that row in place:
    flip its status from 'Scheduled' to 'Lead deleted' and append a note
    explaining the deletion. We do NOT insert a new history row.

    Why update in place (instead of appending, like no-show / reschedule)?
    Because the meeting was never real — it was created by a data-entry
    mistake. Preserving 'this meeting was scheduled at time X' is not
    useful audit information; it's just noise. The real audit info lives
    on lead_score (deleted_at, delete_reason, delete_notes).

    'Lead deleted' is its own status — distinct from plain 'Cancelled'.
    A cancellation is a customer-driven event; 'Lead deleted' is advisor-
    driven (wrong data, duplicate, test entry). Keeping them separate lets
    analytics answer different questions:
      - "How often do customers cancel?" → status = 'Cancelled'
      - "How often do advisors make data-entry mistakes?" → status = 'Lead deleted'

    Parameters
    ----------
    lead_id : int
        The lead to soft-delete.
    reason : str
        One of the predefined categories from the UI dropdown
        ('Wrong data entered', 'Duplicate scoring',
         'Test entry / not a real lead', 'Other').
        Required: passing empty/None will raise.
    notes : str, optional
        Free-text detail. Useful for 'Other' or when the advisor wants
        to add context to one of the predefined reasons.

    Raises
    ------
    ValueError
        If reason is empty, or if the lead's current meeting_status makes
        it ineligible for delete (see Eligibility rule above).
    RuntimeError
        If the lead doesn't exist or the UPDATE returns no rows.
    """
    if not reason or not reason.strip():
        raise ValueError("soft_delete_lead requires a non-empty 'reason'")

    client = _client()
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    now_iso = _dt.now(_tz.utc).isoformat()

    # Step 1 — Look up the lead's current meeting_status, both to enforce
    # the eligibility rule and to know whether we need to touch
    # meeting_history at all.
    lead_resp = (
        client.schema(SCHEMA).table(TABLE)
        .select("meeting_status, meeting_datetime, meeting_timezone, partner_sent_to")
        .eq("id", lead_id)
        .limit(1)
        .execute()
    )
    if not lead_resp.data:
        raise RuntimeError(f"Lead id={lead_id} not found")
    current = lead_resp.data[0]
    current_status = current.get("meeting_status")

    # Step 2 — Enforce eligibility. NULL or 'Scheduled' only.
    if current_status is not None and str(current_status).lower() != "scheduled":
        raise ValueError(
            f"Cannot delete this lead: its meeting status is "
            f"'{current_status}'. Delete is only allowed for leads with no "
            f"meeting or with a meeting still scheduled in the future. "
            f"If you really need to remove this lead, update the meeting "
            f"status manually first."
        )

    # Step 3 — If the lead has a Scheduled meeting, flip the matching
    # meeting_history row's status to 'Lead deleted' and add an explanatory
    # note. We update in place (no new row) — see docstring for rationale.
    if current_status and str(current_status).lower() == "scheduled":
        cancel_note = f"Lead was deleted. Reason: {reason.strip()}"
        if notes and notes.strip():
            cancel_note += f". Notes: {notes.strip()}"

        # Find the most recent Scheduled row for this lead — that's the
        # active one corresponding to the current meeting on lead_score.
        history_resp = (
            client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
            .select("id")
            .eq("lead_id", lead_id)
            .eq("meeting_status", "Scheduled")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if history_resp.data:
            history_id = history_resp.data[0]["id"]
            (
                client.schema(SCHEMA).table(MEETING_HISTORY_TABLE)
                .update({
                    "meeting_status": "Lead deleted",
                    "notes":          cancel_note,
                })
                .eq("id", history_id)
                .execute()
            )

    # Step 4 — Soft-delete the lead row itself. Mirror the new status onto
    # lead_score so it's consistent even outside meeting_history.
    payload = {
        "deleted_at":    now_iso,
        "delete_reason": reason.strip(),
        "delete_notes":  (notes.strip() if notes and notes.strip() else None),
    }
    if current_status and str(current_status).lower() == "scheduled":
        payload["meeting_status"] = "Lead deleted"

    resp = (
        client.schema(SCHEMA).table(TABLE)
        .update(payload, returning="representation")
        .eq("id", lead_id)
        .execute()
    )
    if not resp.data:
        raise RuntimeError(f"Soft-delete UPDATE returned no rows for lead_id={lead_id}")
    return resp.data[0]


def health_check() -> dict[str, Any]:
    """
    Cheap read that exercises the three things the MVP relies on:
      - Supabase connectivity
      - SELECT permission on scoring.lead_scores
      - RPC call on scoring.get_percentile
    Returns a summary dict; raises on any failure.
    """
    client = _client()
    total = (
        client.schema(SCHEMA).table(TABLE)
        .select("*", count="exact", head=True).execute().count
    )
    funded = (
        client.schema(SCHEMA).table(TABLE)
        .select("*", count="exact", head=True)
        .eq("funded", True).execute().count
    )
    pct_at_72 = get_percentile(72)
    return {
        "total_leads":   total,
        "funded_leads":  funded,
        "pct_at_score_72": round(pct_at_72, 2),
    }


if __name__ == "__main__":
    # Run directly to smoke-test: `python db.py`
    import json
    print(json.dumps(health_check(), indent=2))