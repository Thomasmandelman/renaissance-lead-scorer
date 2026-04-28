"""
app.py — Renaissance Lead Scorer (Streamlit UI)

Two pages, selected from the sidebar:

  1. Score Lead
     Advisor fills a form, we enrich via APIs, score via the model, and save
     to Supabase. The UI only shows which partner to send the lead to — all
     detailed features & diagnostics go to Supabase for later analysis.

  2. Update Meeting
     Advisor looks up a previously-scored lead by email and sets its
     meeting_datetime (the moment the meeting is booked with the partner).

Run:
    streamlit run app.py

This script intentionally does NOT do anything async at import time, so
Streamlit's fast reruns stay fast. Each button click triggers one work pass.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st

import db
import reply_features
import scoring
from enrichment import (
    AllApisFailedError,
    LeadOutsideUSAError,
    enrich_lead,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Renaissance Lead Scorer",
    page_icon="🏆",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------
# Lightweight access control for the Streamlit Community Cloud free tier
# (which doesn't support per-email allowlists). The shared password lives
# in st.secrets["APP_PASSWORD"] (configured in the Streamlit Cloud dashboard
# or in local .env via app_secrets.get_secret).
#
# Once a session passes the gate, st.session_state["authenticated"] = True
# and the gate is skipped on subsequent reruns within that session. Closing
# the browser / starting a new session requires re-entering the password.
def _check_password() -> bool:
    """
    Returns True if the user has already authenticated this session, or
    if they just submitted the correct password. Returns False otherwise
    and renders the password form.
    """
    if st.session_state.get("authenticated"):
        return True

    from app_secrets import get_secret
    expected = get_secret("APP_PASSWORD")
    if not expected:
        # If APP_PASSWORD isn't configured at all, fail closed rather than
        # silently letting everyone in. Surface the misconfiguration to the
        # operator so they can set the secret.
        st.error(
            "🔒 App is not configured for access. "
            "Set APP_PASSWORD in Streamlit secrets (or .env locally)."
        )
        return False

    st.title("🏆 Renaissance Lead Scorer")
    st.caption("Internal tool — please sign in to continue.")
    pwd = st.text_input(
        "Password",
        type="password",
        key="_pwd_input",
        placeholder="Enter shared team password",
    )
    if st.button("Sign in", type="primary"):
        if pwd == expected:
            st.session_state["authenticated"] = True
            # Clear the input from session_state so it doesn't persist.
            st.session_state.pop("_pwd_input", None)
            st.rerun()
        else:
            st.error("Wrong password. Try again.")
    return False


if not _check_password():
    st.stop()
# ---------------------------------------------------------------------------


# Load scoring tables once per session. Cached so reruns don't re-read disk.
@st.cache_resource
def load_scoring_tables() -> dict:
    return scoring.load_tables("scoring_tables.json")


def pick_partner(percentile: float, tables: dict) -> str:
    """Map a percentile (0-100, lower = better) to a partner using the
    routing rules in scoring_tables.json."""
    for rule in tables["partner_routing"]:
        if percentile <= rule["max_percentile"]:
            return rule["partner"]
    return tables["partner_routing"][-1]["partner"]   # fallback: last rule


# ---------------------------------------------------------------------------
# Score Lead page
# ---------------------------------------------------------------------------

def render_score_lead_page() -> None:
    st.title("🏆 Score Lead")
    st.caption("Fill in the lead's details. The system enriches, scores, and tells you which partner to send them to.")

    # Form fields (kept in session_state so they persist across spinner reruns)
    with st.form(key="score_form", clear_on_submit=False):
        st.subheader("📝 Lead details")

        col1, col2 = st.columns(2)
        with col1:
            company    = st.text_input("Company *", key="company")
            email      = st.text_input("Email *", key="email")
            first_name = st.text_input("First name (first OR last required)", key="first_name")
            last_name  = st.text_input("Last name (first OR last required)", key="last_name")
            phone      = st.text_input("Phone (optional)", key="phone")
        with col2:
            website    = st.text_input("Website (optional)", key="website",
                                       help="If empty, we'll try to find it from the company name")
            advisor    = st.text_input("Advisor (your name) *", key="advisor")
            reply_date = st.date_input(
                "Reply date (UTC) *", key="reply_date",
                format="YYYY-MM-DD",
                value=None,
                help="Date the prospect replied (UTC)",
            )
            # Two numeric fields so the advisor types digits only — no need
            # to remember the ':' separator.
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                reply_hour = st.number_input(
                    "Reply hour (UTC) *", key="reply_hour",
                    min_value=0, max_value=23, step=1, value=None,
                    placeholder="0-23",
                )
            with tcol2:
                reply_minute = st.number_input(
                    "Reply minute (UTC) *", key="reply_minute",
                    min_value=0, max_value=59, step=1, value=None,
                    placeholder="0-59",
                )

        reply_text = st.text_area(
            "First reply text *", key="reply_text", height=150,
            help="Paste the full reply the prospect sent",
        )

        submitted = st.form_submit_button("🚀 SCORE & SAVE LEAD", type="primary",
                                          use_container_width=True)

    if not submitted:
        return

    # -------- Validation --------
    # First name and last name: at least one of the two must be provided.
    # Other mandatory fields: company, email, advisor, date, hour, minute, reply.
    missing = [name for name, val in [
        ("Company", company), ("Email", email),
        ("Advisor", advisor), ("Reply date", reply_date),
        ("First reply text", reply_text),
    ] if not val]

    if not first_name and not last_name:
        missing.append("First name OR Last name")

    if reply_hour is None:   missing.append("Reply hour")
    if reply_minute is None: missing.append("Reply minute")

    if missing:
        st.error(f"Please fill in these required fields: {', '.join(missing)}")
        return

    # Compose the UTC timestamp from the date + hour + minute inputs.
    from datetime import time as _time
    reply_ts_utc = datetime.combine(
        reply_date,
        _time(int(reply_hour), int(reply_minute), 0),
    ).replace(tzinfo=timezone.utc).isoformat()

    # -------- Run the pipeline --------
    with st.spinner("Scoring lead… this can take around 25 seconds"):
        try:
            result = _score_and_save(
                company=company.strip(),
                website=website.strip() or None,
                email=email.strip().lower(),
                first_name=first_name.strip(),
                last_name=last_name.strip(),
                phone=phone.strip(),
                advisor=advisor.strip(),
                reply_ts_utc=reply_ts_utc,
                reply_text=reply_text,
            )
        except LeadOutsideUSAError as e:
            st.error(f"❌ {e}\n\nLead NOT saved.")
            return
        except AllApisFailedError as e:
            st.error(f"❌ All external APIs failed. Lead NOT saved.\n\n{e}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error: {type(e).__name__}: {e}\n\nLead NOT saved.")
            return

    # -------- Show score, percentile, and partner assignment --------
    score_raw  = result.get("score_raw")
    percentile = result.get("percentile")
    partner    = result.get("partner")

    st.success("✅ Lead saved")
    score_txt      = f"{score_raw:.1f}" if score_raw is not None else "—"
    percentile_txt = f"top {percentile:.2f}%" if percentile is not None else "—"
    st.markdown(
        f"**Score:** {score_txt}  \n"
        f"**Percentile:** {percentile_txt}  \n"
        f"**Send to:** {partner or '—'}"
    )


def _score_and_save(
    *,
    company: str,
    website: str | None,
    email: str,
    first_name: str,
    last_name: str,
    phone: str,
    advisor: str,
    reply_ts_utc: str,
    reply_text: str,
) -> dict:
    """
    Orchestrate: enrichment -> reply_features -> scoring -> percentile ->
    partner -> insert to Supabase. Returns a small dict with 'partner' at
    minimum (that's what the UI shows).
    """
    tables = load_scoring_tables()

    # 1) Enrich (runs all external APIs)
    enriched = asyncio.run(enrich_lead(
        company=company, email=email, first_name=first_name,
        last_name=last_name, website=website,
    ))

    # 2) Classify the reply text via regex
    reply_buckets = reply_features.process_reply(reply_text) or {}

    # 3) Build the scoring features dict
    feats = enriched.to_scoring_features()
    feats["reply_timestamp_utc"] = reply_ts_utc
    feats.update(reply_buckets)

    # 4) Score
    scored = scoring.compute_score(feats, tables)

    # Grab time_bucket from the scoring breakdown (same value used for the
    # TimeOfReply lookup) and derive local_hour from the reply timestamp
    # converted to the state's local timezone.
    time_bucket = scored["buckets"].get("TimeOfReply")
    try:
        local_hour = scoring.to_local(reply_ts_utc, enriched.state).strftime("%H:%M:%S")
    except Exception:
        local_hour = None

    # 5) Live percentile from Supabase
    percentile = db.get_percentile(scored["score_raw"])

    # 6) Partner routing
    partner = pick_partner(percentile, tables)

    # 7) Build the row to insert. Matches scoring.lead_scores schema.
    row = {
        # Identity & operational
        "company":     company,
        "website":     enriched.resolved_website,
        "email":       email,
        "first_name":  first_name,
        "last_name":   last_name,
        "phone_number": phone,
        "date":          reply_ts_utc[:10],       # reply date
        "reply_timestamp": reply_ts_utc,
        "first_reply":     reply_text,
        "local_hour":    local_hour,
        "time_bucket":   time_bucket,
        # meeting_datetime stays NULL — set via 'Update Meeting' page later
        # Enrichment features
        "city":              enriched.city,
        "state":             enriched.state,
        "country":           enriched.country,
        "industry":          enriched.industry,
        "years_in_business": enriched.years_in_business,
        "employees":         str(enriched.employees) if enriched.employees is not None else None,
        "job_title":         enriched.job_title,
        "num_locations":     enriched.num_locations,
        # Digital presence
        "has_website":    enriched.has_website,
        "has_gmb":        enriched.has_gmb,
        "has_linkedin":   enriched.has_linkedin,
        "has_facebook":   enriched.has_facebook,
        "has_instagram":  enriched.has_instagram,
        "has_trustpilot": enriched.has_trustpilot,
        "digital_presence_score": int(round(scored["digital_presence_score"])),
        # Score & routing
        "score_raw":  scored["score_raw"],
        "percentile": percentile,
        "partner":    partner,
        # partner_sent_to defaults to the recommended partner; can be overridden
        # via the 'Update Meeting' page if the advisor routes differently.
        "partner_sent_to": partner,
        # NOTE: funded/funded_amount/funded_at/fast_fund/high_value/days_to_fund
        # used to live on this table from the original 22k historical import.
        # They are NOT inserted from the MVP — funding outcomes now live in
        # scoring.funded_events (a separate table, populated when partners
        # report funded deals). Use the VIEW scoring.lead_score_with_funded
        # if you need the funded flag joined per lead.
    }

    # 8) Insert into Supabase
    saved = db.insert_lead(row)

    return {
        "partner":   partner,
        "lead_id":   saved.get("id"),
        "score_raw": scored["score_raw"],
        "percentile": percentile,
    }


# ---------------------------------------------------------------------------
# Update Meeting page
# ---------------------------------------------------------------------------

# Short codes -> IANA zones, pinned to the most-populated US city in each zone.
# Daylight Saving Time is handled automatically by zoneinfo at conversion time,
# so we don't need to distinguish "ET" from "EDT" here.
MEETING_TIMEZONES: dict[str, str] = {
    "ET": "America/New_York",
    "CT": "America/Chicago",
    "MT": "America/Denver",
    "PT": "America/Los_Angeles",
}

MEETING_STATUSES = ["Scheduled", "No-show", "Completed", "Cancelled"]

# Ordered list of partners shown in the 'Sent to partner' dropdown.
# Matches scoring_tables.json partner_routing exactly.
PARTNERS_LIST = [
    "NEW PPA #1", "NEW PPA #2", "NEW PPA #3",
    "Llama Loans", "GBC", "BTC", "GoQualifi",
]


def _format_meeting_datetime_in_tz(utc_iso: str | None, tz_code: str) -> str:
    """
    Render a UTC-stored timestamp in the timezone the advisor originally
    used when scheduling (e.g. '2026-04-25 15:00 CT' for a meeting that's
    actually 2026-04-25 20:00 UTC in Supabase).

    Keeps Supabase purely UTC while showing the user the same wall-clock
    time the partner wrote on the invitation.
    """
    if not utc_iso:
        return "—"
    iana = MEETING_TIMEZONES.get(tz_code)
    if iana is None:
        # Unknown TZ — fall back to showing the raw UTC string
        return f"{utc_iso} (UTC)"
    try:
        # Parse the ISO8601 string. Supabase returns 'YYYY-MM-DDTHH:MM:SS+00:00'
        # or similar; Python's fromisoformat handles it natively in 3.11+.
        s = utc_iso.replace("Z", "+00:00") if isinstance(utc_iso, str) else utc_iso
        dt_utc = datetime.fromisoformat(s) if isinstance(s, str) else s
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_local = dt_utc.astimezone(ZoneInfo(iana))
        return f"{dt_local.strftime('%Y-%m-%d %H:%M')} {tz_code}"
    except Exception:
        return f"{utc_iso} ({tz_code})"


def render_update_meeting_page() -> None:
    st.title("📅 Update Meeting")
    st.caption("Look up a scored lead, then schedule / reschedule the meeting with the partner.")

    # If the advisor arrived here by clicking a pending-meeting shortcut in
    # the sidebar, auto-run the lookup. We DO NOT write to the text_input's
    # session_state key here (that raises StreamlitAPIException on the next
    # widget instantiation). Instead we just fetch the lead and store it;
    # the text field is left empty but the lead appears below it.
    jump_email = st.session_state.pop("jump_to_email", None)
    if jump_email:
        try:
            found = db.find_lead_by_email(jump_email)
            if found is not None:
                st.session_state["found_lead"] = found
                st.info(f"Loaded lead: **{found.get('company')}** (`{jump_email}`)")
            else:
                st.warning(f"No lead found with email {jump_email!r}.")
                st.session_state.pop("found_lead", None)
        except Exception as e:
            st.error(f"❌ Supabase error while loading {jump_email}: {type(e).__name__}: {e}")

    # -------- Lookup --------
    email = st.text_input("Lead email *", key="update_email",
                          placeholder="john@acmeplumbing.com")
    if st.button("🔎 Find lead"):
        email_clean = email.strip()
        if not email_clean:
            st.error("Enter an email first.")
        elif "@" not in email_clean or "." not in email_clean.split("@")[-1]:
            st.error(
                f"{email_clean!r} doesn't look like a valid email address. "
                f"Use the full email (e.g. 'john@acmeplumbing.com'), not just the domain."
            )
        else:
            try:
                lead = db.find_lead_by_email(email_clean)
            except Exception as e:
                st.error(f"❌ Supabase error: {type(e).__name__}: {e}")
                return
            if lead is None:
                st.warning(f"No lead found with email {email_clean!r}.")
                st.session_state.pop("found_lead", None)
            else:
                st.session_state["found_lead"] = lead
            # A fresh search resets any 'just scheduled' confirmation view
            st.session_state.pop("just_scheduled", None)
            st.session_state.pop("show_new_meeting_form", None)

    lead = st.session_state.get("found_lead")
    if not lead:
        return

    # -------- Lead summary --------
    st.markdown("---")
    st.subheader(f"Lead: {lead.get('company')}")

    partner         = lead.get("partner") or "—"
    partner_sent_to = lead.get("partner_sent_to") or "—"
    score_raw       = lead.get("score_raw")
    percentile      = lead.get("percentile")
    scored_at       = (lead.get("scored_at") or "")[:10] or "—"

    score_txt      = f"{score_raw:.1f}" if score_raw is not None else "—"
    percentile_txt = f"top {percentile:.2f}%" if percentile is not None else "—"

    # Show 'Sent to' only if it differs from the recommendation, to keep
    # the header tight in the common case where it matches.
    if partner_sent_to and partner_sent_to != partner and partner_sent_to != "—":
        partner_line = f"**Recommended:** {partner}  ·  **Sent to:** {partner_sent_to}"
    else:
        partner_line = f"**Partner:** {partner}"

    st.markdown(
        f"{partner_line}  \n"
        f"**Score:** {score_txt}  ·  **Percentile:** {percentile_txt}  \n"
        f"**Scored at:** {scored_at}"
    )

    # -------- Delete lead (soft) --------
    # Destructive action hidden behind an expander + confirm checkbox so it
    # can't be triggered by a single click. Uses soft delete — the row stays
    # in Supabase with deleted_at set, but disappears from all app views.
    #
    # Eligibility (mirrors db.soft_delete_lead): only allowed when the lead
    # has no meeting yet, or its meeting is currently 'Scheduled' (which is
    # always future-dated — past Scheduleds get auto-completed elsewhere).
    # Any other status (Completed / No-show / Cancelled / Rescheduled /
    # Lead deleted) means real-world history exists; the advisor must
    # update that meeting's status manually before deleting. This protects
    # against accidentally erasing leads that produced actual events.
    #
    # All deletes are framed as advisor errors (wrong data, duplicate, test
    # entry, etc.) — a customer canceling is NOT a delete, that goes through
    # meeting_status='Cancelled' instead. This keeps the analytics clean:
    # delete_reason tells us where WE are losing accuracy, not customer churn.
    DELETE_REASONS = [
        "— Select a reason —",          # placeholder, blocks the button
        "Wrong data entered",
        "Duplicate scoring",
        "Test entry / not a real lead",
        "Other",
    ]

    # Determine eligibility up front so the UI can explain *why* the delete
    # button is unavailable instead of silently disabling it.
    _ms = lead.get("meeting_status")
    delete_eligible = (_ms is None) or (str(_ms).lower() == "scheduled")

    with st.expander("⚠️ Delete this lead", expanded=False):
        if not delete_eligible:
            st.warning(
                f"This lead can't be deleted because its meeting is "
                f"**{_ms}**. Delete is only available for leads with no "
                f"meeting or a meeting still scheduled in the future. "
                f"If you really need to remove this lead, update the "
                f"meeting status appropriately first."
            )
        else:
            st.caption(
                "Soft-delete: the lead is hidden from the app but stays in Supabase "
                "for audit. Useful if you scored the wrong lead by mistake."
            )
            confirm = st.checkbox(
                f"I confirm I want to delete **{lead.get('company')}** (`{lead.get('email')}`)",
                key="delete_confirm",
            )
            delete_reason = st.selectbox(
                "Reason for deletion *",
                DELETE_REASONS,
                key="delete_reason",
            )
            delete_notes = st.text_input(
                "Additional details (optional)",
                key="delete_notes",
                placeholder="e.g. Typed wrong email, duplicate of lead #12345",
            )
            # The button is enabled only when BOTH the confirm checkbox is ticked
            # AND a real reason is selected (not the placeholder).
            reason_chosen = delete_reason and delete_reason != DELETE_REASONS[0]
            if st.button(
                "🗑️ Delete lead",
                type="secondary",
                disabled=not (confirm and reason_chosen),
            ):
                try:
                    db.soft_delete_lead(
                        lead["id"],
                        reason=delete_reason,
                        notes=delete_notes or None,
                    )
                    st.success(f"✅ Lead deleted. It will no longer appear in the app.")
                    # Clear all session state for this lead
                    for k in ("found_lead", "just_scheduled", "show_new_meeting_form",
                              "delete_confirm", "delete_reason", "delete_notes"):
                        st.session_state.pop(k, None)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Delete failed: {type(e).__name__}: {e}")

    # -------- Meeting history timeline --------
    try:
        history = db.list_meetings_for_lead(lead["id"])
    except Exception as e:
        history = []
        st.caption(f"(Could not load meeting history: {e})")

    current_status = lead.get("meeting_status")
    has_meeting = bool(history)

    if has_meeting:
        with st.expander("📜 Meeting history", expanded=True):
            for i, m in enumerate(history, start=1):
                tz_code = m.get("meeting_timezone") or "UTC"
                stat    = m.get("meeting_status") or "—"
                sent_to = m.get("partner_sent_to")
                dt_display = _format_meeting_datetime_in_tz(
                    m.get("meeting_datetime"), tz_code,
                )
                icon = {
                    "Scheduled": "🕐", "Completed": "✅",
                    "No-show":   "🚫", "Cancelled": "❌",
                }.get(stat, "•")
                # Only show 'sent to' if the history row has it populated
                # (old rows predating this feature will have NULL)
                sent_to_txt = f" · sent to **{sent_to}**" if sent_to else ""
                st.markdown(
                    f"{icon} **Meeting #{i}** — {dt_display}{sent_to_txt} · "
                    f"status: **{stat}**"
                )

    # -------- Action chooser --------
    st.markdown("---")

    # If the advisor JUST scheduled/rescheduled in this visit, show a clean
    # confirmation view instead of the action chooser. They can click 'Find
    # lead' again to re-open the full options.
    if st.session_state.get("just_scheduled"):
        st.success("✅ Meeting saved. Come back later via 'Update Meeting' to change status or reschedule.")
        return

    if not has_meeting:
        # First-ever scheduling
        st.info("No meeting scheduled yet. Use the form below to schedule the first one.")
        _render_schedule_form(lead, is_reschedule=False)
        return

    if current_status == "Scheduled":
        # Meeting is open — advisor can update status OR reschedule
        action = st.radio(
            "What would you like to do?",
            ["Update status of current meeting", "Reschedule to a new date/time"],
            key="update_action",
            horizontal=True,
        )
        if action.startswith("Update"):
            _render_status_update_form(lead)
        else:
            _render_schedule_form(lead, is_reschedule=True)
        return

    # Meeting already closed (Completed / No-show / Cancelled)
    # Advisors still need to be able to correct the status (e.g. auto-Completed
    # was wrong because the lead actually no-showed). Show BOTH options:
    #   1) Update status (change to any other valid status)
    #   2) Schedule a new meeting (for reschedule scenarios)
    status_msg = {
        "Completed": "✅ This meeting is marked as **Completed**.",
        "No-show":   "🚫 This meeting was marked as **No-show**.",
        "Cancelled": "❌ This meeting was marked as **Cancelled**.",
    }.get(current_status, f"Current status: **{current_status}**.")
    st.info(status_msg)

    action = st.radio(
        "What would you like to do?",
        ["Change the status", "Schedule a new meeting"],
        key="closed_action",
        horizontal=True,
    )
    if action == "Change the status":
        _render_status_update_form(lead)
    else:
        _render_schedule_form(lead, is_reschedule=True)


def _render_status_update_form(lead: dict) -> None:
    """Form for marking a Scheduled meeting as Completed / No-show / Cancelled."""
    current_status = lead.get("meeting_status") or "Scheduled"

    new_status = st.selectbox(
        "New status *",
        options=[s for s in MEETING_STATUSES if s != current_status],
        key="new_meeting_status",
    )
    if st.button("💾 Update status", type="primary"):
        try:
            db.update_meeting_status(lead["id"], new_status)
            st.success(f"✅ Meeting status updated to **{new_status}**")
            st.session_state["found_lead"] = db.find_lead_by_email(lead["email"])
            st.session_state["just_scheduled"] = True   # reuse the 'clean view' flag
            st.session_state.pop("show_new_meeting_form", None)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Update failed: {type(e).__name__}: {e}")


def _render_schedule_form(lead: dict, *, is_reschedule: bool) -> None:
    """Form to schedule a new meeting (first or rescheduled) with numeric
    date/time fields and a timezone dropdown."""
    verb = "Reschedule" if is_reschedule else "Schedule"
    st.markdown(f"### {verb} meeting")
    st.caption("Enter the time as it appears on the partner's invitation (e.g. '15:00 ET').")

    # Date: YYYY / MM / DD as three numeric fields
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        m_year  = st.number_input("Year *",  min_value=2025, max_value=2030,
                                  step=1, value=None, placeholder="YYYY",
                                  key=f"m_year_{is_reschedule}")
    with dc2:
        m_month = st.number_input("Month *", min_value=1, max_value=12,
                                  step=1, value=None, placeholder="MM",
                                  key=f"m_month_{is_reschedule}")
    with dc3:
        m_day   = st.number_input("Day *",   min_value=1, max_value=31,
                                  step=1, value=None, placeholder="DD",
                                  key=f"m_day_{is_reschedule}")

    # Time: HH / MM + TZ
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        m_hour   = st.number_input("Hour *",   min_value=0, max_value=23,
                                   step=1, value=None, placeholder="0-23",
                                   key=f"m_hour_{is_reschedule}")
    with tc2:
        m_minute = st.number_input("Minute *", min_value=0, max_value=59,
                                   step=1, value=None, placeholder="0-59",
                                   key=f"m_minute_{is_reschedule}")
    with tc3:
        m_tz = st.selectbox(
            "Timezone *",
            options=list(MEETING_TIMEZONES.keys()),
            format_func=lambda k: {
                "ET": "ET · Eastern",
                "CT": "CT · Central",
                "MT": "MT · Mountain",
                "PT": "PT · Pacific",
            }[k],
            key=f"m_tz_{is_reschedule}",
        )

    # Partner this lead is actually being sent to. Defaults to the recommended
    # partner from the scoring model, but the advisor can override if the
    # actual routing differs (partner unavailable, capacity, relationship, etc.)
    recommended = lead.get("partner") or PARTNERS_LIST[0]
    current_sent_to = lead.get("partner_sent_to") or recommended
    try:
        default_idx = PARTNERS_LIST.index(current_sent_to)
    except ValueError:
        default_idx = PARTNERS_LIST.index(recommended) if recommended in PARTNERS_LIST else 0

    sent_to = st.selectbox(
        f"Sent to partner *  (recommended: **{recommended}**)",
        options=PARTNERS_LIST,
        index=default_idx,
        key=f"m_sent_to_{is_reschedule}",
        help="Partner who actually got this lead. Defaults to the scoring recommendation.",
    )

    if st.button(f"💾 {verb} meeting", type="primary"):
        # Validate all fields present
        missing = [name for name, v in [
            ("Year", m_year), ("Month", m_month), ("Day", m_day),
            ("Hour", m_hour), ("Minute", m_minute),
        ] if v is None]
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
            return

        # Compose the datetime in the chosen IANA zone, then convert to UTC
        try:
            tz_iana = MEETING_TIMEZONES[m_tz]
            local_dt = datetime(
                int(m_year), int(m_month), int(m_day),
                int(m_hour), int(m_minute), 0,
                tzinfo=ZoneInfo(tz_iana),
            )
        except ValueError as e:
            st.error(f"❌ Invalid date/time: {e}")
            return
        utc_dt = local_dt.astimezone(timezone.utc).isoformat()

        # Reject meetings scheduled in the past (common user error). Compare
        # in UTC since that's the canonical representation.
        now_utc = datetime.now(timezone.utc)
        if local_dt.astimezone(timezone.utc) < now_utc:
            st.error(
                f"❌ Can't schedule a meeting in the past. "
                f"The time you entered is {local_dt.strftime('%Y-%m-%d %H:%M')} {m_tz}, "
                f"which is before the current time."
            )
            return

        # Call the right backend function
        try:
            if is_reschedule:
                db.reschedule_meeting(lead["id"], utc_dt, m_tz, sent_to)
            else:
                db.create_meeting(lead["id"], utc_dt, m_tz, sent_to)
        except Exception as e:
            st.error(f"❌ Failed to save meeting: {type(e).__name__}: {e}")
            return

        # Success message shows the local time as entered by the advisor
        local_str = local_dt.strftime("%Y-%m-%d %H:%M")
        st.success(
            f"✅ Meeting {'rescheduled' if is_reschedule else 'scheduled'} for "
            f"**{local_str} {m_tz}**."
        )
        # Refresh cached lead so the header + history reflect new state.
        # Also mark 'just_scheduled' so the next render shows a clean
        # confirmation view (no action chooser, no forms) until the advisor
        # explicitly searches for the lead again.
        st.session_state["found_lead"] = db.find_lead_by_email(lead["email"])
        st.session_state["just_scheduled"] = True
        st.session_state.pop("show_new_meeting_form", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar router
# ---------------------------------------------------------------------------

PAGES = {
    "Score Lead":     render_score_lead_page,
    "Update Meeting": render_update_meeting_page,
}
PAGE_NAMES = list(PAGES.keys())

st.sidebar.title("Renaissance Lead Scorer")

# If a pending-button click on the previous run requested a page switch,
# set the radio's session_state value BEFORE the widget is instantiated.
# Writing to session_state BEFORE widget creation on the same run is legal
# (what's forbidden is writing AFTER). The widget will then use this value
# as its selected state, and subsequent reruns will preserve it naturally.
_requested_page = st.session_state.pop("_goto_page", None)
if _requested_page in PAGE_NAMES:
    st.session_state["page_selector"] = _requested_page

page = st.sidebar.radio("Go to", PAGE_NAMES, key="page_selector")
st.sidebar.markdown("---")

# Auto-complete past-due Scheduled meetings — runs silently in background
# (no UI notification, as requested). Sweep happens once per session to avoid
# re-running on every rerun.
if not st.session_state.get("_auto_complete_ran"):
    try:
        n_completed = db.auto_complete_past_meetings()
        st.session_state["_auto_complete_ran"] = True
        # TEMPORARY DEBUG — show in sidebar how many meetings got auto-completed
        # this pass. Remove once we're confident the sweep is working.
        st.sidebar.caption(f"🔄 Auto-complete: {n_completed} meeting(s) updated")
    except Exception as e:
        # TEMPORARY DEBUG — surface the error instead of swallowing it.
        st.sidebar.error(f"Auto-complete failed: {type(e).__name__}: {e}")
        st.session_state["_auto_complete_ran"] = True

# Pending meetings alert — shows only leads scored from today onward that
# have NEVER been assigned a meeting (meeting_status IS NULL). A lead leaves
# the pending list as soon as its first meeting is scheduled, and doesn't
# come back even if the meeting is later cancelled or rescheduled.
from datetime import date as _date
_start_of_today = datetime.combine(_date.today(), datetime.min.time()).replace(
    tzinfo=timezone.utc
).isoformat()

try:
    pending = db.list_leads_without_meeting(limit=50, scored_since=_start_of_today)
except Exception as e:
    pending = None
    st.sidebar.caption(f"(Could not load pending meetings: {e})")

if pending is not None:
    if pending:
        st.sidebar.warning(f"⚠️ {len(pending)} pending meeting{'s' if len(pending) != 1 else ''} today")
        with st.sidebar.expander("View pending", expanded=False):
            for lead in pending[:10]:
                company = lead.get("company") or "—"
                email   = lead.get("email") or "—"
                partner = lead.get("partner") or "—"
                scored  = (lead.get("scored_at") or "")[:16]
                # One button per pending lead. Clicking it:
                #   1) Stores the email in session_state so the Update
                #      Meeting page pre-fills + auto-searches.
                #   2) Requests the page switch via '_goto_page' — the
                #      radio above will honor it on the NEXT render (we
                #      can't write to 'page_selector' after the widget
                #      is instantiated, which it already is at this point).
                #   3) st.rerun() triggers that next render immediately.
                btn_label = f"{company}\n{email}\n→ {partner}"
                if st.button(btn_label, key=f"pending_btn_{lead['id']}", use_container_width=True):
                    st.session_state["jump_to_email"] = email
                    st.session_state["_goto_page"]    = "Update Meeting"
                    st.session_state.pop("just_scheduled", None)
                    st.session_state.pop("show_new_meeting_form", None)
                    st.rerun()
                st.caption(f"scored {scored}")
            if len(pending) > 10:
                st.caption(f"…and {len(pending) - 10} more")
    else:
        st.sidebar.success("✅ No pending meetings today")

st.sidebar.markdown("---")
st.sidebar.caption("All leads are saved to Supabase. The advisor only sees the assigned partner.")

PAGES[page]()