"""
scoring.py — Compute a weighted composite score for a Renaissance lead.

This module contains NO external API calls. It takes already-enriched feature
values (from enrichment.py in production, or from the CSV + Claude classifier
in the historical re-score), applies bucketizers, looks up per-bucket scores
in scoring_tables.json, and returns:

    score_raw = base_score (weighted sum, 50-100) + digital_presence (0-6)

The JSON file 'scoring_tables.json' is the single source of truth for weights,
per-bucket scores, digital presence signals, and partner routing.

Design:
  - Every bucketizer is a pure function. Feed it garbage -> returns "Unknown"
    (or closest safe bucket). Never raises on bad data; that would block the UI.
  - Missing timestamp -> Month/Day/TimeOfReply fall through to UNKNOWN_SCORE
    (no bucket match). Small score loss, no crash.
  - All string comparisons are case-insensitive where it matters.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

UNKNOWN_SCORE = 50             # Fallback when no bucket matches
DEFAULT_TZ = "America/New_York"  # When state is missing / not recognized

# State name -> IANA timezone. Multi-timezone states use the largest-metro TZ
# (Florida/Kentucky/Indiana -> ET; Tennessee/Texas -> CT; Oregon -> PT).
STATE_TO_TZ: dict[str, str] = {
    "Alabama": "America/Chicago",        "Alaska": "America/Anchorage",
    "Arizona": "America/Phoenix",        "Arkansas": "America/Chicago",
    "California": "America/Los_Angeles", "Colorado": "America/Denver",
    "Connecticut": "America/New_York",   "Delaware": "America/New_York",
    "District of Columbia": "America/New_York",
    "Florida": "America/New_York",       "Georgia": "America/New_York",
    "Hawaii": "Pacific/Honolulu",        "Idaho": "America/Boise",
    "Illinois": "America/Chicago",
    "Indiana": "America/Indiana/Indianapolis",
    "Iowa": "America/Chicago",           "Kansas": "America/Chicago",
    "Kentucky": "America/New_York",      "Louisiana": "America/Chicago",
    "Maine": "America/New_York",         "Maryland": "America/New_York",
    "Massachusetts": "America/New_York", "Michigan": "America/Detroit",
    "Minnesota": "America/Chicago",      "Mississippi": "America/Chicago",
    "Missouri": "America/Chicago",       "Montana": "America/Denver",
    "Nebraska": "America/Chicago",       "Nevada": "America/Los_Angeles",
    "New Hampshire": "America/New_York", "New Jersey": "America/New_York",
    "New Mexico": "America/Denver",      "New York": "America/New_York",
    "North Carolina": "America/New_York","North Dakota": "America/Chicago",
    "Ohio": "America/New_York",          "Oklahoma": "America/Chicago",
    "Oregon": "America/Los_Angeles",     "Pennsylvania": "America/New_York",
    "Rhode Island": "America/New_York",  "South Carolina": "America/New_York",
    "South Dakota": "America/Chicago",   "Tennessee": "America/Chicago",
    "Texas": "America/Chicago",          "Utah": "America/Denver",
    "Vermont": "America/New_York",       "Virginia": "America/New_York",
    "Washington": "America/Los_Angeles", "West Virginia": "America/New_York",
    "Wisconsin": "America/Chicago",      "Wyoming": "America/Denver",
}

# Abbreviations -> full names (for when enrichment returns 'CA' etc.)
STATE_ABBREV: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# ----------------------------------------------------------------------------
# Table loader
# ----------------------------------------------------------------------------

def load_tables(path: str | Path = "scoring_tables.json") -> dict[str, Any]:
    """Load the scoring tables JSON into memory."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Bucketizers — Company features
# ----------------------------------------------------------------------------

def bucketize_tib(years: float | int | str | None) -> str:
    """Time in Business bucket. Uses floor() on fractional years (per Thomas)."""
    if years is None or (isinstance(years, float) and math.isnan(years)):
        return "Unknown"
    try:
        y = int(math.floor(float(years)))
    except (TypeError, ValueError):
        return "Unknown"
    if y < 0:           return "Unknown"
    if y <= 2:          return "0-2 yrs"
    if y <= 5:          return "3-5 yrs"
    if y <= 10:         return "6-10 yrs"
    if y <= 20:         return "11-20 yrs"
    return "21+ yrs"


def bucketize_employees(value: int | str | None) -> str:
    """
    Accepts either an int (exact count) or a string.  Apollo sometimes returns
    already-bucketed strings like '11-50' or '2-10'; we accept those directly.
    Out-of-distribution (>500) falls to Unknown per Thomas's confirmed rule.
    """
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        v = value.strip()
        if v in ("1", "2-10", "11-50", "51-200", "201-500"):
            return v
        try:
            n = int(v)
        except ValueError:
            return "Unknown"
    else:
        try:
            n = int(value)
        except (TypeError, ValueError):
            return "Unknown"
    if n <= 0:    return "Unknown"
    if n == 1:    return "1"
    if n <= 10:   return "2-10"
    if n <= 50:   return "11-50"
    if n <= 200:  return "51-200"
    if n <= 500:  return "201-500"
    return "Unknown"   # >500 OOD


def bucketize_locations(n: int | str | None) -> str:
    """Number of locations bucket."""
    if n is None:
        return "Unknown"
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "Unknown"
    if n <= 0:    return "Unknown"
    if n == 1:    return "1 location"
    if n <= 3:    return "2-3 locations"
    return "4+ locations"


# Seniority priority — first pattern that matches wins.
# Order deliberate: Owner outranks CEO/President (confirmed with Thomas).
_SENIORITY_PRIORITY: list[tuple[str, re.Pattern[str]]] = [
    ("Owner",         re.compile(r"\b(owner|sole proprietor|self[-\s]?employed|proprietor)\b", re.I)),
    ("CEO/President", re.compile(r"\b(ceo|chief executive officer|president)\b", re.I)),
    ("Founder",       re.compile(r"\b(co[-\s]?founder|founder)\b", re.I)),
    ("Partner",       re.compile(r"\b(managing partner|partner|principal)\b", re.I)),
    ("C-Suite",       re.compile(r"\b(cfo|coo|cto|chro|cmo|cro|cao|cpo|chief\s+\w+\s+officer)\b", re.I)),
    ("Management",    re.compile(r"\b(vp|vice president|director|manager|head of|gm|general manager|controller)\b", re.I)),
]

def bucketize_seniority(job_title: str | None) -> str:
    """Map a free-text job title to one of the Excel buckets."""
    if not job_title:
        return "Unknown"
    t = str(job_title).strip()
    if not t:
        return "Unknown"
    for bucket, pattern in _SENIORITY_PRIORITY:
        if pattern.search(t):
            return bucket
    return "Other"  # has a title but doesn't match any pattern


# ----------------------------------------------------------------------------
# Bucketizers — Timestamps (DST-aware via zoneinfo)
# ----------------------------------------------------------------------------

def _parse_utc(ts: str | datetime) -> datetime:
    """Parse ISO timestamp into a UTC-aware datetime."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        s = str(ts).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt


def to_local(ts_utc: str | datetime, state: str | None) -> datetime:
    """UTC timestamp -> local datetime for the given state."""
    dt = _parse_utc(ts_utc)
    norm = _normalize_state(state) if state else ""
    tz_name = STATE_TO_TZ.get(norm, DEFAULT_TZ)
    return dt.astimezone(ZoneInfo(tz_name))


def bucketize_time_of_reply(ts_utc: str | datetime, state: str | None) -> str:
    """6 buckets of local hour of reply."""
    local = to_local(ts_utc, state)
    h = local.hour
    if 6 <= h < 9:    return "1_Early (6-9am)"
    if 9 <= h < 12:   return "2_Morning (9am-12pm)"
    if 12 <= h < 17:  return "3_Afternoon (12-5pm)"
    if 17 <= h < 20:  return "4_Late Afternoon (5-8pm)"
    if 20 <= h < 24:  return "5_Night (8pm-12am)"
    return "6_Late Night (12-6am)"   # 0 <= h < 6


def bucketize_month_contacted(ts_utc: str | datetime, state: str | None) -> str:
    return to_local(ts_utc, state).strftime("%B")


def bucketize_day_of_week(ts_utc: str | datetime, state: str | None) -> str:
    return to_local(ts_utc, state).strftime("%A")


# ----------------------------------------------------------------------------
# Bucketizers — Reply text
# ----------------------------------------------------------------------------
# NOTE: The 5 reply-text buckets (ReplyLength, Professionalism, Cleanliness,
# Urgency, Intent) are NOT produced here. They come from reply_features.py,
# which holds the regex logic copied 1:1 from the chat that scored the 22k
# historical leads. Keeping that logic in its own module guarantees zero
# drift between the Supabase baseline and leads scored live in the MVP.


# ----------------------------------------------------------------------------
# Normalizers
# ----------------------------------------------------------------------------

def _normalize_state(state: str | None) -> str:
    """Accept full name, abbreviation, any casing. Returns Excel name or ''."""
    if not state:
        return ""
    s = str(state).strip()
    if not s:
        return ""
    up = s.upper()
    if up in STATE_ABBREV:
        return STATE_ABBREV[up]
    for full in STATE_TO_TZ:
        if full.lower() == s.lower():
            return full
    return ""


def _match_industry(industry: str | None, valid_industries: dict[str, int]) -> str:
    """Case-insensitive match to one of the 99 industry buckets; else 'Unknown'."""
    if not industry:
        return "Unknown"
    s = str(industry).strip()
    if s in valid_industries:
        return s
    for name in valid_industries:
        if name.lower() == s.lower():
            return name
    return "Unknown"


# ----------------------------------------------------------------------------
# Digital Presence (additive bonus, 0-6, per-signal weights)
# ----------------------------------------------------------------------------

def _to_bool(v: Any) -> bool:
    if v is None:                 return False
    if isinstance(v, bool):       return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "t", "y")
    return False


def compute_digital_presence(
    has_dict: dict[str, Any],
    signal_weights: dict[str, float],
    max_bonus: float,
) -> float:
    """Weighted sum of 6 digital signals, capped at max_bonus."""
    total = sum(w * _to_bool(has_dict.get(sig)) for sig, w in signal_weights.items())
    return min(total, max_bonus)


# ----------------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------------

def _lookup(tables: dict, variable: str, bucket: str) -> int:
    """Return the per-bucket score; fall back to Unknown (or 50) if missing."""
    table = tables["scores"].get(variable, {})
    if bucket in table:
        return int(table[bucket])
    if "Unknown" in table:
        return int(table["Unknown"])
    return UNKNOWN_SCORE


# ----------------------------------------------------------------------------
# Main scorer
# ----------------------------------------------------------------------------

def compute_score(features: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single lead.  See module docstring for input/output shape.
    Input `features` should contain any subset of:
        years_in_business, industry, employees, state, num_locations, job_title,
        reply_timestamp_utc (or reply_timestamp),
        core_message, professionalism, cleanliness, urgency, intent,
        has_website, has_gmb, has_linkedin, has_facebook,
        has_instagram, has_trustpilot
    Missing values resolve to Unknown (score 50 for that feature).
    """
    weights = tables["weights"]
    valid_industries = tables["scores"]["Industry"]

    state_norm = _normalize_state(features.get("state"))
    reply_ts   = features.get("reply_timestamp_utc") or features.get("reply_timestamp")

    # 1) Bucketize each feature
    buckets: dict[str, str] = {
        "TIB":              bucketize_tib(features.get("years_in_business")),
        "Industry":         _match_industry(features.get("industry"), valid_industries),
        "Employees":        bucketize_employees(features.get("employees")),
        "State":            state_norm or "Unknown",
        "Locations":        bucketize_locations(features.get("num_locations")),
        "Seniority":        bucketize_seniority(features.get("job_title")),
        # Reply-text buckets come already classified from reply_features.py
        "ReplyLength":      features.get("reply_length_bucket") or "Short (30-75)",
        "Professionalism":  features.get("prof_bucket")         or "0 Signals",
        "Cleanliness":      features.get("cleanliness_bucket")  or "Clean Message (no markers)",
        "Urgency":          features.get("urgency_bucket")      or "No Urgency Words",
        "Intent":           features.get("intent_bucket")       or "Other/Unclear",
        # Time-based (derive from reply_timestamp in local TZ)
        "MonthContacted":       bucketize_month_contacted(reply_ts, state_norm) if reply_ts else "",
        "DayOfWeekContacted":   bucketize_day_of_week(reply_ts, state_norm)     if reply_ts else "",
        "TimeOfReply":          bucketize_time_of_reply(reply_ts, state_norm)   if reply_ts else "",
    }

    # 2) Look up per-bucket scores and compute weighted contributions
    breakdown: dict[str, dict[str, Any]] = {}
    base_score = 0.0
    for var, weight in weights.items():
        bucket = buckets[var]
        bucket_score = _lookup(tables, var, bucket)
        contribution = weight * bucket_score
        base_score += contribution
        breakdown[var] = {
            "bucket":       bucket,
            "score":        bucket_score,
            "weight":       weight,
            "contribution": round(contribution, 4),
        }

    # 3) Digital presence bonus (0-6, additive)
    dp_cfg = tables["digital_presence"]
    dp_bonus = compute_digital_presence(
        has_dict={k: features.get(k) for k in dp_cfg["signals"]},
        signal_weights=dp_cfg["signals"],
        max_bonus=dp_cfg["max_bonus"],
    )

    return {
        "score_raw":              round(base_score + dp_bonus, 4),
        "base_score":             round(base_score, 4),
        "digital_presence_score": round(dp_bonus, 4),
        "features":               breakdown,
        "buckets":                buckets,
    }