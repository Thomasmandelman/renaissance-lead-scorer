"""
enrichment.py — Orchestrate all external API calls for a Renaissance lead.

Given the minimal form input (company + optional website + email + names),
calls Gemini 2.5 Flash grounded, Apollo, and Google Places (New) in parallel
(with sequential fallbacks where specified) and returns a fully-enriched
feature dict ready for scoring.compute_score().

Design principles:
  1. A missing data point (e.g. "no LinkedIn page") is NOT an error — it's a
     valid False/None that flows through to the scoring. Only *technical*
     failures (auth, rate limits, API down) get reported as errors.
  2. USA-only. Any country != "United States" aborts with LeadOutsideUSAError.
  3. Parallel where possible (asyncio.gather), sequential only where there's
     a data dependency (find_website must complete before Apollo starts).
  4. Each API call returns (value, ApiCallStatus). The caller composes these
     into the final EnrichmentResult, which carries a per-API diagnostic panel
     that the UI renders.
  5. No SDKs — raw httpx. Fewer moving parts, no version drift, same code on
     Linux and Windows.

Required env vars (in .env via python-dotenv):
  GEMINI_API_KEY, GOOGLE_PLACES_API_KEY, APOLLO_API_KEY

Typical latency:
  ~4-5 s if advisor provided website
  ~6-7 s if we have to find website first via Gemini
  ~8-10 s worst case with fallbacks triggered
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

GEMINI_KEY  = os.environ.get("GEMINI_API_KEY")
PLACES_KEY  = os.environ.get("GOOGLE_PLACES_API_KEY")
APOLLO_KEY  = os.environ.get("APOLLO_API_KEY")

GEMINI_MODEL      = "gemini-2.5-flash"       # grounded + reasoning (industry/TIB/location)
GEMINI_FAST_MODEL = "gemini-2.5-flash-lite"  # grounded + simple enum (digital presence, job title)
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
PLACES_URL   = "https://places.googleapis.com/v1/places:searchText"
APOLLO_ORG_URL    = "https://api.apollo.io/api/v1/organizations/enrich"
APOLLO_PEOPLE_URL = "https://api.apollo.io/api/v1/people/match"

TIMEOUT_GEMINI = 15.0    # grounded calls can be slow
TIMEOUT_APOLLO = 10.0
TIMEOUT_PLACES = 8.0
TIMEOUT_HTTP_CHECK = 3.0

# ----------------------------------------------------------------------------
# Constants: valid US states, personal email domains, invalid website patterns
# ----------------------------------------------------------------------------

US_STATES: list[str] = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
]
US_STATES_SET = set(US_STATES)

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "live.com", "msn.com", "me.com", "mac.com", "comcast.net",
    "protonmail.com", "ymail.com", "gmx.com", "mail.com", "zoho.com",
}

# Domains to reject as "not a real website"
_INVALID_WEBSITE_DOMAINS = {
    "facebook.com", "fb.com", "instagram.com", "linkedin.com",
    "twitter.com", "x.com", "tiktok.com", "youtube.com",
    "wordpress.com", "wixsite.com", "blogspot.com", "weebly.com",
    "squarespace.com", "sites.google.com", "carrd.co", "godaddysites.com",
    "tumblr.com", "medium.com",
}
_VALID_TLDS = {
    "com", "net", "org", "io", "co", "biz", "info",
    "us", "ca", "uk", "au", "nz",
}

# Loaded lazily from scoring_tables.json — the 99 valid industries
_INDUSTRY_ENUM: list[str] | None = None


def _load_industries() -> list[str]:
    """Load the 99 valid industry names from scoring_tables.json, once."""
    global _INDUSTRY_ENUM
    if _INDUSTRY_ENUM is None:
        path = Path(__file__).parent / "scoring_tables.json"
        with open(path, "r", encoding="utf-8") as f:
            tables = json.load(f)
        _INDUSTRY_ENUM = [k for k in tables["scores"]["Industry"] if k != "Unknown"]
    return _INDUSTRY_ENUM


# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------

class EnrichmentError(Exception):
    """Base class for enrichment-level failures that should block scoring."""


class LeadOutsideUSAError(EnrichmentError):
    """Raised when the resolved country is not United States."""

    def __init__(self, country: str | None):
        self.country = country or "unknown"
        super().__init__(f"Lead outside USA (country={self.country}). Scoring cancelled.")


class AllApisFailedError(EnrichmentError):
    """Raised when every external API failed technically — can't score blindly."""


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

StatusLevel = Literal["ok", "no_data", "tech_fail", "skipped"]


@dataclass
class ApiCallStatus:
    """Per-API diagnostic shown in the UI panel after scoring."""
    api: str                      # "Gemini: location", "Apollo: org", etc.
    status: StatusLevel           # see below
    duration_ms: int              # wall-clock time for the call
    note: str | None = None       # human-readable note (e.g. "company not in DB")
    error_type: str | None = None # "rate_limit", "auth", "quota", "timeout", ...
    error_message: str | None = None


# status legend:
#   ok        → API responded with useful data (green)
#   no_data   → API responded but no match for this company (blue ⓘ — normal)
#   tech_fail → API had a technical problem (yellow ⚠ — flagged)
#   skipped   → call was not attempted (e.g. Apollo skipped for personal email)


@dataclass
class EnrichmentResult:
    """All features produced by enrichment, plus per-API diagnostics."""
    # Location
    city: str | None = None
    state: str | None = None
    country: str | None = None
    # Company data
    industry: str | None = None
    years_in_business: int | None = None
    employees: int | str | None = None
    job_title: str | None = None
    num_locations: int = 1
    # Digital presence signals (booleans, never None — default False)
    has_website: bool = False
    has_gmb: bool = False
    has_linkedin: bool = False
    has_facebook: bool = False
    has_instagram: bool = False
    has_trustpilot: bool = False
    # Metadata
    resolved_website: str | None = None   # The website actually used (input or Gemini-found)
    statuses: list[ApiCallStatus] = field(default_factory=list)

    def to_scoring_features(self) -> dict[str, Any]:
        """Produce the dict format expected by scoring.compute_score()."""
        return {
            "industry":          self.industry,
            "years_in_business": self.years_in_business,
            "employees":         self.employees,
            "state":             self.state,
            "num_locations":     self.num_locations,
            "job_title":         self.job_title,
            "has_website":    self.has_website,
            "has_gmb":        self.has_gmb,
            "has_linkedin":   self.has_linkedin,
            "has_facebook":   self.has_facebook,
            "has_instagram":  self.has_instagram,
            "has_trustpilot": self.has_trustpilot,
        }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _extract_domain_from_email(email: str) -> str | None:
    """'john@acmeplumbing.com' -> 'acmeplumbing.com'. None if invalid."""
    if not email or "@" not in email:
        return None
    domain = email.split("@", 1)[1].strip().lower()
    return domain if "." in domain else None


def _is_personal_email(email: str) -> bool:
    domain = _extract_domain_from_email(email)
    return domain in PERSONAL_EMAIL_DOMAINS if domain else True


def _normalize_website(website: str | None) -> str | None:
    """
    Accepts 'https://www.foo.com/bar', 'www.foo.com', 'foo.com'. Returns
    'foo.com' (root domain, lowercase), or None if not a valid website shape.
    """
    if not website:
        return None
    s = str(website).strip().lower()
    if not s:
        return None
    # Strip scheme
    if "://" in s:
        s = urlparse(s).netloc or urlparse(s).path
    # Strip leading www.
    if s.startswith("www."):
        s = s[4:]
    # Strip path
    s = s.split("/")[0]
    # Minimal shape check: must contain a dot and valid TLD
    if "." not in s:
        return None
    return s


def _is_valid_website(domain: str | None) -> bool:
    """True if the domain is a real business website (not a social media page
    or free platform subdomain)."""
    if not domain:
        return False
    d = domain.lower()
    # Reject known social/free-platform domains
    for bad in _INVALID_WEBSITE_DOMAINS:
        if d == bad or d.endswith("." + bad):
            return False
    # Require a recognised TLD as the last segment
    parts = d.split(".")
    if len(parts) < 2:
        return False
    return parts[-1] in _VALID_TLDS


def _normalize_company_name(name: str) -> str:
    """Strip suffixes like LLC/Inc/Corp + punctuation for comparison."""
    s = str(name).strip().lower()
    s = re.sub(r"[\u2019']", "", s)                  # delete apostrophes (Joe's -> joes)
    s = re.sub(r"[,.\u2013\u2014\-]", " ", s)        # other punctuation -> space
    s = re.sub(r"\b(llc|inc|corp|corporation|ltd|limited|co|company|l\.l\.c\.)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _company_name_matches(place_name: str, company: str) -> bool:
    """Conservative match: place name starts with company name (normalized)."""
    pn = _normalize_company_name(place_name)
    cn = _normalize_company_name(company)
    if not pn or not cn:
        return False
    return pn == cn or pn.startswith(cn) or cn in pn


def _same_website(a: str | None, b: str | None) -> bool:
    """Compare two website domains (after normalization)."""
    na = _normalize_website(a)
    nb = _normalize_website(b)
    return bool(na and nb and na == nb)


# ----------------------------------------------------------------------------
# Low-level: Gemini call wrapper (grounded, JSON-in-prompt parsing)
# ----------------------------------------------------------------------------

async def _call_gemini_grounded(
    client: httpx.AsyncClient,
    prompt: str,
    *,
    retry_on_parse_fail: bool = True,
    model: str = GEMINI_MODEL,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """
    Calls Gemini with Google Search grounding enabled, asking the model to
    respond in JSON.

    Model defaults to gemini-2.5-flash (good reasoning). For simpler tasks
    like enum classification, pass model=GEMINI_FAST_MODEL (flash-lite, ~2-3x
    faster).

    Gemini does NOT support response_schema + tools simultaneously, so we
    instruct the model to return JSON in the prompt and extract the first
    balanced JSON object from the (possibly prose-wrapped) response.

    Returns (parsed_json, error_type, error_message).
    """
    if not GEMINI_KEY:
        return None, "auth", "GEMINI_API_KEY not set in .env"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.0,
            # NOTE: responseMimeType='application/json' is INCOMPATIBLE with
            # tool use (grounding) — Gemini returns 400. We instruct the model
            # to return JSON via prompt, then extract it from the text.
        },
    }
    params = {"key": GEMINI_KEY}

    last_err: tuple[str, str] | None = None

    for attempt in range(2):  # 1 retry
        try:
            resp = await client.post(
                url, json=payload, params=params, timeout=TIMEOUT_GEMINI,
            )
        except httpx.TimeoutException:
            last_err = ("timeout", f"Gemini timed out after {TIMEOUT_GEMINI}s")
            await asyncio.sleep(1)
            continue
        except httpx.RequestError as e:
            last_err = ("network", f"Gemini network error: {e}")
            await asyncio.sleep(1)
            continue

        if resp.status_code == 401 or resp.status_code == 403:
            return None, "auth", f"Gemini auth error {resp.status_code}: invalid API key"
        if resp.status_code == 429:
            last_err = ("rate_limit", "Gemini rate-limited (429)")
            await asyncio.sleep(2)
            continue
        if resp.status_code >= 500:
            last_err = ("api_down", f"Gemini server error {resp.status_code}")
            await asyncio.sleep(1)
            continue
        if resp.status_code >= 400:
            # Probably a quota/billing issue; body has details
            body = resp.text[:300]
            return None, "quota", f"Gemini {resp.status_code}: {body}"

        # 200 OK — parse body
        try:
            body = resp.json()
        except ValueError as e:
            last_err = ("parse", f"Gemini response was not valid JSON: {e}")
            if retry_on_parse_fail and attempt == 0:
                continue
            return None, *last_err

        # Check for prompt-level block (whole prompt rejected by safety)
        if body.get("promptFeedback", {}).get("blockReason"):
            reason = body["promptFeedback"]["blockReason"]
            return None, "safety", f"Gemini blocked the prompt: {reason}"

        candidates = body.get("candidates") or []
        if not candidates:
            last_err = ("no_candidates", "Gemini returned no candidates")
            if retry_on_parse_fail and attempt == 0:
                continue
            return None, *last_err

        cand = candidates[0]
        finish_reason = cand.get("finishReason", "")

        # Response blocked by safety / recitation / prohibited content — no retry
        if finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST"):
            return None, "safety", f"Gemini blocked response: finishReason={finish_reason}"

        # Some responses have no content.parts (empty response, tool-only, etc.)
        content = cand.get("content") or {}
        parts   = content.get("parts") or []
        if not parts:
            # MAX_TOKENS with no output is worth reporting distinctly
            if finish_reason == "MAX_TOKENS":
                return None, "truncated", "Gemini ran out of tokens before producing output"
            last_err = ("empty", f"Gemini returned no content parts (finishReason={finish_reason})")
            if retry_on_parse_fail and attempt == 0:
                await asyncio.sleep(0.5)
                continue
            return None, *last_err

        text = parts[0].get("text")
        if not text:
            last_err = ("empty", "Gemini returned an empty text part")
            if retry_on_parse_fail and attempt == 0:
                await asyncio.sleep(0.5)
                continue
            return None, *last_err

        # Strip markdown fences if the model added them
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

        # Extract the first top-level JSON object from the text. Grounded
        # responses often include prose before/after the JSON, so we scan
        # for a balanced {...} block instead of demanding the whole response
        # be pure JSON.
        json_str = _extract_first_json_object(text)
        if json_str is None:
            last_err = ("parse", f"Gemini response has no JSON object. Raw: {text[:200]}")
            if retry_on_parse_fail and attempt == 0:
                continue
            return None, *last_err

        try:
            return json.loads(json_str), None, None
        except json.JSONDecodeError as e:
            last_err = ("parse", f"Gemini returned malformed JSON: {e}. Block: {json_str[:200]}")
            if retry_on_parse_fail and attempt == 0:
                continue
            return None, *last_err

    return None, *(last_err or ("unknown", "Gemini failed for unknown reason"))


def _extract_first_json_object(text: str) -> str | None:
    """
    Find the first balanced {...} JSON object in a free-form text. Handles
    nested braces and string literals (including escaped quotes) correctly.
    Returns the substring or None if no balanced block is found.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ----------------------------------------------------------------------------
# Gemini: find the website of a company
# ----------------------------------------------------------------------------

async def gemini_find_website(
    client: httpx.AsyncClient, company: str, email: str,
) -> tuple[str | None, ApiCallStatus]:
    """Find an official website for the company. None if not found."""
    t0 = _now_ms()
    prompt = f"""Find the official website of this US small business.

Company: {company}
Contact email: {email}

Return ONLY this JSON (no markdown, no commentary):

{{"website": "domain.com"}}

Rules:
- Return only the ROOT DOMAIN (e.g. "acmeplumbing.com", not "https://www.acmeplumbing.com/contact")
- Do NOT return social media pages (facebook.com, instagram.com, linkedin.com, twitter.com)
- Do NOT return free platform subdomains (wordpress.com, wixsite.com, blogspot.com, squarespace.com)
- If the email is from a corporate domain (e.g. john@acmeplumbing.com), that domain is likely the website — verify with a search
- If you cannot find with high confidence, return {{"website": null}}"""

    data, err_type, err_msg = await _call_gemini_grounded(client, prompt)
    dt = _now_ms() - t0

    if err_type:
        return None, ApiCallStatus(
            api="Gemini: find website", status="tech_fail", duration_ms=dt,
            error_type=err_type, error_message=err_msg,
        )

    raw = (data or {}).get("website")
    normalized = _normalize_website(raw)
    if normalized and _is_valid_website(normalized):
        return normalized, ApiCallStatus(
            api="Gemini: find website", status="ok", duration_ms=dt,
            note=f"Found {normalized}",
        )
    return None, ApiCallStatus(
        api="Gemini: find website", status="no_data", duration_ms=dt,
        note="No valid website found for this company",
    )


# ----------------------------------------------------------------------------
# Gemini: industry + TIB + location (city/state/country) in one call
# ----------------------------------------------------------------------------

async def gemini_industry_tib_location(
    client: httpx.AsyncClient, company: str, website: str | None, email: str,
) -> tuple[dict[str, Any], ApiCallStatus]:
    """
    Single grounded call returning industry, years_in_business, city, state,
    country. Any missing fields come back as None; caller applies Places
    fallback for location.
    """
    t0 = _now_ms()
    industries = _load_industries()

    prompt = f"""You are a business data analyst enriching lead data for Renaissance Growth, \
a US small business funding company. Use Google Search to research this company.

Company: {company}
Website: {website or 'unknown'}
Email: {email}

Return ONLY this JSON (no markdown, no commentary):

{{
  "industry": "<one of the values below>",
  "years_in_business": <integer or null>,
  "city": "<US city name>" or null,
  "state": "<full US state name>" or null,
  "country": "United States" or "<actual country name>" or null
}}

Allowed industry values (pick the single best fit; if nothing fits, use "Unknown"):
{json.dumps(industries)}

Allowed state values (full English name, not abbreviation):
{json.dumps(US_STATES)}

Rules:
- industry: MUST be exactly one of the values in the list above — copy verbatim
- years_in_business: integer number of years since founding, based on reliable sources. null if unknown
- state: MUST be from the list above. Full name like "California", never "CA". null if unknown
- country: "United States" if the business operates in USA. Otherwise the actual country name. null if completely unknown
- If this is not a US business, set country to the actual country — we will handle the rejection
- Return valid JSON only"""

    data, err_type, err_msg = await _call_gemini_grounded(client, prompt)
    dt = _now_ms() - t0

    if err_type:
        return {}, ApiCallStatus(
            api="Gemini: industry/TIB/location", status="tech_fail", duration_ms=dt,
            error_type=err_type, error_message=err_msg,
        )

    data = data or {}
    # Validate + sanitize each field
    industry = data.get("industry")
    if industry not in industries:
        industry = None

    tib = data.get("years_in_business")
    try:
        tib = int(tib) if tib is not None else None
        if tib is not None and (tib < 0 or tib > 200):
            tib = None
    except (TypeError, ValueError):
        tib = None

    state = data.get("state")
    if state not in US_STATES_SET:
        state = None

    city = data.get("city") or None
    country = data.get("country") or None

    filled = sum(1 for x in (industry, tib, city, state, country) if x is not None)
    note = f"Filled {filled}/5 fields"
    return {
        "industry": industry,
        "years_in_business": tib,
        "city": city,
        "state": state,
        "country": country,
    }, ApiCallStatus(
        api="Gemini: industry/TIB/location",
        status="ok" if filled >= 3 else "no_data",
        duration_ms=dt, note=note,
    )


# ----------------------------------------------------------------------------
# Gemini: digital presence (LinkedIn / FB / IG / Trustpilot)
# ----------------------------------------------------------------------------

async def gemini_digital_presence(
    client: httpx.AsyncClient,
    company: str,
    website: str | None,
    *,
    needed: list[str] | None = None,
    timeout: float | None = None,
) -> tuple[dict[str, bool], ApiCallStatus]:
    """
    Search for the company's presence on social platforms.

    Parameters:
      needed  — subset of the 4 keys (has_linkedin, has_facebook, has_instagram,
                has_trustpilot). When passed, only those are queried. Keeps the
                prompt focused. Defaults to all 4.
      timeout — wall-clock timeout (seconds). Exceeded -> tech_fail(timeout).
                Defaults to TIMEOUT_GEMINI.
    """
    t0 = _now_ms()

    all_platforms = ["has_linkedin", "has_facebook", "has_instagram", "has_trustpilot"]
    if needed is None:
        needed = all_platforms
    needed = [k for k in all_platforms if k in needed]   # valid keys, preserve order

    default = {k: False for k in all_platforms}
    if not needed:
        return default, ApiCallStatus(
            api="Gemini: digital presence", status="skipped", duration_ms=0,
            note="All platforms already resolved by scrape; Gemini not called",
        )

    # Build an explicit list of Google search queries we want Gemini to run.
    # Telling the model WHICH queries to execute (instead of 'check each
    # platform') dramatically improves recall because Gemini stops deciding
    # 'how to search' and just runs the queries we dictate.
    query_templates = {
        "has_linkedin":   [
            'site:linkedin.com/company "{company}"',
            'site:linkedin.com "{company}" {website}',
        ],
        "has_facebook":   [
            'site:facebook.com "{company}"',
            '"{company}" facebook page {website}',
        ],
        "has_instagram":  [
            'site:instagram.com "{company}"',
            '"{company}" instagram {website}',
        ],
        "has_trustpilot": [
            'site:trustpilot.com/review "{company}"',
            '"{company}" trustpilot reviews',
        ],
    }

    def _platform_block(key: str) -> str:
        label = {
            "has_linkedin":   "LinkedIn",
            "has_facebook":   "Facebook",
            "has_instagram":  "Instagram",
            "has_trustpilot": "Trustpilot",
        }[key]
        queries = query_templates[key]
        formatted = "\n".join(
            f"     - {q.format(company=company, website=website or '')}"
            for q in queries
        )
        return f"  {label} ({key}):\n{formatted}"

    platforms_block = "\n\n".join(_platform_block(k) for k in needed)
    json_fields = ",\n  ".join(f'"{k}": true or false' for k in needed)

    prompt = (
        f"You are verifying a US small business's social media presence.\n\n"
        f"Company: {company}\n"
        f"Website: {website or 'unknown'}\n\n"
        f"For each platform below, run the Google searches listed and decide "
        f"whether the company has an official presence there.\n\n"
        f"{platforms_block}\n\n"
        f"Return ONLY this JSON (no markdown, no commentary):\n\n"
        f"{{\n  {json_fields}\n}}\n\n"
        f"Decision rules:\n"
        f"- TRUE if the searches return a page/profile whose name, handle, or "
        f"linked website reasonably matches this company (minor variations in "
        f"capitalization, punctuation, or added 'Official'/'NYC'/etc. are fine).\n"
        f"- TRUE if a search result shows the company's own website LINKING OUT "
        f"to a page on that platform.\n"
        f"- FALSE only if the searches return NO plausible match for this company.\n"
        f"- Small businesses commonly have Facebook and Instagram even without "
        f"LinkedIn — do not assume they won't exist.\n"
        f"- Do not require explicit verification of ownership — a page that "
        f"looks like the company's is enough evidence."
    )

    tmo = timeout if timeout is not None else TIMEOUT_GEMINI
    try:
        data, err_type, err_msg = await asyncio.wait_for(
            _call_gemini_grounded(client, prompt),
            timeout=tmo,
        )
    except asyncio.TimeoutError:
        return default, ApiCallStatus(
            api="Gemini: digital presence", status="tech_fail",
            duration_ms=int(tmo * 1000), error_type="timeout",
            error_message=f"Gemini DP exceeded aggressive timeout ({tmo}s); scrape kept",
        )

    dt = _now_ms() - t0

    if err_type:
        return default, ApiCallStatus(
            api="Gemini: digital presence", status="tech_fail", duration_ms=dt,
            error_type=err_type, error_message=err_msg,
        )

    data = data or {}
    result = dict(default)
    for k in needed:
        result[k] = bool(data.get(k, False))
    found = sum(result[k] for k in needed)
    return result, ApiCallStatus(
        api="Gemini: digital presence", status="ok", duration_ms=dt,
        note=f"Checked {len(needed)}/4 platforms, found {found} positives",
    )


# ----------------------------------------------------------------------------
# Website HTML scraping — complement to Gemini for digital presence detection
# ----------------------------------------------------------------------------
# Most small businesses link their social media in the website footer. If the
# website HTML contains a link to facebook.com/<handle>, instagram.com/<handle>
# etc., that's very strong evidence the company has that social presence.
#
# We OR-merge this with Gemini's answer: if EITHER Gemini OR the scrape
# found the link, we consider the platform present. This compensates for
# Gemini being occasionally too conservative.

# Regex patterns. Negative lookaheads exclude sharing widgets, ad tracking,
# and other non-account URLs on each platform.
_SOCIAL_PATTERNS = {
    "has_facebook":   re.compile(
        r"facebook\.com/(?!sharer|plugins?|dialog|tr[/?]|events/|login|public|profile\.php|2008|home\.php)"
        r"[a-z0-9._\-]+",
        re.IGNORECASE),
    "has_instagram":  re.compile(
        r"instagram\.com/(?!embed|accounts|p/|reel/|explore|stories|direct)"
        r"[a-z0-9._\-]+",
        re.IGNORECASE),
    "has_linkedin":   re.compile(
        r"linkedin\.com/(?:company|in|school|showcase)/[a-z0-9._\-]+",
        re.IGNORECASE),
    "has_trustpilot": re.compile(
        r"trustpilot\.com/review/[a-z0-9._\-]+",
        re.IGNORECASE),
}


async def scrape_social_links(
    client: httpx.AsyncClient, website: str,
) -> tuple[dict[str, bool], ApiCallStatus]:
    """
    GET the website's home + /contact + /about pages (in parallel) and search
    the combined HTML for links to the 4 social platforms. Used as OR-merge
    complement to Gemini digital presence.

    Scanning 3 common pages instead of just the home page catches sites that
    only put social icons in a contact/about section (fairly common).
    """
    t0 = _now_ms()
    base = website if website.startswith("http") else f"https://{website}"
    base = base.rstrip("/")
    paths = ["", "/contact", "/about"]
    urls = [base + p for p in paths]

    default = {k: False for k in _SOCIAL_PATTERNS}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RenaissanceBot/1.0)"}

    async def _fetch(u: str) -> str:
        try:
            r = await client.get(u, timeout=5.0, follow_redirects=True, headers=headers)
            if r.status_code >= 400:
                return ""
            return r.text[:200_000]
        except (httpx.TimeoutException, httpx.RequestError):
            return ""

    try:
        pages = await asyncio.gather(*(_fetch(u) for u in urls))
    except Exception as e:
        return default, ApiCallStatus(
            api="Website scrape", status="tech_fail", duration_ms=_now_ms()-t0,
            error_type="network", error_message=str(e),
        )

    combined = "\n".join(pages)
    dt = _now_ms() - t0

    if not combined.strip():
        return default, ApiCallStatus(
            api="Website scrape", status="no_data", duration_ms=dt,
            note=f"No reachable pages at {base}",
        )

    result = {k: bool(pat.search(combined)) for k, pat in _SOCIAL_PATTERNS.items()}
    found = sum(result.values())
    return result, ApiCallStatus(
        api="Website scrape", status="ok" if found else "no_data", duration_ms=dt,
        note=f"Found {found}/4 social links across home/contact/about pages",
    )


# ----------------------------------------------------------------------------
# Gemini fallback for job title (runs only if Apollo people-match fails)
# ----------------------------------------------------------------------------

async def gemini_find_job_title(
    client: httpx.AsyncClient,
    company: str, first_name: str, last_name: str, email: str,
) -> tuple[str | None, ApiCallStatus]:
    """
    Grounded search for a person's job title. Used as fallback when Apollo
    didn't match. Uses Flash-Lite because the task is simple.
    """
    t0 = _now_ms()
    prompt = f"""Find the current job title of this specific person at the specified company.

Name: {first_name} {last_name}
Company: {company}
Email: {email}

Use Google Search — check LinkedIn, the company website, press releases, \
or other public sources.

Return ONLY this JSON (no markdown, no commentary):

{{"job_title": "Owner"}} or {{"job_title": null}}

Rules:
- Return the title verbatim as found publicly (e.g. "Chief Executive Officer", \
"Owner", "Operations Manager")
- Only return a title if you are confident it's the right person at the right company
- Return null if you cannot find it with confidence
- Do NOT invent or guess titles"""

    data, err_type, err_msg = await _call_gemini_grounded(
        client, prompt, model=GEMINI_FAST_MODEL,
    )
    dt = _now_ms() - t0

    if err_type:
        return None, ApiCallStatus(
            api="Gemini: find job title", status="tech_fail", duration_ms=dt,
            error_type=err_type, error_message=err_msg,
        )

    title = (data or {}).get("job_title")
    if not title or not isinstance(title, str):
        return None, ApiCallStatus(
            api="Gemini: find job title", status="no_data", duration_ms=dt,
            note="No title found for this person",
        )
    title = title.strip()
    return title, ApiCallStatus(
        api="Gemini: find job title", status="ok", duration_ms=dt,
        note=f"title={title}",
    )


async def gemini_find_employees(
    client: httpx.AsyncClient,
    company: str,
    website: str | None,
) -> tuple[int | str | None, ApiCallStatus]:
    """
    Grounded search for a company's employee count. Used as fallback when
    Apollo organization_enrich didn't match. Uses Flash-Lite because the
    task is focused.

    Returns either:
      - int (exact count), or
      - str in {"1", "2-10", "11-50", "51-200", "201-500"} (already-bucketed), or
      - None if Gemini could not find a value.
    """
    t0 = _now_ms()
    prompt = (
        f"Find the current number of employees at this US small business.\n\n"
        f"Company: {company}\n"
        f"Website: {website or 'unknown'}\n\n"
        f"Use Google Search — check LinkedIn company page, Crunchbase, "
        f"ZoomInfo, or the company's own website.\n\n"
        f"Return ONLY this JSON (no markdown, no commentary):\n\n"
        f'{{"employees": <integer>}} or {{"employees": "<bucket>"}} or {{"employees": null}}\n\n'
        f"Rules:\n"
        f"- Prefer an exact number if you can find one (e.g. 25).\n"
        f'- If only a range is available, return one of these exact bucket strings: '
        f'"1", "2-10", "11-50", "51-200", "201-500".\n'
        f"- Return null if you cannot find a credible source.\n"
        f"- Do NOT invent or guess — better null than wrong.\n"
        f'- If the business is clearly a small family-owned operation with no '
        f'published data, "2-10" is a reasonable default.'
    )

    data, err_type, err_msg = await _call_gemini_grounded(
        client, prompt, model=GEMINI_FAST_MODEL,
    )
    dt = _now_ms() - t0

    if err_type:
        return None, ApiCallStatus(
            api="Gemini: find employees", status="tech_fail", duration_ms=dt,
            error_type=err_type, error_message=err_msg,
        )

    emp = (data or {}).get("employees")
    if emp is None:
        return None, ApiCallStatus(
            api="Gemini: find employees", status="no_data", duration_ms=dt,
            note="No credible employee count found",
        )
    # Accept int or a valid bucket string. Anything else -> None.
    valid_buckets = {"1", "2-10", "11-50", "51-200", "201-500"}
    if isinstance(emp, int):
        return emp, ApiCallStatus(
            api="Gemini: find employees", status="ok", duration_ms=dt,
            note=f"employees={emp}",
        )
    if isinstance(emp, str) and emp.strip() in valid_buckets:
        return emp.strip(), ApiCallStatus(
            api="Gemini: find employees", status="ok", duration_ms=dt,
            note=f"employees={emp.strip()}",
        )
    # Unrecognized shape — try int() one last time
    try:
        n = int(str(emp).strip())
        return n, ApiCallStatus(
            api="Gemini: find employees", status="ok", duration_ms=dt,
            note=f"employees={n}",
        )
    except (TypeError, ValueError):
        return None, ApiCallStatus(
            api="Gemini: find employees", status="no_data", duration_ms=dt,
            note=f"Got unparseable value: {emp!r}",
        )


async def gemini_check_usa_operations(
    client: httpx.AsyncClient,
    company: str,
    website: str | None,
    detected_country: str,
) -> tuple[dict[str, Any], ApiCallStatus]:
    """
    Secondary country check. Used when the primary Gemini/Places lookup says
    the company is based outside the US — we still want to accept the lead if
    the company has real USA operations (stores, offices, warehouses).

    Uses flash-lite because it's a focused yes/no + state pick.

    Returns a dict with:
      - has_usa_operations: bool
      - usa_state: str (one of the 50 + DC) or None
    """
    t0 = _now_ms()
    empty = {"has_usa_operations": False, "usa_state": None}

    prompt = (
        f"The company \"{company}\" ({website or 'unknown website'}) "
        f"appears to be based in {detected_country}.\n\n"
        f"Use Google Search to verify whether this company has PHYSICAL "
        f"OPERATIONS in the United States — stores, offices, warehouses, "
        f"or service locations. Do NOT count merely shipping to USA, "
        f"online-only availability, or having USA customers.\n\n"
        f"If yes, identify the US state where it has the MOST presence "
        f"(headquarters or largest operation).\n\n"
        f"Return ONLY this JSON (no markdown, no commentary):\n\n"
        f'{{"has_usa_operations": true_or_false, "usa_state": "<full state name>" or null}}\n\n'
        f"Rules:\n"
        f"- has_usa_operations = TRUE only if you find concrete evidence of "
        f"physical US operations\n"
        f"- usa_state MUST be one of the following (full English name, not "
        f"abbreviation): {json.dumps(US_STATES)}\n"
        f"- usa_state = null if you cannot determine the state\n"
        f"- If has_usa_operations is FALSE, usa_state must be null"
    )

    data, err_type, err_msg = await _call_gemini_grounded(
        client, prompt, model=GEMINI_FAST_MODEL,
    )
    dt = _now_ms() - t0

    if err_type:
        return empty, ApiCallStatus(
            api="Gemini: USA re-check", status="tech_fail", duration_ms=dt,
            error_type=err_type, error_message=err_msg,
        )

    data = data or {}
    has_usa = bool(data.get("has_usa_operations", False))
    state = data.get("usa_state")
    if state not in US_STATES_SET:
        state = None
    # If no state, can't really say USA operations
    if has_usa and state is None:
        return {"has_usa_operations": False, "usa_state": None}, ApiCallStatus(
            api="Gemini: USA re-check", status="no_data", duration_ms=dt,
            note=f"Claimed USA ops but no state identified for {company!r}",
        )

    if has_usa:
        return {"has_usa_operations": True, "usa_state": state}, ApiCallStatus(
            api="Gemini: USA re-check", status="ok", duration_ms=dt,
            note=f"Found USA operations in {state}",
        )

    return empty, ApiCallStatus(
        api="Gemini: USA re-check", status="no_data", duration_ms=dt,
        note=f"No USA operations found for {company!r} (based in {detected_country})",
    )



# ----------------------------------------------------------------------------
# Google Places (New) — single searchText call gives num_locations + has_gmb
# ----------------------------------------------------------------------------

async def places_search(
    client: httpx.AsyncClient, company: str, website: str | None = None,
) -> tuple[dict[str, Any], ApiCallStatus]:
    """
    One searchText call. Extracts from the response:
      - has_gmb: True if any place matches this company
      - num_locations: count of matching places (capped at 20)
      - city/state/country: from the first strong match (used as location fallback)
      - websiteUri of first match (not used here but returned for completeness)
    """
    t0 = _now_ms()
    if not PLACES_KEY:
        return {}, ApiCallStatus(
            api="Google Places", status="tech_fail", duration_ms=0,
            error_type="auth", error_message="GOOGLE_PLACES_API_KEY not set in .env",
        )

    field_mask = (
        "places.id,places.displayName,places.formattedAddress,"
        "places.addressComponents,places.websiteUri,places.types,"
        "places.businessStatus"
    )
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_KEY,
        "X-Goog-FieldMask": field_mask,
    }
    body = {"textQuery": company, "maxResultCount": 20}

    try:
        resp = await client.post(PLACES_URL, json=body, headers=headers, timeout=TIMEOUT_PLACES)
    except httpx.TimeoutException:
        return {}, ApiCallStatus(
            api="Google Places", status="tech_fail", duration_ms=_now_ms()-t0,
            error_type="timeout", error_message=f"Places timed out after {TIMEOUT_PLACES}s",
        )
    except httpx.RequestError as e:
        return {}, ApiCallStatus(
            api="Google Places", status="tech_fail", duration_ms=_now_ms()-t0,
            error_type="network", error_message=str(e),
        )

    dt = _now_ms() - t0

    if resp.status_code in (401, 403):
        return {}, ApiCallStatus(
            api="Google Places", status="tech_fail", duration_ms=dt,
            error_type="auth", error_message=f"Places auth error {resp.status_code}",
        )
    if resp.status_code == 429:
        return {}, ApiCallStatus(
            api="Google Places", status="tech_fail", duration_ms=dt,
            error_type="rate_limit", error_message="Places rate-limited",
        )
    if resp.status_code >= 400:
        return {}, ApiCallStatus(
            api="Google Places", status="tech_fail", duration_ms=dt,
            error_type="api_error", error_message=f"Places {resp.status_code}: {resp.text[:200]}",
        )

    try:
        data = resp.json()
    except ValueError:
        return {}, ApiCallStatus(
            api="Google Places", status="tech_fail", duration_ms=dt,
            error_type="parse", error_message="Places returned non-JSON",
        )

    places = data.get("places", []) or []

    # Filter to matches — prefer websiteUri match, fall back to name match
    if website:
        matches = [p for p in places if _same_website(p.get("websiteUri"), website)]
        if not matches:
            matches = [p for p in places
                       if _company_name_matches(p.get("displayName", {}).get("text", ""), company)]
    else:
        matches = [p for p in places
                   if _company_name_matches(p.get("displayName", {}).get("text", ""), company)]

    has_gmb = len(matches) > 0
    num_locations = min(len(matches), 20) if matches else 1

    # Pull city/state/country from the first match (used only as fallback)
    city = state = country = None
    if matches:
        comps = matches[0].get("addressComponents", []) or []
        for c in comps:
            types = c.get("types", [])
            long_name = c.get("longText") or c.get("shortText")
            if "locality" in types:
                city = long_name
            elif "administrative_area_level_1" in types:
                state = long_name
            elif "country" in types:
                country = long_name

    status: StatusLevel = "ok" if has_gmb else "no_data"
    note = (f"{num_locations} location(s) matched" if has_gmb
            else "Company not listed on Google Maps")

    return {
        "has_gmb": has_gmb,
        "num_locations": num_locations,
        "city": city,
        "state": state,
        "country": country,
    }, ApiCallStatus(api="Google Places", status=status, duration_ms=dt, note=note)


# ----------------------------------------------------------------------------
# Apollo: organization enrich (employees) + people match (job_title)
# ----------------------------------------------------------------------------

async def apollo_organization_enrich(
    client: httpx.AsyncClient, domain: str,
) -> tuple[dict[str, Any], ApiCallStatus]:
    """
    Enrich a company by domain. Returns a dict with:
      - employees: int | str | None
      - has_linkedin:  bool  (True if Apollo has linkedin_url)
      - has_facebook:  bool  (idem)
      - has_twitter:   bool  (kept for completeness; not a scoring feature)

    These social booleans are a high-precision shortcut — Apollo stores the
    actual URLs, so no ambiguity. They feed into the digital-presence layer
    before falling back to website scrape or Gemini.
    """
    t0 = _now_ms()
    empty = {"employees": None, "has_linkedin": False,
             "has_facebook": False, "has_twitter": False}

    if not APOLLO_KEY:
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=0,
            error_type="auth", error_message="APOLLO_API_KEY not set in .env",
        )

    try:
        resp = await client.post(
            APOLLO_ORG_URL,
            headers={"x-api-key": APOLLO_KEY, "Content-Type": "application/json"},
            json={"domain": domain},
            timeout=TIMEOUT_APOLLO,
        )
    except httpx.TimeoutException:
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=_now_ms()-t0,
            error_type="timeout", error_message=f"Apollo timed out after {TIMEOUT_APOLLO}s",
        )
    except httpx.RequestError as e:
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=_now_ms()-t0,
            error_type="network", error_message=str(e),
        )

    dt = _now_ms() - t0
    if resp.status_code in (401, 403):
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=dt,
            error_type="auth", error_message="Apollo API key invalid",
        )
    if resp.status_code == 402:
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=dt,
            error_type="quota", error_message="Apollo out of credits — top up at apollo.io",
        )
    if resp.status_code == 429:
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=dt,
            error_type="rate_limit", error_message="Apollo rate-limited",
        )
    if resp.status_code >= 400:
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=dt,
            error_type="api_error", error_message=f"Apollo {resp.status_code}: {resp.text[:200]}",
        )

    try:
        data = resp.json()
    except ValueError:
        return empty, ApiCallStatus(
            api="Apollo: organization", status="tech_fail", duration_ms=dt,
            error_type="parse", error_message="Apollo returned non-JSON",
        )

    org = data.get("organization") or {}
    emp = org.get("estimated_num_employees") or org.get("num_employees")

    # Social URLs — treat any non-empty string as presence (True)
    def _present(u):
        return bool(u) and isinstance(u, str) and u.strip()

    result = {
        "employees":   emp,
        "has_linkedin": _present(org.get("linkedin_url")),
        "has_facebook": _present(org.get("facebook_url")),
        "has_twitter":  _present(org.get("twitter_url")),
    }

    if emp is None and not any((result["has_linkedin"], result["has_facebook"], result["has_twitter"])):
        # No data at all — Apollo doesn't have this company
        return result, ApiCallStatus(
            api="Apollo: organization", status="no_data", duration_ms=dt,
            note=f"Domain {domain} not in Apollo database",
        )

    social_found = [k.replace("has_", "") for k in ("has_linkedin","has_facebook","has_twitter") if result[k]]
    note_parts = []
    if emp is not None:
        note_parts.append(f"employees={emp}")
    if social_found:
        note_parts.append(f"socials={'+'.join(social_found)}")
    note = ", ".join(note_parts) or "partial match"

    return result, ApiCallStatus(
        api="Apollo: organization", status="ok", duration_ms=dt, note=note,
    )


async def apollo_people_match(
    client: httpx.AsyncClient, email: str, first_name: str, last_name: str,
) -> tuple[str | None, ApiCallStatus]:
    """Returns the matched person's job title, or None."""
    t0 = _now_ms()
    if not APOLLO_KEY:
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=0,
            error_type="auth", error_message="APOLLO_API_KEY not set in .env",
        )

    try:
        resp = await client.post(
            APOLLO_PEOPLE_URL,
            headers={"x-api-key": APOLLO_KEY, "Content-Type": "application/json"},
            json={
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "reveal_personal_emails": False,
            },
            timeout=TIMEOUT_APOLLO,
        )
    except httpx.TimeoutException:
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=_now_ms()-t0,
            error_type="timeout", error_message=f"Apollo timed out after {TIMEOUT_APOLLO}s",
        )
    except httpx.RequestError as e:
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=_now_ms()-t0,
            error_type="network", error_message=str(e),
        )

    dt = _now_ms() - t0
    if resp.status_code in (401, 403):
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=dt,
            error_type="auth", error_message="Apollo API key invalid",
        )
    if resp.status_code == 402:
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=dt,
            error_type="quota", error_message="Apollo out of credits",
        )
    if resp.status_code == 429:
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=dt,
            error_type="rate_limit", error_message="Apollo rate-limited",
        )
    if resp.status_code >= 400:
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=dt,
            error_type="api_error", error_message=f"Apollo {resp.status_code}: {resp.text[:200]}",
        )

    try:
        data = resp.json()
    except ValueError:
        return None, ApiCallStatus(
            api="Apollo: people", status="tech_fail", duration_ms=dt,
            error_type="parse", error_message="Apollo returned non-JSON",
        )

    person = data.get("person") or {}
    title = person.get("title")
    if not title:
        return None, ApiCallStatus(
            api="Apollo: people", status="no_data", duration_ms=dt,
            note=f"No Apollo match for {first_name} {last_name} <{email}>",
        )
    return title, ApiCallStatus(
        api="Apollo: people", status="ok", duration_ms=dt,
        note=f"title={title}",
    )


# ----------------------------------------------------------------------------
# HTTP check — verify a website actually responds
# ----------------------------------------------------------------------------

async def http_check_website(
    client: httpx.AsyncClient, website: str,
) -> tuple[bool, ApiCallStatus]:
    t0 = _now_ms()
    url = website if website.startswith("http") else f"https://{website}"
    try:
        resp = await client.head(url, timeout=TIMEOUT_HTTP_CHECK, follow_redirects=True)
        # Some servers don't support HEAD; fall back to GET
        if resp.status_code == 405:
            resp = await client.get(url, timeout=TIMEOUT_HTTP_CHECK, follow_redirects=True)
        dt = _now_ms() - t0
        alive = resp.status_code < 400
        return alive, ApiCallStatus(
            api="HTTP: website check",
            status="ok" if alive else "no_data",
            duration_ms=dt,
            note=f"HTTP {resp.status_code} — {'alive' if alive else 'not responding'}",
        )
    except httpx.TimeoutException:
        return False, ApiCallStatus(
            api="HTTP: website check", status="no_data", duration_ms=_now_ms()-t0,
            note="Website did not respond within 3s (treated as has_website=False)",
        )
    except httpx.RequestError as e:
        return False, ApiCallStatus(
            api="HTTP: website check", status="no_data", duration_ms=_now_ms()-t0,
            note=f"Website unreachable: {e}",
        )


# ----------------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------------

async def enrich_lead(
    *,
    company: str,
    email: str,
    first_name: str,
    last_name: str,
    website: str | None = None,
) -> EnrichmentResult:
    """
    Enrich a lead. Company + email + first_name + last_name are mandatory.
    Website is optional; if missing, Gemini will try to find it.

    Returns EnrichmentResult. Raises LeadOutsideUSAError if country != US, or
    AllApisFailedError if every API failed technically.
    """
    result = EnrichmentResult()

    async with httpx.AsyncClient() as client:

        # ---------- STEP 1: Resolve website (sequential if missing) ----------
        provided_website = _normalize_website(website)
        if provided_website and _is_valid_website(provided_website):
            resolved_website = provided_website
            website_found_by_gemini = False
        else:
            if website:  # advisor provided something invalid
                result.statuses.append(ApiCallStatus(
                    api="Input: website", status="no_data", duration_ms=0,
                    note=f"Advisor-provided website {website!r} rejected (invalid/social/free-platform)",
                ))
            found, status = await gemini_find_website(client, company, email)
            result.statuses.append(status)
            resolved_website = found
            website_found_by_gemini = bool(found)

        result.resolved_website = resolved_website

        # ---------- STEP 2: Determine Apollo domain input ----------
        apollo_domain: str | None = None
        if resolved_website:
            apollo_domain = resolved_website
        else:
            email_domain = _extract_domain_from_email(email)
            if email_domain and email_domain not in PERSONAL_EMAIL_DOMAINS:
                apollo_domain = email_domain
        # if still None -> Apollo org call is skipped

        # ---------- STEP 3: Fire parallel API calls (EXCEPT Gemini DP) ----------
        # Gemini DP is held back intentionally — we want to see the website-scrape
        # result first, so we can skip Gemini DP entirely if scrape found all 4
        # platforms, or ask only for the missing ones with an aggressive timeout.
        tasks = {
            "gemini_loc":    gemini_industry_tib_location(client, company, resolved_website, email),
            "places":        places_search(client, company, resolved_website),
            "apollo_people": apollo_people_match(client, email, first_name, last_name),
        }
        if apollo_domain:
            tasks["apollo_org"] = apollo_organization_enrich(client, apollo_domain)
        if resolved_website and not website_found_by_gemini:
            tasks["http_check"] = http_check_website(client, resolved_website)
        if resolved_website:
            tasks["scrape_social"] = scrape_social_links(client, resolved_website)

        # Run all in parallel
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        outputs = dict(zip(tasks.keys(), gathered))

        # ---------- STEP 4: Process each result ----------

        # Gemini location
        loc_data, loc_status = _unpack(outputs["gemini_loc"], default=({}, _fallback_status("Gemini: industry/TIB/location")))
        result.statuses.append(loc_status)
        result.industry          = loc_data.get("industry")
        result.years_in_business = loc_data.get("years_in_business")
        result.city              = loc_data.get("city")
        result.state             = loc_data.get("state")
        result.country           = loc_data.get("country")

        # Places
        places_data, places_status = _unpack(outputs["places"], default=({}, _fallback_status("Google Places")))
        result.statuses.append(places_status)
        result.has_gmb       = bool(places_data.get("has_gmb", False))
        result.num_locations = int(places_data.get("num_locations") or 1)

        # LOCATION FALLBACK: if Gemini missing city/state/country, use Places values
        if not result.city    and places_data.get("city"):    result.city    = places_data["city"]
        if not result.state   and places_data.get("state") in US_STATES_SET:
            result.state   = places_data["state"]
        if not result.country and places_data.get("country"): result.country = places_data["country"]

        # ---------------------------------------------------------------
        # Consume Apollo organization enrich FIRST — it gives us employees +
        # direct links to LinkedIn/Facebook/Twitter (when present). Apollo is
        # the highest-precision signal we have because it stores actual URLs,
        # so we use it as the primary source and layer scrape+Gemini on top.
        # ---------------------------------------------------------------
        apollo_org_data = {"employees": None, "has_linkedin": False,
                           "has_facebook": False, "has_twitter": False}
        if "apollo_org" in outputs:
            apollo_org_data, org_status = _unpack(
                outputs["apollo_org"],
                default=(dict(apollo_org_data), _fallback_status("Apollo: organization")),
            )
            result.statuses.append(org_status)
        else:
            reason = ("no website and email is personal"
                      if _is_personal_email(email) and not resolved_website
                      else "no Apollo domain input available")
            result.statuses.append(ApiCallStatus(
                api="Apollo: organization", status="skipped", duration_ms=0,
                note=f"Skipped: {reason}",
            ))

        # Digital presence: Apollo signals → scrape → Gemini (only missing)
        # ---------------------------------------------------------------
        dp_data = {k: False for k in ("has_linkedin", "has_facebook", "has_instagram", "has_trustpilot")}
        dp_sources: dict[str, str] = {}   # track where each positive came from

        # 1) Apollo signals (highest precision)
        for k in ("has_linkedin", "has_facebook"):
            if apollo_org_data.get(k):
                dp_data[k] = True
                dp_sources[k] = "apollo"
        # (Apollo doesn't cover Instagram or Trustpilot)

        # 2) Website scrape — fills in platforms Apollo didn't have
        if "scrape_social" in outputs:
            scrape_data, scrape_status = _unpack(
                outputs["scrape_social"],
                default=({k: False for k in _SOCIAL_PATTERNS}, _fallback_status("Website scrape")),
            )
            result.statuses.append(scrape_status)
            for k in dp_data:
                if not dp_data[k] and bool(scrape_data.get(k)):
                    dp_data[k] = True
                    dp_sources[k] = "scrape"

        # 3) Gemini DP — only for platforms still missing
        missing_keys = [k for k in dp_data if not dp_data[k]]
        if not missing_keys:
            found_summary = ", ".join(f"{k.replace('has_','')}[{dp_sources[k]}]" for k in dp_data if dp_data[k])
            result.statuses.append(ApiCallStatus(
                api="Gemini: digital presence", status="skipped", duration_ms=0,
                note=f"All 4 platforms resolved by Apollo/scrape ({found_summary}); Gemini not called",
            ))
        else:
            gemini_tmo = 10.0 if resolved_website else TIMEOUT_GEMINI
            gemini_dp, gemini_dp_status = await gemini_digital_presence(
                client, company, resolved_website,
                needed=missing_keys, timeout=gemini_tmo,
            )
            result.statuses.append(gemini_dp_status)
            for k in missing_keys:
                if bool(gemini_dp.get(k)):
                    dp_data[k] = True
                    dp_sources[k] = "gemini"

        result.has_linkedin   = dp_data["has_linkedin"]
        result.has_facebook   = dp_data["has_facebook"]
        result.has_instagram  = dp_data["has_instagram"]
        result.has_trustpilot = dp_data["has_trustpilot"]

        # Apollo people (title) + employees — BOTH can need a Gemini fallback
        # when Apollo couldn't match. Run the two fallbacks CONCURRENTLY to
        # avoid paying 2× latency; they're independent queries.
        title, people_status = _unpack(outputs["apollo_people"], default=(None, _fallback_status("Apollo: people")))
        result.statuses.append(people_status)
        apollo_emp = apollo_org_data.get("employees")

        need_title_fb = (title is None)
        need_emp_fb   = (apollo_emp is None)

        if need_title_fb and need_emp_fb:
            # Both missing — fire both Gemini fallbacks in parallel
            (title_res, emp_res) = await asyncio.gather(
                gemini_find_job_title(client, company, first_name, last_name, email),
                gemini_find_employees(client, company, resolved_website),
                return_exceptions=True,
            )
            # Unpack each (handles exceptions gracefully)
            gemini_title, gt_status = _unpack(
                title_res, default=(None, _fallback_status("Gemini: find job title")),
            )
            result.statuses.append(gt_status)
            title = gemini_title

            gemini_emp, ge_status = _unpack(
                emp_res, default=(None, _fallback_status("Gemini: find employees")),
            )
            result.statuses.append(ge_status)
            result.employees = gemini_emp

        elif need_title_fb:
            gemini_title, gt_status = await gemini_find_job_title(
                client, company, first_name, last_name, email,
            )
            result.statuses.append(gt_status)
            title = gemini_title
            result.employees = apollo_emp

        elif need_emp_fb:
            gemini_emp, ge_status = await gemini_find_employees(
                client, company, resolved_website,
            )
            result.statuses.append(ge_status)
            result.employees = gemini_emp

        else:
            # Both came from Apollo
            result.employees = apollo_emp

        result.job_title = title

        # Website HTTP check (or trust Gemini)
        if "http_check" in outputs:
            alive, web_status = _unpack(outputs["http_check"], default=(False, _fallback_status("HTTP: website check")))
            result.statuses.append(web_status)
            result.has_website = alive
        elif website_found_by_gemini:
            result.has_website = True
            result.statuses.append(ApiCallStatus(
                api="HTTP: website check", status="skipped", duration_ms=0,
                note=f"Trusted Gemini-found website ({resolved_website}) without HTTP verification",
            ))
        else:
            result.has_website = False
            result.statuses.append(ApiCallStatus(
                api="HTTP: website check", status="skipped", duration_ms=0,
                note="No website to check",
            ))

        # ---------- STEP 5: Validate ----------

        # USA-only guard
        # ---------- STEP 5: Validate — USA-only cascade ----------
        # The business rule: accept the lead if the company operates in the USA,
        # even if its headquarters is elsewhere. Cascade:
        #   1) Primary Gemini said USA → accept.
        #   2) Primary Gemini said another country → re-ask Gemini whether the
        #      company has USA operations (and in which state).
        #   3) Primary Gemini said null → use Places data if it found a USA
        #      match (city/state/country from address components).
        #   4) Nothing indicates USA presence → reject.

        primary_country = (result.country or "").strip().lower()
        is_usa = primary_country in ("united states", "usa", "us", "united states of america")

        if is_usa:
            # Normalize to canonical form
            result.country = "United States"

        elif result.country:
            # Non-USA primary — do a focused re-check for USA operations
            usa_check, usa_status = await gemini_check_usa_operations(
                client, company, resolved_website, result.country,
            )
            result.statuses.append(usa_status)
            if usa_check["has_usa_operations"] and usa_check["usa_state"]:
                result.country = "United States"
                # Override state with the USA state where company actually operates,
                # since the primary `state` probably points to the foreign HQ.
                result.state = usa_check["usa_state"]
                result.statuses.append(ApiCallStatus(
                    api="Validation: country", status="ok", duration_ms=0,
                    note=f"Primary said {primary_country!r}, but company has USA ops in {result.state}",
                ))
            else:
                # Re-check said no USA ops — genuine non-USA lead, reject
                raise LeadOutsideUSAError(result.country)

        else:
            # Primary returned null for country — rely on Places
            if places_data.get("country") and \
               places_data.get("state") in US_STATES_SET:
                result.country = "United States"
                result.state   = places_data["state"]
                if not result.city and places_data.get("city"):
                    result.city = places_data["city"]
                result.statuses.append(ApiCallStatus(
                    api="Validation: country", status="ok", duration_ms=0,
                    note=f"Country resolved via Places fallback → {result.state}",
                ))
            else:
                # No signal at all — reject (can't confirm USA)
                raise LeadOutsideUSAError(None)

        # If EVERY technical call failed, abort — we can't score blindly
        tech_fails = sum(1 for s in result.statuses if s.status == "tech_fail")
        ok_data    = sum(1 for s in result.statuses if s.status == "ok")
        if tech_fails >= 4 and ok_data == 0:
            raise AllApisFailedError(
                "All external APIs failed. Check API keys, quotas, and network. "
                "Details: " + "; ".join(
                    f"{s.api}: {s.error_type} — {s.error_message}"
                    for s in result.statuses if s.status == "tech_fail"
                )
            )

    return result


# ----------------------------------------------------------------------------
# Internal helpers for parsing asyncio.gather results
# ----------------------------------------------------------------------------

def _unpack(result, default):
    """asyncio.gather returns the raw return value OR an Exception when
    return_exceptions=True. Normalize to (data, status)."""
    if isinstance(result, Exception):
        default_data, default_status = default
        default_status.error_type = type(result).__name__
        default_status.error_message = str(result)
        default_status.status = "tech_fail"
        return default_data, default_status
    return result


def _fallback_status(api: str) -> ApiCallStatus:
    return ApiCallStatus(api=api, status="tech_fail", duration_ms=0,
                         error_type="unknown", error_message="unexpected exception")


# ----------------------------------------------------------------------------
# CLI smoke test: python enrichment.py
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from dataclasses import asdict

    TEST_CASES = [
        {
            "label": "TEST 1 — Website provided by advisor (normal flow)",
            "sample": {
                "company":    "Joe's Pizza",
                "website":    "joespizzanyc.com",
                "email":      "info@joespizzanyc.com",
                "first_name": "Joe",
                "last_name":  "Pozzuoli",
            },
        },
        {
            "label": "TEST 2 — No website (Gemini has to find it first)",
            "sample": {
                "company":    "Joe's Pizza",
                "website":    None,
                "email":      "info@joespizzanyc.com",
                "first_name": "Joe",
                "last_name":  "Pozzuoli",
            },
        },
    ]

    async def _run_one(label: str, sample: dict) -> None:
        print("\n" + "#" * 72)
        print(f"# {label}")
        print(f"# Input: {sample}")
        print("#" * 72)
        try:
            result = await enrich_lead(**sample)
        except LeadOutsideUSAError as e:
            print(f"✗ {e}")
            return
        except AllApisFailedError as e:
            print(f"✗ {e}")
            return

        print("\n--- FEATURES ---")
        for k, v in asdict(result).items():
            if k != "statuses":
                print(f"  {k:<22s} {v!r}")

        print("\n--- API DIAGNOSTICS ---")
        total = 0
        for s in result.statuses:
            icon = {"ok": "✓", "no_data": "ⓘ", "tech_fail": "⚠", "skipped": "—"}[s.status]
            print(f"  {icon} [{s.duration_ms:>5} ms] {s.api:<34s} {s.note or ''}")
            if s.error_type:
                print(f"             ERROR: {s.error_type} — {s.error_message}")
            total += s.duration_ms
        print(f"\n  Wall-clock work: ~{total} ms (parallel-adjusted, shorter than sum)")

    async def _demo():
        for tc in TEST_CASES:
            await _run_one(tc["label"], tc["sample"])

    asyncio.run(_demo())