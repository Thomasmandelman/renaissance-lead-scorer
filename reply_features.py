"""
reply_features.py — Classify a first_reply email into 5 buckets used by
Renaissance's lead-scoring model.

Code here is COPIED 1:1 from the original chat that scored the 21,763
historical leads (and produced the scores now living in Supabase).
DO NOT MODIFY — any divergence introduces classification drift between the
historical leads and leads scored by the MVP.

Public API:
    process_reply(text) -> dict with keys:
        reply_length_bucket, prof_bucket, cleanliness_bucket,
        urgency_bucket, intent_bucket
    Returns None if the text is empty/missing.
    OOO auto-replies return all buckets = "Unknown" (they were excluded from
    training, so they must not compete for partner tiers).
"""
from __future__ import annotations

import re


# ================================================================
# 1) STRIP_NOISE — limpia thread headers, forwards, disclaimers
# ================================================================
def strip_noise(text: str) -> str:
    patterns = [
        r'\n-+\s*On\s+\w+.*?wrote.*?-+\n.*',
        r'\nOn\s+\w+,?\s+\w+.*?wrote:.*',
        r'\nFrom:.*?Subject:.*',
        r'\n-----Original Message-----.*',
        r'\nSent from my (iPhone|Android|Samsung|mobile).*',
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.DOTALL | re.IGNORECASE)

    disclaimer_patterns = [
        r'This (e-?mail|message).*?confidential.*',
        r'Confidentiality Note.*',
        r'The information contained.*?confidential.*',
        r'Nothing in this message.*?electronic signature.*',
    ]
    for p in disclaimer_patterns:
        text = re.sub(p, '', text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()


# ================================================================
# 2) EXTRACT_FEATURES — genera todos los boolean flags
# ================================================================
def extract_features(text):
    if not text or str(text).strip() == 'nan':
        return None

    text = str(text)
    core = strip_noise(text)
    core_len = len(core)

    has_phone = bool(re.search(r'\b\d{3}[-.\s)]?\d{3}[-.\s]?\d{4}\b', text) or
                     re.search(r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}', text))

    has_disclaimer = bool(re.search(
        r'confidential|intended (solely|only) for|unauthorized (use|review|disclosure)|privileged',
        text, re.IGNORECASE))

    has_sig_phrase = bool(re.search(
        r'\b(regards|sincerely|best|thanks|cheers|thank you|best wishes)\s*[,.]?\s*\n',
        text, re.IGNORECASE))

    has_website = bool(re.search(r'(?:https?://|www\.)[\w.-]+\.\w+', text) or
                       re.search(r'\b[\w-]+\.(com|co|net|org|io)\b', text, re.IGNORECASE))

    has_title = bool(re.search(
        r'\b(CEO|CFO|COO|CTO|President|Owner|Founder|Director|Manager|Principal|Chief|VP)\b',
        text))

    has_question = '?' in core

    has_action = bool(re.search(
        r'\b(call|schedule|meet|discuss|talk|connect|chat)\b', core, re.IGNORECASE))

    has_urgency_high = bool(re.search(
        r'\b(asap|urgent|immediately|right away|right now|need (it |funds )?now)\b',
        core, re.IGNORECASE))
    has_urgency_med = bool(re.search(
        r'\b(soon|this week|tomorrow|today|quickly|next few days)\b',
        core, re.IGNORECASE))

    is_affirmative_only = bool(re.search(
        r'^\s*(yes|sure|ok|okay|sounds good|interested)[\s.!]*$',
        core, re.IGNORECASE | re.MULTILINE)) and core_len < 40

    has_need_amount = bool(re.search(
        r'\b(need|require|looking for|want)\s+(?:\$|about\s+\$|around\s+\$)?\s*\d+\s*(k|,\d{3}|million|thousand)',
        core, re.IGNORECASE))

    has_dollar_amount = bool(re.search(
        r'\$\s*\d+\s*(k|,?\d{3}|million|thousand)?', core, re.IGNORECASE) or
        re.search(r'\b\d+\s*k\b', core, re.IGNORECASE))

    has_specific_use = bool(re.search(
        r'\b(for (equipment|expansion|hir\w+|payroll|inventory|renovat\w+|open|buy|purchase|stock)|to (buy|hire|open|expand|stock|purchase))\b',
        core, re.IGNORECASE))

    has_debt = bool(re.search(
        r'\b(pay\s*off|consolidat|refinanc|existing (loan|debt|advance|mca)|buried in debt)\b',
        core, re.IGNORECASE))

    has_skeptical = bool(re.search(
        r"(what'?s the catch|who are you|is this legit|is this real|scam|too good to be true|not interested)",
        core, re.IGNORECASE))

    has_availability = bool(re.search(
        r'\b(available|free|open)\s+(on\s+)?\w+(day)?|\b(tomorrow|today|this week|next week|monday|tuesday|wednesday|thursday|friday)\b.*?\b(at|@)\s*\d+',
        core, re.IGNORECASE))

    is_ooo = bool(re.search(
        r'out of (the )?office|vacation|on leave|will (be|return) back|automatic reply',
        text, re.IGNORECASE))

    prof_signals = sum([has_phone, has_website, has_title, has_sig_phrase])

    return {
        'core_len': core_len,
        'has_phone': int(has_phone),
        'has_disclaimer': int(has_disclaimer),
        'has_sig_phrase': int(has_sig_phrase),
        'has_website': int(has_website),
        'has_title': int(has_title),
        'has_question': int(has_question),
        'has_action': int(has_action),
        'has_urgency_high': int(has_urgency_high),
        'has_urgency_med': int(has_urgency_med),
        'is_affirmative_only': int(is_affirmative_only),
        'has_need_amount': int(has_need_amount),
        'has_dollar_amount': int(has_dollar_amount),
        'has_specific_use': int(has_specific_use),
        'has_debt': int(has_debt),
        'has_skeptical': int(has_skeptical),
        'has_availability': int(has_availability),
        'is_ooo': int(is_ooo),
        'prof_signals': prof_signals,
    }


# ================================================================
# 3) BUCKETIZE — convierten los flags a bins del Excel scoring
# ================================================================
def bin_length(c):
    if c < 30:  return "Very Short (<30)"
    if c < 75:  return "Short (30-75)"
    if c < 150: return "Medium (75-150)"
    if c < 300: return "Detailed (150-300)"
    return "Long (300+)"


def bin_prof(n):
    if n == 0: return "0 Signals"
    if n == 1: return "1 Signal"
    if n == 2: return "2 Signals"
    return "3+ Signals"


def classify_clean(row):
    # Mutually exclusive — orden de prioridad importa
    if row['has_disclaimer'] and row['has_phone'] and row['has_title']:
        return "Disclaimer + Phone + Title"
    if row['has_disclaimer']:
        return "Has Legal Disclaimer only"
    if row['has_phone'] and row['has_title']:
        return "Phone + Title (no disclaimer)"
    if row['has_phone']:
        return "Phone only"
    if row['has_sig_phrase']:
        return "Signature phrase only"
    return "Clean Message (no markers)"


def classify_urgency(row):
    if row['has_urgency_high']:  return "High Urgency"
    if row['has_urgency_med']:   return "Medium Urgency"
    if row['has_availability']:  return "Specific Availability"
    return "No Urgency Words"


def classify_intent(row):
    # Orden intencional — primer match gana (más específico primero)
    if row['has_debt']:            return "Debt/Refinance"
    if row['has_specific_use']:    return "States Use"
    if row['has_need_amount'] or row['has_dollar_amount']: return "Mentions $ Amount"
    if row['has_availability']:    return "Provides Availability"
    if row['has_action']:          return "Has Action Word"
    if row['has_question']:        return "Asks Questions"
    if row['is_affirmative_only']: return "Simple Yes"
    return "Other/Unclear"


# ================================================================
# 4) PIPELINE público
# ================================================================
def process_reply(reply_text):
    """
    Take a reply text and return the 5 buckets ready for scoring-table lookup.
    OOO autoresponders return all buckets = 'Unknown' (excluded from training).
    Returns None if text is empty/missing.
    """
    features = extract_features(reply_text)
    if features is None:
        return None

    if features['is_ooo']:
        return {
            'reply_length_bucket': 'Unknown',
            'prof_bucket':         'Unknown',
            'cleanliness_bucket':  'Unknown',
            'urgency_bucket':      'Unknown',
            'intent_bucket':       'Unknown',
        }

    return {
        'reply_length_bucket': bin_length(features['core_len']),
        'prof_bucket':         bin_prof(features['prof_signals']),
        'cleanliness_bucket':  classify_clean(features),
        'urgency_bucket':      classify_urgency(features),
        'intent_bucket':       classify_intent(features),
    }