"""
rescore_all_leads_fast.py — Fast version.

Computes new scores locally (identical to rescore_all_leads.py), then writes
a single SQL file with all the UPDATEs. You paste that file into Supabase SQL
Editor and it runs in under a minute.

Why this is faster: the HTTP latency of the supabase-py client dominates
(~500ms per request). Bundling thousands of UPDATEs into one SQL file run by
the SQL Editor eliminates that overhead — Postgres processes the batch in
seconds.

Output: rescored_updates.sql  (you paste it into Supabase SQL Editor)

Nothing is touched in Supabase by this script. It only READS to fetch current
leads, then writes a local .sql file. The actual UPDATEs happen when you
execute that SQL.
"""
import os
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

import scoring
import reply_features

from supabase import create_client

SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

SCHEMA = "scoring"
TABLE  = "lead_score"
OUTPUT_SQL = "rescored_updates.sql"

tables = scoring.load_tables("scoring_tables.json")
PARTNER_ROUTING = tables["partner_routing"]


def partner_for_percentile(pct: float) -> str:
    for entry in PARTNER_ROUTING:
        if pct <= entry["max_percentile"]:
            return entry["partner"]
    return PARTNER_ROUTING[-1]["partner"]


def reprocess_reply_text(reply_text):
    default = {
        "reply_length_bucket": "Short (30-75)",
        "prof_bucket":         "0 Signals",
        "cleanliness_bucket":  "Clean Message (no markers)",
        "urgency_bucket":      "No Urgency Words",
        "intent_bucket":       "Other/Unclear",
    }
    if not reply_text:
        return default
    classified = reply_features.process_reply(reply_text)
    if classified is None:
        return default
    return {k: classified[k] for k in default.keys()}


def clean_timestamp(v):
    """
    Normalize timestamps. Some historical rows store the literal string 'nan'
    in reply_timestamp — scoring.py's _parse_utc can't handle that and raises
    ValueError. Treat those (and other empty-ish values) as None so the
    scoring falls back to Unknown for Month/Day/Time features.
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "nat", "null"):
        return None
    return str(v)


def sql_escape(s):
    """Escape a value for a SQL literal. None -> NULL. Strings: single-quote escape."""
    if s is None:
        return "NULL"
    if isinstance(s, (int, float)):
        return str(s)
    # String
    return "'" + str(s).replace("'", "''") + "'"


def main():
    print("="*72)
    print("FAST RESCORE — generates SQL file, no slow HTTP round-trips")
    print("="*72)
    print("Adjusted bucket values in scoring_tables.json:")
    print(f"  ReplyLength['Very Short (<30)']         = {tables['scores']['ReplyLength']['Very Short (<30)']}")
    print(f"  Cleanliness['Signature phrase only']    = {tables['scores']['Cleanliness']['Signature phrase only']}")
    print(f"  Intent['Has Action Word']               = {tables['scores']['Intent']['Has Action Word']}")
    print(f"  Urgency['Medium Urgency']               = {tables['scores']['Urgency']['Medium Urgency']}")
    print()

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    print("Step 1/3: Fetching all leads...")
    all_leads = []
    page = 0
    page_size = 1000
    while True:
        resp = (
            client.schema(SCHEMA).table(TABLE)
            .select(
                "id, first_reply, reply_timestamp, industry, job_title, "
                "employees, years_in_business, num_locations, state, "
                "has_website, has_gmb, has_linkedin, has_facebook, "
                "has_instagram, has_trustpilot, "
                "score_raw, percentile, partner"
            )
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        all_leads.extend(batch)
        print(f"   ...fetched {len(all_leads):,}")
        if len(batch) < page_size:
            break
        page += 1

    total = len(all_leads)
    if total == 0:
        print("No leads to process. Exiting.")
        return
    print(f"Total: {total:,}\n")

    print("Step 2/3: Recomputing scores locally...")
    new_scores = []
    errors = 0
    for i, lead in enumerate(all_leads, 1):
        try:
            reply_buckets = reprocess_reply_text(lead.get("first_reply"))
            features = {
                "years_in_business":     lead.get("years_in_business"),
                "industry":              lead.get("industry"),
                "employees":             lead.get("employees"),
                "state":                 lead.get("state"),
                "num_locations":         lead.get("num_locations"),
                "job_title":             lead.get("job_title"),
                "reply_timestamp_utc":   clean_timestamp(lead.get("reply_timestamp")),
                **reply_buckets,
                "has_website":           lead.get("has_website"),
                "has_gmb":               lead.get("has_gmb"),
                "has_linkedin":          lead.get("has_linkedin"),
                "has_facebook":          lead.get("has_facebook"),
                "has_instagram":         lead.get("has_instagram"),
                "has_trustpilot":        lead.get("has_trustpilot"),
            }
            result = scoring.compute_score(features, tables)
            new_scores.append({
                "id":             lead["id"],
                "new_score_raw":  result["score_raw"],
                "old_score_raw":  lead.get("score_raw"),
                "old_partner":    lead.get("partner"),
            })
        except Exception as e:
            errors += 1
            print(f"   ⚠️ Error on id={lead.get('id')}: {type(e).__name__}: {e}")
    print(f"Rescored: {len(new_scores):,}  Errors: {errors}\n")

    print("Step 3/3: Computing percentiles + writing SQL file...")
    sorted_scores = sorted([r["new_score_raw"] for r in new_scores], reverse=True)
    # Map score -> rank (lowest rank for ties = best percentile)
    score_to_rank = {}
    for rank, s in enumerate(sorted_scores):
        if s not in score_to_rank:
            score_to_rank[s] = rank
    n = len(sorted_scores)
    for r in new_scores:
        rank = score_to_rank[r["new_score_raw"]]
        r["new_percentile"] = round(100 * rank / n, 2) if n > 0 else 0.0
        r["new_partner"]    = partner_for_percentile(r["new_percentile"])

    # Write SQL file using a CTE with VALUES — single statement, fast.
    with open(OUTPUT_SQL, "w", encoding="utf-8") as f:
        f.write("-- Auto-generated by rescore_all_leads_fast.py\n")
        f.write(f"-- Rescores {len(new_scores):,} leads with the adjusted scoring_tables.json\n")
        f.write("-- Single-statement UPDATE from a VALUES list — runs in seconds.\n\n")
        f.write("UPDATE scoring.lead_score AS ls SET\n")
        f.write("  score_raw  = v.new_score_raw,\n")
        f.write("  percentile = v.new_percentile,\n")
        f.write("  partner    = v.new_partner\n")
        f.write("FROM (VALUES\n")

        lines = []
        for r in new_scores:
            lines.append(
                f"  ({r['id']}, {r['new_score_raw']}, "
                f"{r['new_percentile']}, {sql_escape(r['new_partner'])})"
            )
        f.write(",\n".join(lines))
        f.write("\n) AS v(id, new_score_raw, new_percentile, new_partner)\n")
        f.write("WHERE ls.id = v.id;\n")

    print(f"\n✅ SQL file written: {OUTPUT_SQL}")
    print(f"   Contains {len(new_scores):,} row updates.")
    print()
    print("─" * 72)
    print("NEXT STEPS:")
    print(f"  1. Open {OUTPUT_SQL} in a text editor")
    print("  2. Copy ALL the contents")
    print("  3. Paste into Supabase SQL Editor")
    print("  4. Click Run")
    print("  5. You'll see '{n:,} rows affected' — done in seconds.".format(n=len(new_scores)))
    print("─" * 72)
    print()

    # Summary of changes
    partner_changes = Counter()
    for r in new_scores:
        if r["new_partner"] != r["old_partner"]:
            partner_changes[(r["old_partner"] or "—", r["new_partner"])] += 1

    if partner_changes:
        print(f"Partner transitions: {sum(partner_changes.values()):,} leads moved tier")
        for (old, new), count in sorted(partner_changes.items(), key=lambda x: -x[1])[:15]:
            print(f"   {old:<14} → {new:<14} : {count:,}")
    else:
        print("No partner transitions.")

    diffs = [r["new_score_raw"] - (r["old_score_raw"] or 0) for r in new_scores if r["old_score_raw"] is not None]
    if diffs:
        avg = sum(diffs) / len(diffs)
        print(f"\nScore delta: avg {avg:+.2f}   min {min(diffs):+.2f}   max {max(diffs):+.2f}")


if __name__ == "__main__":
    main()