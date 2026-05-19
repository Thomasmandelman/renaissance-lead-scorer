# Renaissance Lead Scoring System
### Architecture & Impact Summary

---

## 📌 Overview
> 🔒 **Confidentiality Notice:** Due to non-disclosure agreements (NDAs) regarding financial partners, proprietary scoring algorithms, and sensitive lead data, the source code of certain modules, live demo access, and datasets cannot be publicly shared. The architecture described below reflects the real production system.
The **Renaissance Lead Scoring System** is a production-grade AI-powered platform that automates the routing of inbound leads to the optimal financial partner. It combines real-time data enrichment from multiple external APIs, a calibrated predictive scoring model, and a relational database backbone that turns every lead interaction into structured, queryable data — enabling both operational efficiency and long-term analytical value.

The system was designed end-to-end as a single integrated solution: data enrichment, scoring, application interface, and database layer all work together to remove manual decision-making from the lead-routing process while building a complete data asset for the company.

---

## 🛠️ System Architecture

### Layer 1 — Data Enrichment Pipeline
When a lead enters the system, it carries minimal information: company name, email, contact name, and optionally a website. The enrichment pipeline transforms that minimal input into a **structured profile of 13 features** used by the scoring model.

*   **How it works:** The pipeline orchestrates four data sources in a coordinated cascade. Sources run in parallel where possible to minimize latency, and sequentially only where data dependencies require it.
    *   **Google Gemini (grounded search):** The workhorse for industry classification, years in business, location resolution, website discovery, USA validation, and digital-presence detection. Output is constrained to closed enums and validated as JSON to prevent hallucinations.
    *   **Apollo.io:** Precise but narrow. Used for two specific signals: employee count and contact job title. Skipped automatically when the lead has a personal email and no website to save API quota.
    *   **Google Places API:** Used for Google Business Profile presence and multi-location detection. Acts as a location fallback when Gemini returns null.
    *   **Lightweight HTTP scraping:** A free signal layer that detects direct social media links (Facebook, Instagram, LinkedIn, Trustpilot) directly from the company's homepage.
*   **Reliability and cost:** Per-lead latency averages **6 to 10 seconds** end-to-end (down from ~30s in sequential). Cost lands around **USD 0.03 to 0.05 per lead** — significantly cheaper than off-the-shelf alternatives. Each API call returns its own status (`ok`, `no_data`, `tech_fail`, `skipped`), allowing intelligent fallbacks or retries.

### Layer 2 — Predictive Scoring Engine
Once enriched, each lead is scored using a **13-feature predictive model** calibrated against thousands of historical funding outcomes. The model produces a raw score, a percentile rank against the entire active lead base, and a recommended partner.

*   **Feature composition:** The 13 features combine company attributes, digital presence (aggregated into a 0-6 score), and reply characteristics (length, professionalism, urgency, intent).
*   **Routing logic:** Leads are routed to one of seven financial partners based on percentile rank. The system distinguishes between the model's recommendation and the partner the lead was actually sent to, preserving historical truth for ongoing calibration.
*   **Honest scoring:** When enrichment cannot confirm a critical feature (e.g., USA operations), the affected features are neutralized rather than fabricated, preventing bad data from contaminating the database.

### Layer 3 — Production Application
The application is built in Python with **Streamlit** and deployed to Streamlit Cloud. It serves as the daily operational interface for Inbox Managers (IMs).

*   **Core workflows:**
    *   **Score Lead:** Runs enrichment, scoring, and routing in ~25 seconds and persists results with full audit metadata.
    *   **Update Meeting:** Tracks the lifecycle of meetings (scheduled, completed, no-show, cancelled) distinguishing between user rescheduling and data-entry corrections.
    *   **Duplicate detection:** Surfaces existing records if an email exists, letting the IM resolve or create a separate engagement.
    *   **Soft-delete with audit trail:** Leads marked as test data or errors are soft-deleted with reasons and notes, preserving history without polluting active operations.
*   **Authentication and access:** Uses session-based authentication. All sensitive operations are gated, and external API keys are managed through Streamlit secrets — *never committed to source control.*

### Layer 4 — Database & Analytics Layer
Behind the application sits a **PostgreSQL database (Supabase)** designed as a long-term data asset.

*   **Schema design:**
    *   `Lead Score` table (operational backbone with full metadata).
    *   `Meeting History` table (tracks state transitions with timestamps).
    *   `Funded Events` table (handles multiple funding outcomes like line of credit renewals).
    *   `Analytical views` (SQL views aggregating data for clean reporting).
*   **Data integrity:** Foreign key relationships protect referential integrity. Bulk migrations are validated by sampling via verification queries before execution.
*   **Historical consolidation:** The database was seeded with **22,000+ historical leads and 1,400+ funded events** (representing approx. **USD 77.8M** in tracked funding) consolidated from legacy CSV and Excel sources.

---

## 📈 Impact

### Operational
*   Lead routing is automated end-to-end based on a calibrated model, removing arbitrary criteria.
*   Per-lead processing time dropped from minutes of manual lookup to **under 30 seconds** of automated enrichment + scoring.
*   Data entry errors are caught at input time rather than discovered weeks later.

### Analytical
*   Every lead can now be analyzed by campaign, workspace, advisor, industry, geography, and outcome.
*   Funded outcomes are matched back to predictions, enabling continuous calibration of the scoring model.
*   Top-performing campaigns are identified by **funded conversion rate**, not vanity metrics.

### Strategic
*   Turns lead operations into a real data asset that compounds in value over time, driving a structural increase in company valuation.

---

## 🚀 Technology Stack

*   **Languages:** Python (`asyncio`, `httpx`, `dataclasses`), SQL (PostgreSQL, T-SQL).
*   **AI / APIs:** Google Gemini 2.5 Flash (grounded search), Anthropic Claude (development tooling), Apollo.io, Google Places API.
*   **Backend:** Supabase (PostgreSQL) with validated schemas, foreign keys, and CHECK constraints.
*   **Frontend & DevOps:** Streamlit (Streamlit Cloud with secrets management), Git/GitHub, resumable batch jobs.
