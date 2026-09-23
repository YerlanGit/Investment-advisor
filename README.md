# RAMP — Institutional Portfolio Risk-Analysis Telegram Bot

RAMP is a Telegram bot that runs an **institutional-grade, MAC3/Barra-style
risk engine** over a user's real brokerage portfolio and returns a banker-grade
HTML report — multi-factor risk decomposition, tail risk, stress tests, and an
AI-written narrative — in Russian.

> Pipeline: **Engine (`src/finance/*`) → Adapter (`src/pdf_payload.py`) →
> Premium V2 React (`src/premium_payload.py` + `src/premium_assets/`, default) /
> Jinja2 v3 fallback (`src/templates/report_*_v3.html`)**, with
> `src/ai_narrative.py` (Claude) for the prose layer.

---

## What it does

1. **Connects to a portfolio** — Freedom Broker (Tradernet) API via encrypted
   per-user credentials, or a template portfolio.
2. **Runs the MAC3 risk engine** (`src/finance/investment_logic.py`):
   - **Base Currency Approach** — every price is converted to the reporting
     currency *before* returns/covariance, so FX risk flows into the matrix
     correctly (`src/finance/currency.py`). Assets with no cross-rate are
     dropped, never mixed into one covariance matrix.
   - **Factor model + Euler decomposition** — Ledoit-Wolf/EWMA covariance,
     marginal & component contribution to risk (TRC).
   - **Geometric annualisation** of the book's own log return
     `ln(1 + Σ wᵢ·Rᵢ)` (the log of the weighted SIMPLE sum — not a weighted sum
     of logs, which drops the diversification return), currency-matched
     **Sharpe / Sortino**, and a per-currency risk-free rate (static default
     USD 4.5% / KZT 14%, override via `US_RFR_ANNUAL` / `KZ_RFR_ANNUAL`; the QC
     panel prints which one was used). Margin debt accrues at that rate.
   - **Historical VaR / bootstrap CVaR** (deterministic, reproducible seed),
     **stress tests** with a smooth convexity cap, **Black-Litterman** views
     (posterior returns are EXCESS returns over rf), and a 4-pillar composite
     score.
3. **Adds context** — RAG over bank-analytic PDFs (ChromaDB) and macro/regime
   signals.
4. **Renders a report** — `pdf_payload.build_payload()` maps engine output to
   the template schema; `html_renderer.render_report_html()` renders the v3
   banker template and ships a signed Cloud Storage URL to the user.
5. **Bills tokens** — atomic SQLite tokenomics; the user is charged only after
   the report is delivered. Pricing (2026-07-17): **1 token = 2 500 KZT**,
   pack of 10 = 25 000 KZT (`tg_bot.TOKEN_PRICE_KZT`; tiers: base/scenario
   1 token, deep 2, demo reports free).
6. **Mandate menu** — `/mandate` shows the current mandate and edits it in
   place (benchmark in two taps, asset classes, risk profile, full re-quiz) —
   free of charge; the chosen benchmark propagates end-to-end, including the
   DEEP factor-decomposition column (`benchmark_factor_profile`).

## Two services, one image

| Service | Entry point | Writes | Isolation |
|---|---|---|---|
| **Report bot** | `src/entrypoint.py` (image `CMD`) | tokenomics, key vault, report HTML | own token, own Cloud Run service |
| **Data bot** (quote loader) | `src/ingest_entrypoint.py` (`--command` override) | `stooq/prices.sqlite` via compare-and-swap | own token (a shared token is refused at start) |

Both run from the same image but in separate containers, event loops and
state files; the report bot reads a local read-only copy of the quote base.
The code boundary (shared core / report bot / data bot, lazy imports included)
is an executable manifest — `tests/test_service_boundaries.py`; the physical
split plan is `docs/ARCHITECTURE_FOR_AGENTS.md §8`.

## Repository layout

```text
src/
├── entrypoint.py            # Cloud Run entry: health server + bot polling + ChromaDB sync
├── ingest_entrypoint.py     # SECOND service: quote-loader bot (health + scheduled check)
├── ingest_bot.py            # loader bot: Stooq daily/history files → prices.sqlite (CAS)
├── tg_bot.py                # aiogram 3.x bot: FSM onboarding, analysis flow, billing
├── db_tokenomics.py         # async SQLite token ledger (atomic, WAL)
├── pdf_payload.py           # ADAPTER: engine results -> template schema
├── html_renderer.py         # Jinja2 env + template selection (v3)
├── pdf_charts.py            # inline SVG charts (equity curve, sector donut, sparklines)
├── ai_narrative.py          # Claude narrative (prompt-injection-fenced)
├── profile_manager.py       # risk-mandate profiles & benchmarks
├── finance/                 # THE RISK ENGINE
│   ├── investment_logic.py  #   MAC3 engine + UniversalPortfolioManager facade
│   ├── currency.py          #   Base-currency FX transformation
│   ├── scoring.py / scoring_orchestrator.py   # 4-pillar scoring
│   ├── black_litterman.py / simulate.py / stress.py / regime.py
│   ├── period_returns.py / technicals.py / action_plan.py
│   ├── manual_portfolio.py / stooq_*.py       # manual book + Stooq quote base
│   ├── broker_api.py / security.py            # broker + Fernet vault
│   └── ...
├── freedom_portfolio/       # Tradernet client + price history (history.py)
├── agent/                   # advisor bot, RAG engine, gatekeeper
├── services/                # GCS report storage, FRED macro feed, FX feed, quote publisher/ingest
├── premium_assets/          # built Premium V2 bundles (from design/premium_v2 via build.sh)
└── templates/               # report_{basic,deep,scenario}_v3.html (Jinja fallback + scenario tier)
cloud_function/              # RAG-ingest Cloud Function (bank PDFs -> ChromaDB)
tests/                       # test_phase*.py — hermetic engine suite (deploy gate)
```

## Running the tests

The hermetic engine suite (no network, mocks FRED/Anthropic/Tradernet) is the
Cloud Build deploy gate:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_phase*.py"
# or
PYTHONPATH=src python -m pytest tests/ -q
```

Mobile/DOM gates (`test_phase55`, `test_phase62`) need Playwright + Chromium and
skip themselves otherwise. Exact expected counts for both verification runs
(full checkout and the deploy-image mirror) live in `CLAUDE.md`.

## Deployment

Containerised (`Dockerfile`, runs as a non-root user) and deployed to **Google
Cloud Run** via `cloudbuild.yaml`:

- **Build → hermetic test gate → push → deploy** (a red test blocks the deploy).
- **Durable state** — `tokenomics.db` and the Fernet `users_vault.db` live on a
  gcsfuse-mounted Cloud Storage volume (`/mnt/state`); the app refuses to boot
  if these point at ephemeral storage (`assert_persistent_state`).
- **Secrets** — `OMBRI_BOT_TOKEN` (прежнее имя `RAMP_BOT_TOKEN` ещё принимается), `FINTECH_MASTER_KEY`, `ANTHROPIC_API_KEY`,
  `FREEDOM_API_KEY/SECRET`, `FRED_API_KEY` are injected from Secret Manager,
  never baked into the image. `FINTECH_MASTER_KEY` supports `MultiFernet`
  rotation (`NEW_KEY,OLD_KEY`).
- **Single instance** (`max-instances=1`, `concurrency=1`) — SQLite is not a
  multi-writer store.

### Required environment

See `.env.template`. Key variables: `OMBRI_BOT_TOKEN`, `FINTECH_MASTER_KEY`,
`ANTHROPIC_API_KEY`, `FREEDOM_API_KEY`, `FREEDOM_API_SECRET`, `FRED_API_KEY`,
`TOKENOMICS_DB_PATH`, `VAULT_DB_PATH`, `REPORTING_CURRENCY`, `ADMIN_USER_IDS`.

## Disclaimer

RAMP produces analytical reports for informational purposes only; it is not
investment advice. Stress-test figures are illustrative (smooth convexity cap).
