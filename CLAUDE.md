# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Detected stack
- Languages: Python 3.11 (aiogram 3.x Telegram bot + quant engine: numpy / pandas / scikit-learn). `requirements.txt` carries **upper bounds** (major caps; human intent file); `requirements.lock` is the **hash-locked resolution** the Docker image and CI install (`--require-hashes`) — regenerate via `pip-compile --generate-hashes --strip-extras -o requirements.lock requirements.txt` on linux/py3.11 whenever requirements.txt changes.
- LLM: Anthropic API (Sonnet 4.6 base tier / Opus 4.8 deep tier) via `src/ai_narrative.py` — env-overridable (`ANTHROPIC_MODEL_BASE` / `ANTHROPIC_MODEL_DEEP`); Opus omits `temperature`, so DEEP idea variety comes from the prompt-level freshness directive.
- Report rendering: **Premium V2 React is the production default** (`PREMIUM_REPORT_ENABLED` default true in `src/html_renderer.py`). Classic v3 Jinja (`src/templates/report_*_v3.html`) is the RETAINED auto-fallback (try/except in `render_report_html`) and is still test-pinned — set `PREMIUM_REPORT_ENABLED=false` to force it. Pipeline: `finance/*` → `pdf_payload.build_payload` (compute/format) → `premium_payload.build_design_data` (view-map) → `premium_renderer`.
- Infra: GCP Cloud Run (long-polling bot), Cloud Function (RAG ingest), ChromaDB, SQLite on gcsfuse.

## Verification
- Run the Python suite from the repo root: `python -m pytest tests/ -q`
  (CI mirrors this in `.github/workflows/python-ci.yml`; the deploy gate re-runs it in Cloud Build). Baseline: **511 passed, 10 skipped**.
- `src/` and `tests/` are both present; update both surfaces together when behavior changes.
- Report templates live in `src/templates/` — smoke-render via `html_renderer.render_report_html(None, ...)`; Premium bundles rebuild via `design/premium_v2/build.sh` (Tailwind step runs from repo root) → synced to `src/premium_assets/`.

## Repository shape
- `src/` — bot (`tg_bot.py`, `entrypoint.py`; UI/routing only — series math lives in `finance/portfolio_series.py`), quant engine (`finance/`), report layer (`pdf_payload.py`, `premium_payload.py`, `premium_renderer.py`, `html_renderer.py`, `templates/`), LLM narrative (`ai_narrative.py`), advisory audit (`agent/gatekeeper.py`).
- `tests/` — pytest/unittest suites (`test_phase*.py`); validation surfaces reviewed alongside code changes.
- `cloud_function/` — GCS-triggered RAG ingest (ChromaDB build from bank PDFs). Two-bucket flow: PDFs land in **INBOX** `ramp-bot-chroma-db-inbox-investadv` (the Cloud Function trigger, **path A**); the built ChromaDB lives in **STORE** `ramp-bot-chroma-db-investadv` and the bot mirrors it on boot (`entrypoint._download_chroma_db`). **Reliable fallback (path B, 2026-07-04):** `entrypoint._boot_ingest_from_inbox` — if the synced STORE is empty but the INBOX holds PDFs, the bot ingests them in-container on a daemon thread (deps + pre-baked ONNX model + pinned chromadb present) and publishes the store back to STORE, bypassing the fragile Cloud-Function/Eventarc/region chain (`RAG_BOOT_INGEST=0` to disable; `cloud_function/rag_engine.py` is kept identical to `src/agent/rag_engine.py`; the function's `chromadb` is pinned `==0.6.3`). `scripts/ingest_bank_report.py` — admin CLI for fast/manual PDF ingest (`--upload` publishes ChromaDB to STORE). RAG engine `src/agent/rag_engine.py` (chromadb imported lazily): section-aware size-bounded chunking + `bank`/`section`/`tickers` metadata + recency ranking + optional ticker filter. RAG observability (2026-07-04): `_fetch_rag_context` surfaces KB inventory (docs/chunks); CoVe carries a Bank-RAG inventory row + an `ИИ-цитирование банк-аналитики` checker that flags bank consensus the model emits from memory vs. verified `[RAG:file]` citations.
- Quant notes: 4-Pillar F-pillar carries a bounded ±0.5 fundamental-momentum bonus (`SEC_Fundamental_Momentum` = YoY margin trend; opt-in, `None`→no effect; Piotroski-F stays in the Credit pillar — no double-count). Regime macro-overlay (`REGIME_MACRO_OVERLAY`, default-on) tilts Growth×Cycle with GDP/unemployment/10Y-breakeven-inflation level ⊕ rate-of-change (inflation → growth axis via the policy channel), each bounded ±0.05.
- `AUDIT.md` — living institutional audit (findings, Было/Стало, dashboard). `MCP_STRATEGY.md` — MCP/LLM strategy. `REPORT_SECTIONS.md` — map of report sections → builders → templates. `RAG_INGESTION.md` — step-by-step bank-report ingestion guide; `docs/RAG_TROUBLESHOOTING.md` — 8-step runbook for a stubbornly-empty RAG (bucket-name reconciliation, function trigger/deploy, boot-only sync). `docs/REGIME_SECTION_DEEP.md` / `docs/REPORT_PAGES_DEEP.md` — DEEP section deep-dives (regime overlay incl. inflation nudge, CoVe rows).

## Working agreement
- Prefer small, reviewable changes and keep generated bootstrap files aligned with actual repo workflows.
- Keep shared defaults in `.claude.json`; reserve `.claude/settings.local.json` for machine-local overrides.
- Do not overwrite existing `CLAUDE.md` content automatically; update it intentionally when repo workflows change.
