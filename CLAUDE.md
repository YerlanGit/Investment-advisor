# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Detected stack
- Languages: Python 3.11 (aiogram 3.x Telegram bot + quant engine: numpy / pandas / scikit-learn).
- LLM: Anthropic API (Sonnet 4.6 base tier / Opus 4.8 deep tier) via `src/ai_narrative.py` — env-overridable (`ANTHROPIC_MODEL_BASE` / `ANTHROPIC_MODEL_DEEP`); Opus omits `temperature`, so DEEP idea variety comes from the prompt-level freshness directive.
- Infra: GCP Cloud Run (long-polling bot), Cloud Function (RAG ingest), ChromaDB, SQLite on gcsfuse.

## Verification
- Run the Python suite from the repo root: `python -m pytest tests/ -q`
  (CI mirrors this in `.github/workflows/python-ci.yml`; the deploy gate re-runs it in Cloud Build).
- `src/` and `tests/` are both present; update both surfaces together when behavior changes.
- Report templates live in `src/templates/` — smoke-render via `html_renderer.render_report_html(None, ...)` when touching them.

## Repository shape
- `src/` — bot (`tg_bot.py`, `entrypoint.py`), quant engine (`finance/`), report layer (`pdf_payload.py`, `html_renderer.py`, `templates/`), LLM narrative (`ai_narrative.py`), advisory audit (`agent/gatekeeper.py`).
- `tests/` — pytest/unittest suites (`test_phase*.py`); validation surfaces reviewed alongside code changes.
- `cloud_function/` — GCS-triggered RAG ingest (ChromaDB build from bank PDFs).
- `AUDIT.md` — living institutional audit (findings, Было/Стало, dashboard). `MCP_STRATEGY.md` — MCP/LLM strategy. `REPORT_SECTIONS.md` — map of report sections → builders → templates.

## Working agreement
- Prefer small, reviewable changes and keep generated bootstrap files aligned with actual repo workflows.
- Keep shared defaults in `.claude.json`; reserve `.claude/settings.local.json` for machine-local overrides.
- Do not overwrite existing `CLAUDE.md` content automatically; update it intentionally when repo workflows change.
