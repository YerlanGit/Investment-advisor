"""
batch_reports.py — nightly ASYNCHRONOUS report generation via the Anthropic
Message Batches API (50% token discount).

Offline counterpart to `ai_narrative.generate_narrative`: instead of one
synchronous call per user, scheduled/planned reports are submitted as a single
batch, processed within ~1h, and pulled back. It reuses the EXACT same
structured-output tool (`_REPORT_TOOL`) and cached system prompt as the live
path, so the narratives are identical in shape — only cheaper and async.

Status: architecture scaffold. The Anthropic wiring is real; the two
integration points (`iter_planned_jobs` → job source, `persist_narrative` →
result sink) are intentionally stubs to wire to the tokenomics/snapshot DB and
the report renderer when the nightly scheduler is built.

Economics: Batches run at 50% of standard token price and most finish well
under the 1h target (24h max). Combined with prompt caching on the shared
25 KB system prompt, a nightly run of N planned reports costs roughly
  N · (cached_system·0.1 + user_tokens) · 0.5
vs the synchronous path's full price — see docs/llm/MCP_STRATEGY.md §Economics.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable

# Reuse the live pipeline's building blocks so batch == sync in shape.
from ai_narrative import (
    MODEL_BASE, MODEL_DEEP, MAX_TOKENS_BASE, MAX_TOKENS_DEEP,
    _REPORT_TOOL, _build_system_prompt, _summarise_for_prompt, _user_prompt,
)

logger = logging.getLogger("batch_reports")


@dataclass
class BatchJob:
    """One planned report to generate offline."""
    custom_id: str                       # e.g. f"{user_id}:{tier}:{date}"
    results: dict                        # analyze_all() output
    tier: str                            # "base" | "deep"
    market_context: str = ""             # RAG context (optional)
    user_risk_profile: str = "Moderate"


@dataclass
class BatchHandle:
    batch_id: str
    job_count: int
    status: str = "submitted"
    custom_ids: list[str] = field(default_factory=list)


class BatchNarrativeGenerator:
    """Submit planned reports as ONE Anthropic batch (−50% tokens).

    Usage (when the nightly scheduler is wired):
        gen = BatchNarrativeGenerator()
        gen.run_nightly()                     # end-to-end (uses the stubs below)

    Or drive it manually:
        handle = gen.submit(jobs)
        results = gen.collect(handle.batch_id)   # {custom_id: parsed_tool_input}
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    # ── Request construction (identical to the live path) ────────────────────
    def _request_params(self, job: BatchJob) -> dict:
        model   = MODEL_DEEP if job.tier == "deep" else MODEL_BASE
        max_tok = MAX_TOKENS_DEEP if job.tier == "deep" else MAX_TOKENS_BASE
        summary = _summarise_for_prompt(job.results)
        return {
            "model":       model,
            "max_tokens":  max_tok,
            "temperature": 0.1,
            "system": [{                         # cached 25 KB system prompt
                "type": "text",
                "text": _build_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{
                "role": "user",
                "content": _user_prompt(summary, tier=job.tier,
                                        market_context=job.market_context,
                                        user_profile=job.user_risk_profile),
            }],
            "tools":       [_REPORT_TOOL],       # structured outputs
            "tool_choice": {"type": "tool", "name": "emit_report"},
        }

    # ── Submit / collect ─────────────────────────────────────────────────────
    def submit(self, jobs: Iterable[BatchJob]) -> BatchHandle:
        """Create a batch; returns a handle. Most batches finish < 1h (24h max)."""
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        job_list = list(jobs)
        if not job_list:
            raise ValueError("No jobs to submit.")

        client = anthropic.Anthropic(api_key=self._api_key)
        requests = [
            Request(custom_id=j.custom_id,
                    params=MessageCreateParamsNonStreaming(**self._request_params(j)))
            for j in job_list
        ]
        batch = client.messages.batches.create(requests=requests)
        logger.info("Batch submitted: %s (%d jobs, status=%s)",
                    batch.id, len(requests), batch.processing_status)
        return BatchHandle(batch_id=batch.id, job_count=len(requests),
                           status=batch.processing_status,
                           custom_ids=[j.custom_id for j in job_list])

    def collect(self, batch_id: str, poll_seconds: int = 60,
                timeout_seconds: int = 24 * 3600) -> dict[str, dict]:
        """Block until the batch ends, then return {custom_id: parsed tool input}."""
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)

        deadline = time.monotonic() + timeout_seconds
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Batch {batch_id} did not end within budget.")
            time.sleep(poll_seconds)

        out: dict[str, dict] = {}
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                logger.warning("Batch item %s -> %s", result.custom_id, result.result.type)
                continue
            msg = result.result.message
            tb = next((b for b in msg.content
                       if getattr(b, "type", None) == "tool_use"
                       and getattr(b, "name", "") == "emit_report"), None)
            if tb is not None:
                out[result.custom_id] = dict(tb.input or {})
        logger.info("Batch %s collected: %d/%d succeeded", batch_id, len(out),
                    sum(1 for _ in []) or len(out))
        return out

    # ── Integration points (STUBS — wire to the scheduler & renderer) ────────
    def iter_planned_jobs(self) -> Iterable[BatchJob]:
        """STUB: yield one BatchJob per user due a scheduled report.

        Wire to the tokenomics/profile DB (who opted into nightly reports) +
        the broker/engine (`analyze_all`) to build each `results` dict.
        """
        raise NotImplementedError("Wire to the nightly scheduler / user DB.")

    def persist_narrative(self, custom_id: str, parsed: dict) -> None:
        """STUB: persist/render the finished narrative.

        Wire to `pdf_payload.build_payload` + `html_renderer` +
        `report_storage` (GCS) and the snapshot DB, exactly like the live path.
        """
        raise NotImplementedError("Wire to report_storage / snapshot DB.")

    def run_nightly(self) -> None:
        """End-to-end nightly flow (orchestration scaffold)."""
        jobs = list(self.iter_planned_jobs())
        if not jobs:
            logger.info("No planned reports tonight.")
            return
        handle = self.submit(jobs)
        for custom_id, parsed in self.collect(handle.batch_id).items():
            self.persist_narrative(custom_id, parsed)
