"""
Smart-Money / insider-flow signal layer (BLOCK 3.5 — foundation).

Goal
────
Surface "smart money" conviction — corporate INSIDERS buying/selling their own
stock — as an additional, auditable signal for the DEEP report.  Insider
clusters (several officers/directors buying in the same window) are one of the
few legal signals with documented forward-return predictive power.

Scope of THIS module (deliberately bounded — see "убери ненужное")
──────────────────────────────────────────────────────────────────
This is the FREE, self-hostable FOUNDATION, not the full build:

  ✅ data model            — InsiderSignal (one row per ticker)
  ✅ ingestion INTERFACE   — build_insider_signals(...), pluggable `fetch`
  ✅ scoring               — deterministic cluster/net-flow → [-2, +2] score
  ✅ CoVe hook             — insider_lineage_row(...) for the data-lineage panel
  ✅ gating                — OFF by default (SMART_MONEY_INSIDERS=1 to enable)

  ⏸ DEFERRED (need paid / heavyweight infra, not needed at this stage):
     • LIVE SEC Form 4 XML parsing at scale (rate-limited EDGAR crawl + a
       robust ownershipDocument parser + a nightly Cloud Function cache).
     • Congressional / politician trades (STOCK Act PTR filings) — only
       available cleanly via PAID aggregators (Quiver, Capitol Trades).
     • Government-stimulus / contract-award flows (USASpending bulk data) —
       large ingest + entity-resolution problem.
   The interface below is shaped so these become additional `fetch` providers
   without touching the scoring or the report wiring.

Data source (free)
──────────────────
SEC EDGAR Form 4 ("Statement of Changes in Beneficial Ownership"), filed within
2 business days of an insider transaction.  Full-text + structured XML:
    https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4&CIK=<cik>
    https://data.sec.gov/submissions/CIK<10-digit>.json   (filing index)
A compliant client MUST send a descriptive User-Agent and respect ≤10 req/s.

Predictive-model design (brief)
───────────────────────────────
The signal this module produces is meant to feed, not replace, the 4-Pillar
score.  Three model tiers, increasing in ambition:

  1. RULE / EVENT-STUDY (this module's scorer; ship first):
       net_flow_usd, buy_count, sell_count, distinct_insiders over a 90d window
       → cluster_flag = (distinct_buyers ≥ 3 AND net_flow_usd > 0).
       Forward signal: cluster buys historically precede positive 1–3 month
       abnormal returns; CEO/CFO buys weigh more than 10%-owner buys.
       Output: an insider_score ∈ [-2, +2] usable as a Pillar-A (Fundamentals)
       OR Pillar-D (Credit/conviction) tilt — additive, capped, never a
       standalone Buy/Sell.

  2. CROSS-SECTIONAL ML (next):
       features = {net_flow/mktcap, role-weighted buy intensity, 10b5-1 vs
       discretionary flag, cluster size, days-since-last-cluster, sector
       z-scored} → gradient-boosted ranker predicting 21-day forward residual
       (alpha) return.  Trained out-of-sample, walk-forward; output a percentile
       the report shows as "insider conviction percentile".

  3. SEQUENCE / REGIME-CONDITIONED (later):
       condition the ML head on the macro regime (BLOCK 3.4) — insider buying
       in a Recovery regime is a stronger long signal than in late Expansion.

All tiers stay ADVISORY (CoVe-tagged "не ИИР") and feed the existing
hallucination/math CoVe checkers — no model output bypasses the gatekeeper.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("SmartMoney")


# ── Gating ───────────────────────────────────────────────────────────────────
SMART_MONEY_ENV = "SMART_MONEY_INSIDERS"


def insiders_enabled() -> bool:
    """True only when the operator explicitly opts in (default OFF)."""
    return str(os.getenv(SMART_MONEY_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


# ── Scoring tunables ─────────────────────────────────────────────────────────
_CLUSTER_MIN_BUYERS = 3        # distinct insiders buying ⇒ "cluster"
_SCORE_CLIP         = 2.0      # insider score is on the same ±2 pillar scale
# Role weights — a CEO/CFO open-market buy is a stronger tell than a 10%-owner's.
_ROLE_WEIGHT = {
    "CEO": 1.5, "CFO": 1.5, "President": 1.3, "COO": 1.2,
    "Director": 1.0, "Officer": 1.0, "10%Owner": 0.6,
}


@dataclass(frozen=True)
class InsiderSignal:
    """One aggregated insider-flow reading for a single ticker."""
    ticker:           str
    net_flow_usd:     float = 0.0      # Σ buys − Σ sells (open-market, 90d)
    buy_count:        int   = 0
    sell_count:       int   = 0
    distinct_buyers:  int   = 0
    cluster_flag:     bool  = False    # ≥ _CLUSTER_MIN_BUYERS distinct buyers, net +
    score:            float = 0.0      # [-2, +2] advisory tilt
    as_of:            Optional[str] = None
    status:           str   = "missing"   # ok | warn | missing | disabled | error
    note:             str   = ""

    def as_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "net_flow_usd":    round(float(self.net_flow_usd), 2),
            "buy_count":       int(self.buy_count),
            "sell_count":      int(self.sell_count),
            "distinct_buyers": int(self.distinct_buyers),
            "cluster_flag":    bool(self.cluster_flag),
            "score":           round(float(self.score), 3),
            "as_of":           self.as_of,
            "status":          self.status,
            "note":            self.note,
        }


def score_insider_flow(*,
                       net_flow_usd:    float,
                       distinct_buyers: int,
                       sell_count:      int,
                       market_cap_usd:  Optional[float] = None,
                       role_weight:     float = 1.0) -> tuple[float, bool]:
    """
    Deterministic rule-tier scorer (model tier 1).

    Returns ``(score, cluster_flag)`` where score ∈ [-2, +2].  Magnitude scales
    with net flow relative to market cap (when known) and with the role weight;
    a buy CLUSTER (≥3 distinct buyers, net positive) gets a conviction bump.
    Pure + side-effect-free so it is trivially unit-testable.
    """
    net  = float(net_flow_usd or 0.0)
    base = 0.0
    if market_cap_usd and market_cap_usd > 0:
        # bps of market cap, saturating: 50 bps net buying ⇒ ~full unit.
        base = max(-1.0, min(1.0, (net / float(market_cap_usd)) / 0.005))
    else:
        # No market cap → sign-only with a mild magnitude from buyer breadth.
        base = max(-1.0, min(1.0, (1.0 if net > 0 else -1.0 if net < 0 else 0.0)
                             * min(1.0, distinct_buyers / 5.0)))

    cluster = bool(distinct_buyers >= _CLUSTER_MIN_BUYERS and net > 0)
    score = base * float(role_weight)
    if cluster:
        score += 0.5                         # conviction bump for clustered buys
    if sell_count > 0 and net < 0:
        score -= 0.25                        # distribution by insiders
    return float(max(-_SCORE_CLIP, min(_SCORE_CLIP, score))), cluster


def build_insider_signals(
        tickers: list[str],
        *,
        fetch: Optional[Callable[[str], dict]] = None,
        market_caps: Optional[dict[str, float]] = None,
) -> dict[str, dict]:
    """
    Produce one InsiderSignal per ticker.

    Args:
        tickers      : equity tickers to look up (ETFs/cash are skipped upstream).
        fetch        : injectable provider ``fetch(ticker) -> {net_flow_usd,
                       distinct_buyers, buy_count, sell_count, top_role, as_of}``.
                       When None (the default), NO network call is made and every
                       row is status="disabled"/"missing" — this is the safe
                       foundation behaviour until a real EDGAR Form-4 provider is
                       wired (see module docstring).
        market_caps  : optional {ticker: market_cap_usd} to scale the score.

    Returns ``{ticker: InsiderSignal.as_dict()}``.  Never raises — a failing
    provider degrades that ticker to status="error" and keeps going.
    """
    out: dict[str, dict] = {}
    market_caps = market_caps or {}

    if not insiders_enabled():
        for t in tickers:
            out[str(t)] = InsiderSignal(
                ticker=str(t), status="disabled",
                note=f"{SMART_MONEY_ENV}=0 (insider layer off)").as_dict()
        return out

    if fetch is None:
        for t in tickers:
            out[str(t)] = InsiderSignal(
                ticker=str(t), status="missing",
                note="no SEC Form-4 provider wired").as_dict()
        return out

    for t in tickers:
        t = str(t)
        try:
            raw = fetch(t) or {}
            net      = float(raw.get("net_flow_usd", 0.0) or 0.0)
            buyers   = int(raw.get("distinct_buyers", 0) or 0)
            buys     = int(raw.get("buy_count", 0) or 0)
            sells    = int(raw.get("sell_count", 0) or 0)
            role_w   = _ROLE_WEIGHT.get(str(raw.get("top_role", "Officer")), 1.0)
            score, cluster = score_insider_flow(
                net_flow_usd=net, distinct_buyers=buyers, sell_count=sells,
                market_cap_usd=market_caps.get(t), role_weight=role_w)
            out[t] = InsiderSignal(
                ticker=t, net_flow_usd=net, buy_count=buys, sell_count=sells,
                distinct_buyers=buyers, cluster_flag=cluster, score=score,
                as_of=raw.get("as_of"),
                status="ok" if (buys or sells) else "warn",
                note=("insider cluster buy" if cluster else ""),
            ).as_dict()
        except Exception as exc:               # one bad ticker can't sink the run
            logger.debug("Insider fetch failed for %s: %s", t, exc)
            out[t] = InsiderSignal(ticker=t, status="error",
                                   note=str(exc)[:80]).as_dict()
    return out


def insider_lineage_row(signals: Optional[dict]) -> dict:
    """
    CoVe data-lineage row for the insider layer (BLOCK 4.8 hook).

    Mirrors the dict shape data_lineage._row produces so build_lineage can
    append it directly.  Status rolls up the per-ticker signals.
    """
    base = {
        "name":   "Smart-Money (инсайдеры SEC Form 4)",
        "source": "SEC EDGAR · Form 4",
        "method": "90д нетто-поток + кластер покупок · role-weighted → score[-2..+2]",
        "as_of":  None, "freshness_days": None,
    }
    if not signals:
        return {**base, "status": "missing",
                "note": "insider layer off / no data"}
    statuses = [s.get("status") for s in signals.values()]
    if all(s == "disabled" for s in statuses):
        return {**base, "status": "missing",
                "note": f"{SMART_MONEY_ENV}=0 (выкл. по умолчанию)"}
    n_ok      = sum(1 for s in statuses if s == "ok")
    n_cluster = sum(1 for s in signals.values() if s.get("cluster_flag"))
    if n_ok == 0:
        return {**base, "status": "missing", "note": "нет покрытия Form-4"}
    return {**base, "status": "ok",
            "note": f"{n_ok} тикеров · {n_cluster} кластерных покупок"}


__all__ = [
    "InsiderSignal",
    "build_insider_signals",
    "score_insider_flow",
    "insider_lineage_row",
    "insiders_enabled",
    "SMART_MONEY_ENV",
]
