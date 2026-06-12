"""
Simulate portfolio risk metrics on a NEW weight vector.

This module powers the "Ожидаемый эффект до/после" panel in the DEEP
report — given the target weights produced by Black-Litterman optimisation
(or any other reweighting), it re-evaluates every metric the cover page
shows under the new allocation, WITHOUT re-running the factor regression.

All inputs are precomputed by analyze_all() so the module is sklearn-free
and trivially unit-testable on synthetic data.

Math
────
  Volatility (annualised)  σ = √(w' · Σ_structural · w)
        reuses the structural cov returned by calculate_structural_risk;
        identical to the "before" formula on cover.

  TRC (Euler decomposition)
        MCTR_i = (Σ · w)_i / σ
        ERC%_i = w_i · MCTR_i / σ · 100
        Max TRC = max_i ERC%_i — used for the concentration component of
        the composite gauge.

  CVaR / Sharpe / MDD       SAMPLE REPLAY:
        port_daily_new = (daily log-returns matrix) @ w_new
        cvar_new       = mean(port_daily_new[port_daily_new ≤ VaR_5%])
        sharpe_new     = (mean·252 − rfr) / std·√252
        mdd_new        = min((exp(cumsum(port_daily_new)) / running_max) − 1)
        Assumes "as if the new weights had always been held" — standard
        ex-post approximation; for forward Monte-Carlo use a separate tool.

  Composite Risk Score      Same formula as MAC3RiskEngine._composite_risk_score:
        0.40 · min(vol/0.40, 1)·100  +
        0.40 · min(|CVaR|/0.10, 1)·100  +
        0.20 · min(max_TRC/50, 1)·100
        Replicated here so the simulator carries no engine dependency.

  Expected return           E[r_ann] = Σ w_i · μ_i  where μ_i is the BL
        posterior mean (annualised).  Fallback: realised annualised return
        of each asset derived from daily_log_returns.

  IT share                  Σ w_i for assets classified as Technology
        (case-insensitive sector lookup; fallback to ticker-prefix heuristic).

Coverage semantics
──────────────────
Every metric reports whether it improved.  "Improved" means moving in the
DESIRED direction:
  • Risk metrics (vol, |CVaR|, MDD-magnitude, max_TRC, IT_share,
    composite_risk_index) → improved when DECREASED.
  • Return metrics (Sharpe, expected_return)                → improved
    when INCREASED.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── Composite Risk Score — re-exported from the single source of truth ───────
# finance.scoring.composite_risk_score is the canonical implementation.  We
# alias it under the historical private name so `_composite_risk_score` and
# the module's `__all__` export keep working for every existing caller/test.
from finance.scoring import composite_risk_score as _composite_risk_score  # noqa: E402,F401


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _weight_vector(tickers: list[str], weights_dict: dict[str, float]) -> np.ndarray:
    """Build an aligned weight vector for `tickers` from `weights_dict`."""
    return np.array([_safe_float(weights_dict.get(t), 0.0) for t in tickers], dtype=float)


def _structural_vol_and_trc(weights: np.ndarray,
                              cov_ann:  np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute σ_port and Euler ERC% vector from annualised structural cov.

    Returns (sigma_ann, erc_pct_vector).  erc_pct_vector[i] is the % of
    total portfolio risk contributed by asset i.  Sum of erc_pct ≈ 100.
    """
    if weights.size == 0 or cov_ann.size == 0:
        return 0.0, np.zeros(0)
    port_var = float(weights @ cov_ann @ weights)
    if port_var <= 0:
        return 0.0, np.zeros_like(weights)
    sigma = math.sqrt(port_var)
    mctr  = (cov_ann @ weights) / sigma
    erc   = (weights * mctr) / sigma           # decimal fractions
    return sigma, erc * 100.0                  # → percent


def _sample_metrics(daily_log_matrix: np.ndarray,
                     weights:           np.ndarray,
                     rfr_ann:           float,
                     trading_days:      int = 252,
                     ) -> dict:
    """
    Recompute CVaR / Sharpe / Max Drawdown on weights via sample replay.

    daily_log_matrix : shape (n_days, n_assets) of daily log-returns
    weights          : length n_assets (aligned to columns of the matrix)

    Returns dict with cvar_95, sharpe, max_drawdown — all decimal scalars.
    Returns NaNs when the window is too short to be meaningful (< 60 days).
    """
    if daily_log_matrix.size == 0 or weights.size == 0:
        return {"cvar_95": float("nan"), "sharpe": float("nan"),
                "max_drawdown": float("nan"), "n_days": 0}

    port_daily = daily_log_matrix @ weights              # length n_days
    n = port_daily.size
    if n < 60:
        return {"cvar_95": float("nan"), "sharpe": float("nan"),
                "max_drawdown": float("nan"), "n_days": int(n)}

    # CVaR 95% — mean of bottom 5%.
    var_95  = float(np.percentile(port_daily, 5))
    tail    = port_daily[port_daily <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else var_95

    # Sharpe — annualised, geometric-anchored mean / sample stdev.
    # F-7: geometric annualisation (exp(mean·252)−1) to match the engine's
    # headline Sharpe basis (H1) — arithmetic mean·252 overstated the
    # before/after rebalance delta by ~σ²/2.
    # Threshold std at 1e-12 to avoid FP-noise blowing up Sharpe on
    # near-constant series (e.g. an all-cash portfolio or a unit-test
    # fixture).  Real portfolios have daily σ > 1e-3, never near zero.
    ann_ret = float(np.exp(float(np.mean(port_daily)) * trading_days) - 1.0)
    daily_std = float(np.std(port_daily, ddof=1))
    sharpe = (ann_ret - rfr_ann) / (daily_std * math.sqrt(trading_days)) \
             if daily_std > 1e-12 else float("nan")

    # Max drawdown — peak-to-trough on equity curve.
    eq_curve    = np.exp(np.cumsum(port_daily))
    running_max = np.maximum.accumulate(eq_curve)
    drawdowns   = eq_curve / running_max - 1.0
    max_dd      = float(drawdowns.min())

    return {"cvar_95": cvar_95, "sharpe": sharpe,
            "max_drawdown": max_dd, "n_days": int(n)}


def _expected_return_from_bl(target_w_by_ticker: dict[str, float],
                              bl_records:          Optional[list[dict]]) -> Optional[float]:
    """
    Σ w_i · μ_i where μ_i is the BL posterior mean for ticker i.

    Returns None when bl_records is missing or empty (so the caller can
    fall back to a realised-return estimator).
    """
    if not bl_records:
        return None
    # F-11: when BL folded in ZERO active views, posterior_mu is the raw
    # equilibrium π = δΣw — a function of current weights and covariance,
    # not a forward forecast.  Decline it so the caller falls back to the
    # realised-return estimator instead of presenting circular "expected
    # return" to the user.  (Old records without the key keep legacy
    # behaviour.)
    if all(r.get("n_views", 1) == 0 for r in bl_records):
        return None
    mu_by_t = {r["ticker"]: _safe_float(r.get("posterior_mu"), 0.0)
               for r in bl_records}
    er = 0.0
    for t, w in target_w_by_ticker.items():
        er += float(w) * mu_by_t.get(t, 0.0)
    return float(er)


def _realised_expected_return(daily_log_matrix: np.ndarray,
                                weights:           np.ndarray,
                                trading_days:      int = 252) -> Optional[float]:
    """Annualised realised return of the portfolio under `weights`."""
    if daily_log_matrix.size == 0 or weights.size == 0:
        return None
    port_daily = daily_log_matrix @ weights
    return float(np.exp(float(np.mean(port_daily)) * trading_days) - 1.0)


def _it_share(tickers: list[str],
              weights: np.ndarray,
              sector_by_ticker: Optional[dict[str, str]] = None) -> float:
    """
    Fraction of portfolio classified as Technology.

    Uses an explicit sector lookup when provided; otherwise falls back to
    a permissive prefix heuristic on common IT tickers.  Returns 0.0 when
    no IT exposure is identifiable.
    """
    _IT_PREFIXES = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN",
                    "TSLA", "PLTR", "CRWD", "NET", "AMD", "INTC", "ORCL",
                    "ADBE", "CRM", "NOW", "SNOW", "SHOP", "AVGO", "QCOM",
                    "SMCI", "ASML", "TSM"}
    total = 0.0
    sector_by_ticker = sector_by_ticker or {}
    for t, w in zip(tickers, weights):
        sec = (sector_by_ticker.get(t) or "").lower()
        is_it = (
            "tech" in sec or "info" in sec or "software" in sec
            or t.upper().split(".")[0] in _IT_PREFIXES
        )
        if is_it:
            total += float(w)
    return float(total)


# ── Public API ───────────────────────────────────────────────────────────────

# Metrics whose IMPROVEMENT is a DECREASE of the raw `after - before` delta.
#
# CVaR and Max-Drawdown are intentionally NOT here: they are negative-valued
# (e.g. −0.05), and an improvement means moving TOWARD zero, i.e. a POSITIVE
# raw delta.  _delta_row compares raw values, not magnitudes — listing them
# as "lower is better" inverted the favourable flag (a worse drawdown showed
# green, an improved CVaR showed red).
_RISK_METRICS_LOWER_IS_BETTER = {"volatility_ann", "max_trc", "risk_index"}

# Sprint-5.1 (A2): metrics whose direction is NOT a universal value judgment.
# `it_share` used to sit in LOWER_IS_BETTER, so ANY cut of tech exposure
# rendered green ("improved") — backwards for an Aggressive growth mandate.
# The sector share is INFORMATIONAL: we report the delta, but improved=None
# (the template renders it neutral, neither green nor red).
_NEUTRAL_METRICS = {"it_share"}


def _delta_row(before: Optional[float], after: Optional[float],
                metric_name: str,
                *, as_pp: bool = False) -> dict:
    """Build a {before, after, delta, improved} cell for one metric.

    `improved` is None (neutral) when: an input is missing, the metric is
    direction-neutral (`_NEUTRAL_METRICS`), or the delta is exactly zero
    (a no-op must not render as red "not improved").
    """
    if before is None or after is None or \
       (isinstance(before, float) and math.isnan(before)) or \
       (isinstance(after,  float) and math.isnan(after)):
        return {"before": before, "after": after,
                "delta": None, "improved": None}
    delta = after - before
    improved: Optional[bool]
    if metric_name in _NEUTRAL_METRICS or abs(delta) < 1e-12:
        improved = None
    elif metric_name in _RISK_METRICS_LOWER_IS_BETTER:
        improved = delta < 0
    else:
        improved = delta > 0
    out = {"before": before, "after": after,
           "delta": float(delta), "improved": improved}
    if as_pp:
        out["delta_pp"] = round(float(delta) * 100, 2)
    return out


def simulate_after_plan(*,
                          perf_df:         pd.DataFrame,
                          risk_matrix:     pd.DataFrame,
                          daily_log_returns: Optional[pd.DataFrame],
                          bl_records:      Optional[list[dict]],
                          current_metrics: dict,
                          risk_free_rate:  float,
                          target_weights:  dict[str, float],
                          sector_by_ticker: Optional[dict[str, str]] = None,
                          ) -> Optional[dict]:
    """
    Re-evaluate the cover-page metrics under `target_weights`.

    Args:
        perf_df          : performance table (Ticker, Current_Value,
                            Fundamental_Sector).
        risk_matrix      : annualised structural covariance matrix, indexed
                            by ticker (same as engine.calculate_structural_risk
                            output).
        daily_log_returns: optional DataFrame of daily log-returns; columns
                            must be the engine's resolved tickers.  When
                            None, CVaR/Sharpe/MDD/expected return fall back
                            to BL-based values (or None).
        bl_records       : optional list of {ticker, current_w, target_w,
                            posterior_mu, ...} from Black-Litterman; used
                            for expected_return when present.
        current_metrics  : portfolio_metrics dict from analyze_all (used
                            for "before" values).
        risk_free_rate   : annualised risk-free rate.
        target_weights   : {ticker: target_weight_decimal}.
        sector_by_ticker : optional {ticker: sector_label} mapping for the
                            IT-share calculation.

    Returns None when inputs are too thin to simulate ANY metric meaningfully
    (e.g. empty risk matrix + empty daily returns).
    """
    if (risk_matrix is None or risk_matrix.empty) and \
       (daily_log_returns is None or daily_log_returns.empty):
        return None

    # ── Build aligned ticker universe and weight vectors ───────────────────
    total_val = 0.0
    cur_w_by_ticker: dict[str, float] = {}
    sector_by_ticker = dict(sector_by_ticker or {})
    if perf_df is not None and not perf_df.empty:
        total_val = float(perf_df["Current_Value"].fillna(0).sum())
        for _, row in perf_df.iterrows():
            t  = str(row.get("Ticker", "")).strip()
            cv = _safe_float(row.get("Current_Value"), 0.0)
            if t and total_val > 0:
                cur_w_by_ticker[t] = cv / total_val
            if t and "Fundamental_Sector" in perf_df.columns:
                sec = row.get("Fundamental_Sector")
                if sec is not None and str(sec).lower() not in {"none", "nan"}:
                    sector_by_ticker.setdefault(t, str(sec))

    # Ticker universe for structural metrics: cov matrix's index (only
    # assets that survived the factor regression).
    if risk_matrix is not None and not risk_matrix.empty:
        struct_tickers = list(risk_matrix.index)
        cov_ann        = risk_matrix.values
    else:
        struct_tickers = list(cur_w_by_ticker.keys())
        cov_ann        = np.zeros((len(struct_tickers), len(struct_tickers)))

    w_before = _weight_vector(struct_tickers, cur_w_by_ticker)
    w_after  = _weight_vector(struct_tickers, target_weights)

    # ── Structural metrics (vol, max_TRC) — always computable ──────────────
    vol_before, erc_before = _structural_vol_and_trc(w_before, cov_ann)
    vol_after,  erc_after  = _structural_vol_and_trc(w_after,  cov_ann)
    # max_TRC is the largest CONCENTRATION of risk in a single position —
    # only positive ERC contributors count.  Negative ERC means the asset is
    # NET hedging the portfolio (its covariance with the rest is negative),
    # so |erc| would mis-flag a strong diversifier as a concentration.
    _pos_before    = erc_before[erc_before > 0] if erc_before.size else erc_before
    _pos_after     = erc_after[erc_after > 0]   if erc_after.size  else erc_after
    max_trc_before = float(np.max(_pos_before)) if _pos_before.size else 0.0
    max_trc_after  = float(np.max(_pos_after))  if _pos_after.size  else 0.0

    # ── Sample-replay metrics — need daily returns ─────────────────────────
    sample_before: dict = {"cvar_95": float("nan"), "sharpe": float("nan"),
                            "max_drawdown": float("nan"), "n_days": 0}
    sample_after:  dict = dict(sample_before)
    if daily_log_returns is not None and not daily_log_returns.empty:
        # Align tickers with what's available in daily_log_returns.
        avail = [t for t in struct_tickers if t in daily_log_returns.columns]
        if avail:
            dl_matrix = daily_log_returns[avail].dropna().values
            w_b_aligned = _weight_vector(avail, cur_w_by_ticker)
            w_a_aligned = _weight_vector(avail, target_weights)
            sample_before = _sample_metrics(dl_matrix, w_b_aligned, risk_free_rate)
            sample_after  = _sample_metrics(dl_matrix, w_a_aligned, risk_free_rate)

    # ── Anchor "before" to the HEADLINE metrics, keep the simulated delta ──
    # The sample-replay window differs from the engine's main Sharpe/CVaR/MDD
    # calc, so the raw `sample_before` disagreed with the KPI strip (Sharpe
    # 0.81 headline vs 0.76 here).  We anchor the displayed "before" to the
    # headline value and carry the simulated Δ forward to "after", so the
    # Expected-Effect panel is consistent with the rest of the report while
    # still showing the true effect of the rebalance.
    def _anchor(metric_key: str, headline_key: str) -> None:
        headline = current_metrics.get(headline_key)
        b, a = sample_before.get(metric_key), sample_after.get(metric_key)
        if (headline is None or b is None or a is None
                or (isinstance(b, float) and math.isnan(b))
                or (isinstance(a, float) and math.isnan(a))):
            return
        try:
            delta = float(a) - float(b)
            sample_before[metric_key] = float(headline)
            sample_after[metric_key]  = float(headline) + delta
        except (TypeError, ValueError):
            return

    _anchor("sharpe",       "Sharpe_Ratio")
    _anchor("cvar_95",      "CVaR_95_Daily")
    _anchor("max_drawdown", "Max_Drawdown")

    # ── Composite risk index ───────────────────────────────────────────────
    cvar_b = sample_before["cvar_95"]
    cvar_a = sample_after["cvar_95"]
    risk_index_before = _composite_risk_score(vol_before,
                                                cvar_b if not math.isnan(cvar_b) else 0.0,
                                                max_trc_before)
    risk_index_after  = _composite_risk_score(vol_after,
                                                cvar_a if not math.isnan(cvar_a) else 0.0,
                                                max_trc_after)

    # ── Expected return: BL μ preferred, realised fallback ────────────────
    er_before = _expected_return_from_bl(cur_w_by_ticker, bl_records)
    er_after  = _expected_return_from_bl(target_weights, bl_records)
    if er_before is None and daily_log_returns is not None and not daily_log_returns.empty:
        avail = [t for t in struct_tickers if t in daily_log_returns.columns]
        if avail:
            dl_matrix = daily_log_returns[avail].dropna().values
            w_b_aligned = _weight_vector(avail, cur_w_by_ticker)
            w_a_aligned = _weight_vector(avail, target_weights)
            er_before = _realised_expected_return(dl_matrix, w_b_aligned)
            er_after  = _realised_expected_return(dl_matrix, w_a_aligned)

    # ── IT share ───────────────────────────────────────────────────────────
    it_before = _it_share(struct_tickers, w_before, sector_by_ticker)
    it_after  = _it_share(struct_tickers, w_after,  sector_by_ticker)

    # ── Compose the result ─────────────────────────────────────────────────
    metrics = {
        "risk_index":      _delta_row(risk_index_before, risk_index_after, "risk_index"),
        "volatility_ann":  _delta_row(vol_before, vol_after, "volatility_ann", as_pp=True),
        "cvar_95":         _delta_row(
                                None if math.isnan(cvar_b) else cvar_b,
                                None if math.isnan(cvar_a) else cvar_a,
                                "cvar_95_magnitude",       # lower |cvar| is better
                                as_pp=True),
        "sharpe":          _delta_row(
                                None if math.isnan(sample_before["sharpe"]) else sample_before["sharpe"],
                                None if math.isnan(sample_after["sharpe"])  else sample_after["sharpe"],
                                "sharpe"),
        "max_drawdown":    _delta_row(
                                None if math.isnan(sample_before["max_drawdown"]) else sample_before["max_drawdown"],
                                None if math.isnan(sample_after["max_drawdown"])  else sample_after["max_drawdown"],
                                "max_drawdown_magnitude",  # less-negative is better
                                as_pp=True),
        "max_trc":         _delta_row(max_trc_before, max_trc_after, "max_trc"),
        "it_share":        _delta_row(it_before, it_after, "it_share", as_pp=True),
        "expected_return": _delta_row(er_before, er_after, "expected_return", as_pp=True),
    }

    # Weight-change summary (for transparency / per-asset audit).
    weight_changes: list[dict] = []
    for t in struct_tickers:
        wb = cur_w_by_ticker.get(t, 0.0)
        wa = float(target_weights.get(t, wb))
        if abs(wa - wb) >= 0.001:                # show only meaningful moves
            weight_changes.append({
                "ticker":      t,
                "before_pct":  round(wb * 100, 2),
                "after_pct":   round(wa * 100, 2),
                "delta_pp":    round((wa - wb) * 100, 2),
            })
    weight_changes.sort(key=lambda d: abs(d["delta_pp"]), reverse=True)

    # ── Composite verdict: "improvement" or "tradeoff" ────────────────────
    # An honest verdict for the rebalance.  The legacy headline read
    # "снижение риска" any time vol went down — but a rebalance that drops
    # vol while concentration (max_trc) or drawdown gets worse is a
    # COMPROMISE, not an improvement.  Flag that explicitly so the AI
    # narrative and the template show the user what they're trading off.
    def _cell_worsened(name: str) -> bool:
        cell = metrics.get(name) or {}
        if cell.get("delta") is None or cell.get("improved") is None:
            return False
        return cell["improved"] is False and abs(cell["delta"]) > 1e-6

    vol_cell       = metrics.get("volatility_ann") or {}
    vol_delta      = vol_cell.get("delta") or 0.0
    vol_improved   = vol_cell.get("improved") is True
    vol_worsened   = vol_cell.get("improved") is False and abs(vol_delta) > 1e-6
    trc_worsened   = _cell_worsened("max_trc")
    dd_worsened    = _cell_worsened("max_drawdown")

    if vol_improved and (trc_worsened or dd_worsened):
        worsened = []
        if trc_worsened: worsened.append("max_trc")
        if dd_worsened:  worsened.append("max_drawdown")
        verdict = {
            "kind":     "tradeoff",
            "headline": "Компромисс: снижение волатильности за счёт "
                        "роста концентрации/просадки",
            "worsened": worsened,
        }
    elif vol_improved:
        verdict = {"kind": "improvement",
                   "headline": "Снижение риска без значимых компромиссов",
                   "worsened": []}
    elif vol_worsened:
        verdict = {"kind": "degradation",
                   "headline": "Рост волатильности — план не улучшает риск",
                   "worsened": ["volatility_ann"]}
    else:
        # Either vol delta is zero (target == current) or improvement flag is
        # None.  Either way the rebalance is materially a no-op.
        verdict = {"kind": "neutral",
                   "headline": "Эффект ребалансировки маржинален",
                   "worsened": []}

    return {
        "metrics":          metrics,
        "verdict":          verdict,
        "weight_changes":   weight_changes,
        "n_days_used":      sample_after.get("n_days", 0),
        "uses_bl_returns":  bl_records is not None and len(bl_records) > 0
                              and er_before is not None,
        "method":           ("vol/TRC via √(w'Σw) on structural cov · "
                             "CVaR/Sharpe/MDD via sample replay on daily log-returns · "
                             "composite = 0.4·vol+0.4·|CVaR|+0.2·max_TRC normalised"),
    }


__all__ = [
    "simulate_after_plan",
    "_composite_risk_score",
]
