"""
Black-Litterman target-weight engine.

Combines the equilibrium-implied prior (reverse-optimised from market caps
or current weights) with discretionary "views" — typically derived from
RAG-extracted bank insights or the 4-pillar score — to produce a posterior
mean and a corresponding target-weight vector.

The implementation follows the standard Idzorek-style formulation:

  π    = δ · Σ · w_mkt
  μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q]
  w*   = (δΣ)⁻¹ μ_BL                      then renormalised to sum to 1

We deliberately keep this small (~70 lines of math) and depend only on
numpy / pandas — no scipy.optimize, no portfoliopy, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BLResult:
    tickers:        list[str]
    target_weights: np.ndarray   # length = len(tickers); sums to ~1.0
    current_weights: np.ndarray
    delta_weights:  np.ndarray   # target - current
    posterior_mu:   np.ndarray   # implied annual expected returns
    # F-11: number of active views folded into the posterior.  0 means
    # posterior_mu is the raw reverse-optimised equilibrium π = δΣw — a
    # function of current weights and covariance, NOT a forward forecast.
    # Consumers (simulate's Expected-Effect panel) must not present it as one.
    n_views:        int = 0

    def as_records(self) -> list[dict]:
        return [
            {
                "ticker":       t,
                "current_w":    float(self.current_weights[i]),
                "target_w":     float(self.target_weights[i]),
                "delta_w_pp":   float((self.target_weights[i] - self.current_weights[i]) * 100),
                "posterior_mu": float(self.posterior_mu[i]),
                "n_views":      int(self.n_views),
            }
            for i, t in enumerate(self.tickers)
        ]


def _as_array(weights: pd.Series | dict | Sequence[float],
              tickers: list[str]) -> np.ndarray:
    if isinstance(weights, dict):
        return np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
    if isinstance(weights, pd.Series):
        return np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
    arr = np.asarray(weights, dtype=float)
    if arr.size != len(tickers):
        raise ValueError("weights length does not match tickers")
    return arr


def _cap_and_redistribute(w: np.ndarray, cap: float, iters: int = 32) -> np.ndarray:
    """Cap each long-only weight at ``cap`` and redistribute the excess across
    the uncapped names (water-filling), keeping the vector summing to 1.

    Soft constraint: when the cap is INFEASIBLE for the name count (n·cap ≤ 1 —
    e.g. a 10% cap on a 4-name book, where the weights MUST sum to 100%), the
    constraint cannot be honoured, so we leave the target unchanged rather than
    forcing a meaningless equal-weight or oscillating.  The mandate's risk
    aversion + turnover cap still differentiate the optimisation in that case.
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0:
        return w
    w = w / s
    n = w.size
    if cap * n <= 1.0 + 1e-9:
        return w                       # cap infeasible for this name count — no-op
    for _ in range(iters):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        pool = float(w[under].sum())
        if pool <= 1e-12:
            break
        w[under] += excess * (w[under] / pool)
    total = w.sum()
    return w / total if total > 0 else w


def black_litterman(
    *,
    cov: pd.DataFrame | np.ndarray,
    tickers: list[str],
    current_weights: pd.Series | dict | Sequence[float],
    views_P: Optional[np.ndarray] = None,
    views_Q: Optional[np.ndarray] = None,
    view_confidence: Optional[np.ndarray] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    max_active_share: float = 0.25,
    max_single_weight: Optional[float] = None,
) -> BLResult:
    """
    Run Black-Litterman optimisation.

    Args:
        cov              : Annualised covariance matrix (n x n) or DataFrame.
        tickers          : Ordered list matching cov rows/cols.
        current_weights  : Reverse-optimisation prior weights (treat as
                           "market" implied weights for this user's universe).
        views_P, views_Q : Optional view matrix (k x n) and view returns (k,)
                           describing discretionary views.  If both are None
                           the result reduces to the implied prior.
        view_confidence  : Optional vector (k,) ∈ (0, 1] giving Idzorek
                           confidence per view; converted internally to Ω.
        risk_aversion    : δ; default 2.5 ≈ long-term equity premium / vol².
        tau              : Scaling of prior covariance.  0.05 is the canonical
                           Black-Litterman value.
        max_active_share : Soft cap on |target - current| total turnover.
                           Targets are pulled toward current weights to
                           respect this — protects against runaway turnover
                           on noisy views.

    Returns:
        BLResult.  Target weights are clipped to [0, 1] and renormalised so
        they sum to exactly 1.
    """
    if isinstance(cov, pd.DataFrame):
        # Reorder cov to match tickers
        sigma = cov.reindex(index=tickers, columns=tickers).values
    else:
        sigma = np.asarray(cov, dtype=float)
    n = len(tickers)
    if sigma.shape != (n, n):
        raise ValueError(f"cov shape {sigma.shape} != ({n},{n})")

    w_curr = _as_array(current_weights, tickers)
    if w_curr.sum() > 0:
        w_curr = w_curr / w_curr.sum()      # normalise prior to 1
    else:
        w_curr = np.ones(n) / n             # equal-weight fallback

    # Implied equilibrium returns: π = δ · Σ · w
    pi = risk_aversion * (sigma @ w_curr)

    active_views = 0          # F-11: stays 0 whenever posterior_mu ends up = π
    if views_P is None or views_Q is None or len(views_Q) == 0:
        posterior_mu = pi.copy()
    else:
        P = np.asarray(views_P, dtype=float).reshape(-1, n)
        Q = np.asarray(views_Q, dtype=float).reshape(-1)
        if view_confidence is None:
            # Default Ω = diag(P τΣ Pᵀ) — Black-Litterman canonical.
            omega = np.diag(np.diag(P @ (tau * sigma) @ P.T))
        else:
            # Idzorek-style confidence (0..1] → Ω diagonal scaled by (1-c)/c.
            confidences = np.clip(np.asarray(view_confidence, dtype=float).reshape(-1),
                                  1e-3, 0.999)
            base = np.diag(P @ (tau * sigma) @ P.T)
            omega = np.diag(base * (1.0 - confidences) / confidences)

        # Posterior mean (Idzorek 2005, eq. 10).
        try:
            tau_sigma_inv = np.linalg.inv(tau * sigma)
            omega_inv     = np.linalg.inv(omega)
            posterior_cov_inv = tau_sigma_inv + P.T @ omega_inv @ P
            posterior_cov     = np.linalg.inv(posterior_cov_inv)
            posterior_mu      = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
            active_views      = int(len(Q))
        except np.linalg.LinAlgError:
            # Singular; fall back to prior (no views folded in → stays 0).
            posterior_mu = pi.copy()

    # Target weights: w* = (δ Σ)⁻¹ μ_BL  (long-only, normalised, capped turnover).
    try:
        w_star = np.linalg.solve(risk_aversion * sigma, posterior_mu)
    except np.linalg.LinAlgError:
        w_star = w_curr.copy()

    w_star = np.clip(w_star, 0.0, 1.0)
    if w_star.sum() <= 0:
        w_star = w_curr.copy()
    else:
        w_star = w_star / w_star.sum()

    # Soft turnover cap: blend toward current if active share too high.
    active_share = float(np.abs(w_star - w_curr).sum() / 2.0)
    if active_share > max_active_share and active_share > 0:
        # blend so the new active share == max_active_share
        scale = max_active_share / active_share
        w_star = w_curr + scale * (w_star - w_curr)
        w_star = np.clip(w_star, 0.0, 1.0)
        if w_star.sum() > 0:
            w_star = w_star / w_star.sum()

    # Sprint-5 (Task 4 — mandate-aware optimisation): a per-name weight cap
    # derived from the investor's mandate.  A Conservative mandate caps each
    # risky position tighter (e.g. 10%) than an Aggressive one (30%), so the BL
    # target actually respects the investor's risk appetite instead of being
    # mandate-agnostic.  Applied AFTER the turnover blend so the cap is the
    # binding final constraint.
    if max_single_weight is not None and 0.0 < max_single_weight < 1.0:
        w_star = _cap_and_redistribute(w_star, float(max_single_weight))

    return BLResult(
        tickers         = list(tickers),
        target_weights  = w_star,
        current_weights = w_curr,
        delta_weights   = w_star - w_curr,
        posterior_mu    = posterior_mu,
        n_views         = active_views,
    )


def views_from_scores(
    asset_scores: dict,
    tickers: list[str],
    *,
    score_to_view_pct: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert per-asset Total Scores (-6..+6) into single-asset views.
    Each non-zero score yields a view 'asset i will outperform by
    score · score_to_view_pct annualised'.  Confidence scales with |score|/6.

    Returns (P, Q, confidence) suitable for `black_litterman`.
    Tickers without a score, or with score 0, are skipped.
    """
    P_rows: list[np.ndarray] = []
    Q_vals: list[float] = []
    conf:   list[float] = []

    for i, t in enumerate(tickers):
        sc = asset_scores.get(t)
        if sc is None:
            continue
        total = sc.get("total") if isinstance(sc, dict) else getattr(sc, "total", None)
        if total is None or abs(total) < 1e-6:
            continue
        row = np.zeros(len(tickers), dtype=float)
        row[i] = 1.0
        P_rows.append(row)
        Q_vals.append(float(total) * score_to_view_pct)
        conf.append(min(0.95, max(0.10, abs(float(total)) / 6.0)))

    if not P_rows:
        return np.zeros((0, len(tickers))), np.zeros(0), np.zeros(0)

    return np.vstack(P_rows), np.array(Q_vals), np.array(conf)


__all__ = ["black_litterman", "BLResult", "views_from_scores"]
