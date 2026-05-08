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

    def as_records(self) -> list[dict]:
        return [
            {
                "ticker":       t,
                "current_w":    float(self.current_weights[i]),
                "target_w":     float(self.target_weights[i]),
                "delta_w_pp":   float((self.target_weights[i] - self.current_weights[i]) * 100),
                "posterior_mu": float(self.posterior_mu[i]),
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
        except np.linalg.LinAlgError:
            # Singular; fall back to prior.
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

    return BLResult(
        tickers         = list(tickers),
        target_weights  = w_star,
        current_weights = w_curr,
        delta_weights   = w_star - w_curr,
        posterior_mu    = posterior_mu,
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
