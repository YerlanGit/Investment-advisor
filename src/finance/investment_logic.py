import os
import hashlib
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf

from finance.broker_api import RealPortfolioRequired
from freedom_portfolio import TradernetClient, get_history_frame

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Antigravity_RiskEngine")

# Sprint-5 Task 4 — mandate → Black-Litterman constraint table.  Maps the
# canonical 3-state risk mandate to the optimiser's risk-aversion (δ), turnover
# cap (active share) and per-name weight cap.  Conservative ⇒ less aggressive
# tilts, less turnover and tighter single-name caps; Aggressive ⇒ the reverse.
_MANDATE_BL_CONSTRAINTS: dict[str, dict[str, float]] = {
    "CONSERVATIVE": {"risk_aversion": 4.0, "max_active_share": 0.15, "max_single_weight": 0.10},
    "MODERATE":     {"risk_aversion": 2.5, "max_active_share": 0.25, "max_single_weight": 0.20},
    "AGGRESSIVE":   {"risk_aversion": 2.0, "max_active_share": 0.35, "max_single_weight": 0.30},
}

# Multi-period (1м / 3м / 6м / 12м / YTD) returns live in a separate module
# so they can be unit-tested without pulling in sklearn / engine dependencies.
from finance.period_returns import (
    MIN_OVERLAP_TDAYS as _MIN_OVERLAP_TDAYS,
    compute_period_returns_table as _compute_period_returns_table,
    build_portfolio_log_returns as _build_portfolio_log_returns,
    compute_benchmark_stats as _compute_benchmark_stats,
)
# Stress engine (parametric factor shocks) — also sklearn-free.
from finance.stress import (
    run_stress_scenarios as _run_stress_scenarios,
    STRESS_TEST_DISCLAIMER as _STRESS_TEST_DISCLAIMER,
)
# Base-currency layer (H2): reporting-currency-aware FX transformation +
# currency-matched RFR with geometric daily compounding (H3).
from finance.currency import (
    ReportingCurrency,
    convert_price_matrix,
    daily_rfr_geometric,
    get_rfr_for_currency,
    infer_currencies_for_tickers,
)
# Simulator (Black-Litterman "after-plan" portfolio metrics) — sklearn-free.
from finance.simulate import simulate_after_plan as _simulate_after_plan
# FRED macro feed (yield curve / HY spread / PMI / VIX / breakeven) — also
# sklearn-free; gracefully degrades when FRED_API_KEY is missing or network
# is unreachable.
from services.macro_data import MacroFeed as _MacroFeed
# P-5 (audit risk-methodology): small-window confidence intervals — χ² for
# the sample σ, Fisher z for correlations.  Dependency-light (stdlib only).
from finance.inference import (
    fisher_rho_ci as _fisher_rho_ci,
    sigma_ci_multiplier as _sigma_ci_multiplier,
)

# P-3 (audit A2/D2): historical VaR/CVaR quantiles need ~100 observations for
# a stable 95% tail; below this the point estimates are published with an
# explicit "insufficient_history" reliability verdict instead of silently.
MIN_RELIABLE_TAIL_OBS = 100

# P-4 (audit B2, roadmap METHODOLOGY_SPARSE_AND_LEVERAGED §2.2): Vasicek
# shrinkage of regression betas toward a prior — GATED, default OFF, because
# it changes Σ on staggered books and needs its own validation pass before
# production.  w = n/(n+n₀); prior: Market → 1.0 (or the ETP's leverage L for
# registry names), style/EM factors → 0.0.
BETA_SHRINKAGE_ENV = "BETA_SHRINKAGE"
BETA_SHRINKAGE_PRIOR_OBS_ENV = "BETA_SHRINKAGE_PRIOR_OBS"


def beta_shrinkage_enabled() -> bool:
    return str(os.getenv(BETA_SHRINKAGE_ENV, "")).strip().lower() in (
        "1", "true", "yes", "on")


# ── BLOCK 3.5 — hierarchical factor orthogonalization (gated) ────────────────
# The factor set mixes a CORE macro axis (Market, Rates, Commodities) with
# long-only style ETFs (Momentum/Value/Quality/Size/Volatility) and EM proxies
# that all carry the SAME market/rates beta — so the raw factor covariance F is
# near-collinear (condition number κ ≈ 62 in production), the Ridge betas get
# unstable, and B·F·Bᵀ double-counts the shared market risk.
#
# Rather than PCA (which would destroy the NAMED factors the report depends on),
# we residualize each style/EM factor against its CORE PARENT via OLS — a
# hierarchical Gram-Schmidt: «основные макро-факторы → суб-факторы стилей».
# Each factor keeps its name and its own mean; only the redundant shared beta is
# removed, so "Momentum" becomes MARKET-NEUTRAL momentum, etc.  κ drops sharply
# and the style premia stop being counted twice.
#
# BLOCK 3 (2026-06-26): multicollinearity fix promoted to DEFAULT-ON.  The
# production factor set carried κ≈60.8 (max|corr|≈0.89) — the style/EM betas
# were near-interpolating noise because their shared market/rates beta was
# counted twice.  The hierarchical residualisation below is the effective math
# solution (chosen over PCA precisely BECAUSE it preserves the named factors the
# report depends on): it removes only the redundant shared beta, dropping κ
# below the 30 collinearity threshold while every factor keeps its name + mean.
# The env var now ONLY exists as an escape hatch — set FACTOR_ORTHOGONALIZE=0 to
# restore the legacy raw-factor decomposition; any other value (incl. unset)
# orthogonalises.
FACTOR_ORTHOGONALIZE_ENV = "FACTOR_ORTHOGONALIZE"
_FACTOR_CORE = ("Market", "Rates", "Commodities")
# Each non-core factor → the core parent(s) to neutralise its shared beta against.
_FACTOR_PARENTS = {
    "Momentum":   ("Market",),
    "Value":      ("Market",),
    "Quality":    ("Market",),
    "Size":       ("Market",),
    "Volatility": ("Market",),
    "EM_Equity":  ("Market",),
    "EM_Bond":    ("Rates",),
}


def factor_orthogonalize_enabled() -> bool:
    # DEFAULT-ON (BLOCK 3): unset → enabled.  Only an explicit off-switch
    # (0/false/no/off) restores the legacy collinear decomposition.
    return str(os.getenv(FACTOR_ORTHOGONALIZE_ENV, "on")).strip().lower() not in (
        "0", "false", "no", "off", "")


# ── F-23 / P-1 — daily-reset leveraged ETPs: structural variance drag ────────
# A k×-daily-reset product's log return is ≈ k·r_u − ½k(k−1)σ_u² per day: the
# −½k(k−1)σ_u² variance drag is CONTRACTUAL (rebalancing cost), not luck.  In
# the factor regression that drag lands in the intercept α̂ — which the F-14
# forward panel rightly EXCLUDES for ordinary names (realised idiosyncratic
# drift is not a persistent premium).
#
# P-1 (2026-07-13, audit risk-methodology): the name-only registry moved to
# `finance/leveraged.py` and now carries the leverage multiplier L, the
# underlying and the expense ratio.  For names with a KNOWN L the forward
# drag is the CONTRACTUAL −½L(L−1)σ_u² (σ_u = σ_ETP/|L| from the ETP's own
# daily history) plus the fee −ER/252, disclosed separately (C5).  Names in
# the registry WITHOUT parameters keep the F-23 empirical fallback
# min(α̂, 0).  Either way the adjustment can only LOWER a forecast (the
# anti-218% guard is preserved).  Extend via LEVERAGED_ETP_EXTRA (names) or
# LEVERAGED_ETP_PARAMS (full parameters) without a redeploy.
from finance.leveraged import (
    contractual_drag_daily as _contractual_drag_daily,
    etp_info as _etp_info,
    is_leveraged_etp as _is_leveraged_etp_impl,
)


def _is_leveraged_etp(ticker: str) -> bool:
    """True when the BASE symbol (suffix stripped) is a known daily-reset
    leveraged/inverse ETP (registry finance/leveraged.py or env extras).
    Kept as the historical name — tests and callers import it from here."""
    return _is_leveraged_etp_impl(ticker)


def apply_leveraged_drag(expected_log_daily: "np.ndarray",
                         tickers,
                         alpha_daily) -> "np.ndarray":
    """F-23: add min(α̂_daily, 0) to the forward daily log expectation of
    leveraged-ETP tickers only.  Pure function so the policy is unit-testable:
    ordinary names and positive alphas pass through untouched.  Retained as
    the EMPIRICAL fallback for registry names without a known multiplier —
    `apply_leveraged_forward` below routes between this and the contractual
    formula."""
    out = np.asarray(expected_log_daily, dtype=float).copy()
    alpha = pd.to_numeric(pd.Series(list(alpha_daily)), errors="coerce")\
              .fillna(0.0).values
    for i, t in enumerate(tickers):
        if i < len(out) and i < len(alpha) and _is_leveraged_etp(t):
            out[i] += min(float(alpha[i]), 0.0)
    return out


def apply_leveraged_forward(expected_log_daily: "np.ndarray",
                            tickers,
                            alpha_daily,
                            sigma_daily_map: "dict | None" = None,
                            ) -> "tuple[np.ndarray, dict]":
    """P-1: forward adjustment for daily-reset leveraged ETPs, with the
    contractual/empirical split disclosed per ticker.

    For each registry ticker:
      • L known AND σ_ETP available → contractual: drag = −½L(L−1)(σ_ETP/|L|)²,
        fee = −ER/252 — applied INSTEAD of the α̂ fallback (α̂ already contains
        the realised drag; applying both would double-count);
      • otherwise → empirical F-23 fallback min(α̂, 0).
    Ordinary names pass through untouched.  Every adjustment is ≤ 0.

    Returns (adjusted_array, details) where details maps ticker →
    {method, L, underlying, drag_daily, fee_daily, drag_ann_pct, fee_ann_pct}
    for the QC panel (audit C5: drag and cost-of-leverage are separate rows).
    """
    out = np.asarray(expected_log_daily, dtype=float).copy()
    alpha = pd.to_numeric(pd.Series(list(alpha_daily)), errors="coerce")\
              .fillna(0.0).values
    sigma_daily_map = sigma_daily_map or {}
    details: dict[str, dict] = {}
    for i, t in enumerate(tickers):
        if i >= len(out) or i >= len(alpha):
            continue
        info = _etp_info(t)
        if info is None:
            continue
        sigma = sigma_daily_map.get(str(t))
        if info.leverage is not None and sigma is not None and \
                np.isfinite(float(sigma)) and float(sigma) > 0:
            drag, fee = _contractual_drag_daily(
                info.leverage, float(sigma), info.expense_ratio)
            out[i] += drag + fee
            details[str(t)] = {
                "method":       "contractual",
                "L":            float(info.leverage),
                "underlying":   info.underlying,
                "drag_daily":   round(drag, 8),
                "fee_daily":    round(fee, 8),
                # Annualised display twins (log→simple on each component so
                # the QC panel reads in pp/yr without re-deriving).
                "drag_ann_pct": round((np.exp(drag * 252) - 1.0) * 100, 2),
                "fee_ann_pct":  round((np.exp(fee * 252) - 1.0) * 100, 2),
            }
        else:
            emp = min(float(alpha[i]), 0.0)
            out[i] += emp
            details[str(t)] = {
                "method":       "empirical_alpha",
                "L":            (float(info.leverage)
                                 if info.leverage is not None else None),
                "underlying":   info.underlying,
                "drag_daily":   round(emp, 8),
                "fee_daily":    0.0,
                "drag_ann_pct": round((np.exp(emp * 252) - 1.0) * 100, 2),
                "fee_ann_pct":  0.0,
            }
    return out, details


def orthogonalize_factors_hierarchical(f_data: "pd.DataFrame",
                                       return_betas: bool = False):
    """
    Residualize each style/EM factor against its core macro parent(s).

    Pure + deterministic.  Returns a NEW DataFrame with identical columns/index:
    core factors pass through untouched; each child factor is replaced by
    (child − OLS_fit_on_parents) + child_mean, so its level/scale is preserved
    while the shared parent beta is removed.  Falls back to the input unchanged
    when the core parents are absent or history is too short (<10 rows).

    F-1 (2026-07-10): with ``return_betas=True`` the function ALSO returns the
    fitted child→parent OLS coefficients ``{child: {parent: beta}}``.  The
    stress engine needs them to transform its RAW-factor shock catalog into
    the SAME residual space these betas live in — otherwise a raw «Momentum
    −15%» shock (which historically INCLUDED MTUM's market component) is
    applied to a market-neutral residual beta and the market leg of the shock
    is double-counted.  See ``finance.stress.residualize_shocks``.
    """
    if f_data is None or f_data.empty or len(f_data) < 10:
        return (f_data, {}) if return_betas else f_data
    cols = list(f_data.columns)
    out = f_data.copy()
    ortho_betas: dict[str, dict[str, float]] = {}
    for child, parents in _FACTOR_PARENTS.items():
        if child not in cols:
            continue
        par = [p for p in parents if p in cols]
        if not par:
            continue
        y = f_data[child].values.astype(float)
        X = np.column_stack(
            [np.ones(len(f_data))] + [f_data[p].values.astype(float) for p in par])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            out[child] = resid + float(np.mean(y))   # keep the factor's own mean
            # beta[0] is the intercept; beta[1:] align with `par`.
            ortho_betas[child] = {p: float(b) for p, b in zip(par, beta[1:])}
        except Exception:
            continue
    return (out, ortho_betas) if return_betas else out


def _nearest_psd(mat: "np.ndarray", floor: float = 1e-12) -> "np.ndarray":
    """
    Project a symmetric matrix onto the nearest positive-semi-definite matrix
    (Higham-style eigenvalue clipping).

    Sprint-3 #9: the factor covariance F is a blend (0.7·EWMA + 0.3·Ledoit-Wolf)
    and is NOT guaranteed PSD; a non-PSD F can produce slightly negative w'Σw and
    biased per-asset MCTR/ERC.  This symmetrises F and clamps negative eigenvalues
    to `floor`, then rebuilds it.  When F is already PSD the operation is a no-op
    within float tolerance, so production decompositions are unchanged in the
    common case — it only repairs the rare degenerate blend.
    """
    M = np.asarray(mat, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.size == 0:
        return M
    M = 0.5 * (M + M.T)                      # symmetrise
    try:
        w, V = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        return M
    if np.all(w >= -floor):                  # already PSD → leave untouched
        return M
    w_clipped = np.clip(w, floor, None)
    return (V * w_clipped) @ V.T


class MAC3RiskEngine:
    """
    Институциональный движок рисков (RAMP Style).
    Внедрены: Структурная ковариация (Barra/MAC3), Euler Risk Decomposition, CVaR.
    """
    # Все тикеры используют формат Tradernet "SYMBOL.EXCHANGE".
    # KSPI → KSPI.KZ потому что Tradernet нативно котирует KZ-биржу.
    # KAP/HSBK → .IL (Лондон) — там их основная ликвидность.
    TICKER_MAP = {
        'KAZATOMPROM':   'KAP.IL',
        'KASPI':         'KSPI.KZ',
        'KSPI':          'KSPI.KZ',
        'HALYK':         'HSBK.IL',
        'HSBK':          'HSBK.IL',
        'KAZAKHTELECOM': 'KZTK.KZ',
        'BITCOIN':       'BTC-USD',
        'ETHEREUM':      'ETH-USD',
    }

    # Инструменты нулевого риска (или которые не прогоняются через регрессию)
    NON_RISK_ASSETS = ['USD', 'EUR', 'CASH', 'RUB', 'KZT']

    # Proxy-ETF для облигаций (формат Tradernet — все .US).
    BOND_PROXIES = {
        'RF': 'BIL.US',   # Risk Free (1-3 Month T-Bill ETF)
        'IG': 'LQD.US',   # Investment Grade (Corporate Bond ETF)
        'HY': 'HYG.US',   # High Yield (Junk Bond ETF)
    }

    # Маппинг известных облигаций к категориям (для автоопределения)
    BOND_CLASSIFICATION_MAP = {
        'KZ_GOV_BOND': 'RF',
        'US_TREASURY': 'RF',
        'KASPI_BOND':  'IG',
    }

    # Прокси для инструментов, у которых либо нет ценовой истории, либо
    # история слишком короткая/иллквидная для надёжного факторного анализа.
    # Применяется ТОЛЬКО к ковариации/корреляции — текущая цена и P&L
    # берутся из реального тикера (брокера).
    #
    # Логика подбора прокси: похожая duration + кредит-рейтинг + валюта.
    INSTRUMENT_PROXY_MAP = {
        # Astana Exchange Special Purpose Companies (структурные ноты Freedom).
        # Обычно — короткая duration USD/KZT corporate, аналог LQD.
        'AIX_SPC':           'LQD.US',
        # Любой ISIN-тикер с .AIX суффиксом (пример: FFSPC6.1028.AIX).
        # Если префикс другой — fallback к консервативному T-Bill.
        'AIX_DEFAULT':       'BIL.US',
        # KZ корпоративные бонды без прямого Tradernet-маппинга.
        'KZ_CORPORATE_BOND': 'LQD.US',
        # Развивающиеся рынки — KAP.IL/HSBK.IL/KSPI.KZ если основная серия слишком короткая.
        'EM_EQUITY_FALLBACK': 'EEM.US',
    }

    # Sector ETF index map — used for sector exposure analysis in reports.
    # Format: sector_name -> Tradernet ETF ticker.
    SECTOR_ETF_MAP = {
        'Technology':     'XLK.US',
        'Semiconductors': 'SOXX.US',
        'Finance':        'XLF.US',
        'Energy':         'XLE.US',
        'Healthcare':     'XLV.US',
        'Commodities':    'DBC.US',
        'Gold':           'GLD.US',
        'Silver':         'SLV.US',
        'Oil':            'USO.US',
        'Industrials':    'XLI.US',
        'Consumer':       'XLP.US',
        'Materials':      'XME.US',
    }

    # Ticker-to-sector classification for automatic sector detection.
    # KZ/KASE tickers are classified as EM_Kazakhstan — they are benchmarked
    # against the MSCI EM index (EEM.US), not the S&P 500.
    TICKER_SECTOR = {
        # US Technology
        'AAPL':  'Technology',
        'MSFT':  'Technology',
        'GOOGL': 'Technology',
        'META':  'Technology',
        'AMZN':  'Technology',
        'ORCL':  'Technology',
        # US Semiconductors
        'NVDA':  'Semiconductors',
        'AVGO':  'Semiconductors',
        'AMD':   'Semiconductors',
        'INTC':  'Semiconductors',
        'QBTS':  'Semiconductors',
        'SOXX':  'Semiconductors',
        'TSM':   'Semiconductors',
        # US Finance
        'JPM':   'Finance',
        'GS':    'Finance',
        'BAC':   'Finance',
        # US Energy
        'XOM':   'Energy',
        'CVX':   'Energy',
        'COP':   'Energy',
        # US Healthcare
        'JNJ':   'Healthcare',
        'UNH':   'Healthcare',
        'PFE':   'Healthcare',
        # Commodities / Precious Metals
        'GLD':   'Gold',
        'SLV':   'Silver',
        'GDX':   'Gold',
        'USO':   'Oil',
        # Kazakhstan / KASE / AIX — EM_Kazakhstan (benchmarked vs MSCI EM)
        'KSPI':  'EM_Kazakhstan',
        'HSBK':  'EM_Kazakhstan',
        'KAP':   'EM_Kazakhstan',
        'KZTK':  'EM_Kazakhstan',
        'KCEL':  'EM_Kazakhstan',
        'BAST':  'EM_Kazakhstan',
        'HRGL':  'EM_Kazakhstan',
        'KZAP':  'EM_Kazakhstan',
        # US Bond ETFs
        'TLT':   'Bonds',
        'AGG':   'Bonds',
        'BND':   'Bonds',
        'LQD':   'Bonds',
        'HYG':   'Bonds',
        'IEF':   'Bonds',
        'BIL':   'Bonds',
        'EMB':   'Bonds',
        'SHY':   'Bonds',
        'VCIT':  'Bonds',
    }

    # Benchmark / catalogue ETFs fetched alongside factor ETFs.
    # NOT used as regression factors — only for benchmark_comparison + TE.
    # Includes all tickers from profile_manager.BENCHMARK_LIST so they are
    # pre-loaded in a single batch (avoids a second download pass).
    BENCHMARK_EXTRA = [
        'QQQ.US', 'AGG.US', 'URTH.US',  # broad indices
        # Factor ETFs already in factor_tickers (SPY, MTUM, VLUE, QUAL, IWM, DBC, IEF, EEM, EMB)
    ]

    def __init__(self, trading_days=252, ewma_halflife=63,
                 reporting_currency: ReportingCurrency | str | None = None,
                 fx_provider=None, risk_mandate: str = "MODERATE"):
        self.trading_days = trading_days
        # halflife=63 торговых дня (~3 мес) ⇒ дневной decay λ = 0.5^(1/63) ≈ 0.989.
        # Это СОЗНАТЕЛЬНО плавнее RiskMetrics λ=0.94 (что соответствовало бы
        # halflife ≈ 11 дн): факторная ковариация стабильнее, меньше шумовых
        # разворотов весов. (Sprint-5.1: прежний коммент «≈ λ=0.94» был неверен.)
        self.ewma_halflife = ewma_halflife
        # H4: investor risk mandate (CONSERVATIVE/MODERATE/AGGRESSIVE) — drives
        # the composite-risk-score weighting + CVaR base divisor.
        from finance.scoring import normalize_risk_mandate as _nrm
        self.risk_mandate = _nrm(risk_mandate)
        # Факторные прокси (формат Tradernet — все .US ETF).
        # EM_Equity (EEM.US) и EM_Bond (EMB.US) добавлены для корректного
        # моделирования казахстанских и EM-активов (.KZ/.IL тикеры).
        self.factor_tickers = {
            'Market':      'SPY.US',    # S&P 500 — глобальный рыночный фактор
            'Momentum':    'MTUM.US',
            'Value':       'VLUE.US',
            'Quality':     'QUAL.US',
            'Size':        'IWM.US',
            'Volatility':  'SPLV.US',   # Invesco S&P 500 Low Volatility — low-vol factor
            'Commodities': 'DBC.US',
            'Rates':       'IEF.US',    # 7-10y Treasury — фактор процентных ставок
            'EM_Equity':   'EEM.US',    # MSCI Emerging Markets — EM equity premium
            'EM_Bond':     'EMB.US',    # JP Morgan EM Bond ETF — EM credit/FX premium
        }
        # ── H2: Base-currency / RFR layer ────────────────────────────────────
        # Reporting currency drives BOTH the FX transformation of the price
        # matrix and the RFR applied in Sharpe/Sortino — they must agree to
        # avoid the cross-currency premium leak (USD assets vs KZ-RFR bug).
        if reporting_currency is None:
            self.reporting_currency = ReportingCurrency.from_env()
        elif isinstance(reporting_currency, ReportingCurrency):
            self.reporting_currency = reporting_currency
        else:
            self.reporting_currency = ReportingCurrency(str(reporting_currency).upper())
        self.current_rfr_annual, self.rfr_source = get_rfr_for_currency(self.reporting_currency)
        # Geometric daily RFR (H3) — cached once, used by both Sharpe and
        # Sortino's downside filter so they stay consistent.
        self.current_rfr_daily = daily_rfr_geometric(self.current_rfr_annual,
                                                     trading_days)
        # FX provider hook: callable (base_ccy, quote_ccy) -> pd.Series, or
        # None to bypass FX (USD-only portfolios stay on the fast path).
        # When None AND a real FRED_API_KEY is set in env, we auto-wire the
        # FredFxProvider so Cloud Run picks up DEXKZUS without any caller
        # change.  Import is local to avoid pulling `requests` into the
        # sklearn-free unit-test path.
        if fx_provider is None:
            try:
                from services.fx_feed import default_fx_provider
                fx_provider = default_fx_provider()
            except Exception as exc:                  # pragma: no cover
                logger.warning("Auto FX provider unavailable: %s", exc)
                fx_provider = None
        self.fx_provider = fx_provider
        # Кешируемый клиент Tradernet — переиспользуется между запросами одного
        # вызова (внутри analyze_all). Keys читаются из env (Cloud Run secret).
        self._tradernet_client: TradernetClient | None = None
        # Audit trail filled by `_apply_fx_conversion` — surfaced into the
        # report's QC panel so the user can verify what was converted.
        self._last_fx_records: list = []
        self._last_asset_currencies: dict[str, str] = {}

    def math_firewall(self, df):
        """
        Защита от 'битых' данных (галлюцинаций API).

        Replaces ±Inf with NaN BEFORE the directional fill so a single
        infinite price tick can't survive the fill and poison the covariance
        matrix or Sharpe/CVaR downstream.  ffill then carries the last finite
        value across interior gaps and up to the frame's end.

        F-6 (2026-07-10): the old trailing ``.bfill()`` is REMOVED — it filled
        LEADING NaNs by copying an asset's first quote backwards over its
        pre-listing period.  Those flat phantom prices are a look-ahead bias:
        zero log-returns that shrink a young asset's σ/β toward zero and
        distort its correlations.  Leading NaNs now survive the firewall by
        design; the covariance path handles them via the row-level dropna
        (honest common-date window) plus the sparse-asset guard in
        ``calculate_structural_risk``.
        """
        cleaned = df.replace([np.inf, -np.inf], np.nan)
        return cleaned.ffill()

    # ── Bootstrap CVaR (stationary block bootstrap, Politis-Romano 1994) ─────
    @staticmethod
    def _bootstrap_cvar(returns: np.ndarray, n_boot: int = 2000,
                        alpha: float = 0.05, seed: int | None = None) -> dict:
        """
        Compute CVaR(alpha) point estimate plus 95% bootstrap CI.

        Stationary block-bootstrap with geometric block lengths (mean ≈ √N)
        preserves daily-return autocorrelation — naive iid resampling
        underestimates tail risk on serially-dependent series.

        H5 — deterministic-yet-data-driven seed
        ────────────────────────────────────────
        When `seed is None`, the seed is derived from the returns vector
        itself (`int(abs(hash(tuple(round(r, 6)))) % 1e8)`).  This keeps
        the call deterministic for unit tests (same input → same CI) but
        makes the CI move when the input window rolls forward — fixing
        the H5 "CI never widens between runs" UX bug while preserving
        reproducibility.  Tests that explicitly pin `seed=N` keep working.

        Returns: {'point': float, 'lo95': float, 'hi95': float}
        """
        if seed is None:
            # Sprint-3 #10: derive the seed with hashlib.sha256, NOT the builtin
            # hash().  Python's hash() of containers is salted per-process for
            # some types and its algorithm is an implementation detail — so the
            # same return window could seed differently across interpreters /
            # versions, breaking the "same input → same CI" contract.  sha256 of
            # the rounded byte buffer is stable on every Python/OS.  Round to 6
            # decimals first so float noise on identical inputs can't move it.
            try:
                arr = np.round(np.asarray(returns, dtype=float), 6)
                digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).digest()
                seed = int.from_bytes(digest[:8], "big") % 100_000_000
            except (TypeError, ValueError):
                seed = 42       # fall back if returns is exotic (object dtype)
        rng = np.random.default_rng(seed)
        N = len(returns)
        if N == 0:
            return {"point": 0.0, "lo95": None, "hi95": None}
        block = max(1, int(np.sqrt(N)))
        cvars = np.empty(n_boot, dtype=float)
        ret_arr = np.asarray(returns, dtype=float)
        for b in range(n_boot):
            idx = np.empty(N, dtype=np.int64)
            filled = 0
            while filled < N:
                start = int(rng.integers(0, N))
                L = int(rng.geometric(1.0 / block))
                take = min(L, N - filled)
                for k in range(take):
                    idx[filled + k] = (start + k) % N
                filled += take
            sample = ret_arr[idx]
            var_b = np.percentile(sample, alpha * 100)
            tail = sample[sample <= var_b]
            cvars[b] = float(tail.mean()) if tail.size else float(var_b)
        return {
            "point": float(np.mean(cvars)),
            "lo95":  float(np.percentile(cvars, 2.5)),
            "hi95":  float(np.percentile(cvars, 97.5)),
        }

    # ── Marginal VaR (numerical sensitivity dVaR/dw_i) ───────────────────────
    @staticmethod
    def _marginal_var(a_data: pd.DataFrame, weights: np.ndarray,
                      var_p: float = 0.05, h: float = 0.005) -> pd.Series:
        """
        Symmetric finite-difference marginal VaR per asset, BUDGET-NEUTRAL.
        dVaR/dw_i ≈ (VaR(w⁺) − VaR(w⁻)) / (2h), where w± bumps w_i by ±h and
        rescales the REST of the book so Σw stays constant — i.e. the +1%
        is funded by selling the other names pro-rata, not by adding leverage.

        (F-1 fix: the previous version bumped w_i alone, which changed gross
        exposure and measured a leveraged sensitivity — systematically
        inflating |M-VaR| versus the reallocation the report describes.)

        Result is in the same units as one-day return (decimal).  A negative
        Marginal VaR for a long position means reallocating INTO that position
        would REDUCE 1-day downside (rare, usually negatively-correlated hedges).
        """
        cols = list(a_data.columns)
        ret_mat = a_data.values
        total_w = float(weights.sum())
        out: dict[str, float] = {}

        def _bumped(i: int, dh: float) -> np.ndarray:
            w = weights.copy()
            rest = total_w - w[i]
            if rest > 1e-12:
                # Fund the bump pro-rata from the rest of the book: Σw unchanged.
                w_other_scale = (rest - dh) / rest
                w *= w_other_scale
                w[i] = weights[i] + dh
            else:
                # Single-asset book — nothing to fund from; fall back to the
                # raw bump (documented leveraged sensitivity for this edge).
                w[i] = weights[i] + dh
            return w

        for i, ticker in enumerate(cols):
            v_plus  = float(np.percentile(ret_mat @ _bumped(i, +h), var_p * 100))
            v_minus = float(np.percentile(ret_mat @ _bumped(i, -h), var_p * 100))
            out[ticker] = (v_plus - v_minus) / (2.0 * h)
        return pd.Series(out)

    # ── Composite Risk Score (0..100) ────────────────────────────────────────
    # Thin delegate to the single source of truth in finance.scoring.
    # Kept as a staticmethod so existing callers / tests that reference
    # MAC3RiskEngine._composite_risk_score keep working unchanged.
    @staticmethod
    def _composite_risk_score(volatility: float, cvar: float,
                              max_erc_pct: float, mandate: str = "MODERATE",
                              *, max_drawdown: float | None = None,
                              leverage_ratio: float | None = None,
                              sector_top_pct: float | None = None) -> int:
        from finance.scoring import composite_risk_score as _crs
        return _crs(volatility, cvar, max_erc_pct, mandate=mandate,
                    max_drawdown=max_drawdown, leverage_ratio=leverage_ratio,
                    sector_top_pct=sector_top_pct)

    def _get_tradernet_client(self) -> TradernetClient:
        """Lazy-init Tradernet client.  Reused across calls inside analyze_all."""
        if self._tradernet_client is None:
            api_key    = (os.getenv("FREEDOM_API_KEY")    or "").strip()
            secret_key = (os.getenv("FREEDOM_API_SECRET") or "").strip()
            self._tradernet_client = TradernetClient(api_key, secret_key)
        return self._tradernet_client

    def resolve_tickers(self, tickers):
        """
        Преобразует пользовательский тикер в формат Tradernet ("SYMBOL.EXCHANGE").

        Спецслучаи:
          - cash currencies → пропускаются (NON_RISK_ASSETS)
          - известные KZ облигации → ETF-прокси (BOND_PROXIES)
          - крипто → '*-USD' (исторический формат, для совместимости с
            форком CCXT/Yahoo, не Tradernet — но если тикер известен через
            Tradernet, его можно подменить позже)
          - AIX-инструменты (FFSPC*.AIX) → INSTRUMENT_PROXY_MAP['AIX_SPC'] для
            фактор-моделирования.  Ценообразование остаётся через брокера.
        """
        resolved = []
        aix_found: list[str] = []
        for t in tickers:
            t_str = str(t).upper().strip()
            if t_str in self.NON_RISK_ASSETS:
                continue   # Cash отдельно

            # 1. Известные облигации → ETF-прокси
            if t_str in self.BOND_CLASSIFICATION_MAP:
                category = self.BOND_CLASSIFICATION_MAP[t_str]
                resolved.append(self.BOND_PROXIES.get(category, self.BOND_PROXIES['RF']))
                continue
            # 2. Эвристика по имени тикера
            if 'BOND' in t_str or 'OVD' in t_str:
                resolved.append(self.BOND_PROXIES['IG'])
                continue

            # 3. AIX (Astana International Exchange) — структурные ноты Freedom
            if t_str.endswith('.AIX') or 'FFSPC' in t_str:
                proxy = self.INSTRUMENT_PROXY_MAP.get('AIX_SPC',
                          self.INSTRUMENT_PROXY_MAP['AIX_DEFAULT'])
                aix_found.append(t_str)
                resolved.append(proxy)
                continue

            # 4. Явный маппинг → формат Tradernet
            if t_str in self.TICKER_MAP:
                resolved.append(self.TICKER_MAP[t_str])
                continue

            # 5. Крипто (Tradernet их не торгует напрямую — оставляем *-USD формат
            #    как маркер; отсутствующие колонки пропускаются без падения)
            if t_str in ('BTC', 'ETH', 'SOL', 'BNB'):
                resolved.append(f"{t_str}-USD")
                continue

            # 6. Если уже в формате SYM.EXCHANGE — оставляем
            if '.' in t_str:
                resolved.append(t_str)
                continue

            # 7. Голый US-тикер → добавить .US
            resolved.append(f"{t_str}.US")
        if aix_found:
            proxy = self.INSTRUMENT_PROXY_MAP.get('AIX_SPC', self.INSTRUMENT_PROXY_MAP['AIX_DEFAULT'])
            logger.info("AIX instruments proxied to %s for factor model: %s", proxy, ", ".join(aix_found))
        return resolved

    def get_market_data(self, tickers, period_days: int | None = None):
        """
        Загрузка дневных закрытий через Tradernet API (replaces yfinance).

        period_days (Фаза 4 · Data Resilience §4-б): default читается из env
        ``HISTORY_LOOKBACK_DAYS``.  С 2026-07-07 дефолт = **1825 КАЛЕНДАРНЫХ
        дней ≈ 1260 ТОРГОВЫХ ≈ 5 лет** (ровно целевое окно §4-б: «1260 торговых
        ≈ 1825 кал.»).  Параметр `days` в `history.py` — КАЛЕНДАРНЫЙ
        (`today − timedelta(days=days)`), поэтому 5 торговых лет = 1825 кал., а
        НЕ 1260 кал. (то было бы ~3.45 года).  Расширение окна меняет все
        метрики; молодые активы не роняют матрицу — sparse-history дроп +
        row-level `dropna` дают общее пересечение дат (эффективное окно =
        min-история среди активов и фактор-ETF), а `min_obs` гейт защищает
        от вырожденной регрессии.  Откат на прежнее 2-летнее окно —
        `HISTORY_LOOKBACK_DAYS=730`.

        Returns (data, history_result) tuple where history_result contains
        loaded/failed/retried ticker details for user-facing messages.

        H2: After Tradernet loads native-currency prices, we transform
        every column whose native currency differs from
        `self.reporting_currency` by multiplying with the matching FX
        series.  USD-only portfolios short-circuit (no FX calls, no copy).
        """
        if period_days is None:
            try:
                # Дефолт 1825 кал.дн ≈ 1260 торговых ≈ 5 лет (§4-б Data Resilience).
                period_days = int(os.getenv("HISTORY_LOOKBACK_DAYS", "1825"))
            except (TypeError, ValueError):
                period_days = 1825
            period_days = max(90, min(3650, period_days))   # sanity clamp

        valid_tickers = self.resolve_tickers(tickers)
        all_req = list(dict.fromkeys(
            valid_tickers + list(self.factor_tickers.values()) + self.BENCHMARK_EXTRA
        ))

        logger.info("Загрузка цен через Tradernet для %d инструментов: %s",
                    len(all_req), ", ".join(all_req))

        client = self._get_tradernet_client()
        history_result = get_history_frame(client, all_req, days=period_days)

        if history_result.data.empty:
            logger.error("Tradernet вернул пустой набор цен — все тикеры провалились")
            return history_result.data, history_result

        firewalled = self.math_firewall(history_result.data)
        # ── H2: FX conversion to reporting currency ──────────────────────────
        converted = self._apply_fx_conversion(firewalled)
        return converted, history_result

    def _apply_fx_conversion(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the price matrix into the reporting currency (H2).

        Safety:
          • If every column is already in `reporting_currency`, returns the
            input frame unchanged (no allocation).
          • Never mutates the caller's frame: convert_price_matrix copies
            internally before writing.
          • Updates `self._last_fx_records` and `self._last_asset_currencies`
            so the report's QC panel can show what was actually converted.
        """
        asset_currencies = infer_currencies_for_tickers(prices.columns)
        # Factor ETFs are all .US → USD; explicitly pin them so an exotic
        # symbol that sneaks in doesn't trigger a spurious FX call.
        for factor_tkr in self.factor_tickers.values():
            asset_currencies.setdefault(factor_tkr, "USD")
        result = convert_price_matrix(
            prices            = prices,
            asset_currencies  = asset_currencies,
            reporting         = self.reporting_currency,
            fx_provider       = self.fx_provider,
            lag_one_day       = True,
        )
        self._last_fx_records       = result.fx_records
        self._last_asset_currencies = result.asset_currencies
        if not result.no_op:
            logger.info(
                "FX → %s applied to %d pair(s): %s",
                self.reporting_currency.value, len(result.fx_records),
                ", ".join(r.pair for r in result.fx_records),
            )
        return result.prices_base

    def get_ticker_sector(self, ticker: str) -> str:
        """Возвращает сектор для тикера (или 'Other').

        KZ-тикеры с суффиксами .KZ или .IL автоматически классифицируются как
        EM_Kazakhstan, если они не найдены в TICKER_SECTOR по базовому имени.
        """
        base = ticker.split('.')[0].upper() if '.' in ticker else ticker.upper()
        suffix = ticker.upper().rsplit('.', 1)[-1] if '.' in ticker else ''
        sector = self.TICKER_SECTOR.get(base)
        if sector:
            return sector
        # KZ/IL exchange suffix → treat as EM Kazakhstan
        if suffix in ('KZ', 'IL') or ticker.upper().endswith('.AIX'):
            return 'EM_Kazakhstan'
        return 'Other'

    def get_sector_exposure(self, tickers: list, weights: dict) -> dict:
        """
        Рассчитывает секторное распределение портфеля.

        Returns dict mapping sector name to total weight %.
        """
        sectors: dict[str, float] = {}
        for t in tickers:
            sector = self.get_ticker_sector(t)
            sectors[sector] = sectors.get(sector, 0) + weights.get(t, 0)
        # Sort by weight descending
        return dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True))

    def calculate_structural_risk(self, data, asset_tickers, weights_dict):
        """
        Ядро MAC3: Построение факторной модели и декомпозиция Эйлера.
        Облигации и Кэш в asset_tickers не передаются!
        Но weights_dict содержит их доли от ОБЩЕГО портфеля, поэтому сумма весов asset_tickers < 1.0.
        """
        # Reset the additive-analytics slot up-front: the early returns below
        # (no data / no factors / short history / zero weights) must never
        # leave a STALE decomposition from a previous call on the engine.
        self._last_factor_decomposition = {}
        # F-1: reset the orthogonalization betas alongside — a stale map from a
        # previous call would residualize stress shocks that were fitted on
        # different data (or when this run skips orthogonalization entirely).
        self._last_ortho_betas = {}
        # F-14: number of common observations the Ridge regression actually
        # used — BLOCK 5 gates the FORWARD expected-return panel on it (an
        # annualised forecast off a sub-year window is not publishable).
        self._last_regression_nobs = 0
        # F-20: reset the sparse-exclusion list alongside (same staleness
        # contract as the ortho betas above).
        self._last_sparse_dropped = []
        resolved_assets = self.resolve_tickers(asset_tickers)

        # Restrict to columns we actually need and drop all-NaN columns BEFORE
        # the row-level dropna. Without this, a single broker-only ticker with
        # no Tradernet history (e.g. FFSPC6.1028.AIX on Astana exchange) leaves
        # an all-NaN column whose NaNs propagate through the row-level dropna
        # and erase every row — Ridge then gets shape (0, K) and raises.
        needed_cols = [*self.factor_tickers.values(), *resolved_assets]
        available = [c for c in needed_cols if c in data.columns]
        data = data[available].dropna(axis=1, how='all')

        # F-6: sparse-history guard.  With leading NaNs no longer bfilled (the
        # look-ahead fix in math_firewall / history.py), the row-level dropna
        # below shrinks the common window to the YOUNGEST surviving column.  A
        # thinly-listed asset with a few weeks of quotes must not collapse the
        # whole book's 5-year window — drop it from the STRUCTURAL model
        # instead (same graceful contract as a missing ticker: its weight in
        # weights_dict still dilutes portfolio risk; price/PnL keep coming
        # from the broker).  Factor ETFs are exempt — long histories by
        # construction, and losing one would silently change the model.
        _factor_cols = set(self.factor_tickers.values())
        sparse_assets = [
            c for c in data.columns
            if c not in _factor_cols
            and int(data[c].notna().sum()) < _MIN_OVERLAP_TDAYS
        ]
        if sparse_assets:
            logger.warning(
                "Структурная модель: исключены %d актив(ов) с историей "
                "< %d торговых дней (защита общего окна дат): %s",
                len(sparse_assets), _MIN_OVERLAP_TDAYS, ", ".join(sparse_assets))
            data = data.drop(columns=sparse_assets)
        # F-20: expose the exclusions so the Action Plan can annotate WHY a
        # name has no beta / BL target / quantity instead of printing a bare
        # directional call (live report: «SELL SPCX» with qty=null).
        self._last_sparse_dropped = list(sparse_assets)

        # Логарифмические доходности для агрегации факторов
        returns = np.log(data / data.shift(1)).dropna()

        if returns.empty:
            logger.warning("Нет общих ценовых данных для факторной модели — пропускаем структурный риск.")
            return pd.DataFrame(), pd.DataFrame(), {}

        # 1. Выделяем факторы и активы
        present_factors = {k: v for k, v in self.factor_tickers.items() if v in returns.columns}
        if not present_factors:
            logger.warning("Ни одна фактор-серия не доступна — пропускаем структурный риск.")
            return pd.DataFrame(), pd.DataFrame(), {}
        f_data = returns[list(present_factors.values())]
        f_data.columns = list(present_factors.keys())

        # BLOCK 3.5: optionally orthogonalize style/EM factors against the core
        # macro factors to kill multicollinearity (κ≈62) BEFORE fitting betas +
        # covariance — so B and F are built on independent, still-named factors.
        # Gated; default OFF leaves the production decomposition unchanged.
        _orthogonalized = factor_orthogonalize_enabled()
        if _orthogonalized:
            # F-1: capture the child→parent OLS betas so the stress engine can
            # transform its RAW-space shock catalog into this residual space
            # (shock_resid = shock_raw − Σ β̂·shock_parent).  Without the
            # transform the market leg of a style shock is double-counted.
            f_data, self._last_ortho_betas = orthogonalize_factors_hierarchical(
                f_data, return_betas=True)

        # F-3: underdetermined-regression guard.  Ridge fits K factor betas per
        # asset and LedoitWolf/EWMA estimate a KxK factor covariance — with
        # fewer observations than ~2K the betas are near-interpolating noise
        # and the covariance is degenerate/rank-deficient, silently poisoning
        # structural vol, every ERC% and the Black-Litterman prior.  Skip the
        # structural model entirely (same graceful contract as the guards
        # above); callers already tolerate the empty return.
        k_factors = len(present_factors)
        min_obs   = max(2 * k_factors, 30)
        if len(returns) < min_obs:
            logger.warning(
                "Структурный риск пропущен: %d общих наблюдений < %d "
                "(нужно ≥2×%d факторов; короткая общая история).",
                len(returns), min_obs, k_factors,
            )
            return pd.DataFrame(), pd.DataFrame(), {}
        # F-14: record the effective regression window (common date rows) so
        # analyze_all can gate the forward expected-return panel on it.
        self._last_regression_nobs = int(len(returns))
        
        # Защита от отсутствующих тикеров:
        existing_resolved = [r for r in resolved_assets if r in returns.columns]
        # Сопоставляем обратно с оригинальными (чтобы отчет был красивым)
        valid_originals = []
        valid_resolved = []
        for orig, res in zip(asset_tickers, resolved_assets):
            if res in returns.columns:
                valid_originals.append(orig)
                valid_resolved.append(res)
                
        if not valid_resolved:
            logger.warning("Нет рисковых активов для расчета.")
            return pd.DataFrame(), pd.DataFrame(), {}
            
        a_data = returns[valid_resolved]
        a_data.columns = valid_originals # Возвращаем оригинальные имена
        
        # Вектор весов (БЕЗ нормализации к 1.0! Если вложено 50% в акции и 50% в Кэш, сумма здесь будет 0.5)
        # Это обеспечит правильное разбавление риска
        weights = np.array([weights_dict.get(ticker, 0) for ticker in valid_originals])
        
        # 2. Факторная регрессия (Поиск Бета коэффициентов)
        B_matrix = [] # Экспозиции
        specific_variances = [] # Специфический риск
        exposures_report = {}
        # P-4 (audit B1): SE(β_k) ≈ σ_ε / (σ_{f_k}·√T) — точная OLS-формула
        # для ОРТОГОНАЛЬНЫХ регрессоров; факторы ортогонализованы по умолчанию
        # (BLOCK 3.5), для legacy raw-режима это нижняя граница SE.  Кэшируем
        # входы один раз на весь цикл.
        _f_std_daily = f_data.std(ddof=1)
        _reg_T = int(len(f_data))
        beta_se_map: dict[str, dict[str, float]] = {}
        # P-4b: Vasicek-сжатие (default OFF — см. beta_shrinkage_enabled).
        _shrink_on = beta_shrinkage_enabled()
        try:
            _shrink_n0 = max(1, int(os.getenv(BETA_SHRINKAGE_PRIOR_OBS_ENV, "120")))
        except (TypeError, ValueError):
            _shrink_n0 = 120

        for asset in valid_originals:
            y = a_data[asset]
            X = f_data

            # alpha=1.0 shrinks daily-return betas by ~98% (penalty >> RSS on
            # 0.01-scale returns).  Use 0.001 — enough to stabilise correlated
            # factors while keeping betas economically meaningful.
            model = Ridge(alpha=0.001, fit_intercept=True).fit(X, y)
            betas = model.coef_
            alpha = model.intercept_

            # P-4b: Vasicek shrinkage toward the prior (Market → 1.0, либо L
            # для реестровых ETP; стили/EM → 0).  Интерсепт и остатки
            # ПЕРЕсчитываются с сжатыми бетами, чтобы D в Σ = B·F·Bᵀ + D
            # остался согласованным.  Выключено по умолчанию — прод не
            # меняется, пока сжатие не пройдёт отдельную валидацию.
            if _shrink_on:
                _w_shr = _reg_T / (_reg_T + _shrink_n0)
                _lev = None
                try:
                    from finance.leveraged import leverage_of as _lev_of
                    _lev = _lev_of(asset)
                except Exception:
                    _lev = None
                prior = np.array([
                    (float(_lev) if (_lev is not None and k == "Market") else
                     1.0 if k == "Market" else 0.0)
                    for k in f_data.columns], dtype=float)
                betas = _w_shr * np.asarray(betas, dtype=float) \
                    + (1.0 - _w_shr) * prior
                alpha = float(np.mean(y)
                              - np.asarray(X.mean().values, dtype=float) @ betas)
                residuals = np.asarray(
                    y - (X.values @ betas + alpha), dtype=float)
            else:
                residuals = np.asarray(y - model.predict(X), dtype=float)

            # Guard: ddof=1 variance needs ≥2 finite residuals, else it
            # divides by (n-1)=0 → NaN/Inf that would poison D and the
            # structural covariance.  A 1-sample (or degenerate) regression
            # carries no measurable specific risk → treat as 0.
            finite_res = residuals[np.isfinite(residuals)]
            if finite_res.size < 2:
                spec_var = 0.0
                resid_vol_ann = 0.0
            else:
                spec_var = float(np.var(finite_res, ddof=1))
                resid_vol_ann = float(np.std(finite_res, ddof=1)) * np.sqrt(self.trading_days)

            B_matrix.append(betas)
            specific_variances.append(spec_var)

            # P-4 (audit B1): standard errors of the fitted betas.
            # SE(β_k) = σ_ε / (σ_{f_k}·√T); NaN-safe on degenerate factors.
            if finite_res.size >= 2 and _reg_T > 0:
                _res_sd = float(np.std(finite_res, ddof=1))
                _se_row: dict[str, float] = {}
                for _k in f_data.columns:
                    _fsd = float(_f_std_daily.get(_k, 0.0) or 0.0)
                    if _fsd > 0 and np.isfinite(_res_sd):
                        _se_row[str(_k)] = round(
                            _res_sd / (_fsd * np.sqrt(_reg_T)), 6)
                if _se_row:
                    beta_se_map[str(asset)] = _se_row

            exposures_report[asset] = {f'Beta_{k}': b for k, b in zip(f_data.columns, betas)}
            exposures_report[asset]['Specific_Alpha_Daily'] = alpha
            # ddof=1 for self-consistency with specific_variances above
            # (both are estimators of residual dispersion; biased ddof=0
            # under-reports the per-asset residual vol by ~√((n-1)/n)).
            exposures_report[asset]['Residual_Vol_Ann'] = resid_vol_ann
            exposures_report[asset]['Weight_Pct'] = weights_dict.get(asset, 0) * 100

        B = np.array(B_matrix) # (N_assets, K_factors)
        D = np.diag(specific_variances) # (N_assets, N_assets)
        
        # 3. Структурная Ковариационная Матрица
        # EWMA-ковариация: halflife=63 дн ⇒ λ_daily ≈ 0.989 (плавнее
        # RiskMetrics λ=0.94 ≈ hl 11 дн — выбрано для стабильности факторной
        # матрицы).  Недавние дни влияют сильнее, чем старые.
        ewma_cov = f_data.ewm(halflife=self.ewma_halflife).cov()
        # Берём последний срез (самый актуальный)
        last_date = f_data.index[-1]
        cov_factors_ewma = ewma_cov.loc[last_date].values
        
        # Ledoit-Wolf Shrinkage (стабилизация при малых N)
        try:
            lw = LedoitWolf().fit(f_data.values)
            cov_factors_shrunk = lw.covariance_
            # Блендим EWMA (70%) + Shrinkage (30%) для баланса адаптивности и стабильности
            cov_factors = 0.7 * cov_factors_ewma + 0.3 * cov_factors_shrunk
        except Exception:
            cov_factors = cov_factors_ewma
        # Sprint-3 #9: the EWMA+LW blend is not guaranteed PSD — project it onto
        # the nearest PSD matrix so Σ = B·F·Bᵀ + D stays PSD (well-defined ERC,
        # no negative w'Σw).  No-op within tolerance when the blend is already PSD.
        cov_factors = _nearest_psd(cov_factors)

        # ── BLOCK 4.6: factor-multicollinearity diagnostic (read-only) ───────
        # The structural model ALREADY avoids double-counting correlated risk:
        # Σ = B·F·Bᵀ + D carries the FULL off-diagonal factor covariance F, so
        # two correlated factors contribute their covariance exactly once.  A
        # PCA / orthogonalisation would only DESTROY the named-factor
        # interpretability the report depends on — so instead of forcing it we
        # MEASURE collinearity and surface a diagnostic:
        #   • max |off-diagonal correlation| across the factor set
        #   • condition number κ of the factor CORRELATION matrix
        # |corr| > 0.95 or κ > 30 ⇒ near-collinear factors make the Ridge betas
        # unstable; we log a warning and feed the numbers to a CoVe row so the
        # operator can prune/merge a redundant factor (e.g. SPY vs QQQ).
        factor_diagnostic: dict = {}
        try:
            _F = np.asarray(cov_factors, dtype=float)
            if _F.ndim == 2 and _F.shape[0] >= 2:
                _d    = np.sqrt(np.clip(np.diag(_F), 1e-18, None))
                _corr = _F / np.outer(_d, _d)
                _off  = _corr - np.eye(_corr.shape[0])
                _max_off = float(np.max(np.abs(_off))) if _off.size else 0.0
                _cond    = float(np.linalg.cond(_corr))
                # P-5 (audit D4): Fisher z-ДИ на максимальную |off-diag|
                # корреляцию — на коротком окне регрессии знак/величина ρ̂
                # статистически не определены, и ДИ делает это видимым.
                _corr_ci = _fisher_rho_ci(_max_off, int(len(returns)))
                factor_diagnostic = {
                    "n_factors":        int(_F.shape[0]),
                    "factors":          list(f_data.columns),
                    "max_abs_corr":     round(_max_off, 4),
                    "max_corr_ci95":    ([round(x, 4) for x in _corr_ci]
                                         if _corr_ci else None),
                    "corr_window_days": int(len(returns)),
                    "condition_number": round(_cond, 2),
                    "near_collinear":   bool(_max_off > 0.95 or _cond > 30.0),
                    # BLOCK 3.5: whether the hierarchical orthogonalization was
                    # applied (κ above is then the POST-orthogonalization value).
                    "orthogonalized":   bool(_orthogonalized),
                }
                if factor_diagnostic["near_collinear"]:
                    logger.warning(
                        "Факторная мультиколлинеарность: max|corr|=%.2f, κ=%.1f "
                        "(порог 0.95 / 30 → беты Ridge нестабильны; рассмотрите "
                        "слияние коррелирующих факторов).",
                        _max_off, _cond,
                    )
        except Exception as exc:
            logger.debug("Factor diagnostic skipped: %s", exc)
        self._last_factor_diagnostic = factor_diagnostic

        structural_cov = B @ cov_factors @ B.T + D
        structural_cov_ann = structural_cov * self.trading_days 
        
        cov_df = pd.DataFrame(structural_cov_ann, index=valid_originals, columns=valid_originals)
        
        # Если портфель пустой по рисковым активам
        if weights.sum() == 0:
            return cov_df, pd.DataFrame(exposures_report).T, {}
            
        # 4. Риск портфеля
        port_variance = weights.T @ structural_cov_ann @ weights
        # BLK-3: the blended factor covariance (EWMA + Ledoit-Wolf) is not
        # guaranteed positive-semi-definite, so w'Σw can land marginally below
        # zero from float noise / a non-PSD matrix.  Clamp at 0 before the
        # sqrt so port_volatility is never NaN (a NaN here silently poisons
        # Sharpe/Sortino and zeroes every Euler risk contribution).
        port_variance = float(max(port_variance, 0.0))
        port_volatility = np.sqrt(port_variance)
        
        # 5. Декомпозиция Эйлера
        # MCTR = (Sigma * w) / Port_Vol
        mctr = (structural_cov_ann @ weights) / port_volatility if port_volatility > 0 else np.zeros_like(weights)
        # ERC (%) = w * MCTR / Port_Vol
        erc_pct = (weights * mctr) / port_volatility if port_volatility > 0 else np.zeros_like(weights)
        
        for i, asset in enumerate(valid_originals):
            exposures_report[asset]['Euler_Risk_Contribution_Pct'] = erc_pct[i] * 100

        # ── Additive analytics layer: factor-VARIANCE decomposition (по
        # источникам риска, а не по активам) + marginal overlap («факторные
        # двойники» + unique-risk share).  Pure add-on из отдельного модуля —
        # читает ТЕ ЖЕ B/F/D/w, что структурная модель уже построила, ничего
        # не мутирует; любая ошибка деградирует в {} (тот же graceful-паттерн,
        # что и κ-диагностика BLOCK 4.6 выше).
        try:
            from finance.factor_decomposition import build_factor_decomposition
            self._last_factor_decomposition = build_factor_decomposition(
                B=B,
                F=cov_factors,
                D=np.asarray(specific_variances, dtype=float),
                weights=weights,
                factor_names=list(f_data.columns),
                asset_names=list(valid_originals),
            )
        except Exception as exc:
            logger.debug("Factor decomposition skipped: %s", exc)
            self._last_factor_decomposition = {}

        # 6. Хвостовые метрики
        # `port_returns_daily` carries LOG returns in the REPORTING CURRENCY
        # (H2 pre-converted the price matrix → a_data is base-currency log
        # returns).  Sum(weights) < 1 by design when cash sits in the
        # portfolio — that dilutes risk correctly without renormalisation.
        port_returns_daily = a_data.values @ weights
        port_series_index  = a_data.index
        # F-22 (2026-07-11): realized-metrics basis — masked composite series.
        # The intersection window `a_data` shrinks to the YOUNGEST kept listing
        # (honest windows after F-6), silently re-basing Sharpe / Annualised
        # Return / MaxDD / VaR / CVaR onto a sub-year window when the book
        # holds one young name.  Rebuild the portfolio series over the FULL
        # price panel with per-day renormalised weights of the names actually
        # trading that day (the same composite-backfill convention as the
        # period-returns table and the equity curve), re-scaled by Σw so cash
        # dilution is preserved: on full-coverage days the composite equals
        # `a_data @ weights` EXACTLY, and when it adds no history the legacy
        # path is kept bit-for-bit (β/Σ regression stays on the intersection
        # window — only the REALIZED panel gains history).
        try:
            _w_by_res = {res: float(weights_dict.get(orig, 0.0) or 0.0)
                         for orig, res in zip(valid_originals, valid_resolved)}
            _total_w  = float(sum(_w_by_res.values()))
            _comp, _comp_info = _build_portfolio_log_returns(
                data[valid_resolved], _w_by_res)
            if _comp is not None and _total_w > 0 and len(_comp) > len(a_data):
                port_returns_daily = _comp.values * _total_w
                port_series_index  = _comp.index
                logger.info(
                    "Realized-metrics basis: composite %d дн "
                    "(окно пересечения %d дн, покрыто весов %.0f%%).",
                    len(_comp), len(a_data),
                    float(_comp_info.get("covered_weight", 0.0)) * 100)
        except Exception as _exc:
            logger.warning(
                "Composite series unavailable — realized metrics fall back "
                "to the intersection window: %s", _exc)
        # H3 (Phase-3): expose the date-indexed portfolio log-return series so
        # the Telegram equity-curve SVG consumes it instead of recomputing
        # `log(prices/prices.shift) @ weights` in the bot layer.  Stored on
        # the engine (same pattern as _last_fx_records); analyze_all copies it
        # into results["port_log_returns"].
        try:
            self._last_port_log_returns = pd.Series(port_returns_daily,
                                                    index=port_series_index)
        except Exception:
            self._last_port_log_returns = None
        # H3: geometric daily RFR — `(1+r)^(1/252)-1` rather than `r/252`.
        # Keeps Sortino's downside filter consistent with Sharpe's annual
        # excess-return numerator.
        risk_free_rate_daily = self.current_rfr_daily

        excess_returns = port_returns_daily - risk_free_rate_daily
        # H-4: industry-standard downside deviation (Sortino/Satchell).
        # Take the shortfalls min(excess, 0) against the MAR (=0), square them,
        # and average over the TOTAL number of observations N — not just the
        # losing days.  The previous `np.std(excess[excess<0])` used the wrong
        # denominator (count of negatives) and de-meaned around the downside
        # mean instead of the target, making the ratio non-comparable to the
        # standard definition.
        downside_shortfall = np.minimum(excess_returns, 0.0)
        downside_vol = np.sqrt(np.mean(downside_shortfall ** 2)) * np.sqrt(self.trading_days)

        # H1: geometric annualisation of log-returns.
        # Arithmetic `mean·252` over-states annual return by ~σ²/2 (it
        # ignores the Itô correction).  For 24% annual vol that's ~+2.9
        # pp of phantom return per year — directly inflates Sharpe.
        # `exp(mean_log·252) - 1` is the exact equivalent simple return.
        ann_return = float(np.exp(np.mean(port_returns_daily) * self.trading_days) - 1.0)

        var_95 = np.percentile(port_returns_daily, 5) if len(port_returns_daily)>0 else 0
        cvar_95 = port_returns_daily[port_returns_daily <= var_95].mean() if len(port_returns_daily)>0 else 0

        # Bootstrap CVaR with 95% CI (stationary block bootstrap, Politis-Romano).
        # The point CVaR above is just the empirical mean of the bottom-5%
        # observations on a finite window — its standard error is ~15-20% of
        # the magnitude.  Bootstrap gives a confidence interval the user can
        # weight against the lower bound when sizing tail-risk hedges.
        if len(port_returns_daily) >= 60:
            cvar_boot = self._bootstrap_cvar(port_returns_daily, n_boot=2000, alpha=0.05)
        else:
            cvar_boot = {"point": float(cvar_95), "lo95": None, "hi95": None}

        # Real Max Drawdown (peak-to-trough on equity curve).
        # port_returns_daily are daily LOG returns → equity curve = exp(cumsum).
        # MaxDD ≠ VaR_95: VaR is a 1-day quantile, MaxDD is the worst peak-to-trough
        # of the realised equity path. Both are needed for a complete risk picture.
        if len(port_returns_daily) > 0:
            eq_curve     = np.exp(np.cumsum(port_returns_daily))
            running_max  = np.maximum.accumulate(eq_curve)
            drawdowns    = eq_curve / running_max - 1.0
            max_drawdown = float(drawdowns.min())
        else:
            max_drawdown = 0.0

        # Marginal VaR per asset — sensitivity dVaR/dw_i.
        #
        # Useful complement to Euler TRC: TRC says "what fraction of CURRENT
        # portfolio risk is driven by asset i", M-VaR says "by how many bps
        # would 1-day VaR move if I add 1% weight to asset i".  M-VaR is in
        # the same units as VaR itself (decimal daily return per unit weight).
        if len(port_returns_daily) >= 60 and a_data.shape[1] > 0:
            mvar_series = self._marginal_var(a_data, weights, var_p=0.05, h=0.005)
            for asset, mvar_val in mvar_series.items():
                if asset in exposures_report:
                    exposures_report[asset]['Marginal_VaR_Daily'] = float(mvar_val)

        # Composite Risk Score 0..100 — blends the three independent signals
        # (volatility, tail, concentration) into a single user-facing gauge.
        # Replaces the prior gauge that looked at vol alone.
        max_erc = 0.0
        if len(weights) > 0 and port_volatility > 0:
            max_erc = float(np.max(np.abs(erc_pct))) * 100
        composite_risk = self._composite_risk_score(
            volatility=port_volatility, cvar=cvar_95, max_erc_pct=max_erc,
            mandate=getattr(self, "risk_mandate", "MODERATE"),
        )

        # P-4 (audit B1): сводный вердикт надёжности бет — максимальная
        # ОТНОСИТЕЛЬНАЯ SE по Market-оси; > 0.5 на окне регрессии короче
        # торгового года ⇒ беты статистически шумные (β=1.8 при SE=1.0
        # неотличима от 1.0) — warning в лог + вердикт в metrics.
        _beta_reliability = "ok"
        try:
            _rel_ses = []
            for _a, _ses in beta_se_map.items():
                _b_mkt = float(exposures_report.get(_a, {}).get("Beta_Market", 0.0) or 0.0)
                _se_mkt = float(_ses.get("Market", 0.0) or 0.0)
                if abs(_b_mkt) > 1e-9 and _se_mkt > 0:
                    _rel_ses.append(_se_mkt / abs(_b_mkt))
            if _rel_ses and max(_rel_ses) > 0.5 and _reg_T < self.trading_days:
                _beta_reliability = "noisy_short_window"
                logger.warning(
                    "Беты статистически шумные: max SE(β_Market)/|β| = %.2f "
                    "на окне %d торг. дней (< %d).",
                    max(_rel_ses), _reg_T, self.trading_days)
        except Exception:
            _beta_reliability = "ok"

        # Sharpe / Sortino with currency-matched, geometrically-compounded RFR
        # (H1+H3).  ann_return is geometric simple-return, RFR is annual
        # simple-return — both already in the same units (no fractional/log mix).
        sharpe  = ((ann_return - self.current_rfr_annual) / port_volatility
                   if port_volatility > 0 else np.nan)
        sortino = ((ann_return - self.current_rfr_annual) / downside_vol
                   if downside_vol > 0 else np.nan)

        portfolio_metrics = {
            "Total_Volatility_Ann":  port_volatility,
            "Annualised_Return":     ann_return,            # H1: geometric simple return
            "Sharpe_Ratio":          sharpe,
            "Sortino_Ratio":         sortino,
            "VaR_95_Daily":          var_95,
            "CVaR_95_Daily":         cvar_95,
            "CVaR_95_Bootstrap":     cvar_boot,
            "Max_Drawdown":          max_drawdown,
            "Max_Euler_Risk_Pct":    max_erc,
            "Composite_Risk_Score":  composite_risk,
            "Positive_Days_Pct":     (port_returns_daily > 0).mean() * 100 if len(port_returns_daily) > 0 else 0,
            # F-22: how many trading days the REALIZED panel (Sharpe numerator,
            # Annualised_Return, VaR/CVaR, MaxDD) was computed on — composite
            # window when it extends the intersection, else the intersection.
            "realized_window_days":  int(len(port_returns_daily)),
            # P-3 (audit A2/D2): explicit reliability verdict for the
            # historical VaR/CVaR quantiles — ~100 obs is the floor for a
            # stable 95% tail; below it the numbers print WITH the flag.
            "var_reliability":       ("ok"
                                      if len(port_returns_daily) >= MIN_RELIABLE_TAIL_OBS
                                      else "insufficient_history"),
            # P-5 (audit A3): χ²-ДИ множители на выборочную σ реализованного
            # окна — «σ известна с точностью ×[lo, hi]».  None на окнах < 2.
            "volatility_ci":         (
                {"lo_mult": round(_vci[0], 3), "hi_mult": round(_vci[1], 3),
                 "window_days": int(len(port_returns_daily)),
                 "confidence": 0.95}
                if (_vci := _sigma_ci_multiplier(int(len(port_returns_daily))))
                else None),
            # P-4 (audit B1): SE(β) per asset/factor + сводный флаг
            # надёжности: max относительная SE по Market-оси > 0.5 на окне
            # короче года ⇒ беты статистически шумные.
            "beta_standard_errors":  beta_se_map,
            "beta_reliability":      _beta_reliability,
            "beta_shrinkage_applied": bool(_shrink_on),
            # H2 audit trail — surfaced to the report's QC panel.
            "reporting_currency":    self.reporting_currency.value,
            "risk_free_rate_annual": self.current_rfr_annual,
            "risk_free_rate_daily":  self.current_rfr_daily,
            "risk_free_rate_source": self.rfr_source,
            "fx_conversion": [
                {"pair": r.pair, "coverage_pct": r.coverage_pct,
                 "last": r.last_value, "fallback_used": r.fallback_used}
                for r in self._last_fx_records
            ],
            # H4 disclaimer is always present — even on portfolios where
            # no scenario hits the convex cap, the user sees the policy.
            "stress_test_disclaimer": _STRESS_TEST_DISCLAIMER,
            # F-4 (2026-07-10): the Sharpe/Sortino estimator mixes horizons BY
            # DESIGN — numerator = realised geometric return over the FULL
            # lookback window (equal-weighted), denominator = STRUCTURAL vol
            # from the EWMA(hl=63)⊕Ledoit-Wolf factor covariance (weighted
            # toward the last ~3 months).  A calm recent regime therefore
            # reads HIGHER than a classical sample-Sharpe and vice versa.
            # Surfaced to the QC/Integrity panel so the basis is auditable.
            "sharpe_basis_note": (
                "Sharpe/Sortino: числитель — реализованная геометрическая "
                "доходность за всё окно наблюдений (композитная серия: дни "
                "считаются по именам, реально торговавшимся, веса "
                "ренормализуются по-дневно); знаменатель — структурная "
                "волатильность EWMA(hl=63)⊕Ledoit-Wolf, взвешенная к последним "
                "~3 месяцам. В спокойном свежем рынке оценка выше классической "
                "sample-Sharpe, после недавнего стресса — ниже."
            ),
            # BLOCK 4.6: factor-multicollinearity diagnostic (κ + max|corr|),
            # surfaced to the CoVe panel so collinear factors are visible.
            "factor_diagnostics": getattr(self, "_last_factor_diagnostic", {}),
            # Additive layer (finance/factor_decomposition): дисперсия по
            # ИСТОЧНИКАМ риска + факторные двойники; {} когда не определена.
            "factor_decomposition": getattr(self, "_last_factor_decomposition", {}),
        }

        return cov_df, pd.DataFrame(exposures_report).T, portfolio_metrics

    def fit_factor_betas(self, data, target_ticker: str) -> "dict[str, float] | None":
        """
        Факторные беты ПРОИЗВОЛЬНОГО тикера (B1 2026-07-17: мандатный бенчмарк
        в секции «Факторное разложение»).

        Прогоняет дневные лог-доходности `target_ticker` через ТОТ ЖЕ
        Ridge-пайплайн, что и активы портфеля в `calculate_structural_risk`:
        общая факторная панель `factor_tickers`, та же иерархическая
        ортогонализация (BLOCK 3.5, gated `FACTOR_ORTHOGONALIZE`), тот же
        guard на недоопределённую регрессию (≥ max(2K, 30) наблюдений) и тот
        же `Ridge(alpha=0.001)` — иначе беты бенчмарка несопоставимы с бетами
        активов в таблице.

        Возвращает {factor_name: beta} или None (нет данных / короткое
        перекрытие истории) — потребители обязаны переживать None (фолбэк на
        S&P-константу в `tg_bot._build_factor_betas_table`).  Чистая
        read-only функция: модель Σ = B·F·Bᵀ + D и её downstream-потребители
        (стресс, BL, Euler, κ) НЕ затрагиваются (ADR Вариант A).
        """
        try:
            if data is None or getattr(data, "empty", True):
                return None
            if target_ticker not in data.columns:
                return None
            factor_cols = [v for v in self.factor_tickers.values()
                           if v in data.columns]
            if not factor_cols:
                return None
            cols = list(dict.fromkeys([*factor_cols, target_ticker]))
            sub = data[cols].dropna(axis=1, how="all")
            if target_ticker not in sub.columns:
                return None
            returns = np.log(sub / sub.shift(1)).dropna()
            present = {k: v for k, v in self.factor_tickers.items()
                       if v in returns.columns}
            if not present or target_ticker not in returns.columns:
                return None
            f_data = returns[list(present.values())].copy()
            f_data.columns = list(present.keys())
            if factor_orthogonalize_enabled():
                f_data = orthogonalize_factors_hierarchical(f_data)
            min_obs = max(2 * len(present), 30)
            if len(returns) < min_obs:
                logger.info(
                    "Беты бенчмарка %s пропущены: %d наблюдений < %d "
                    "(короткое перекрытие истории).",
                    target_ticker, len(returns), min_obs)
                return None
            y = returns[target_ticker]
            model = Ridge(alpha=0.001, fit_intercept=True).fit(f_data, y)
            return {str(k): float(b)
                    for k, b in zip(f_data.columns, model.coef_)}
        except Exception as exc:                      # noqa: BLE001
            logger.warning("fit_factor_betas(%s) failed: %s",
                           target_ticker, exc)
            return None

    def calculate_atr(self, data, tickers, ohlc_data=None, period=14):
        """
        True ATR (Wilder’s Average True Range) with OHLC when available.

        True Range = max(H-L, |H-Cp|, |L-Cp|) where Cp is previous Close.
        Falls back to Close-only approximation (|ΔClose|) when OHLC is unavailable.

        Args:
            data: Close-price DataFrame (columns = resolved tickers)
            tickers: list of original (unresolved) tickers
            ohlc_data: optional dict {resolved_ticker: DataFrame[Open,High,Low,Close]}
            period: ATR lookback window (default: 14 days, Wilder standard)
        """
        resolved = self.resolve_tickers(tickers)
        atr_results = {}

        for orig, res in zip(tickers, resolved):
            if res not in data.columns:
                continue

            # Prefer True ATR from OHLC if available
            if ohlc_data and res in ohlc_data:
                ohlc = ohlc_data[res]
                if all(c in ohlc.columns for c in ('High', 'Low', 'Close')):
                    high = ohlc['High']
                    low = ohlc['Low']
                    close_prev = ohlc['Close'].shift(1)
                    tr = pd.concat([
                        high - low,
                        (high - close_prev).abs(),
                        (low - close_prev).abs(),
                    ], axis=1).max(axis=1)
                    # Wilder's RMA (α = 1/period). NOT pandas span (which gives
                    # α = 2/(span+1) — that's EMA, ~2× faster than Wilder, and
                    # would under-smooth True Range. SEC 34-105226 / Wilder 1978.
                    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
                    last_close = ohlc['Close'].iloc[-1]
                    atr_pct = (atr / last_close) * 100 if last_close > 0 else 0
                    atr_results[orig] = {'ATR_Absolute': atr, 'ATR_Pct': atr_pct}
                    continue

            # Fallback: Close-only ATR approximation (Wilder RMA on |ΔClose|)
            prices = data[res]
            daily_range = prices.diff().abs()
            atr = daily_range.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
            atr_pct = (atr / prices.iloc[-1]) * 100 if prices.iloc[-1] > 0 else 0
            atr_results[orig] = {'ATR_Absolute': atr, 'ATR_Pct': atr_pct}

        return pd.DataFrame(atr_results).T if atr_results else pd.DataFrame()



@dataclass
class MarketDataPreview:
    """Serialisable summary of the Step-1 market-data load (H-3 facade).

    Lets the Telegram layer render its progress panel without reaching into
    MAC3RiskEngine private config (NON_RISK_ASSETS / factor_tickers /
    BENCHMARK_EXTRA / resolve_tickers / get_market_data).
    """
    data: pd.DataFrame                       # full price matrix incl. factors
    history_result: object                   # HistoryResult (retried/failed/…)
    risky_tickers: list                      # portfolio tickers minus cash
    resolved_portfolio: list                 # resolved + de-duplicated
    internal_tickers: set                    # factor ETFs + benchmark infra
    loaded_count: int                        # non-all-NaN columns loaded
    portfolio_loaded: int                    # portfolio tickers with data
    portfolio_total: int                     # resolved portfolio size
    proxy_map: dict = field(default_factory=dict)   # original → proxy ticker


class UniversalPortfolioManager:
    COLUMN_ALIASES = {
        'Ticker': ['Symbol', 'Asset', 'Тикер', 'Name'],
        'Quantity': ['Qty', 'Amount', 'Shares', 'Кол-во'],
        'Purchase_Price': ['Buy_Price', 'Price_Buy', 'Entry_Price', 'Цена_Покупки', 'Avg_Price']
    }

    def __init__(self):
        self.engine = MAC3RiskEngine()

    def prefetch_market_data(self, candidate_tickers) -> "MarketDataPreview":
        """Public facade for the bot's Step-1 progress panel (H-3).

        Encapsulates every engine-internal access the Telegram layer used to
        reach for directly, returning a ready-to-render summary.  The bot must
        call THIS instead of poking `manager.engine.*`.
        """
        eng = self.engine
        risky_tickers = [t for t in (candidate_tickers or [])
                         if str(t).upper() not in eng.NON_RISK_ASSETS]
        data, history_result = eng.get_market_data(risky_tickers)

        loaded_count = (
            len([c for c in data.columns if not data[c].isna().all()])
            if not data.empty else 0
        )
        internal_tickers = set(list(eng.factor_tickers.values()) + eng.BENCHMARK_EXTRA)
        resolved_portfolio = list(dict.fromkeys(eng.resolve_tickers(risky_tickers)))
        portfolio_loaded = len([
            t for t in resolved_portfolio
            if t in data.columns and not data[t].isna().all()
        ])
        # Proxy replacements (original → resolved) when the engine remapped a
        # raw ticker to a tradable proxy.
        proxy_map: dict = {}
        for t in risky_tickers:
            resolved = eng.resolve_tickers([t])
            if resolved and resolved[0] != f"{str(t).upper()}.US" and resolved[0] != str(t).upper():
                proxy_map[t] = resolved[0]

        return MarketDataPreview(
            data               = data,
            history_result     = history_result,
            risky_tickers      = risky_tickers,
            resolved_portfolio = resolved_portfolio,
            internal_tickers   = internal_tickers,
            loaded_count       = loaded_count,
            portfolio_loaded   = portfolio_loaded,
            portfolio_total    = len(resolved_portfolio),
            proxy_map          = proxy_map,
        )

    def _standardize_columns(self, df):
        for standard, aliases in self.COLUMN_ALIASES.items():
            actual_col = next((c for c in df.columns if c in aliases or c == standard), None)
            if actual_col: df = df.rename(columns={actual_col: standard})
        return df

    def analyze_all(self, source, scenario_shocks=None, profile_benchmark: str | None = None,
                    risk_mandate: str | None = None):
        """
        profile_benchmark: Tradernet ETF ticker for the user's mandate benchmark
        (e.g. 'AGG.US' for Conservative, 'SPY.US' for Moderate, 'QQQ.US' for Aggressive).
        When provided it is computed first and labelled 'Профильный бенчмарк' in results.

        risk_mandate: CONSERVATIVE / MODERATE / AGGRESSIVE (or an RU/EN profile
        name / numeric score) — drives the composite-risk-score calibration
        (H4).  When None, the engine keeps its current mandate (MODERATE
        default).
        """
        if risk_mandate is not None:
            from finance.scoring import normalize_risk_mandate as _nrm
            self.engine.risk_mandate = _nrm(risk_mandate)
        if isinstance(source, pd.DataFrame): raw_df = source
        else: raise ValueError("Неверный источник данных.")

        # ── MAC3 GATE: portfolio source verification ──────────────────────────
        # Detect the three possible DataFrame origins stamped by FreedomConnector:
        #   _ramp_is_mock=True, _ramp_is_fallback=True  → broker API failed, fallback mock
        #   _ramp_is_mock=True, _ramp_source='demo'     → user explicitly chose template/demo
        #   (no attrs)                                  → live broker data
        _is_fallback = raw_df.attrs.get('_ramp_is_fallback', False)
        _is_mock     = raw_df.attrs.get('_ramp_is_mock',     False)

        # ── EMPTY-PORTFOLIO GUARD ────────────────────────────────────────────
        # 0-row frame → `total = df['Current_Value'].sum()` = 0 →
        # `df['Current_Value']/total` = all-NaN → Ridge fit + bootstrap CVaR
        # explode with cryptic errors.  Bail out with a user-facing message
        # the existing handler (`RealPortfolioRequired` is caught in
        # tg_bot._run_analysis_background) already understands.
        if raw_df is None or raw_df.empty:
            logger.warning("MAC3 ENGINE GATE ▸ empty portfolio (0 rows) — halting.")
            raise RealPortfolioRequired(
                "У вас нет открытых позиций для анализа.\n\n"
                "Откройте хотя бы одну позицию у брокера или выберите "
                "«Демо-режим» в настройках подключения."
            )

        if _is_fallback:
            logger.error(
                "MAC3 ENGINE GATE ▸ FALLBACK mock portfolio detected "
                "(broker API returned no live data). "
                "Halting immediately — skipping market data fetch and all MAC3 calculations."
            )
            raise RealPortfolioRequired(
                "Не удалось получить данные портфеля от Freedom Broker. "
                "Проверьте ваши API-ключи и попробуйте снова.\n\n"
                "Если хотите запустить анализ на шаблонном портфеле, "
                "выберите «Демо-режим» в настройках подключения."
            )

        if _is_mock:
            logger.info(
                "MAC3 ENGINE ▸ Portfolio source: DEMO (template). "
                "Running analysis on synthetic mock positions."
            )
        else:
            logger.info(
                "MAC3 ENGINE ▸ Portfolio source: REAL (live Freedom Broker data). "
                "Initiating full MAC3 Risk Engine pipeline."
            )
        # ─────────────────────────────────────────────────────────────────────

        df = self._standardize_columns(raw_df).set_index('Ticker')

        # Extract broker-provided current prices before any further processing.
        # Used as fallback for instruments with no Tradernet history
        # (KZ bonds with ISIN tickers, cash balances from Freedom API "acc").
        broker_current_prices: dict = {}
        if "Broker_Current_Price" in df.columns:
            broker_current_prices = df["Broker_Current_Price"].dropna().to_dict()
            df = df.drop(columns=["Broker_Current_Price"])

        # Скачиваем цены на Рисковые активы
        risky_tickers = [t for t in df.index if t not in self.engine.NON_RISK_ASSETS]

        all_data, history_result = self.engine.get_market_data(risky_tickers)
        # Фундаментальные метрики берутся ТОЛЬКО из SEC EDGAR (см. ниже,
        # batch_fundamental_scan). SEC EDGAR покрывает ROE/Debt/Margins/Growth
        # точнее и без зависимости от стороннего API (10-K/10-Q филинги).

        # Установка Current Price.
        # Приоритет:
        #   (1) NON_RISK_ASSETS (кэш) → 1.0
        #   (2) Tradernet или proxy-ETF (для облигаций через BOND_PROXIES)
        #   (3) Цена брокера (Freedom API mkt_price) — fallback для КЗ облигаций без данных
        #   (4) Purchase_Price — крайний fallback для облигаций с известным паттерном имени
        #
        # ВАЖНО: resolve_tickers() пропускает NON_RISK_ASSETS → список короче df.index.
        # Используем per-ticker резолюцию вместо zip, чтобы избежать сдвига индексов.
        current_prices = {}
        broker_priced_only: set = set()  # тикеры, оцениваемые только через брокера (без фактор-модели)
        for orig in df.index:
            orig_str = str(orig).upper().strip()
            if orig_str in self.engine.NON_RISK_ASSETS:
                current_prices[orig] = 1.0
                continue
            resolved_list = self.engine.resolve_tickers([orig])
            res = resolved_list[0] if resolved_list else orig_str
            if res in all_data.columns:
                # Включает облигации, разрешённые в proxy-ETF (BIL, LQD, HYG)
                current_prices[orig] = all_data[res].iloc[-1]
            elif orig in broker_current_prices:
                # KZ облигации с ISIN-тикером: используем mkt_price от Freedom API
                current_prices[orig] = broker_current_prices[orig]
                broker_priced_only.add(orig)
                logger.info(
                    "Используется цена брокера для %s = %.4f (нет данных Tradernet)",
                    orig, broker_current_prices[orig],
                )
            elif orig_str in self.engine.BOND_CLASSIFICATION_MAP or 'BOND' in orig_str or 'OVD' in orig_str:
                # Облигация без брокерской цены — удержание до погашения по цене покупки
                p_price = df.loc[orig, 'Purchase_Price']
                current_prices[orig] = p_price if pd.notna(p_price) else 100.0
                broker_priced_only.add(orig)

        df['Current_Price'] = pd.Series(current_prices)
        df = df.dropna(subset=['Current_Price'])

        # Стоимости и Веса
        df['Total_Cost'] = df['Quantity'] * df['Purchase_Price']
        df['Current_Value'] = df['Quantity'] * df['Current_Price']
        df['PnL'] = df['Current_Value'] - df['Total_Cost']
        # BLK-2: guard against Purchase_Price == 0 (gifted shares, corporate
        # actions, broker data gaps).  A bare PnL/0 yields ±inf — and inf is
        # NOT caught by a downstream .fillna(0) — so one zero-cost row would
        # corrupt the whole portfolio return.  Replace the 0 denominator with
        # NaN first, then fill the resulting NaN with 0.0 (flat return).
        df['Return_Pct'] = df['PnL'].div(df['Total_Cost'].replace(0, np.nan)).fillna(0.0)

        total_portfolio_value = float(df['Current_Value'].sum())
        # Second guard: even with rows, if every Current_Price is 0 or all
        # rows are NON_RISK_ASSETS with zero balance, total is 0.  Don't
        # divide — propagate the same user-friendly error.
        if not np.isfinite(total_portfolio_value) or total_portfolio_value <= 0:
            logger.warning("MAC3 ENGINE ▸ total portfolio value is 0 or NaN — halting.")
            raise RealPortfolioRequired(
                "Стоимость портфеля = 0. Проверьте подключение к брокеру "
                "или откройте позиции — анализ невозможен на пустом счёте."
            )
        weights_dict = (df['Current_Value'] / total_portfolio_value).to_dict()

        # MAC3 Structural Risk.
        # Исключены из факторной модели:
        #   • NON_RISK_ASSETS (кэш) — нулевая волатильность по определению
        #   • broker_priced_only — КЗ облигации без ценовой истории в Tradernet;
        #     их веса в weights_dict корректно разбавляют риск акций.
        # Облигации, разрешённые в proxy-ETF (BIL/LQD/HYG), ОСТАЮТСЯ в actual_risky —
        # у них есть ценовая история и они корректно обрабатываются Rates-фактором.
        actual_risky = [
            t for t in df.index
            if t not in self.engine.NON_RISK_ASSETS and t not in broker_priced_only
        ]
        cov_matrix, factor_df, port_metrics = self.engine.calculate_structural_risk(all_data, actual_risky, weights_dict)
        
        # ATR (Intraday Margin Standard — SEC 34-105226)
        # Uses True ATR (OHLC) when available, falls back to Close-only
        atr_df = self.engine.calculate_atr(
            all_data, actual_risky,
            ohlc_data=getattr(history_result, 'ohlc_data', None),
        )
        if not atr_df.empty:
            df = df.join(atr_df)

        # SEC EDGAR Fundamental Scores (sector-normalized)
        # Build sector_map from MAC3RiskEngine.TICKER_SECTOR for sector-aware thresholds
        sector_map = {t: self.engine.get_ticker_sector(t) for t in actual_risky}
        try:
            from finance.sec_edgar import batch_fundamental_scan
            sec_df = batch_fundamental_scan(actual_risky, sector_map=sector_map)
            if not sec_df.empty:
                df = df.join(sec_df)
        except Exception as e:
            logger.warning(f"SEC EDGAR scan пропущен: {e}")
        
        # Джойним факторы к общему DF
        df = df.join(factor_df)
        
        # Для кэша выставляем волатильность 0
        if 'Residual_Vol_Ann' in df.columns:
            df['Residual_Vol_Ann'] = df['Residual_Vol_Ann'].fillna(0)
            df['Euler_Risk_Contribution_Pct'] = df['Euler_Risk_Contribution_Pct'].fillna(0)

        # Ожидаемая доходность: annualised CAPM-style return from factor betas.
        #
        # Correct approach: betas are from DAILY log-return regression, so
        # expected return must be built from the SAME daily timescale and then
        # annualised — NOT from the cumulative period log-return (which mixes
        # timescales and produces nonsensical "expected" values).
        #
        # Algorithm:
        #   1. Compute mean daily log-return for each factor ETF.
        #   2. GEOMETRIC annualisation: E[r_ann] = exp(mean_daily_log * 252) - 1
        #      This avoids the arithmetic overestimate (arithmetic: mean*252).
        #   3. Map ETF tickers → factor key names (Beta_Market ← Market ← SPY.US).
        #   4. Expected return = Σ beta_k * E[r_k_ann] — β·μ ONLY.  The Ridge
        #      intercept (realised idiosyncratic drift) is EXCLUDED from the
        #      forward forecast (F-14); it stays visible via Alpha_Specific.
        #   5. Specific Alpha = actual return - expected return (in %).
        # P-1/P-7: σ_ETP (daily, ddof=1) for each registry leveraged-ETP name,
        # from its OWN price history — shared by the forward-drag adjustment
        # and the path-dependent stress branch.  ≥60 obs floor (sparse-guard
        # parity); thinner names simply stay absent from the map.
        letf_sigma_map: dict[str, float] = {}
        try:
            for _t in df.index:
                if not _is_leveraged_etp(_t):
                    continue
                _res_l = self.engine.resolve_tickers([_t])
                _col = _res_l[0] if _res_l else None
                if _col and _col in all_data.columns:
                    _pr = all_data[_col].dropna()
                    if len(_pr) >= _MIN_OVERLAP_TDAYS:
                        _lr = np.log(_pr / _pr.shift(1)).dropna()
                        _sd = float(_lr.std(ddof=1))
                        if np.isfinite(_sd) and _sd > 0:
                            letf_sigma_map[str(_t)] = _sd
        except Exception as _exc:
            logger.warning("LETF sigma map skipped: %s", _exc)
            letf_sigma_map = {}

        present_factor_keys = [
            k for k, v in self.engine.factor_tickers.items() if v in all_data.columns
        ]
        present_factor_etfs = [self.engine.factor_tickers[k] for k in present_factor_keys]
        f_data = all_data[present_factor_etfs] if present_factor_etfs else pd.DataFrame()

        if not f_data.empty:
            factor_daily_log = np.log(f_data / f_data.shift(1)).dropna()
            factor_mean_daily = factor_daily_log.mean()       # index = ETF tickers
            # Keep the per-factor expectations in DAILY LOG space — the single
            # geometric annualisation happens once, on the combined per-asset
            # expected log return below (F-2 fix).
            factor_mu_daily_by_key = pd.Series({
                k: factor_mean_daily.get(v, 0.0)
                for k, v in zip(present_factor_keys, present_factor_etfs)
            })
        else:
            factor_mu_daily_by_key = pd.Series(dtype=float)

        beta_cols = [c for c in df.columns if str(c).startswith('Beta_')]
        if beta_cols and not factor_mu_daily_by_key.empty:
            beta_factor_keys = [c[len('Beta_'):] for c in beta_cols]
            mu_vec = factor_mu_daily_by_key.reindex(beta_factor_keys).fillna(0.0).values
            # F-14 (2026-07-10, пост-релизный фикс «218.4%»): the FORWARD
            # expectation is now β·μ ONLY — the Ridge intercept (realised
            # idiosyncratic drift) is EXCLUDED from the forecast.  Institutional
            # convention: alpha is not a persistent premium, and annualising it
            # geometrically explodes on short regression windows — after F-6
            # removed the bfill look-ahead, the common window honestly shrinks
            # to the youngest listing, and a leveraged single-stock ETF's bull
            # streak extrapolated to «ожид. дох. 218.4% · Sharpe 11.29» on the
            # live cover.  β are bounded by Ridge and μ_f is estimated on the
            # FULL factor history (factor ETFs always span the whole lookback),
            # so the β·μ expectation stays economically sane.  The realised
            # drift remains visible через Alpha_Specific below.
            factor_log_daily = df[beta_cols].fillna(0).values @ mu_vec
            expected_log_daily = factor_log_daily
            # F-23/P-1: leveraged daily-reset ETPs — lower the forward by the
            # CONTRACTUAL variance drag −½L(L−1)σ_u² + fee −ER/252 when the
            # multiplier L is in the registry (finance/leveraged.py), else by
            # the empirical min(α̂,0) fallback.  Ordinary names untouched.
            if 'Specific_Alpha_Daily' in df.columns:
                try:
                    # letf_sigma_map построена выше (≥60 obs floor); имена
                    # тоньше порога падают в α̂-фолбэк (α̂=0 → no-op,
                    # бит-в-бит пре-P-1 поведение).
                    expected_log_daily, _lev_details = apply_leveraged_forward(
                        expected_log_daily, list(df.index),
                        df['Specific_Alpha_Daily'], letf_sigma_map)
                    if _lev_details and isinstance(port_metrics, dict):
                        port_metrics["leveraged_adjustments"] = _lev_details
                        # Back-compat key (pre-P-1 consumers/tests).
                        port_metrics["leveraged_drag_tickers"] = \
                            list(_lev_details.keys())
                except Exception as _exc:
                    logger.warning("Leveraged-ETP drag skipped: %s", _exc)
            df['Expected_Return'] = np.exp(
                expected_log_daily * self.engine.trading_days) - 1.0
            # Belt-and-suspenders: clamp the per-asset FORWARD expectation to a
            # publishable band — a runaway β on a noisy short window must not
            # print a four-digit forecast.  Clamped names are counted for QC.
            _ER_LO, _ER_HI = -0.50, 1.00
            _clamped_n = int(((df['Expected_Return'] < _ER_LO)
                              | (df['Expected_Return'] > _ER_HI)).sum())
            if _clamped_n:
                logger.warning(
                    "Expected_Return clamped to [%.0f%%, %.0f%%] for %d asset(s).",
                    _ER_LO * 100, _ER_HI * 100, _clamped_n)
            df['Expected_Return'] = df['Expected_Return'].clip(_ER_LO, _ER_HI)
            if isinstance(port_metrics, dict):
                port_metrics["expected_return_clamped_n"] = _clamped_n
            # Specific Alpha: actual holding-period return minus the
            # FACTOR-IMPLIED return (in %) — i.e. the realised excess the
            # factor model does NOT explain.  This is exactly the textbook
            # «specific alpha» reading; the forecast column above stays clean.
            df['Alpha_Specific'] = (df['Return_Pct'] - df['Expected_Return']) * 100

        # ── BLOCK 5: portfolio-level FORWARD expected annual return ──────────
        # Aggregate the per-asset CAPM expectations into a single portfolio
        # number the UI can show against risk:
        #     E[r_port] = Σ_i w_i · E[r_i]   +   w_cash · r_f
        # The risky weights generally sum to <1 (cash/bonds sit outside the
        # factor model); the un-invested residual earns the risk-free rate, so
        # cash drag is priced in rather than silently dropped.  This is distinct
        # from `Annualised_Return` (the REALISED trailing CAGR) — it is the
        # forward expectation implied by today's factor betas + factor premia.
        # The result is fed back into the SAME port_metrics dict the structural
        # model returned, so every downstream surface reads one figure.
        if 'Expected_Return' in df.columns and isinstance(port_metrics, dict):
            try:
                # F-14: gate the ANNUALISED forward panel on the regression
                # window.  With the F-6 look-ahead fix the common window is the
                # honest intersection of listings — when it is shorter than one
                # trading year, an annualised forward claim off it is noise
                # (the live «218.4% · Sharpe 11.29» cover bug).  The per-asset
                # Expected_Return column stays (bounded β·μ), only the
                # portfolio-level panel degrades to «—».
                _nobs = int(getattr(self.engine, "_last_regression_nobs", 0) or 0)
                port_metrics["expected_return_window_days"] = _nobs
                if _nobs < self.engine.trading_days:
                    logger.warning(
                        "Forward expected-return panel skipped: regression "
                        "window %d < %d trading days.",
                        _nobs, self.engine.trading_days)
                    raise StopIteration          # → graceful skip below
                _exp = df['Expected_Return']
                _w_risky = 0.0
                _er_weighted = 0.0
                for _t in df.index:
                    _w = float(weights_dict.get(_t, 0.0) or 0.0)
                    _er = _exp.get(_t)
                    if _er is None or not np.isfinite(float(_er)):
                        continue
                    _w_risky += _w
                    _er_weighted += _w * float(_er)
                _rfr = float(port_metrics.get("risk_free_rate_annual") or 0.0)
                _cash_w = max(0.0, 1.0 - _w_risky)
                _port_exp = _er_weighted + _cash_w * _rfr
                _vol = float(port_metrics.get("Total_Volatility_Ann") or 0.0)
                port_metrics["Expected_Return_Annual"] = _port_exp
                # Forward (ex-ante) Sharpe — expected excess return per unit of
                # structural vol.  Complements the realised Sharpe_Ratio above.
                port_metrics["Expected_Sharpe"] = (
                    (_port_exp - _rfr) / _vol if _vol > 0 else None)
                port_metrics["Expected_Return_Cash_Weight"] = round(_cash_w, 4)
                port_metrics["Expected_Return_Invested_Weight"] = round(_w_risky, 4)
            except StopIteration:
                pass                             # sub-year window — panel «—»
            except Exception as _exc:
                logger.warning("portfolio expected-return aggregation skipped: %s", _exc)

        # ═══════════════ BENCHMARK COMPARISON ═══════════════
        # Профильный бенчмарк ставится первым для корректного расчёта Tracking Error.
        benchmarks: dict[str, str] = {}
        if profile_benchmark:
            benchmarks["Профильный бенчмарк"] = profile_benchmark

        # Бенчмарки в формате Tradernet (.US ETF-прокси для индексов).
        # EEM.US и EMB.US уже запрошены как EM-факторы → данные гарантированно есть.
        benchmarks.update({
            'S&P 500':      'SPY.US',   # ETF-прокси на ^GSPC
            'Nasdaq 100':   'QQQ.US',   # ETF-прокси на ^NDX
            'Russell 2000': 'IWM.US',   # ETF-прокси на ^RUT
            'MSCI EM':      'EEM.US',   # Emerging Markets equity index — EM/KASE сравнение
            'EM Bonds':     'EMB.US',   # JP Morgan EM Bond index — для KZ bond holders
        })
        # ── Weighted portfolio log-return series — built ONCE, robustly ────
        # Sparse-history constituents (thinly-traded local listings) are
        # dropped from the panel and the surviving weights renormalised, so
        # one bad name can no longer collapse the overlap window for the
        # whole book.  See period_returns.build_portfolio_log_returns.
        port_resolved = self.engine.resolve_tickers(actual_risky)
        col_weights: dict[str, float] = {}
        for orig, res in zip(actual_risky, port_resolved):
            if res in all_data.columns:
                col_weights[res] = col_weights.get(res, 0.0) + float(weights_dict.get(orig, 0.0))
        port_log_series, return_coverage = _build_portfolio_log_returns(
            all_data[list(col_weights)] if col_weights else None, col_weights
        )
        if return_coverage["dropped"]:
            logger.info("Return series: dropped %d sparse-history name(s): %s "
                        "(covered weight %.1f%%)",
                        len(return_coverage["dropped"]),
                        ", ".join(return_coverage["dropped"]),
                        return_coverage["covered_weight"] * 100)

        # Benchmark log-returns — one Series per benchmark (per-pair join).
        bm_logs: dict[str, pd.Series] = {}
        for bm_name, bm_ticker in benchmarks.items():
            if bm_ticker in all_data.columns:
                bm_prices = all_data[bm_ticker].dropna()
                if len(bm_prices) >= 2:
                    bm_logs[bm_name] = np.log(bm_prices / bm_prices.shift(1)).dropna()

        # ═══════════════ BENCHMARK FACTOR PROFILE (B1 2026-07-17) ═══════════════
        # Реальные факторные беты выбранного бенчмарка (тот же Ridge-пайплайн,
        # что и активы) — столбец «бенчмарк» в секции «Факторное разложение»
        # больше не константа (Market=1, rest=0)=S&P 500.  Для SPY.US беты
        # ≈ (Market≈1, стили≈0) → отчёт обратносовместим по построению.
        # None (нет бенчмарка / нет истории) — потребители обязаны переживать.
        benchmark_factor_profile: dict | None = None
        if profile_benchmark:
            _bm_betas = self.engine.fit_factor_betas(all_data, profile_benchmark)
            if _bm_betas:
                try:
                    from profile_manager import BENCHMARK_LIST as _BM_NAMES
                except Exception:                     # pragma: no cover
                    _BM_NAMES = {}
                benchmark_factor_profile = {
                    "ticker": profile_benchmark,
                    "name":   _BM_NAMES.get(profile_benchmark, profile_benchmark),
                    "betas":  _bm_betas,
                }
            else:
                logger.info(
                    "Benchmark factor profile: беты для %s недоступны — секция "
                    "факторов использует S&P-константу (graceful fallback).",
                    profile_benchmark)

        # ═══════════════ BENCHMARK COMPARISON ═══════════════
        # TE / IR / annualised excess are computed per benchmark on a PAIR
        # inner-join (compute_benchmark_stats) — a benchmark's own short
        # history can no longer null the comparison, and numerator and
        # denominator share one aligned window (no IR scale bug).
        benchmark_results: dict[str, dict] = {}
        port_return = float(df['Return_Pct'].fillna(0).values @ np.array(
            [weights_dict.get(t, 0) for t in df.index]))
        for bm_name, bm_ticker in benchmarks.items():
            if bm_ticker not in all_data.columns:
                continue
            bm_prices = all_data[bm_ticker].dropna()
            if len(bm_prices) <= 1:
                continue
            bm_return = float(bm_prices.iloc[-1] / bm_prices.iloc[0]) - 1.0
            stats = _compute_benchmark_stats(port_log_series, bm_logs.get(bm_name),
                                             trading_days=self.engine.trading_days)
            benchmark_results[bm_name] = {
                "Benchmark_Return":     bm_return,                  # period total (display)
                "Portfolio_Return":     port_return,                # period total (display)
                "Excess_Return":        port_return - bm_return,    # period total (display)
                "Portfolio_Ann_Return": stats["port_ann_return"]   if stats else None,
                "Benchmark_Ann_Return": stats["bm_ann_return"]     if stats else None,
                "Excess_Return_Ann":    stats["excess_ann"]        if stats else None,
                "Tracking_Error":       stats["tracking_error"]    if stats else None,
                "Information_Ratio":    stats["information_ratio"] if stats else None,
                "Beating_Benchmark":    port_return > bm_return,
            }

        # ═══════════════ MULTI-PERIOD RETURNS TABLE ═══════════════
        # Reuses the SAME robust port_log_series + per-benchmark bm_logs, so
        # every period row shares one aligned window with the TE/IR figures.
        period_returns_table: dict[str, dict] = {}
        try:
            if port_log_series is not None and bm_logs:
                period_returns_table = _compute_period_returns_table(
                    port_log_series, bm_logs
                )
        except Exception as exc:  # never block the rest of the pipeline
            logger.warning("period_returns_table build failed: %s", exc)
            period_returns_table = {}

        # ═══════════════ STRESS SCENARIOS ═══════════════
        # Parametric factor-shock scenarios using the betas just fitted.  The
        # default catalog ships 7 scenarios (5 direct, 2 proxy).  Result is a
        # list[dict] — always present, possibly empty when perf_df is too
        # thin to support any scenario.
        try:
            # Recovery rate = portfolio's own annualised return, clamped to a
            # realistic band [8%, 18%].  A growth book recovers faster than
            # the generic 8% market drift, but we cap at 18% so an
            # unsustainable trailing CAGR (e.g. +36%) cannot fabricate an
            # absurdly fast recovery.  Floor 8% = long-run market average.
            _ann_ret = float(port_metrics.get("Annualised_Return") or 0.0)
            _recovery_rate = min(0.18, max(0.08, _ann_ret))
            stress_scenarios = _run_stress_scenarios(
                perf_df      = df.reset_index(),
                total_value  = total_portfolio_value,
                port_metrics = port_metrics,
                ann_return_baseline = _recovery_rate,
                # F-1: the Beta_* columns in perf_df live in the RESIDUAL factor
                # space when orthogonalization ran; hand the fitted child→parent
                # betas to the stress engine so it maps the raw shock catalog
                # into the same space (invariant to FACTOR_ORTHOGONALIZE).
                ortho_betas  = getattr(self.engine, "_last_ortho_betas", {}) or {},
                # P-7: реестровые daily-reset ETP считаются path-dependent —
                # (1+X_u)^L·exp(−½L(L−1)σ_u²·63)−1 вместо линейного β·shock
                # с капом ±35% (кап срезал КОНТРАКТНУЮ выпуклость плеча).
                letf_sigma_daily = letf_sigma_map,
            )
        except Exception as exc:
            logger.warning("stress scenarios build failed: %s", exc)
            stress_scenarios = []

        # Sector exposure analysis
        sector_exposure = self.engine.get_sector_exposure(list(df.index), weights_dict)

        # ── Composite-risk aggravators (2026-07-18) ────────────────────────────
        # Re-score the composite gauge with the STRUCTURAL/TAIL signals the base
        # three (vol/CVaR/single-name-ERC) miss — leverage, single-sector
        # concentration and the realised max drawdown.  Live audit: a 73%-tech,
        # margin-funded book with a −43.5% historical drawdown read «48 ·
        # Умеренный».  The aggravators are bounded and can only RAISE the gauge;
        # a clean, diversified, unlevered book is unchanged.  Computed HERE
        # (sector_exposure + weights available) and passed to BOTH the verdict
        # gauge and the effect simulator so «до/после» stays consistent.
        _agg_lev_ratio = None
        try:
            _lw = sum(max(0.0, float(w)) for w in weights_dict.values())
            _nw = sum(float(w) for w in weights_dict.values())
            _agg_lev_ratio = round(_lw / _nw, 4) if _nw > 1e-9 else 1.0
        except Exception:
            _agg_lev_ratio = None
        _agg_sector_top_pct = None
        try:
            _long_secs = [float(v) for v in (sector_exposure or {}).values()
                          if float(v) > 0]
            _ssum = sum(_long_secs)
            if _ssum > 0:
                _agg_sector_top_pct = round(max(_long_secs) / _ssum * 100.0, 2)
        except Exception:
            _agg_sector_top_pct = None
        try:
            _base_vol = float(port_metrics.get("Total_Volatility_Ann") or 0.0)
            _base_cvar = float(port_metrics.get("CVaR_95_Daily") or 0.0)
            _base_erc = float(port_metrics.get("Max_Euler_Risk_Pct") or 0.0)
            _base_mdd = port_metrics.get("Max_Drawdown")
            _mandate_str = str(getattr(self.engine, "risk_mandate", "MODERATE"))
            port_metrics["Composite_Risk_Score"] = self.engine._composite_risk_score(
                _base_vol, _base_cvar, _base_erc, _mandate_str,
                max_drawdown=(float(_base_mdd) if _base_mdd is not None else None),
                leverage_ratio=_agg_lev_ratio,
                sector_top_pct=_agg_sector_top_pct)
            # Surface the aggravators for the QC/AI layer (why the gauge is
            # higher than the raw vol/CVaR would suggest).
            port_metrics["composite_aggravators"] = {
                "max_drawdown":   (float(_base_mdd) if _base_mdd is not None else None),
                "leverage_ratio": _agg_lev_ratio,
                "sector_top_pct": _agg_sector_top_pct,
            }
        except Exception as _exc:
            logger.warning("composite aggravators skipped: %s", _exc)

        # ═══════════════ FACTOR SCORES GROUPING ═══════════════
        # Group A (Style Factors): MAC3-derived betas, alpha, volatility
        # Group B (Fundamental Factors): SEC EDGAR metrics
        factor_scores = {}
        perf = df.reset_index()
        for _, row in perf.iterrows():
            ticker = row.get('Ticker', '?')
            group_a = {}
            group_b = {}
            # Style factors (Group A)
            for col in beta_cols:
                if col in row.index:
                    group_a[col] = row[col]
            for col in ['Residual_Vol_Ann', 'Euler_Risk_Contribution_Pct',
                        'Expected_Return', 'Alpha_Specific', 'Specific_Alpha_Daily']:
                if col in row.index:
                    group_a[col] = row[col]
            # Fundamental factors (Group B)
            for col in ['Fundamental_Score', 'Fundamental_Sector',
                        'SEC_Op_Margin', 'SEC_Debt_to_Assets',
                        'SEC_ROE', 'SEC_Revenue_Growth_YoY', 'SEC_Filing_Date']:
                if col in row.index:
                    group_b[col] = row[col]
            factor_scores[ticker] = {
                'group_a_style': group_a,
                'group_b_fundamental': group_b,
            }

        # ═══════════════ MACRO DRIVERS (FRED) ═══════════════
        # 6-series pack (yield curve / HY OAS / VIX / breakeven / unemployment /
        # real-GDP growth) used by the DEEP P5 regime page.  Disk-cached for
        # 12h; gracefully degrades to status="missing" when FRED_API_KEY is
        # unset and to status="stale" on transient network failures.  Fetched
        # BEFORE classification so the regime can (optionally) consume it.
        macro_drivers: dict = {}
        try:
            macro_drivers = _MacroFeed().get_regime_drivers()
        except Exception as exc:
            logger.warning("Macro drivers fetch skipped: %s", exc)
            macro_drivers = {}

        # ═══════════════ MACRO REGIME CLASSIFICATION ═══════════════
        # Reuses the factor ETF prices already in `all_data` — no extra calls.
        # BLOCK 3.4: the FRED macro pack is passed in; it is only consulted when
        # REGIME_MACRO_OVERLAY=1 (otherwise the classifier is unchanged).
        try:
            from finance.regime import RegimeClassifier
            regime_reading = RegimeClassifier().classify(all_data, macro=macro_drivers)
        except Exception as exc:
            logger.warning("Regime classification skipped: %s", exc)
            regime_reading = None

        # ═══════════════ TECHNICALS (pillar C) ═══════════════
        # Computes RSI/MACD/SMA/Bollinger/52w-Hi/Mom-12m1m per-asset.
        # Uses the same close-price frame already loaded; no new fetches.
        technicals_map: dict = {}
        try:
            from finance.technicals import compute_technicals
            tech_sector_map = {
                t: self.engine.get_ticker_sector(t) for t in actual_risky
            }
            technicals_map = compute_technicals(
                close_prices = all_data,
                tickers      = actual_risky,
                volume_frame = None,            # OHLC volumes wired in a later phase
                sector_map   = tech_sector_map,
            )
        except Exception as exc:
            logger.warning("Technicals computation skipped: %s", exc)
            technicals_map = {}

        # ═══════════════ CDS FEED (free layer) ═══════════════
        # Lazy-init.  When CDS_DISABLED=1 (e.g. unit tests) we skip the feed
        # entirely.  Failures are caught and the scoring orchestrator falls
        # back to SEC-only Credit signals.
        cds_lookup = None
        if os.getenv("CDS_DISABLED") != "1":
            try:
                from finance.cds_feed import CDSFeed, make_lookup
                cds_lookup = make_lookup(CDSFeed())
            except Exception as exc:
                logger.info("CDS feed unavailable, continuing without it: %s", exc)
                cds_lookup = None

        # Tally CDS coverage so the CoVe lineage row reflects reality
        # (instead of the silent "no per-ticker CDS attached" placeholder).
        # Single extra pass over the cache — microseconds for ≤20 tickers.
        cds_summary: dict = {"enabled": cds_lookup is not None,
                              "checked": 0, "loaded": 0, "gated_out": 0}
        if cds_lookup is not None:
            try:
                checked = list(actual_risky)
                n_loaded = sum(1 for t in checked if cds_lookup(t))
                cds_summary.update({
                    "checked":   len(checked),
                    "loaded":    n_loaded,
                    "gated_out": len(checked) - n_loaded,
                })
            except Exception as exc:
                logger.info("CDS coverage tally skipped: %s", exc)

        # ═══════════════ 4-PILLAR SCORING (F/V/T/C) ═══════════════
        # Produces AssetScore per ticker, including CDS signals when available.
        asset_scores: dict = {}
        try:
            from finance.scoring_orchestrator import score_portfolio
            asset_scores = score_portfolio(
                perf_table = perf,
                technicals = technicals_map,
                regime     = regime_reading,
                cds_lookup = cds_lookup,
                # Sprint-5.1 (S2): in production, small-sector F-pillar
                # z-scores use the LIVE SEC cohort of sector leaders before
                # falling back to the static 2020-25 constants.
                dynamic_benchmarks = os.getenv("SECTOR_COHORT_DISABLED") != "1",
            )
            # Materialise the score columns into the perf table for downstream
            # PDF rendering — this is purely additive, no existing column is
            # overwritten.
            if asset_scores:
                score_rows = []
                for t in perf["Ticker"].tolist():
                    sc = asset_scores.get(str(t))
                    if sc is None:
                        score_rows.append({
                            "Ticker": t,
                            "Score_Fundamentals": None,
                            "Score_Valuations":   None,
                            "Score_Technicals":   None,
                            "Score_Credit":       None,
                            "Score_Total":        None,
                            "Score_Action":       None,
                            "Score_Hotspot":      False,
                        })
                    else:
                        score_rows.append({
                            "Ticker": t,
                            "Score_Fundamentals": sc.fundamentals,
                            "Score_Valuations":   sc.valuations,
                            "Score_Technicals":   sc.technicals,
                            "Score_Credit":       sc.credit,
                            "Score_Total":        sc.total,
                            "Score_Action":       sc.action,
                            "Score_Hotspot":      sc.hotspot,
                        })
                score_df = pd.DataFrame(score_rows).set_index("Ticker")
                # Join back onto perf without losing existing columns.
                perf = perf.set_index("Ticker").join(score_df, how="left").reset_index()
        except Exception as exc:
            logger.warning("Scoring orchestrator skipped: %s", exc)

        # ═══════════════ BLACK-LITTERMAN TARGETS ═══════════════
        # Reverse-optimisation prior + score-derived views → posterior
        # target weights.  Skipped when there's no covariance matrix
        # (degenerate portfolio) or when all scores are zero.
        bl_records: list[dict] | None = None
        try:
            from finance.black_litterman import black_litterman, views_from_scores
            if not cov_matrix.empty and asset_scores:
                bl_tickers = [t for t in cov_matrix.index]
                if bl_tickers:
                    P, Q, conf = views_from_scores(
                        {t: {"total": getattr(s, "total", None)}
                         for t, s in asset_scores.items()},
                        bl_tickers,
                    )
                    # Sprint-5 Task 4 — the investor mandate now CONSTRAINS the
                    # optimiser (was fully mandate-agnostic): a Conservative
                    # book gets a higher risk-aversion δ (smaller tilts), a
                    # tighter turnover cap and a tighter single-name cap than an
                    # Aggressive one, so BL targets respect the approved mandate
                    # instead of being a one-size-fits-all output.
                    _mandate = str(getattr(self.engine, "risk_mandate", "MODERATE")).upper()
                    _bl_cfg = _MANDATE_BL_CONSTRAINTS.get(
                        _mandate, _MANDATE_BL_CONSTRAINTS["MODERATE"])
                    bl_res = black_litterman(
                        cov              = cov_matrix,
                        tickers          = bl_tickers,
                        current_weights  = {t: weights_dict.get(t, 0) for t in bl_tickers},
                        views_P          = P if P.size else None,
                        views_Q          = Q if Q.size else None,
                        view_confidence  = conf if conf.size else None,
                        risk_aversion     = _bl_cfg["risk_aversion"],
                        max_active_share  = _bl_cfg["max_active_share"],
                        max_single_weight = _bl_cfg["max_single_weight"],
                    )
                    bl_records = bl_res.as_records()
        except Exception as exc:
            logger.warning("Black-Litterman skipped: %s", exc)
            bl_records = None

        # ═══════════════ ACTION PLAN ═══════════════
        # Buy zone / Sell target / Stop loss anchored to ATR + SMA + RSI.
        # BLOCK 2.3: computed BEFORE the Expected-Effect simulation so the
        # panel can be driven by the HIGH-PRIORITY (non-deferred) action items
        # rather than the full BL target — keeping Идеи → Action Plan →
        # Ожидаемый эффект one consistent story.
        action_plan_rows: list[dict] = []
        try:
            from finance.action_plan import build_action_plan
            ap_scores = {
                t: {"action":  s.action,
                    "total":   s.total,
                    "hotspot": s.hotspot}
                for t, s in asset_scores.items()
            }
            # F-20: originals of the names the sparse-guard excluded from the
            # structural model (resolved → original via base-symbol match), so
            # the plan can say WHY a row has no beta/target/quantity.
            _sparse_res = set(getattr(self.engine, "_last_sparse_dropped", []) or [])
            _uncovered = {
                orig for orig, res in zip(actual_risky, port_resolved)
                if res in _sparse_res
                or str(res).split(".")[0] in {str(s).split(".")[0]
                                              for s in _sparse_res}
            } if _sparse_res else set()
            rows = build_action_plan(
                perf_table      = perf,
                asset_scores    = ap_scores,
                technicals_map  = technicals_map,
                bl_records      = bl_records,
                portfolio_value = total_portfolio_value,
                # Sprint-5.1 (A3): stop/take distances scale with the mandate
                # (Conservative tighter, Aggressive wider) — levels are no
                # longer one-size-fits-all.
                risk_mandate    = self.engine.risk_mandate,
                uncovered       = _uncovered,
            )
            action_plan_rows = [r.as_dict() for r in rows]
        except Exception as exc:
            logger.warning("Action plan skipped: %s", exc)
            action_plan_rows = []

        # ═══════════════ SMART MONEY (insider SEC Form-4) ═══════════════
        # BLOCK 3.5 / B2.4 — always populate results["smart_money"] so the DEEP
        # report can RENDER the insider layer.  Its status is visible even when
        # gated OFF: the panel then shows "слой готов — источник Form-4 не
        # подключён" instead of the section being absent.  No live EDGAR crawl
        # here (gated provider); default per-ticker state is "disabled".
        smart_money: dict = {}
        try:
            from finance.smart_money import build_insider_signals
            sm_tickers = [str(t) for t in df.index][:25]
            smart_money = build_insider_signals(sm_tickers)
        except Exception as exc:
            logger.warning("Smart Money skipped: %s", exc)
            smart_money = {}

        # ═══════════════ EXPECTED EFFECT (after-plan simulation) ═══════════
        # Re-evaluates cover-page metrics under the HIGH-PRIORITY action items
        # (BLOCK 2.3).  Falls back to the BL target — and then to current
        # weights — when no actionable rows exist; the "after" then equals
        # "before" and every delta is 0, which the report can detect.
        expected_effect: dict | None = None
        try:
            # BLOCK 2.3: the panel simulates ONLY the high-priority moves that
            # survived the turnover cap (Buy/Sell/Trim with |Δw|>0).  Deferred
            # and Hold rows leave their weight unchanged; falls back to the BL
            # target when there are no actionable rows.
            from finance.simulate import high_priority_target_weights
            target_weights, hp_tickers, hp_actions = high_priority_target_weights(
                weights_dict, action_plan_rows, bl_records)

            # Build daily log-returns matrix for assets the cov matrix knows
            # about.  Reuses all_data (already loaded) so no extra fetch.
            sim_daily_log: pd.DataFrame | None = None
            if not cov_matrix.empty:
                cov_tickers = list(cov_matrix.index)
                resolved    = self.engine.resolve_tickers(cov_tickers)
                avail_cols  = [r for r in resolved if r in all_data.columns]
                if avail_cols:
                    sub = all_data[avail_cols].dropna()
                    if len(sub) >= 2:
                        log_df = np.log(sub / sub.shift(1)).dropna()
                        # Map columns back to ORIGINAL (display) tickers.
                        col_map = {res: orig for orig, res in zip(cov_tickers, resolved)
                                   if res in avail_cols}
                        log_df = log_df.rename(columns=col_map)
                        sim_daily_log = log_df

            sector_map_for_sim = {t: self.engine.get_ticker_sector(t)
                                   for t in df.index}

            expected_effect = _simulate_after_plan(
                perf_df           = df.reset_index(),
                risk_matrix       = cov_matrix,
                daily_log_returns = sim_daily_log,
                bl_records        = bl_records,
                current_metrics   = port_metrics,
                risk_free_rate    = self.engine.current_rfr_annual,
                target_weights    = target_weights,
                sector_by_ticker  = sector_map_for_sim,
                # 2026-07-18: same mandate + leverage the verdict gauge uses so
                # the effect «до/после» index mirrors the cover gauge (both now
                # carry the structural/tail aggravators).
                mandate           = str(getattr(self.engine, "risk_mandate", "MODERATE")),
                leverage_ratio    = _agg_lev_ratio,
            )
            # Tag which tickers drove the panel so the UI can show the
            # before/after delta is scoped to the high-priority ideas.
            if isinstance(expected_effect, dict):
                expected_effect["high_priority_tickers"] = hp_tickers
                expected_effect["high_priority_actions"] = hp_actions
                expected_effect["driver"] = ("high_priority_action_plan"
                                             if hp_tickers else "bl_target_fallback")
        except Exception as exc:
            logger.warning("Expected-effect simulator skipped: %s", exc)
            expected_effect = None

        # ── Leverage / gross exposure (read-only, AFTER risk math) ─────────
        # Margin debt manifests as a NEGATIVE weight on the cash leg (e.g.
        # USD wt = -17.7% → leverage 117.7%).  We compute these metrics
        # AFTER all matrix maths so the linear-algebra pipeline still
        # consumes the raw weights_dict unchanged.
        _NON_RISK = self.engine.NON_RISK_ASSETS
        long_weight = sum(max(0.0, float(w)) for w in weights_dict.values())
        gross_expo  = sum(abs(float(w))      for w in weights_dict.values())
        net_expo    = sum(float(w)           for w in weights_dict.values())
        cash_weight = sum(float(weights_dict.get(t, 0.0)) for t in weights_dict
                          if str(t).upper() in _NON_RISK)
        # Strict threshold guards float noise on a fully-funded book.
        is_leveraged = cash_weight < -0.001
        leverage_metrics = {
            "gross_exposure":   round(gross_expo, 6),
            "net_exposure":     round(net_expo,   6),
            "long_weight":      round(long_weight, 6),
            "cash_weight":      round(cash_weight, 6),
            "leverage_ratio":   round(long_weight / net_expo, 4) if net_expo > 1e-9 else 1.0,
            "is_leveraged":     bool(is_leveraged),
        }

        return {
            "performance_table": perf,
            "risk_matrix": cov_matrix,
            "total_portfolio_pnl": df['PnL'].sum(),
            "total_value": total_portfolio_value,
            "portfolio_metrics": port_metrics,
            "leverage_metrics":  leverage_metrics,
            # H3 (Phase-3): date-indexed portfolio log-return series for the
            # Telegram equity-curve SVG — the bot no longer recomputes it.
            "port_log_returns":  getattr(self.engine, "_last_port_log_returns", None),
            # H4: the mandate actually used for the composite-risk calibration.
            "risk_mandate":      getattr(self.engine, "risk_mandate", "MODERATE"),
            "risk_free_rate": self.engine.current_rfr_annual,
            "benchmark_comparison": benchmark_results,
            # The user's chosen benchmark TICKER (e.g. QQQ.US), so downstream
            # visuals (equity-curve line) can honour the actual choice instead
            # of defaulting to the first concrete benchmark.  None → no profile
            # benchmark selected.
            "profile_benchmark_ticker": profile_benchmark,
            # B1 (2026-07-17): факторные беты мандатного бенчмарка для секции
            # «Факторное разложение» — {ticker, name, betas} или None.
            "benchmark_factor_profile": benchmark_factor_profile,
            "period_returns_table": period_returns_table,
            "return_series_coverage": return_coverage,
            "stress_scenarios": stress_scenarios,
            "expected_effect": expected_effect,
            "cds_summary":     cds_summary,
            "history_result": history_result,
            "sector_exposure": sector_exposure,
            "factor_scores": factor_scores,
            # Phase 2 additions
            "regime":            regime_reading.as_dict() if regime_reading else None,
            "macro_drivers":     macro_drivers,
            "technicals":        {t: {"score": r.score, "raw": r.raw,
                                       "components": r.components}
                                  for t, r in technicals_map.items()},
            "asset_scores":      {t: {
                                      "fundamentals": s.fundamentals,
                                      "valuations":   s.valuations,
                                      "technicals":   s.technicals,
                                      "credit":       s.credit,
                                      "total":        s.total,
                                      "action":       s.action,
                                      "hotspot":      s.hotspot,
                                      "credit_applicable": getattr(s, "credit_applicable", True),
                                      "fundamentals_applicable": getattr(s, "fundamentals_applicable", True),
                                  } for t, s in asset_scores.items()},
            # Phase 3 additions
            "black_litterman":   bl_records,
            "action_plan":       action_plan_rows,
            # BLOCK 3.5 / B2.4 — insider (Form-4) signals for the DEEP layer.
            "smart_money":       smart_money,
        }