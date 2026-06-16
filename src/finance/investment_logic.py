import os
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
        infinite price tick can't survive ffill/bfill and poison the
        covariance matrix or Sharpe/CVaR downstream.  ffill→bfill then
        carries the last/next finite value across the gap.
        """
        cleaned = df.replace([np.inf, -np.inf], np.nan)
        return cleaned.ffill().bfill()

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
            # Round to 6 decimals so floating-point noise on identical
            # inputs across runs doesn't change the seed.
            try:
                arr = np.round(np.asarray(returns, dtype=float), 6)
                key = tuple(arr.tolist())
                seed = int(abs(hash(key)) % 100_000_000)
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
                              max_erc_pct: float, mandate: str = "MODERATE") -> int:
        from finance.scoring import composite_risk_score as _crs
        return _crs(volatility, cvar, max_erc_pct, mandate=mandate)

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

    def get_market_data(self, tickers, period_days: int = 730):
        """
        Загрузка дневных закрытий через Tradernet API (replaces yfinance).

        Returns (data, history_result) tuple where history_result contains
        loaded/failed/retried ticker details for user-facing messages.

        H2: After Tradernet loads native-currency prices, we transform
        every column whose native currency differs from
        `self.reporting_currency` by multiplying with the matching FX
        series.  USD-only portfolios short-circuit (no FX calls, no copy).
        """
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
        resolved_assets = self.resolve_tickers(asset_tickers)

        # Restrict to columns we actually need and drop all-NaN columns BEFORE
        # the row-level dropna. Without this, a single broker-only ticker with
        # no Tradernet history (e.g. FFSPC6.1028.AIX on Astana exchange) leaves
        # an all-NaN column whose NaNs propagate through the row-level dropna
        # and erase every row — Ridge then gets shape (0, K) and raises.
        needed_cols = [*self.factor_tickers.values(), *resolved_assets]
        available = [c for c in needed_cols if c in data.columns]
        data = data[available].dropna(axis=1, how='all')

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

        for asset in valid_originals:
            y = a_data[asset]
            X = f_data

            # alpha=1.0 shrinks daily-return betas by ~98% (penalty >> RSS on
            # 0.01-scale returns).  Use 0.001 — enough to stabilise correlated
            # factors while keeping betas economically meaningful.
            model = Ridge(alpha=0.001, fit_intercept=True).fit(X, y)
            betas = model.coef_
            alpha = model.intercept_
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
                factor_diagnostic = {
                    "n_factors":        int(_F.shape[0]),
                    "factors":          list(f_data.columns),
                    "max_abs_corr":     round(_max_off, 4),
                    "condition_number": round(_cond, 2),
                    "near_collinear":   bool(_max_off > 0.95 or _cond > 30.0),
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

        # 6. Хвостовые метрики
        # `port_returns_daily` carries LOG returns in the REPORTING CURRENCY
        # (H2 pre-converted the price matrix → a_data is base-currency log
        # returns).  Sum(weights) < 1 by design when cash sits in the
        # portfolio — that dilutes risk correctly without renormalisation.
        port_returns_daily = a_data.values @ weights
        # H3 (Phase-3): expose the date-indexed portfolio log-return series so
        # the Telegram equity-curve SVG consumes it instead of recomputing
        # `log(prices/prices.shift) @ weights` in the bot layer.  Stored on
        # the engine (same pattern as _last_fx_records); analyze_all copies it
        # into results["port_log_returns"].
        try:
            self._last_port_log_returns = pd.Series(port_returns_daily,
                                                    index=a_data.index)
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
            # BLOCK 4.6: factor-multicollinearity diagnostic (κ + max|corr|),
            # surfaced to the CoVe panel so collinear factors are visible.
            "factor_diagnostics": getattr(self, "_last_factor_diagnostic", {}),
        }

        return cov_df, pd.DataFrame(exposures_report).T, portfolio_metrics

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
        # the existing handler (`RealPortfolioRequired` is caught by
        # tg_bot._build_analysis_payload) already understands.
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
        #   4. Expected return = Alpha_ann + Σ beta_k * E[r_k_ann]
        #      Alpha (intercept) from Ridge regression is included.
        #   5. Specific Alpha = actual return - expected return (in %).
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
            # F-2: assemble the WHOLE expectation in daily-log space first
            # (alpha intercept + Σβ·μ are both Ridge log-return quantities),
            # then geometric-annualise ONCE.  The previous version added an
            # arithmetically-annualised log alpha (α·252) to geometric simple
            # factor returns (exp(μ·252)−1) — a units mismatch inherited by
            # Alpha_Specific.
            factor_log_daily = df[beta_cols].fillna(0).values @ mu_vec
            alpha_col = 'Specific_Alpha_Daily'
            if alpha_col in df.columns:
                alpha_log_daily = df[alpha_col].fillna(0).values
            else:
                alpha_log_daily = 0.0
            expected_log_daily = alpha_log_daily + factor_log_daily
            df['Expected_Return'] = np.exp(
                expected_log_daily * self.engine.trading_days) - 1.0
            # Specific Alpha: actual holding-period return minus expected (in %)
            df['Alpha_Specific'] = (df['Return_Pct'] - df['Expected_Return']) * 100

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
            )
        except Exception as exc:
            logger.warning("stress scenarios build failed: %s", exc)
            stress_scenarios = []

        # Sector exposure analysis
        sector_exposure = self.engine.get_sector_exposure(list(df.index), weights_dict)

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
            )
            action_plan_rows = [r.as_dict() for r in rows]
        except Exception as exc:
            logger.warning("Action plan skipped: %s", exc)
            action_plan_rows = []

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
            target_weights, hp_tickers = high_priority_target_weights(
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
            )
            # Tag which tickers drove the panel so the UI can show the
            # before/after delta is scoped to the high-priority ideas.
            if isinstance(expected_effect, dict):
                expected_effect["high_priority_tickers"] = hp_tickers
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
        }