import os
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf

from finance.broker_api import RealPortfolioRequired
from freedom_portfolio import TradernetClient, get_history_frame

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Antigravity_RiskEngine")

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
    }

    # Benchmark / catalogue ETFs fetched alongside factor ETFs.
    # NOT used as regression factors — only for benchmark_comparison + TE.
    # Includes all tickers from profile_manager.BENCHMARK_LIST so they are
    # pre-loaded in a single batch (avoids a second download pass).
    BENCHMARK_EXTRA = [
        'QQQ.US', 'AGG.US', 'URTH.US',  # broad indices
        # Factor ETFs already in factor_tickers (SPY, MTUM, VLUE, QUAL, IWM, DBC, IEF, EEM, EMB)
    ]

    def __init__(self, trading_days=252, ewma_halflife=63):
        self.trading_days = trading_days
        self.ewma_halflife = ewma_halflife  # 63 дней ≈ λ=0.94 (RiskMetrics)
        # Факторные прокси (формат Tradernet — все .US ETF).
        # EM_Equity (EEM.US) и EM_Bond (EMB.US) добавлены для корректного
        # моделирования казахстанских и EM-активов (.KZ/.IL тикеры).
        self.factor_tickers = {
            'Market':      'SPY.US',    # S&P 500 — глобальный рыночный фактор
            'Momentum':    'MTUM.US',
            'Value':       'VLUE.US',
            'Quality':     'QUAL.US',
            'Size':        'IWM.US',
            'Commodities': 'DBC.US',
            'Rates':       'IEF.US',    # 7-10y Treasury — фактор процентных ставок
            'EM_Equity':   'EEM.US',    # MSCI Emerging Markets — EM equity premium
            'EM_Bond':     'EMB.US',    # JP Morgan EM Bond ETF — EM credit/FX premium
        }
        # Безрисковая ставка: для казахстанских инвесторов ставка НБК актуальнее
        # 4% US T-bill. Задаётся через env KZ_RFR_ANNUAL (default 14%).
        self.current_rfr_annual = float(os.getenv("KZ_RFR_ANNUAL", "0.14"))
        # Кешируемый клиент Tradernet — переиспользуется между запросами одного
        # вызова (внутри analyze_all). Keys читаются из env (Cloud Run secret).
        self._tradernet_client: TradernetClient | None = None

    def math_firewall(self, df):
        """Защита от 'битых' данных (галлюцинаций API)."""
        return df.ffill().bfill()

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
                logger.info("AIX-инструмент %s → прокси %s для фактор-модели", t_str, proxy)
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
        return resolved

    def get_market_data(self, tickers, period_days: int = 730):
        """
        Загрузка дневных закрытий через Tradernet API (replaces yfinance).

        Returns (data, history_result) tuple where history_result contains
        loaded/failed/retried ticker details for user-facing messages.
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

        return self.math_firewall(history_result.data), history_result

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
            
            model = Ridge(alpha=1.0, fit_intercept=True).fit(X, y)
            betas = model.coef_
            alpha = model.intercept_
            residuals = y - model.predict(X)
            
            B_matrix.append(betas)
            specific_variances.append(np.var(residuals))
            
            exposures_report[asset] = {f'Beta_{k}': b for k, b in zip(f_data.columns, betas)}
            exposures_report[asset]['Specific_Alpha_Daily'] = alpha
            exposures_report[asset]['Residual_Vol_Ann'] = np.std(residuals) * np.sqrt(self.trading_days)
            exposures_report[asset]['Weight_Pct'] = weights_dict.get(asset, 0) * 100

        B = np.array(B_matrix) # (N_assets, K_factors)
        D = np.diag(specific_variances) # (N_assets, N_assets)
        
        # 3. Структурная Ковариационная Матрица
        # EWMA коварация (стандарт RiskMetrics: λ=0.94, halflife=63)
        # Недавние дни влияют сильнее, чем старые
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
        
        structural_cov = B @ cov_factors @ B.T + D
        structural_cov_ann = structural_cov * self.trading_days 
        
        cov_df = pd.DataFrame(structural_cov_ann, index=valid_originals, columns=valid_originals)
        
        # Если портфель пустой по рисковым активам
        if weights.sum() == 0:
            return cov_df, pd.DataFrame(exposures_report).T, {}
            
        # 4. Риск портфеля
        port_variance = weights.T @ structural_cov_ann @ weights
        port_volatility = np.sqrt(port_variance)
        
        # 5. Декомпозиция Эйлера
        # MCTR = (Sigma * w) / Port_Vol
        mctr = (structural_cov_ann @ weights) / port_volatility if port_volatility > 0 else np.zeros_like(weights)
        # ERC (%) = w * MCTR / Port_Vol
        erc_pct = (weights * mctr) / port_volatility if port_volatility > 0 else np.zeros_like(weights)
        
        for i, asset in enumerate(valid_originals):
            exposures_report[asset]['Euler_Risk_Contribution_Pct'] = erc_pct[i] * 100

        # 6. Хвостовые метрики
        port_returns_daily = a_data.values @ weights # Это будет "разбавлено" так как sum(weights)<1, что верно для Total Return
        risk_free_rate_daily = self.current_rfr_annual / self.trading_days
        
        excess_returns = port_returns_daily - risk_free_rate_daily
        downside_returns = excess_returns[excess_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(self.trading_days)
        
        ann_return = np.mean(port_returns_daily) * self.trading_days
        
        var_95 = np.percentile(port_returns_daily, 5) if len(port_returns_daily)>0 else 0
        cvar_95 = port_returns_daily[port_returns_daily <= var_95].mean() if len(port_returns_daily)>0 else 0

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

        portfolio_metrics = {
            "Total_Volatility_Ann": port_volatility,
            "Sharpe_Ratio": (ann_return - self.current_rfr_annual) / port_volatility if port_volatility > 0 else np.nan,
            "Sortino_Ratio": (ann_return - self.current_rfr_annual) / downside_vol if downside_vol > 0 else np.nan,
            "VaR_95_Daily": var_95,
            "CVaR_95_Daily": cvar_95,
            "Max_Drawdown": max_drawdown,           # NEW: realised peak-to-trough drawdown
            "Positive_Days_Pct": (port_returns_daily > 0).mean() * 100 if len(port_returns_daily)>0 else 0
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



class UniversalPortfolioManager:
    COLUMN_ALIASES = {
        'Ticker': ['Symbol', 'Asset', 'Тикер', 'Name'],
        'Quantity': ['Qty', 'Amount', 'Shares', 'Кол-во'],
        'Purchase_Price': ['Buy_Price', 'Price_Buy', 'Entry_Price', 'Цена_Покупки', 'Avg_Price']
    }

    def __init__(self):
        self.engine = MAC3RiskEngine()

    def _standardize_columns(self, df):
        for standard, aliases in self.COLUMN_ALIASES.items():
            actual_col = next((c for c in df.columns if c in aliases or c == standard), None)
            if actual_col: df = df.rename(columns={actual_col: standard})
        return df

    def analyze_all(self, source, scenario_shocks=None, profile_benchmark: str | None = None):
        """
        profile_benchmark: Tradernet ETF ticker for the user's mandate benchmark
        (e.g. 'AGG.US' for Conservative, 'SPY.US' for Moderate, 'QQQ.US' for Aggressive).
        When provided it is computed first and labelled 'Профильный бенчмарк' in results.
        """
        if isinstance(source, pd.DataFrame): raw_df = source
        else: raise ValueError("Неверный источник данных.")

        # ── MAC3 GATE: portfolio source verification ──────────────────────────
        # Detect the three possible DataFrame origins stamped by FreedomConnector:
        #   _ramp_is_mock=True, _ramp_is_fallback=True  → broker API failed, fallback mock
        #   _ramp_is_mock=True, _ramp_source='demo'     → user explicitly chose template/demo
        #   (no attrs)                                  → live broker data
        _is_fallback = raw_df.attrs.get('_ramp_is_fallback', False)
        _is_mock     = raw_df.attrs.get('_ramp_is_mock',     False)

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
        df['Return_Pct'] = (df['PnL'] / df['Total_Cost'])

        total_portfolio_value = df['Current_Value'].sum()
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
            # GEOMETRIC annualisation (correct for compounding):
            # E[r_ann] = exp(mean_daily_log * 252) - 1
            factor_ann = np.exp(factor_mean_daily * self.engine.trading_days) - 1
            # Map to factor key names for label-safe matching against Beta_* columns
            factor_ann_by_key = pd.Series({
                k: factor_ann.get(v, 0.0)
                for k, v in zip(present_factor_keys, present_factor_etfs)
            })
        else:
            factor_ann_by_key = pd.Series(dtype=float)

        beta_cols = [c for c in df.columns if str(c).startswith('Beta_')]
        if beta_cols and not factor_ann_by_key.empty:
            beta_factor_keys = [c[len('Beta_'):] for c in beta_cols]
            factor_vec = factor_ann_by_key.reindex(beta_factor_keys).fillna(0.0).values
            # Factor-driven expected return
            factor_component = df[beta_cols].fillna(0).values @ factor_vec
            # Alpha intercept: annualised from daily alpha (stored by Ridge)
            alpha_col = 'Specific_Alpha_Daily'
            if alpha_col in df.columns:
                alpha_ann = df[alpha_col].fillna(0).values * self.engine.trading_days
            else:
                alpha_ann = 0.0
            # Total expected return = alpha + factor contribution
            df['Expected_Return'] = alpha_ann + factor_component
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
        benchmark_results = {}
        
        for bm_name, bm_ticker in benchmarks.items():
            if bm_ticker in all_data.columns:
                bm_prices = all_data[bm_ticker].dropna()
                if len(bm_prices) > 1:
                    # Доходность бенчмарка за весь период
                    bm_return = (bm_prices.iloc[-1] / bm_prices.iloc[0]) - 1
                    
                    # Портфельная доходность (взвешенная)
                    port_return = df['Return_Pct'].fillna(0).values @ np.array(
                        [weights_dict.get(t, 0) for t in df.index]
                    )
                    
                    # Tracking Error (дневная разница портфель - бенчмарк)
                    bm_daily = np.log(bm_prices / bm_prices.shift(1)).dropna()
                    
                    # Дневные доходности портфеля
                    port_resolved = self.engine.resolve_tickers(actual_risky)
                    port_daily_cols = [r for r in port_resolved if r in all_data.columns]
                    if port_daily_cols:
                        port_daily_returns = np.log(all_data[port_daily_cols] / all_data[port_daily_cols].shift(1)).dropna()
                        risky_weights = np.array([weights_dict.get(t, 0) for t in actual_risky 
                                                  if self.engine.resolve_tickers([t])[0] in all_data.columns])
                        if len(risky_weights) == port_daily_returns.shape[1]:
                            port_daily = port_daily_returns.values @ risky_weights

                            # Align lengths
                            min_len            = min(len(port_daily), len(bm_daily))
                            port_daily_aligned = port_daily[-min_len:]
                            bm_daily_aligned   = bm_daily.values[-min_len:]
                            tracking_diff      = port_daily_aligned - bm_daily_aligned
                            tracking_error     = np.std(tracking_diff) * np.sqrt(self.engine.trading_days)

                            # Geometric-annualised returns on the SAME aligned window.
                            # Previously: excess_ann = (port_return - bm_return) which
                            # mixed period-return (~2y) with annualised TE → IR scale bug.
                            # Now both numerator and denominator are annualised consistently.
                            port_ann_return = float(np.exp(np.mean(port_daily_aligned) * self.engine.trading_days) - 1)
                            bm_ann_return   = float(np.exp(np.mean(bm_daily_aligned)   * self.engine.trading_days) - 1)
                            excess_ann      = port_ann_return - bm_ann_return
                            info_ratio      = excess_ann / tracking_error if tracking_error > 0 else 0
                        else:
                            tracking_error  = None
                            info_ratio      = None
                            excess_ann      = None
                            port_ann_return = None
                            bm_ann_return   = None
                    else:
                        tracking_error  = None
                        info_ratio      = None
                        excess_ann      = None
                        port_ann_return = None
                        bm_ann_return   = None

                    benchmark_results[bm_name] = {
                        "Benchmark_Return":     bm_return,                    # period total (display)
                        "Portfolio_Return":     port_return,                  # period total (display)
                        "Excess_Return":        port_return - bm_return,      # period total (display)
                        "Portfolio_Ann_Return": port_ann_return,              # NEW: annualised, aligned w/ TE
                        "Benchmark_Ann_Return": bm_ann_return,                # NEW
                        "Excess_Return_Ann":    excess_ann,                   # NEW: annualised excess (for IR)
                        "Tracking_Error":       tracking_error,
                        "Information_Ratio":    info_ratio,
                        "Beating_Benchmark":    port_return > bm_return,
                    }

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

        return {
            "performance_table": perf,
            "risk_matrix": cov_matrix,
            "total_portfolio_pnl": df['PnL'].sum(),
            "total_value": total_portfolio_value,
            "portfolio_metrics": port_metrics,
            "risk_free_rate": self.engine.current_rfr_annual,
            "benchmark_comparison": benchmark_results,
            "history_result": history_result,
            "sector_exposure": sector_exposure,
            "factor_scores": factor_scores,
        }