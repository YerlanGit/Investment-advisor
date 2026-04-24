import pandas as pd
import yfinance as yf
import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Antigravity_RiskEngine")

class MAC3RiskEngine:
    """
    Институциональный движок рисков (RAMP Style).
    Внедрены: Структурная ковариация (Barra/MAC3), Euler Risk Decomposition, CVaR.
    """
    TICKER_MAP = {
        'KAZATOMPROM': 'KAP.IL', 'KASPI': 'KSPI', 'HALYK': 'HSBK.IL',
        'KAZAKHTELECOM': 'KZTK.KZ',
        'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD'
    }

    # Инструменты нулевого риска (или которые не прогоняются через регрессию)
    NON_RISK_ASSETS = ['USD', 'EUR', 'CASH', 'RUB', 'KZT']

    def __init__(self, trading_days=252, ewma_halflife=63):
        self.trading_days = trading_days
        self.ewma_halflife = ewma_halflife  # 63 дней ≈ λ=0.94 (RiskMetrics)
        # Факторные прокси (Time-Series)
        self.factor_tickers = {
            'Market': '^GSPC', 'Momentum': 'MTUM', 'Value': 'VLUE',
            'Quality': 'QUAL', 'Size': 'IWM', 'Commodities': 'DBC', 
            'Rates': '^TNX'
        }
        self.current_rfr_annual = 0.04 # Хардкод на случай падения API

    def math_firewall(self, df):
        """Защита от 'битых' данных (галлюцинаций API)."""
        return df.ffill().bfill()

    def resolve_tickers(self, tickers):
        resolved = []
        for t in tickers:
            t_str = str(t).upper().strip()
            if t_str in self.NON_RISK_ASSETS:
                continue # Cash goes naturally out of equity fetch
            if t_str in self.TICKER_MAP: resolved.append(self.TICKER_MAP[t_str])
            elif 'KSPI' in t_str: resolved.append('KSPI')
            elif t_str in ['BTC', 'ETH', 'SOL', 'BNB']: resolved.append(f"{t_str}-USD")
            elif t_str in ['KAP', 'HSBK']: resolved.append(f"{t_str}.IL")
            else: resolved.append(t_str)
        return resolved

    def get_market_data(self, tickers, period="2y"):
        """Загрузка сырых котировок (Источник: Yahoo Finance API)."""
        valid_tickers = self.resolve_tickers(tickers)
        all_req = list(set(valid_tickers + list(self.factor_tickers.values())))
        
        logger.info(f"Загрузка цен для {len(all_req)} инструментов...")
        data = yf.download(all_req, period=period, progress=False, auto_adjust=True)['Close']
        
        if isinstance(data, pd.Series): 
            data = data.to_frame()
            
        # Обновляем безрисковую ставку через гособлигации (если подгрузились)
        if '^TNX' in data.columns and not data['^TNX'].dropna().empty:
            # ^TNX приходит в процентах (напр. 4.3 => 4.3%)
            self.current_rfr_annual = data['^TNX'].dropna().iloc[-1] / 100.0

        # Заполнение пропусков (Forward Fill)
        return self.math_firewall(data)

    def get_fundamental_metrics(self, tickers):
        """Слой Quality и Leverage."""
        fundamental_data = {}
        for t in tickers:
            t_str = str(t).upper().strip()
            if t_str in self.NON_RISK_ASSETS:
                continue
                
            resolved_list = self.resolve_tickers([t_str])
            if not resolved_list: continue
            resolved = resolved_list[0]
            
            try:
                stock = yf.Ticker(resolved)
                info = stock.info
                fundamental_data[t] = {
                    'ROE_Quality': info.get('returnOnEquity', 0),
                    'Debt_to_Equity': info.get('debtToEquity', 0) / 100,
                    'Forward_PE': info.get('forwardPE', 0),
                    'Market_Cap_Log': np.log(info.get('marketCap', 1)) if info.get('marketCap') else 0
                }
            except:
                fundamental_data[t] = {k: 0 for k in ['ROE_Quality', 'Debt_to_Equity', 'Forward_PE', 'Market_Cap_Log']}
        return pd.DataFrame(fundamental_data).T

    def calculate_structural_risk(self, data, asset_tickers, weights_dict):
        """
        Ядро MAC3: Построение факторной модели и декомпозиция Эйлера.
        Облигации и Кэш в asset_tickers не передаются! 
        Но weights_dict содержит их доли от ОБЩЕГО портфеля, поэтому сумма весов asset_tickers < 1.0.
        """
        resolved_assets = self.resolve_tickers(asset_tickers)
        
        # Логарифмические доходности для агрегации факторов
        returns = np.log(data / data.shift(1)).dropna()
        
        # 1. Выделяем факторы и активы
        f_data = returns[list(self.factor_tickers.values())]
        f_data.columns = list(self.factor_tickers.keys())
        
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
        
        portfolio_metrics = {
            "Total_Volatility_Ann": port_volatility,
            "Sharpe_Ratio": (ann_return - self.current_rfr_annual) / port_volatility if port_volatility > 0 else np.nan,
            "Sortino_Ratio": (ann_return - self.current_rfr_annual) / downside_vol if downside_vol > 0 else np.nan,
            "VaR_95_Daily": var_95,
            "CVaR_95_Daily": cvar_95,
            "Positive_Days_Pct": (port_returns_daily > 0).mean() * 100 if len(port_returns_daily)>0 else 0
        }

        return cov_df, pd.DataFrame(exposures_report).T, portfolio_metrics

    def calculate_atr(self, data, tickers, period=14):
        """
        ATR (Average True Range) — индикатор волатильности для IMS.
        Используется для оценки внутридневного риска по стандарту SEC 34-105226.
        """
        resolved = self.resolve_tickers(tickers)
        atr_results = {}
        
        for orig, res in zip(tickers, resolved):
            if res not in data.columns:
                continue
            prices = data[res]
            # Для Close-only данных: ATR ≈ |ΔPrice| rolling
            daily_range = prices.diff().abs()
            atr = daily_range.rolling(window=period).mean().iloc[-1]
            atr_pct = (atr / prices.iloc[-1]) * 100 if prices.iloc[-1] > 0 else 0
            atr_results[orig] = {
                'ATR_Absolute': atr,
                'ATR_Pct': atr_pct  # ATR как % от цены — нормализованная волатильность
            }
        
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
        profile_benchmark: yfinance ticker for the user's mandate benchmark
        (e.g. 'AGG' for Conservative, '^GSPC' for Moderate, '^NDX' for Aggressive).
        When provided it is computed first and labelled 'Профильный бенчмарк' in results.
        """
        if isinstance(source, pd.DataFrame): raw_df = source
        else: raise ValueError("Неверный источник данных.")

        df = self._standardize_columns(raw_df).set_index('Ticker')
        
        # Скачиваем цены на Рисковые активы
        risky_tickers = [t for t in df.index if t not in self.engine.NON_RISK_ASSETS]
        
        all_data = self.engine.get_market_data(risky_tickers)
        fund_df = self.engine.get_fundamental_metrics(risky_tickers)
        df = df.join(fund_df)

        resolved_indices = self.engine.resolve_tickers(df.index.tolist())
        
        # Установка Current Price (Для Кэша цена всегда 1.0)
        current_prices = {}
        for orig, res in zip(df.index, resolved_indices):
            if orig in self.engine.NON_RISK_ASSETS:
                current_prices[orig] = 1.0
            elif res in all_data.columns:
                current_prices[orig] = all_data[res].iloc[-1]
                
        df['Current_Price'] = pd.Series(current_prices)
        df = df.dropna(subset=['Current_Price'])

        # Стоимости и Веса
        df['Total_Cost'] = df['Quantity'] * df['Purchase_Price']
        df['Current_Value'] = df['Quantity'] * df['Current_Price']
        df['PnL'] = df['Current_Value'] - df['Total_Cost']
        df['Return_Pct'] = (df['PnL'] / df['Total_Cost'])
        
        total_portfolio_value = df['Current_Value'].sum()
        weights_dict = (df['Current_Value'] / total_portfolio_value).to_dict()

        # MAC3 Structural Risk (Только для акций)
        actual_risky = [t for t in df.index if t not in self.engine.NON_RISK_ASSETS]
        cov_matrix, factor_df, port_metrics = self.engine.calculate_structural_risk(all_data, actual_risky, weights_dict)
        
        # ATR (Intraday Margin Standard — SEC 34-105226)
        atr_df = self.engine.calculate_atr(all_data, actual_risky)
        if not atr_df.empty:
            df = df.join(atr_df)
        
        # SEC EDGAR Fundamental Scores (бесплатный фундаментальный анализ)
        try:
            from finance.sec_edgar import batch_fundamental_scan
            sec_df = batch_fundamental_scan(actual_risky)
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

        # Ожидаемая доходность (ЕДИНАЯ ШКАЛА: log-returns)
        f_data = all_data[list(self.engine.factor_tickers.values())]
        # Log-returns факторов за период (согласовано с calculate_structural_risk)
        factor_returns_log = np.log(f_data.ffill().iloc[-1] / f_data.bfill().iloc[0])
        beta_cols = [c for c in df.columns if str(c).startswith('Beta_')]
        if beta_cols:
            # Expected Return в log-шкале, конвертируем в simple для сравнения с Return_Pct
            expected_log = df[beta_cols].fillna(0).values @ factor_returns_log.values
            df['Expected_Return'] = np.exp(expected_log) - 1  # log → simple conversion
            df['Alpha_Specific'] = (df['Return_Pct'] - df['Expected_Return']) * 100

        # ═══════════════ BENCHMARK COMPARISON ═══════════════
        # Профильный бенчмарк ставится первым для корректного расчёта Tracking Error.
        benchmarks: dict[str, str] = {}
        if profile_benchmark:
            benchmarks["Профильный бенчмарк"] = profile_benchmark

        benchmarks.update({
            'S&P 500':     '^GSPC',
            'Nasdaq 100':  '^NDX',
            'Russell 2000': '^RUT',
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
                            
                            # Выравниваем длины
                            min_len = min(len(port_daily), len(bm_daily))
                            tracking_diff = port_daily[-min_len:] - bm_daily.values[-min_len:]
                            tracking_error = np.std(tracking_diff) * np.sqrt(self.engine.trading_days)
                            
                            # Information Ratio = (Rp - Rb) / TE
                            excess_ann = (port_return - bm_return)
                            info_ratio = excess_ann / tracking_error if tracking_error > 0 else 0
                        else:
                            tracking_error = None
                            info_ratio = None
                    else:
                        tracking_error = None
                        info_ratio = None
                    
                    benchmark_results[bm_name] = {
                        "Benchmark_Return": bm_return,
                        "Portfolio_Return": port_return,
                        "Excess_Return": port_return - bm_return,
                        "Tracking_Error": tracking_error,
                        "Information_Ratio": info_ratio,
                        "Beating_Benchmark": port_return > bm_return
                    }

        return {
            "performance_table": df.reset_index(),
            "risk_matrix": cov_matrix,
            "total_portfolio_pnl": df['PnL'].sum(),
            "total_value": total_portfolio_value,
            "portfolio_metrics": port_metrics,
            "risk_free_rate": self.engine.current_rfr_annual,
            "benchmark_comparison": benchmark_results
        }