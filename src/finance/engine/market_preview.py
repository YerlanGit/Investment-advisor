"""Превью рыночных данных: что удалось загрузить до запуска модели.

Арх-3.10: выделен из `finance/investment_logic.py` (остался фасадом).
"""

import pandas as pd
from dataclasses import dataclass, field


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
