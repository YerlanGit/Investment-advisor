"""Ядро отчёта: риск-движок и оркестратор конвейера (Арх-3.10).

Раскладка пакета:

* `risk_engine`      — `MAC3RiskEngine` + модульные константы и хелперы движка
* `portfolio_manager`— `UniversalPortfolioManager` и семь стадий `analyze_all`
* `market_preview`   — `MarketDataPreview`

`finance/investment_logic.py` остался ФАСАДОМ: ни один существующий импорт не
изменился. Новый код может импортировать и отсюда, и из фасада — оба пути
ведут к одним и тем же объектам.
"""

from .market_preview import MarketDataPreview
from .portfolio_manager import UniversalPortfolioManager, _AnalysisCtx
from .risk_engine import (
    MAC3RiskEngine,
    _MANDATE_BL_CONSTRAINTS,
    _is_leveraged_etp,
    _nearest_psd,
    _safe_num,
    apply_leveraged_drag,
    apply_leveraged_forward,
    beta_shrinkage_enabled,
    factor_orthogonalize_enabled,
    orthogonalize_factors_hierarchical,
)

__all__ = [
    "MAC3RiskEngine",
    "MarketDataPreview",
    "UniversalPortfolioManager",
    "apply_leveraged_drag",
    "apply_leveraged_forward",
    "beta_shrinkage_enabled",
    "factor_orthogonalize_enabled",
    "orthogonalize_factors_hierarchical",
]
