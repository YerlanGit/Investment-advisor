"""ФАСАД ядра отчёта. Содержимое переехало в пакет `finance/engine/` (Арх-3.10).

Где что теперь лежит
--------------------
* `finance/engine/risk_engine.py`       — `MAC3RiskEngine`, модульные константы
                                          и хелперы движка (факторная модель,
                                          ковариация, PSD-проекция, плечевые ETP)
* `finance/engine/portfolio_manager.py` — `UniversalPortfolioManager`, семь
                                          стадий `analyze_all` и `_build_results`
* `finance/engine/market_preview.py`    — `MarketDataPreview`

Зачем фасад, а не массовая правка импортов
------------------------------------------
`from finance.investment_logic import MAC3RiskEngine` встречается в 43 местах,
`UniversalPortfolioManager` — в 21. Переписывать их значило бы смешать переезд
с правкой потребителей и потерять свойство «шаг поведенчески пустой»: сломайся
что-нибудь — не отличить причину. Фасад оставляет ровно один вид изменений:
перемещение текста.

🔴 Ловушка для тех, кто пишет тесты
-----------------------------------
`patch("finance.investment_logic.X")` подменяет имя ЗДЕСЬ, а код исполняется в
`finance.engine.*` и смотрит в СВОЙ модуль. Патч при этом не падает — имя на
фасаде есть, — он просто НЕ ДЕЙСТВУЕТ, и проверка вида `assert_not_called()`
проходит ВАКУУМНО: мок никто не трогал, потому что на него никто и не смотрит.
Патчить нужно настоящий модуль:

    patch("finance.engine.risk_engine.TradernetClient")     # ✅ действует
    patch("finance.investment_logic.TradernetClient")       # 🔴 тихо бесполезен

Тот же класс, что `AUDIT §−64`: проверка выглядит защитой, перестав ею быть.
"""

from .engine.market_preview import MarketDataPreview
from .engine.portfolio_manager import (
    UniversalPortfolioManager,
    _AnalysisCtx,
)
from .engine.risk_engine import (
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
