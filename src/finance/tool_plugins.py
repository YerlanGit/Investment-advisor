"""
Финтех-плагины для инструментов LLM-агента.
Изолированы от upstream tools.py для предотвращения merge-конфликтов.
"""
from __future__ import annotations
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolExecution:
    name: str
    source_hint: str
    payload: str
    handled: bool
    message: str


def execute_mac3_risk(name: str, payload: str) -> ToolExecution:
    """Перехват вызова MAC3 Risk Engine от LLM-агента."""
    try:
        from finance.investment_logic import MAC3RiskEngine

        args = json.loads(payload) if payload else {}
        tickers = args.get("tickers", [])
        period = args.get("period", "2y")

        if not tickers:
            return ToolExecution(
                name=name, source_hint="src.finance.investment_logic",
                payload=payload, handled=False,
                message="Ошибка: не переданы тикеры активов."
            )

        engine = MAC3RiskEngine()
        data = engine.get_market_data(tickers, period)

        # Равные веса по умолчанию
        equal_weight = 1.0 / len(tickers)
        weights = {t: equal_weight for t in tickers}

        cov_df, exposures_df, metrics = engine.calculate_structural_risk(data, tickers, weights)

        result_msg = (
            f"Успешный расчет структурного риска (MAC3 + Euler).\n"
            f"Тикеры: {tickers}\n"
            f"Факторные экспозиции:\n{exposures_df.to_string()}\n\n"
            f"Ковариационная матрица (годовая):\n{cov_df.to_string()}\n\n"
            f"Портфельные метрики: {metrics}\n"
            f"Используй Euler Risk Contribution для оценки скрытых рисков."
        )
        return ToolExecution(
            name=name, source_hint="src.finance.investment_logic",
            payload=payload, handled=True, message=result_msg
        )
    except Exception as e:
        return ToolExecution(
            name=name, source_hint="src.finance.investment_logic",
            payload=payload, handled=False,
            message=f"Критическая ошибка квант-движка: {str(e)}"
        )


# Регестр всех финтех-плагинов
FINTECH_PLUGINS = {
    "calculate_portfolio_risk_matrix": {
        "handler": execute_mac3_risk,
        "responsibility": "Количественный анализ рисков по методологии Bloomberg MAC3 (факторные модели, Euler, CVaR).",
        "source_hint": "src.finance.investment_logic",
    }
}
