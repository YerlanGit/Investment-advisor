"""Гибридный портфель: брокер Freedom + ручной ввод → один отчёт.

<!-- nav | area:hybrid | code:src/portfolio_aggregation/ | read-before:перед правкой агрегированного отчёта, fallback на ручной ввод или правки сохранённого ручного портфеля -->

Задание: `pico_prompt_hybrid_architecture_fallback_v2` (решения владельца
D-1…D-12, инвариант I-15 в `docs/roadmap/manual_portfolio/PHASE_00_CONTRACT.md`).

Что живёт в пакете
------------------
* `sources`    — `FreedomSource` / `ManualSource` → `SourceResult` (мок брокера
  превращается в `ok=False` и наружу не выходит);
* `gate`       — I-15: агрегированный отчёт только после ЖИВОГО фетча по ключам
  vault в этом же запросе;
* `aggregator` — `PortfolioAggregator.merge`: гейт → проверка валют → concat;
* `edits`      — Add / Remove / Trim сохранённого ручного портфеля;
* `flags`      — флаг гибрида и лимиты, читаются функциями.

🔴 Математики здесь нет и быть не должно. Склейку дублей (VWAP, кэш по валюте)
делает движок (`_normalize_positions`), цены — `provider_for_source`, всё
остальное — `analyze_all` без единой правки. Пакет только собирает ВХОДНОЙ
фрейм того же контракта, что у брокера и у ручного ввода.

Слой: L1 (данные) + чистые функции. Не знает `user_id`, не импортирует
`tg_bot`, vault и БД — ключи и текст передаёт L4.
"""

from .aggregator import (AggregateResult, AggregationRefused, PortfolioAggregator,
                         unpriced_positions)
from .edits import (EditResult, apply_edit, canonical_text, entries_of,
                    remove_at, version_tag)
from .flags import (FLAG_ADMINS, FLAG_OFF, FLAG_ON, HYBRID_PORTFOLIO_ENV,
                    aggregated_max_positions, broker_fetch_budget_s, hybrid_flag_on,
                    manual_max_positions, rollout_mode)
from .gate import (AGGREGATED_SOURCE, AggregatedNotPermitted, aggregated_manager,
                   live_broker_proof_failure, require_live_broker_fetch)
from .sources import (FAILURE_REASONS, count_positions, KEY_ORIGIN_ADMIN_SERVICE, KEY_ORIGIN_VAULT,
                      FreedomSource, ManualSource, PortfolioSource, SourceResult,
                      load_with_budget)

__all__ = [
    "AGGREGATED_SOURCE",
    "AggregateResult",
    "AggregatedNotPermitted",
    "AggregationRefused",
    "EditResult",
    "FLAG_ADMINS",
    "FLAG_OFF",
    "FLAG_ON",
    "FAILURE_REASONS",
    "FreedomSource",
    "HYBRID_PORTFOLIO_ENV",
    "KEY_ORIGIN_ADMIN_SERVICE",
    "KEY_ORIGIN_VAULT",
    "ManualSource",
    "PortfolioAggregator",
    "PortfolioSource",
    "SourceResult",
    "aggregated_manager",
    "aggregated_max_positions",
    "apply_edit",
    "broker_fetch_budget_s",
    "canonical_text",
    "count_positions",
    "entries_of",
    "hybrid_flag_on",
    "live_broker_proof_failure",
    "load_with_budget",
    "manual_max_positions",
    "remove_at",
    "require_live_broker_fetch",
    "rollout_mode",
    "unpriced_positions",
    "version_tag",
]
