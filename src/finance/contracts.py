"""SSOT ФОРМЫ данных, пересекающих слои конвейера (Арх-2).

Роль : описать контракт `analyze_all` → потребители ОДИН раз, здесь.
Слой : **L0 (cross)** — импортирует только stdlib/typing и НЕ импортирует
       ничего из `finance`, `pdf_payload`, `tg_bot`. Никакой логики: это
       файл-описание, а не файл-поведение.
Док  : `docs/ARCHITECTURE_FOR_AGENTS.md` §3.2 (три контрактные границы).

Зачем
-----
`analyze_all` возвращает словарь из 35 ключей; его читают 9 модулей. До этого
файла форму можно было узнать ТОЛЬКО прочитав место сборки внутри функции на
1 240 строк, а опечатка в ключе у потребителя давала не ошибку, а тихое `None`
→ «—» в отчёте. Так уже терялись данные:

* **R-6** — `actionPlan[].reason` считался движком и молча терялся маппером;
* **D-5** — в одном пайплайне ДВА разных `data_quality` (см. предупреждение
  у `AnalyzeResults.data_quality`), и число одного смысла прочитали в
  контексте другого.

Как пользоваться
----------------
* Новый ключ в `results{}` → СНАЧАЛА строка здесь, потом сборка в движке.
  Тест `tests/test_contracts_results.py` падает, если множества разошлись.
* Тип-чекер (mypy) в проекте НЕ запущен: ценность файла — самодокументация
  плюс исполняемый AST-гейт. Аннотации намеренно консервативны — там, где
  форма вложенного блока не зафиксирована в одном литерале, стоит `Any`
  и ссылка на место сборки, а не выдуманная структура.

Инварианты, проверенные по коду (2026-08-04)
--------------------------------------------
* У `analyze_all` РОВНО ОДИН контрактный `return` (остальные `return` в файле
  принадлежат вложенным хелперам), поэтому все 35 ключей присутствуют ВСЕГДА
  — отсюда `total=True`. Отсутствие данных выражается ЗНАЧЕНИЕМ (`None`,
  пустой словарь/список), а не отсутствием ключа.
* Частичные словари в моках (`src/report_mocks.py`, смоук-мок в
  `html_renderer`) и в тестах контрактом НЕ являются — они специально урезаны
  и под эту аннотацию не подпадают.
* `html_renderer` и `premium_payload` в списке потребителей НЕТ, хотя имена
  ключей у них встречаются: первый строит смоук-МОК такой же формы, второй
  читает уже payload. Именно это совпадение имён завышало прежний замер
  «17 потребителей» — считать надо обращения к пришедшему словарю, а не строки.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

__all__ = [
    "AnalyzeResults",
    "LeverageMetrics",
    "DataQualityBlock",
    "DataQualityFinding",
    "ModelUncovered",
    "MergedPosition",
    "DroppedRow",
    "FxConvertedRow",
    "AssetScore",
    "TechnicalReading",
    "RESULTS_KEYS",
    "PORTFOLIO_METRICS_KEYS",
]


# ── вложенные блоки, форма которых зафиксирована одним литералом ──────────────

class LeverageMetrics(TypedDict):
    """`investment_logic.analyze_all` — плечо и экспозиция книги.

    `gross_exposure` — Σ|w| по ПОЗИЦИЯМ без кэша: маржа это финансирование, а не
    экспозиция (R-7). `is_leveraged` — строгий порог `cash_weight < -0.001`,
    он гасит float-шум на полностью профинансированной книге.
    """
    gross_exposure: float
    net_exposure: float
    long_weight: float
    cash_weight: float
    leverage_ratio: float
    is_leveraged: bool


class DataQualityFinding(TypedDict):
    """Одна находка чекеров `finance/data_checks.py` (уровень DEGRADE).

    `id` — внутренний код проверки (`B-6`, `C-11`, …). **В текст для
    пользователя коды не выводятся** (`PHASE_03 §4.1`, дефект D-3): отчёт
    говорит темами проверок, а не их номерами.
    """
    id: str
    level: str
    message: str
    ticker: Optional[str]


class DataQualityBlock(TypedDict):
    """🔴 НЕ ПУТАТЬ с `payload["data_quality"]`.

    Здесь (движок)          — находки чекеров качества данных: `{profile, findings}`.
    В payload (`pdf_payload`) — доля УСПЕШНО ЗАГРУЖЕННЫХ фактор-СЕРИЙ:
    `{factors_loaded, factors_total}`.

    Совпадение имён на соседних границах конвейера уже дало дефект **D-5**:
    «покрытие 100%» напечатали рядом с факторной таблицей, хотя это была доля
    загруженных серий, а 5.31% книги бет не имели вовсе. Для НОВЫХ ключей
    такие совпадения запрещены (`ARCHITECTURE_FOR_AGENTS.md §3.2`).

    Уровень BLOCK сюда не попадает: он поднимает `DataQualityBlocked` выше по
    стеку, и отчёта не будет вовсе (токен при этом не списывается).
    """
    profile: Optional[str]
    findings: List[DataQualityFinding]


class ModelUncovered(TypedDict):
    """Доля книги ВНЕ факторной модели (короткая история котировок).

    Считается там, где известны и исключения sparse-гейта, и веса. Отчёт
    обязан назвать и процент, и имена: иначе «покрытие 100%» читается как
    «портфель покрыт полностью» (D-5).
    """
    weight_pct: float
    names: List[str]


class MergedPosition(TypedDict):
    """A-1: сколько строк входного фрейма склеено в одну позицию."""
    ticker: str
    rows: int


class DroppedRow(TypedDict):
    """A-2: отброшенная строка портфеля и ПРИЧИНА.

    Молча терять позицию нельзя — отчёт обязан её назвать.
    """
    ticker: str
    reason: str


class FxConvertedRow(TypedDict, total=False):
    """−37: строка, реально переведённая в базовую валюту, и по какому курсу.

    Реестр заполняется по ФАКТУ конверсии, а не по несовпадению валют (D-2):
    при отсутствии курса строка остаётся в своей валюте и сюда НЕ попадает —
    иначе карточка заявит пересчёт, которого не было.
    """
    ticker: str
    currency: str
    rate: float


class TechnicalReading(TypedDict):
    """Пиллар T: `finance/technicals.py` — RSI/MACD/Bollinger/SMA/momentum."""
    score: float
    raw: Dict[str, Any]
    components: Dict[str, Any]


class AssetScore(TypedDict):
    """4-Pillar скор одной бумаги (`finance/scoring.py`).

    `*_applicable=False` — пиллар НЕПРИМЕНИМ, а не «ноль»: у суверенного долга
    DM нет кредитного риска эмитента, у плечевой ETP-обёртки нет фундаментала.
    Ноль и «неприменимо» — разные вещи, отчёт обязан их различать.
    """
    fundamentals: float
    valuations: float
    technicals: float
    credit: float
    total: float
    action: str
    hotspot: bool
    credit_applicable: bool
    fundamentals_applicable: bool


# ── сам контракт ─────────────────────────────────────────────────────────────

class AnalyzeResults(TypedDict):
    """Возврат `UniversalPortfolioManager.analyze_all` — 35 ключей.

    Потребители — **9 модулей** (замер 2026-08-04, по обращениям к ключам):
    `pdf_payload` (30) · `ai_narrative` (15) · `finance/data_lineage` (15) ·
    `agent/advisor_bot` (6) · `finance/portfolio_series` (6) · `tg_bot` (5) ·
    `agent/gatekeeper` (4) · `finance/scenario_report` (4) · `report_charts` (3).

    `report_charts` появился в списке после Арх-5: билдеры графиков переехали
    туда из `tg_bot` и унаследовали его чтения (`tg_bot` при этом 6 → 5). Сам
    список пришлось править ОТДЕЛЬНЫМ раундом, потому что он устарел в момент
    того же коммита — иллюстрация правила «замер без исполняемого гейта
    разъезжается».

    `total=True`: ключи присутствуют всегда, отсутствие данных — это значение
    (`None` / пустой контейнер). Значения НЕ гарантированы непустыми.
    """

    # ── позиции и стоимость ──────────────────────────────────────────────────
    performance_table: Any            # pd.DataFrame: строка на позицию (+ Ticker в индексе)
    risk_matrix: Any                  # pd.DataFrame: структурная ковариация Σ = B·F·Bᵀ + D
    total_portfolio_pnl: float
    total_value: float                # в `reporting_currency`
    reporting_currency: str

    # ── риск ─────────────────────────────────────────────────────────────────
    # 26 ключей, перечислены в PORTFOLIO_METRICS_KEYS; собираются в
    # `calculate_structural_risk` (строка ~1548).  🔴 На пяти ранних выходах
    # этой функции (мало данных / нет пересечения окон) возвращается ПУСТОЙ
    # словарь — потребитель ОБЯЗАН это учитывать, а не рассчитывать на ключи.
    portfolio_metrics: Dict[str, Any]
    leverage_metrics: LeverageMetrics
    port_log_returns: Any             # pd.Series | None — композитная серия портфеля (F-21)
    risk_mandate: str                 # CONSERVATIVE / MODERATE / AGGRESSIVE (H4)
    risk_free_rate: float             # годовая, в валюте отчёта

    # ── бенчмарк и периоды ───────────────────────────────────────────────────
    benchmark_comparison: Any
    profile_benchmark_ticker: Optional[str]   # выбор клиента, напр. 'QQQ.US'
    benchmark_factor_profile: Optional[Dict[str, Any]]  # B1: {ticker, name, betas}
    period_returns_table: Any
    return_series_coverage: Any

    # ── честность данных (всё это ОБЯЗАНО доезжать до отчёта) ────────────────
    proxy_substitutions: Dict[str, str]   # H-1: {оригинал: прокси}; влияет ТОЛЬКО на риск
    priced_at_cost: List[str]             # нет ни своей истории, ни цены брокера → P&L = 0
    fx_converted_rows: List[FxConvertedRow]
    model_uncovered: ModelUncovered
    data_quality: DataQualityBlock        # 🔴 см. предупреждение в DataQualityBlock
    merged_positions: List[MergedPosition]
    dropped_rows: List[DroppedRow]

    # ── аналитические слои ───────────────────────────────────────────────────
    stress_scenarios: Any
    expected_effect: Any                  # «до/после» плана; см. finance/simulate.py
    cds_summary: Any
    history_result: Any
    sector_exposure: Any
    regime: Optional[Dict[str, Any]]      # 4 квадранта + confidence (проценты, не доли)
    macro_drivers: Any
    technicals: Dict[str, TechnicalReading]
    asset_scores: Dict[str, AssetScore]
    black_litterman: Any
    action_plan: Any
    smart_money: Any

    # ⚠️ МЁРТВЫЙ КЛЮЧ (замер 2026-08-04): `factor_scores` производится движком,
    # но не читается НИКЕМ — ни в `src/`, ни в тестах, ни в шаблонах.
    # Оставлен намеренно: удаление ключа меняет контракт и это отдельное
    # решение владельца, а не побочный эффект типизации.
    factor_scores: Any


#: Ключи контракта. Держится в синхроне с `analyze_all` тестом
#: `tests/test_contracts_results.py` (сверка по AST, а не по вере).
RESULTS_KEYS: frozenset[str] = frozenset(AnalyzeResults.__annotations__)

#: Ключи `results["portfolio_metrics"]` — отдельным множеством, потому что блок
#: собирается в другой функции и на части путей приходит ПУСТЫМ.
PORTFOLIO_METRICS_KEYS: frozenset[str] = frozenset({
    "Total_Volatility_Ann", "Annualised_Return", "Sharpe_Ratio", "Sortino_Ratio",
    "VaR_95_Daily", "CVaR_95_Daily", "CVaR_95_Bootstrap", "Max_Drawdown",
    "Max_Euler_Risk_Pct", "Composite_Risk_Score", "Positive_Days_Pct",
    "realized_window_days", "var_reliability", "volatility_ci",
    "beta_standard_errors", "beta_reliability", "beta_shrinkage_applied",
    "reporting_currency", "risk_free_rate_annual", "risk_free_rate_daily",
    "risk_free_rate_source", "fx_conversion", "stress_test_disclaimer",
    "sharpe_basis_note", "factor_diagnostics", "factor_decomposition",
})
