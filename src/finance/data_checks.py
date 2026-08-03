"""
Чекеры качества данных (Фаза 3 плана ручного портфеля).

Три блока, три уровня строгости:

  **BLOCK**   — отчёт не строится, токен не списывается;
  **DEGRADE** — расчёт идёт, но факт попадает в CoVe и `data_lineage`;
  **PASS**    — молча.

Где что вызывается (`PHASE_00 §3c`, уточнено предвыполненным анализом Фазы 3)
────────────────────────────────────────────────────────────────────────────
| Блок | Точка вызова                    | Почему именно там                    |
|------|---------------------------------|--------------------------------------|
| A    | внутри провайдера (`fetch`)     | проверяет СЫРОЙ ответ; после `fetch` |
|      |                                 | тела уже нет — только DataFrame      |
| B    | `get_market_data`, после fetch  | нужна только матрица цен             |
| C    | слой бота, между Шагом 1 и 2    | нужны веса и СТОИМОСТЬ портфеля      |

Разделение не архитектурное украшение: без портфеля **C-8 физически не сможет
считать покрытие ПО СТОИМОСТИ** и выродится в подсчёт тикеров — ровно ловушка
Т-6, из-за которой потеря одной позиции весом 55% выглядит как «90% покрытия».

Профили строгости
─────────────────
Профиль задаёт ИСТОЧНИК ПОРТФЕЛЯ (`price_source`), а НЕ имя отработавшего
провайдера: при failover целиком по отчёту (I-13) они расходятся, а строгость
проверок должна следовать за источником портфеля.

  ``LEGACY`` (`freedom`) — «не хуже, чем сегодня»: пороги равны тем, что движок
                           реализует прямо сейчас;
  ``STRICT`` (`manual`)  — источник один, резерва нет, поэтому усечение выгрузки
                           блокирует;
  ``DEMO``               — витрина обязана рендериться всегда → не блокирует.

Важное исключение из «DEMO не блокирует»: **C-10 и C-12 блокируют во всех
профилях**.  Это проверки на БАГ В КОДЕ (сумма весов, согласованность кэша),
а не на качество данных; витрина с неверной математикой хуже отсутствующей.

Чего чекеры НЕ делают
─────────────────────
Они не дублируют то, что движок уже чинит: `_candles_to_series` снимает дубли
дат, `math_firewall` заменяет ±Inf, `calculate_structural_risk` гейтит окно,
ковариационный слой делает PSD-проекцию.  Разница в том, что движок чинит
МОЛЧА, а блок B обязан оставить след в CoVe — поэтому B-2 выполняется на СЫРОМ
ряде провайдера, до firewall.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CheckLevel(str, Enum):
    PASS = "pass"
    DEGRADE = "degrade"
    BLOCK = "block"


class CheckProfile(str, Enum):
    LEGACY = "legacy"     # freedom
    STRICT = "strict"     # manual
    DEMO = "demo"         # витрина


def profile_for_source(source) -> CheckProfile:
    """Профиль по ИСТОЧНИКУ ПОРТФЕЛЯ.

    Дефолт — ``LEGACY``: источник, не объявивший профиль, не должен получать
    более строгое поведение (I-10), иначе забытый вызов молча ужесточает гейты.
    """
    src = str(source or "").lower()
    if src == "manual":
        return CheckProfile.STRICT
    if src == "demo":
        return CheckProfile.DEMO
    return CheckProfile.LEGACY


@dataclass(frozen=True)
class CheckFinding:
    id: str
    level: CheckLevel
    message: str                      # человеческий текст, без секретов и str(exc)
    ticker: Optional[str] = None      # None — проверка портфельного уровня
    detail: dict = field(default_factory=dict)


@dataclass
class DataQualityReport:
    profile: CheckProfile
    findings: list[CheckFinding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.level is CheckLevel.BLOCK for f in self.findings)

    @property
    def blocking(self) -> list[CheckFinding]:
        return [f for f in self.findings if f.level is CheckLevel.BLOCK]

    @property
    def degraded(self) -> list[CheckFinding]:
        return [f for f in self.findings if f.level is CheckLevel.DEGRADE]

    def user_message(self) -> str:
        """Текст отказа — человеческий, без кодов проверок.

        Формат по §4.1 задания: что случилось, какой масштаб, и что токен цел.
        """
        if not self.blocked:
            return ""
        head = self.blocking[0]
        return (f"{head.message}\n\n"
                "Отчёт не построен, токен не списан.")

    def lineage_rows(self) -> list[dict]:
        """DEGRADE-строки для CoVe-панели."""
        return [{"name": f"Качество данных · {f.id}",
                 "source": "Data checks",
                 "status": "warn",
                 "note": f.message}
                for f in self.degraded]


class DataQualityBlocked(RuntimeError):
    """Блокирующее срабатывание. Обрабатывается там же, где RealPortfolioRequired."""

    def __init__(self, report: DataQualityReport) -> None:
        super().__init__(report.user_message())
        self.report = report


# ── Реестр проверок ──────────────────────────────────────────────────────────
# Чекеры — ДАННЫЕ, а не разбросанные `if` (гейт §10 задания): добавление
# проверки = добавление строки в реестр.

@dataclass(frozen=True)
class CheckSpec:
    id: str
    fn: Callable
    #: уровень по профилю; отсутствие ключа = проверка в этом профиле не применяется
    levels: dict[CheckProfile, CheckLevel]

    def level_for(self, profile: CheckProfile) -> CheckLevel:
        return self.levels.get(profile, CheckLevel.PASS)


_ALL = (CheckProfile.LEGACY, CheckProfile.STRICT, CheckProfile.DEMO)


def _levels(legacy: CheckLevel, strict: CheckLevel, demo: CheckLevel) -> dict:
    return {CheckProfile.LEGACY: legacy, CheckProfile.STRICT: strict,
            CheckProfile.DEMO: demo}


def _uniform(level: CheckLevel) -> dict:
    return {p: level for p in _ALL}


# ── Блок A — целостность выгрузки (вызывается ВНУТРИ провайдера) ─────────────

_CSV_REFUSAL_MARKERS = (
    "exceeded the daily hits limit",   # исчерпана квота Stooq
    "apikey",                          # страница получения ключа
    "<html",                           # HTML вместо CSV
    "<!doctype",
)


def check_response_body(body: str, *, expected_header: str,
                        max_bytes: int, status: int = 200,
                        redirected: bool = False) -> Optional[CheckFinding]:
    """A-1…A-3, A-5: тело ответа — это действительно CSV, а не страница отказа.

    Ключевой момент: Stooq отдаёт отказ с кодом **HTTP 200** и текстом в теле.
    ``pd.read_csv`` на таком ответе либо падает невнятно, либо разбирает мусор
    в DataFrame — поэтому проверка тела обязательна, а не желательна.
    """
    if redirected:
        return CheckFinding("A-2", CheckLevel.BLOCK,
                            "Источник перенаправил запрос — вероятно, требуется "
                            "повторная авторизация ключа.")
    if status != 200:
        return CheckFinding("A-2", CheckLevel.BLOCK,
                            f"Источник ответил кодом {status}.")
    if body is None:
        return CheckFinding("A-1", CheckLevel.BLOCK, "Пустой ответ источника.")
    raw = body.lstrip("﻿").lstrip()            # A-5: BOM
    if len(body.encode("utf-8", "ignore")) > max_bytes:
        return CheckFinding("A-3", CheckLevel.BLOCK,
                            "Ответ источника превысил допустимый размер.")
    low = raw[:400].lower()
    for marker in _CSV_REFUSAL_MARKERS:
        if marker in low:
            return CheckFinding("A-1", CheckLevel.BLOCK,
                                "Источник вернул не котировки, а служебное "
                                "сообщение (квота или ключ).")
    first = raw.split("\n", 1)[0].strip().lower().replace(" ", "")
    if first != expected_header.strip().lower().replace(" ", ""):
        return CheckFinding("A-6", CheckLevel.BLOCK,
                            "Формат ответа источника не совпадает с ожидаемым.")
    if raw.count("\n") < 1:
        return CheckFinding("A-4", CheckLevel.BLOCK,
                            "Источник вернул заголовок без данных.")
    return None


# ── Блок B — корректность ряда (вызывается после провайдера) ────────────────

_ANOMALY_LOG_RETURN = 0.4          # |log-доходность| > 0.4 ≈ ±49%
_ANOMALY_SHARE = 0.001             # >0.1% дней
_STALE_TRADING_DAYS = 5
_GAP_TRADING_DAYS = 10


def _b_index_sane(series: pd.Series, ticker: str, **_):
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex):
        return "индекс ряда не является датами"
    if idx.has_duplicates:
        return "в ряду есть повторяющиеся даты"
    if not idx.is_monotonic_increasing:
        return "ряд не отсортирован по датам"
    return None


def _b_values_finite(series: pd.Series, ticker: str, **_):
    vals = series.to_numpy(dtype=float)
    if len(vals) == 0:
        return "ряд пуст"
    if not np.all(np.isfinite(vals)):
        return "в ряду есть нечисловые значения"
    if np.any(vals <= 0):
        return "в ряду есть нулевые или отрицательные цены"
    return None


def _b_variance(series: pd.Series, ticker: str, **_):
    vals = series.to_numpy(dtype=float)
    if len(vals) > 1 and float(np.nanstd(vals)) <= 0.0:
        return "ряд постоянный — вырождает ковариацию"
    return None


def _b_fresh(series: pd.Series, ticker: str, *, as_of=None, **_):
    if series.empty:
        return None
    last = pd.Timestamp(series.index[-1])
    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.utcnow().normalize()
    gap = int(np.busday_count(last.date(), ref.date())) if ref > last else 0
    if gap > _STALE_TRADING_DAYS:
        return f"последняя котировка старше {gap} торговых дней"
    return None


def _b_gaps(series: pd.Series, ticker: str, **_):
    if len(series) < 2:
        return None
    days = np.busday_count([d.date() for d in series.index[:-1]],
                           [d.date() for d in series.index[1:]])
    worst = int(days.max()) if len(days) else 0
    if worst > _GAP_TRADING_DAYS:
        return f"в ряду разрыв {worst} торговых дней"
    return None


def _b_anomalies(series: pd.Series, ticker: str, **_):
    vals = series.to_numpy(dtype=float)
    if len(vals) < 2 or np.any(vals <= 0):
        return None
    lr = np.diff(np.log(vals))
    n_jumps = int(np.sum(np.abs(lr) > _ANOMALY_LOG_RETURN))
    share = float(n_jumps) / max(1, len(lr))
    if share > _ANOMALY_SHARE:
        # 2026-08-03: было «аномальные дневные скачки в 0.24% наблюдений» —
        # доля в процентах от числа наблюдений читателю ничего не говорит и
        # звучит как поломка данных. Проверка ЭВРИСТИЧЕСКАЯ: она не отличает
        # реальный обвал на отчётности (AAOI — волатильный small-cap, у него
        # такие дни настоящие) от неучтённого сплита. Поэтому называем ФАКТ
        # (сколько дней и какой величины) и не выносим вердикт.
        pct = (math.exp(_ANOMALY_LOG_RETURN) - 1) * 100
        return (f"{n_jumps} дн. с движением цены более {pct:.0f}% — обычно это "
                "отчётность или корпоративное действие; проверьте, если бумага "
                "обычно спокойная")
    return None


_SERIES_CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec("B-1", _b_index_sane,   _uniform(CheckLevel.BLOCK)),
    CheckSpec("B-2", _b_values_finite, _uniform(CheckLevel.BLOCK)),
    CheckSpec("B-3", _b_variance,     _uniform(CheckLevel.BLOCK)),
    CheckSpec("B-4", _b_fresh,        _uniform(CheckLevel.DEGRADE)),
    CheckSpec("B-5", _b_gaps,         _uniform(CheckLevel.DEGRADE)),
    CheckSpec("B-6", _b_anomalies,    _uniform(CheckLevel.DEGRADE)),
)


def check_price_matrix(data: pd.DataFrame, *, profile: CheckProfile,
                       as_of=None) -> DataQualityReport:
    """Блок B: корректность каждого ряда в матрице."""
    report = DataQualityReport(profile=profile)
    if data is None or data.empty:
        report.findings.append(CheckFinding(
            "B-0", CheckLevel.BLOCK if profile is not CheckProfile.DEMO
            else CheckLevel.DEGRADE,
            "Не удалось загрузить ни одного ценового ряда."))
        return report
    for ticker in data.columns:
        series = data[ticker].dropna()
        for spec in _SERIES_CHECKS:
            level = spec.level_for(profile)
            if level is CheckLevel.PASS:
                continue
            try:
                problem = spec.fn(series, str(ticker), as_of=as_of)
            except Exception as exc:                      # noqa: BLE001
                logger.debug("Чекер %s упал на %s: %s", spec.id, ticker, exc)
                continue
            if problem:
                report.findings.append(CheckFinding(
                    spec.id, level, f"{ticker}: {problem}", ticker=str(ticker)))
                if level is CheckLevel.BLOCK:
                    break        # тикер уже забракован — остальное не уточняет
    return report


# ── Блок C — достаточность для математики (вызывается в слое бота) ─────────

#: 10 факторных ETF + 3 бенчмарка = 13 обязательных инструментов
FACTOR_ETFS = ("SPY.US", "MTUM.US", "VLUE.US", "QUAL.US", "IWM.US", "SPLV.US",
               "DBC.US", "IEF.US", "EEM.US", "EMB.US")
BENCHMARK_ETFS = ("QQQ.US", "AGG.US", "URTH.US")
MARKET_FACTOR = "SPY.US"

#: эталоны с заведомо длинной историей — по ним отличают УСЕЧЕНИЕ выгрузки
#: от молодого листинга (PHASE_00 §3.3)
TRUNCATION_REFERENCES = ("SPY.US", "AGG.US")

_WINDOW_YEAR = 252
_COVERAGE_BLOCK = 0.90
_COVERAGE_DEGRADE = 0.98
_BOOTSTRAP_MIN = 500
_TRUNCATION_SHARE = 0.20


def _is_cash(ticker) -> bool:
    """Денежная позиция — по SSOT `asset_taxonomy` (A-5), не по локальному списку.

    Ф-3 (2026-08-02): здесь лежала СЕДЬМАЯ копия классификации — кортеж
    `("USD","EUR","KZT","RUB","CASH")`, в котором не было `GBP`/`CHF`/`CNY`/
    `JPY`/`RUR`. После −37 строка кэша в такой валюте реальна, и она попадала
    сразу в две ловушки: считалась РИСКОВЫМ активом (C-5/C-7) и НЕПОКРЫТОЙ
    стоимостью (C-8) — то есть могла заблокировать корректный отчёт.
    """
    try:
        from finance.asset_taxonomy import is_cash
    except Exception:                                   # pragma: no cover
        return str(ticker).upper() in ("USD", "EUR", "KZT", "RUB", "CASH")
    return is_cash(str(ticker))


def _effective_window(data) -> int:
    """Окно, которое ДЕЙСТВИТЕЛЬНО использует движок (общие даты после
    выброса разреженных бумаг).

    Порог берётся из `period_returns.MIN_OVERLAP_TDAYS` — того же места, что
    импортирует движок, поэтому разойтись они не могут. Фолбэк на строгое
    пересечение оставлен на случай недоступности модуля: он консервативен
    (занижает окно), то есть в худшем случае даст лишнее предупреждение, а не
    пропустит реальную проблему.
    """
    if data is None or getattr(data, "empty", True):
        return 0
    try:
        from finance.period_returns import MIN_OVERLAP_TDAYS
    except Exception:                                   # pragma: no cover
        return int(len(data.dropna(axis=0, how="any")))
    dense = [c for c in data.columns
             if int(data[c].notna().sum()) >= MIN_OVERLAP_TDAYS]
    if not dense:
        return 0
    rows = int(len(data[dense].dropna(axis=0, how="any")))
    # Движок считает на ЛОГ-ДОХОДНОСТЯХ: `np.log(data / data.shift(1)).dropna()`
    # теряет первую строку. Вычитаем её и здесь, иначе чекер и движок вечно
    # расходятся на единицу («200» в отчёте против 199 в модели).
    return max(0, rows - 1)


def _engine_min_obs(n_factors: int) -> int:
    """Порог движка: ниже него структурная модель не строится в принципе."""
    return max(2 * max(int(n_factors), 1), 30)


def check_portfolio_sufficiency(
    *, data: pd.DataFrame, weights: dict, values: dict,
    profile: CheckProfile, requested_days: Optional[int] = None,
    unconvertible_currencies: Iterable[str] = (),
) -> DataQualityReport:
    """Блок C: хватает ли данных, чтобы математика была осмысленной.

    ``weights`` — доли позиций (могут суммироваться в 1 при ОТРИЦАТЕЛЬНОЙ
    денежной компоненте: маржинальный заём — норма, а не дефект).
    ``values``  — рыночная стоимость позиций; нужна для C-8, который считает
    покрытие ПО СТОИМОСТИ, а не по числу тикеров.
    """
    report = DataQualityReport(profile=profile)
    demo = profile is CheckProfile.DEMO
    strict = profile is CheckProfile.STRICT

    def add(cid: str, level: CheckLevel, message: str, **detail):
        # DEMO не блокируется — КРОМЕ проверок внутренней согласованности,
        # которые сигнализируют о баге в коде, а не о плохих данных.
        if demo and level is CheckLevel.BLOCK and cid not in ("C-10", "C-12"):
            level = CheckLevel.DEGRADE
        report.findings.append(CheckFinding(cid, level, message, detail=detail))

    cols = set(map(str, data.columns)) if data is not None else set()

    # C-1 — обязательные инструменты
    missing_factors = [t for t in FACTOR_ETFS if t not in cols]
    missing_bench = [t for t in BENCHMARK_ETFS if t not in cols]
    if missing_factors:
        market_lost = MARKET_FACTOR in missing_factors
        all_lost = len(missing_factors) == len(FACTOR_ETFS)
        # LEGACY повторяет поведение движка: он переживает потерю ЧАСТИ факторов
        # и падает только если не доехало ни одного.  STRICT строже — у ручного
        # портфеля источник один, и дыра означает отказ источника.
        level = (CheckLevel.BLOCK if (strict or market_lost or all_lost)
                 else CheckLevel.DEGRADE)
        add("C-1", level,
            f"Не загружены факторные индексы: {', '.join(missing_factors)}.",
            missing=missing_factors)
    if missing_bench:
        add("C-1", CheckLevel.DEGRADE,
            f"Не загружены бенчмарки: {', '.join(missing_bench)}.",
            missing=missing_bench)

    # C-2 / C-3 — эффективное общее окно.  Пороги ЗЕРКАЛЯТ движок, а не
    # назначаются: max(2K,30) — граница, ниже которой модель не строится;
    # 252 = engine.trading_days — граница, где гаснет форвардная панель (F-14).
    #
    # 🔴 2026-08-03: окно считалось как СТРОГОЕ ПЕРЕСЕЧЕНИЕ всех колонок
    # (`dropna(how="any")`) — и одна молодая бумага обнуляла его для всей
    # книги.  Живой DEEP 03.08: движок построил модель на ПОЛНОМ окне
    # (покрытие факторами 100%, MaxDD −41.3%, Sharpe 0.49 — это сотни
    # наблюдений), а чекер доложил «Наблюдений (37) мало для 13 активов».
    # 37 — это история самой молодой бумаги книги (AAOI), и ровно она
    # схлопывала пересечение.  Отсюда ТРИ ложных предупреждения сразу:
    # C-3, C-5 и C-13.
    #
    # Это повторение ошибки, которую движок уже прошёл: F-15/F-21 отказались
    # от row-dropna по всем именам именно потому, что «один листинг с историей
    # < 60 дней схлопывал общее окно» (`portfolio_series.py`).  Движок сначала
    # ВЫБРАСЫВАЕТ разреженные бумаги (`MIN_OVERLAP_TDAYS`), и только потом
    # берёт пересечение — чекер обязан делать ТО ЖЕ САМОЕ, иначе он меряет
    # величину, которой в расчёте не существует.
    effective = _effective_window(data)
    n_factors = len([t for t in FACTOR_ETFS if t in cols])
    floor = _engine_min_obs(n_factors)
    if effective < floor:
        add("C-2", CheckLevel.BLOCK,
            f"Общая история слишком коротка ({effective} дней): "
            "факторная модель не строится.", window=effective, floor=floor)
    elif effective < _WINDOW_YEAR:
        add("C-3", CheckLevel.DEGRADE,
            f"Общая история короче года ({effective} дней) — прогнозные "
            "оценки не публикуются.", window=effective)

    # C-4 — усечение выгрузки vs молодой листинг.
    # Признак ПАНЕЛЬНЫЙ: если история коротка и у эталонов, которые торгуются
    # десятилетиями, значит усечён ИСТОЧНИК, а не бумага молодая.  По-тикерная
    # проверка этого не различает (тарифное ограничение даёт полный, но
    # короткий ряд у ВСЕХ инструментов).
    if requested_days and data is not None and not data.empty:
        refs = [t for t in TRUNCATION_REFERENCES if t in cols]
        if refs:
            ref_len = max(int(data[t].dropna().shape[0]) for t in refs)
            expected = requested_days * 0.69          # кал. дни → торговые
            if expected > 0 and ref_len < expected * (1 - _TRUNCATION_SHARE):
                add("C-4", CheckLevel.BLOCK if strict else CheckLevel.DEGRADE,
                    "Источник вернул укороченную историю "
                    f"({ref_len} дней вместо ожидаемых ~{int(expected)}).",
                    reference_len=ref_len)

    # C-5 — обусловленность: наблюдений против числа рисковых активов
    risky = [t for t, w in weights.items() if abs(float(w or 0)) > 0
             and not _is_cash(t)]
    n_risky = len(risky)
    if n_risky and effective < 3 * n_risky:
        add("C-5", CheckLevel.BLOCK if strict else CheckLevel.DEGRADE,
            f"Наблюдений ({effective}) мало для {n_risky} активов — "
            "оценки риска неустойчивы.", obs=effective, assets=n_risky)

    # C-7 — диверсификации нет, TRC не определён.
    #
    # Ф-3 (2026-08-02, калибровка при подключении): уровень стал ЗАВИСЕТЬ ОТ
    # ПРОФИЛЯ.  Раньше стоял безусловный BLOCK, и это прямо противоречило
    # принципу самого профиля LEGACY — «не хуже, чем сегодня: пороги равны тем,
    # что движок реализует прямо сейчас».  Сегодня движок СТРОИТ отчёт по книге
    # из одной рисковой позиции (TRC там тривиально 100%), поэтому в LEGACY это
    # предупреждение, а не отказ: подключение чекеров не имеет права отнять у
    # живого пользователя отчёт, который он получал вчера.
    # В STRICT (ручной ввод) остаётся BLOCK: там пользователь сам набрал состав,
    # и попросить вторую позицию дешевле, чем отдать бессмысленную декомпозицию.
    if n_risky < 2:
        add("C-7", CheckLevel.BLOCK if strict else CheckLevel.DEGRADE,
            "В портфеле меньше двух рисковых позиций — "
            "распределение риска не рассчитывается.", risky=n_risky)

    # C-8 / C-9 — ПОКРЫТИЕ ПО СТОИМОСТИ.
    # Считается по деньгам, а не по числу тикеров: портфель из 10 бумаг, где не
    # загрузилась одна весом 55%, имеет 90% покрытия по счёту и 45% по стоимости.
    # Первая метрика пропустит катастрофу, вторая остановит (ловушка Т-6).
    total_value = sum(abs(float(v or 0)) for v in values.values())
    covered_value = sum(abs(float(values.get(t) or 0)) for t in values
                        if str(t) in cols or _is_cash(t))
    coverage = (covered_value / total_value) if total_value > 0 else 0.0
    if total_value > 0:
        if coverage < _COVERAGE_BLOCK:
            uncovered = [t for t in values
                         if str(t) not in cols
                         and not _is_cash(t)]
            add("C-8", CheckLevel.BLOCK,
                f"Не удалось загрузить цены по {len(uncovered)} из {len(values)} "
                f"бумаг ({(1 - coverage) * 100:.0f}% стоимости портфеля).",
                coverage=round(coverage, 4), uncovered=uncovered)
        elif coverage < _COVERAGE_DEGRADE:
            add("C-9", CheckLevel.DEGRADE,
                f"Цены покрывают {coverage * 100:.1f}% стоимости портфеля.",
                coverage=round(coverage, 4))

    # C-10 — арифметика весов.  БЛОКИРУЕТ ВО ВСЕХ ПРОФИЛЯХ: это признак бага
    # в коде, а не плохих данных.
    #
    # Проверяется именно СУММА, а не знак каждого веса: на портфеле с
    # маржинальным займом денежная компонента ОТРИЦАТЕЛЬНА (живой отчёт:
    # USD −7.45%, сумма ровно 100%), и наивное «все веса ≥ 0» забраковало бы
    # совершенно корректную книгу.
    if weights:
        total_w = sum(float(w or 0) for w in weights.values())
        if not math.isclose(total_w, 1.0, abs_tol=1e-6) and \
           not math.isclose(total_w, 100.0, abs_tol=1e-4):
            add("C-10", CheckLevel.BLOCK,
                "Внутренняя ошибка расчёта долей портфеля.",
                total=round(total_w, 8))

    # C-12 — кэш вне рисковых расчётов, но в знаменателе долей.  Тоже про баг.
    cash_keys = [t for t in weights if _is_cash(t)]
    if cash_keys and any(str(t) in cols for t in cash_keys):
        add("C-12", CheckLevel.BLOCK,
            "Внутренняя ошибка: денежная позиция попала в факторную модель.",
            cash=cash_keys)

    # C-11 — валюты без курса
    unconvertible = [str(c) for c in unconvertible_currencies if str(c)]
    if unconvertible:
        add("C-11", CheckLevel.DEGRADE,
            "Нет курса для валют: " + ", ".join(unconvertible) + ".",
            currencies=unconvertible)

    # C-13 — окно для Bootstrap-CVaR
    if 0 < effective < _BOOTSTRAP_MIN:
        add("C-13", CheckLevel.DEGRADE,
            f"Наблюдений ({effective}) мало для устойчивой оценки хвостового "
            "риска — доверительный интервал широкий.", window=effective)

    return report


#: C-6: насколько отрицательным должно быть собственное значение, чтобы это
#: считалось вырождением, а не арифметическим шумом float64.  Порог на порядки
#: выше `floor=1e-12` самой проекции: клипование на уровне 1e-15 — рутина
#: округления и поводом для сообщения не является.
_PSD_NOISE_TOLERANCE = 1e-8


def check_covariance_health(psd_repair: Optional[dict], *,
                            profile: CheckProfile) -> DataQualityReport:
    """C-6: факторная ковариация оказалась вырожденной и была починена.

    Отдельная функция, а не строка в `check_portfolio_sufficiency`, потому что
    вход появляется ПОЗЖЕ: матрица считается в `calculate_structural_risk`,
    уже после того, как известны веса.  Сигнатуры существующих чекеров при
    этом не тронуты (56 тестов Фазы 3 продолжают действовать как есть).

    Движок чинит вырождение сам (`_nearest_psd`, Higham-клипование) — и это
    правильно: отчёт лучше построить.  Но чинит МОЛЧА, а `PHASE_03 §3a.4`
    прямо требует: «чекеры не должны ПОВТОРЯТЬ то, что движок уже чинит, они
    должны СООБЩАТЬ об этом».  Отрицательное собственное значение бленда
    EWMA⊕Ledoit-Wolf означает, что факторы почти линейно зависимы, а значит
    беты и Euler-вклады на этом прогоне неустойчивы — читатель обязан знать.

    STRICT блокирует: у ручного портфеля источник один, и вырожденная матрица
    там означает негодные данные, а не редкий край рынка.
    """
    report = DataQualityReport(profile=profile)
    if not isinstance(psd_repair, dict):
        return report
    try:
        min_eig = float(psd_repair.get("min_eigenvalue"))
    except (TypeError, ValueError):
        return report
    if not math.isfinite(min_eig) or min_eig >= -_PSD_NOISE_TOLERANCE:
        return report                     # шум округления — не находка
    level = (CheckLevel.BLOCK if profile is CheckProfile.STRICT
             else CheckLevel.DEGRADE)
    n_clipped = int(psd_repair.get("n_clipped") or 0)
    size = int(psd_repair.get("size") or 0)
    report.findings.append(CheckFinding(
        "C-6", level,
        f"Факторная ковариация вырождена ({n_clipped} из {size} направлений "
        "имели отрицательную дисперсию) — матрица восстановлена, но оценки "
        "бет и вкладов в риск на этом прогоне менее устойчивы.",
        detail={"min_eigenvalue": min_eig, "n_clipped": n_clipped, "size": size}))
    return report


def merge_reports(*reports: DataQualityReport) -> DataQualityReport:
    """Слить отчёты блоков в один (профиль берётся у первого)."""
    reports = [r for r in reports if r is not None]
    if not reports:
        return DataQualityReport(profile=CheckProfile.LEGACY)
    out = DataQualityReport(profile=reports[0].profile)
    for r in reports:
        out.findings.extend(r.findings)
    return out


__all__ = [
    "BENCHMARK_ETFS",
    "CheckFinding",
    "CheckLevel",
    "CheckProfile",
    "DataQualityBlocked",
    "DataQualityReport",
    "FACTOR_ETFS",
    "check_covariance_health",
    "check_portfolio_sufficiency",
    "check_price_matrix",
    "check_response_body",
    "merge_reports",
    "profile_for_source",
]
