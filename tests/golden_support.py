"""Детерминированный прогон `analyze_all` + нормализация результата (Арх-3.0).

Зачем это существует
--------------------
Фаза Арх-3 режет функцию на 1 240 строк, у которой шесть каналов НЕВИДИМОГО
состояния (движок пишет `_last_*`, стадии читают через `getattr`). Ошибка
области видимости при таком разрезе не даёт исключения — она тихо меняет
число в отчёте. Единственная защита, которая это ловит, — снимок ВСЕГО
результата, сверяемый после каждого шага.

Модуль намеренно лежит в `tests/`, а не в `scripts/`: деплой-образ копирует
`src/` и `tests/`, но НЕ `scripts/` — тест, который умеет проверить фикстуру
только при наличии `scripts/`, в образе бы пропускался, то есть не работал бы
там, где важнее всего.

Имя без префикса `test_` — pytest такие файлы не собирает, это библиотека.

Что гарантирует детерминизм
---------------------------
1. **Цены синтетические и сеяные** (`numpy.default_rng` с фиксированными
   сидами) — приём взят из уже работающего `test_phase26_report_fixes.py`.
2. **Сеть замокана целиком**: рыночные данные, SEC, FRED, CDS, инсайдеры.
   Если в движке появится НОВЫЙ сетевой вызов, снимок разъедется — и это
   правильный сигнал, а не помеха.
3. **Env закреплены** (`ENV_PINS`): окно истории, ортогонализация, ставка.
   `get_rfr_for_currency` читает только окружение, поэтому ставка фиксируется
   переменной, а не сетью.
4. В движке НЕТ `datetime.now()` — проверено сканом; «сегодня» в снимок не
   протекает.
5. Бутстрап CVaR сеется SHA-256 от самих данных (`investment_logic` §Sprint-3
   #10), то есть одинаков между запусками и версиями Python.

Что покрывает состав книги
--------------------------
Портфель подобран так, чтобы «слепых» ключей осталось как можно меньше —
ключ с пустым значением бесполезен: его ПОТЕРЮ при разрезе снимок не заметит.
Замер: слепых ключей **1 из 35** (был 8).

| Приём | Какой ключ оживает |
|---|---|
| дубль `AAPL.US` | `merged_positions` (склейка A-1) |
| нечисловое `Quantity` у `BADQ.US` | `dropped_rows` (гард A-2) |
| AIX-нота без своей истории | `proxy_substitutions`, `priced_at_cost` (H-1) |
| `YOUNG.US` котируется 40 дней | `model_uncovered` (sparse-guard, F-6) |
| `FRED_API_KEY=""` вместо мока | `macro_drivers` — реальный пак со статусом «missing» |
| инъекция `fetch` в инсайдеров | `smart_money` — реальная логика на сеяных данных |

Осознанное ограничение фикстуры v1 (честно, чтобы не считать её всеохватной)
---------------------------------------------------------------------------
* `fx_converted_rows` остаётся пустым: `get_market_data` подменяется целиком,
  поэтому валютный слой внутри загрузки (`_apply_fx_conversion`) не
  исполняется, а книга ведётся в одной валюте. Покрытие FX — фикстура v2.
* RAG/ИИ в `analyze_all` не участвуют вовсе — это слой отчёта.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "results_golden.json"

#: Округление float перед сериализацией.
#: 10 значащих цифр — компромисс, выбранный осознанно: любое РЕАЛЬНОЕ изменение
#: логики (переставленная стадия, потерянный ключ, сдвинутое окно) двигает
#: числа несопоставимо сильнее, а последний бит мантиссы может отличаться
#: между сборками BLAS и версиями numpy. Строгое сравнение repr(float) сделало
#: бы тест хрупким в CI без выигрыша в чувствительности.
FLOAT_SIGNIFICANT_DIGITS = 10

#: Порог, после которого табличные данные сохраняются ДАЙДЖЕСТОМ, а не целиком.
#: Смысл: `history_result` тащит в снимок всю матрицу цен (1300×17 ≈ 22 000
#: чисел) — это ВХОД прогона, он не может измениться от рефакторинга
#: `analyze_all`, но раздувает фикстуру в десять раз и делает диф нечитаемым.
#: Дайджест сохраняет чувствительность (любое изменение меняет SHA-256) и
#: убирает шум. Мелкие таблицы — perf-таблица, ковариация — пишутся ПОЛНОСТЬЮ,
#: потому что именно их диф нужно читать глазами при расхождении.
DIGEST_ROWS_THRESHOLD = 250

#: Окружение прогона. Пиним всё, что влияет на числа.
ENV_PINS: dict[str, str] = {
    "HISTORY_LOOKBACK_DAYS": "1825",
    "FACTOR_ORTHOGONALIZE": "1",
    "REPORTING_CURRENCY": "USD",
    "RFR_USD": "0.045",          # get_rfr_for_currency читает env, не сеть
    "CDS_DISABLED": "0",         # ветка исполняется, но провайдер замокан
    "REGIME_MACRO_OVERLAY": "1",
    # Пустой ключ FRED — НЕ мок, а штатная ветка: `MacroFeed._get_one` без
    # ключа детерминированно отдаёт полный пак со статусом "missing" и в сеть
    # не ходит. Так покрывается РЕАЛЬНЫЙ код, а не заглушка.
    "FRED_API_KEY": "",
}

N_DAYS = 1300         # ≈5 лет торговых дней: столько же, сколько даёт
                      # прод-окно 1825 календарных. На коротком окне чекеры
                      # C-4/C-13 уводят фикстуру в деградацию, и эталон
                      # закрепил бы нерепрезентативное состояние.
_START = "2023-01-02"

#: Портфель фикстуры. Состав подобран так, чтобы задеть разные ветви:
#: крупная акция, вторая акция того же сектора (концентрация), бондовый ETF
#: (кредитный пиллар неприменим) и строка кэша (не риск-актив).
PORTFOLIO_ROWS = [
    {"Ticker": "AAPL.US", "Quantity": 120, "Purchase_Price": 150.0, "Currency": "USD"},
    # ДУБЛЬ той же бумаги — включает склейку A-1 (`merged_positions`).
    {"Ticker": "AAPL.US", "Quantity": 30,  "Purchase_Price": 165.0, "Currency": "USD"},
    {"Ticker": "MSFT.US", "Quantity": 60,  "Purchase_Price": 300.0, "Currency": "USD"},
    {"Ticker": "TLT.US",  "Quantity": 200, "Purchase_Price": 95.0,  "Currency": "USD"},
    # Нечисловое количество — включает гард A-2 (`dropped_rows`): строка
    # обязана быть отброшена И НАЗВАНА, а метрики остаться конечными.
    {"Ticker": "BADQ.US", "Quantity": "н/д", "Purchase_Price": 10.0, "Currency": "USD"},
    # AIX-нота: своей истории нет, риск считается по прокси VWOB.US, а цена —
    # по цене покупки. Включает `proxy_substitutions` + `priced_at_cost`.
    {"Ticker": "FFSPC6.1028.AIX", "Quantity": 40, "Purchase_Price": 101.5, "Currency": "USD"},
    # Молодой листинг (котируется последние ~40 дней) — включает sparse-guard
    # и долю книги вне факторной модели (`model_uncovered`).
    {"Ticker": "YOUNG.US", "Quantity": 50, "Purchase_Price": 12.0, "Currency": "USD"},
    {"Ticker": "USD",     "Quantity": 5000, "Purchase_Price": 1.0,  "Currency": "USD"},
]

#: Сколько дней котируется «молодая» бумага. Меньше `MIN_OVERLAP_TDAYS` (60),
#: поэтому движок обязан исключить её из факторной модели, но СОХРАНИТЬ вес.
YOUNG_LISTING_DAYS = 40


# ── синтетический рынок ──────────────────────────────────────────────────────

def _price_frame(engine) -> pd.DataFrame:
    """Матрица цен: факторные ETF + бенчмарки + бумаги портфеля.

    Каждая колонка — своя сеяная траектория. Сиды фиксированы позицией имени в
    отсортированном списке, поэтому добавление бумаги НЕ сдвигает остальные.
    """
    idx = pd.date_range(_START, periods=N_DAYS, freq="B")
    cols = list(dict.fromkeys(
        list(engine.factor_tickers.values())
        + list(getattr(engine, "BENCHMARK_EXTRA", []))
        + ["AAPL.US", "MSFT.US", "TLT.US", "SPY.US", "QQQ.US", "AGG.US", "EEM.US",
           "VWOB.US",            # прокси для AIX-ноты (H-1)
           "YOUNG.US"]
    ))
    data: dict[str, np.ndarray] = {}
    for name in sorted(cols):
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        # Умеренный дрейф + общая рыночная компонента, чтобы беты были осмысленными.
        common = np.random.default_rng(12345).normal(0.0003, 0.008, N_DAYS)
        own = rng.normal(0.0001, 0.009, N_DAYS)
        series = 100.0 * np.exp(np.cumsum(0.6 * common + 0.4 * own))
        if name == "YOUNG.US":
            # Ведущие NaN СОХРАНЯЮТСЯ (F-6): bfill создавал бы фиктивные нули и
            # занижал σ/β молодой бумаги. Здесь это вход для sparse-guard.
            series = series.astype(float)
            series[:-YOUNG_LISTING_DAYS] = np.nan
        data[name] = series
    return pd.DataFrame(data, index=idx)[cols]


class _FakeHistory:
    """Ровно те атрибуты `HistoryResult`, которые читает `analyze_all`."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.data = frame
        self.ohlc_data: dict = {}
        self.loaded = list(frame.columns)
        self.failed: list = []
        self.retried: list = []


def _fake_cds_lookup(*_a, **_kw):
    """Детерминированная замена сетевому CDS-провайдеру.

    Контракт `cds_feed.make_lookup` — `Callable[[str], dict]`, и при отсутствии
    данных он возвращает ПУСТОЙ СЛОВАРЬ, а не None. Первая редакция мока
    возвращала None, и скоринг молча падал на трёх бумагах из четырёх
    («'NoneType' object has no attribute 'get'») — фикстура зафиксировала бы
    обеднённое состояние как эталон. Мок обязан соблюдать контракт настоящего.
    """
    return lambda _ticker: {}


def _make_insider_stub(real_fn):
    """Реальная логика `build_insider_signals` с ДЕТЕРМИНИРОВАННЫМ провайдером.

    Функция принимает инъекцию `fetch`, поэтому мокать её целиком не нужно:
    подменяется только источник, а вся обработка (пороги, роли, агрегация)
    исполняется настоящая. Заглушка вместо неё оставила бы `smart_money`
    пустым — то есть слепым ключом, потерю которого снимок бы не заметил.

    🔴 `real_fn` передаётся АРГУМЕНТОМ, а не импортируется внутри: импорт
    `from finance.smart_money import build_insider_signals` выполнялся бы уже
    ПОСЛЕ `mock.patch` и возвращал бы саму заглушку — бесконечная рекурсия
    («maximum recursion depth exceeded»), которая в первой редакции молча
    оставляла ключ пустым, потому что вызов обёрнут в `try/except`.
    """
    def _fetch(ticker: str) -> dict:
        h = int(hashlib.sha256(ticker.encode()).hexdigest()[:6], 16)
        return {
            "net_flow_usd": float((h % 7) - 3) * 500_000.0,
            "distinct_buyers": h % 5,
            "buy_count": h % 4,
            "sell_count": (h // 4) % 3,
            "top_role": ("CEO", "CFO", "Director")[h % 3],
            "as_of": "2026-06-30",
        }

    def _stub(tickers, **_kw):
        return real_fn(list(tickers), fetch=_fetch)

    return _stub


def run_analyze_all() -> dict:
    """Прогнать `analyze_all` в полностью изолированном окружении."""
    from finance.investment_logic import UniversalPortfolioManager
    from finance.smart_money import build_insider_signals as _real_insiders

    saved = {k: os.environ.get(k) for k in ENV_PINS}
    os.environ.update(ENV_PINS)
    try:
        upm = UniversalPortfolioManager()
        frame = _price_frame(upm.engine)
        hist = _FakeHistory(frame)
        upm.engine.get_market_data = (                       # type: ignore[assignment]
            lambda tickers, period_days=1825: (frame, hist))
        # Ставка фиксируется явно: движок мог её вычислить до подмены env.
        upm.engine.current_rfr_annual = 0.045
        upm.engine.rfr_source = "fixture[USD]=0.045"

        portfolio = pd.DataFrame(PORTFOLIO_ROWS)
        with mock.patch("finance.sec_edgar.batch_fundamental_scan",
                        return_value=pd.DataFrame()), \
             mock.patch("finance.cds_feed.make_lookup", _fake_cds_lookup), \
             mock.patch("finance.smart_money.build_insider_signals",
                        _make_insider_stub(_real_insiders)):
            return upm.analyze_all(source=portfolio,
                                   profile_benchmark="SPY.US",
                                   risk_mandate="MODERATE")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── нормализация ─────────────────────────────────────────────────────────────

def _num(x: float) -> Any:
    if isinstance(x, bool):
        return x
    f = float(x)
    if math.isnan(f):
        return "<NaN>"
    if math.isinf(f):
        return "<+Inf>" if f > 0 else "<-Inf>"
    if f == 0.0:
        return 0.0            # гасим −0.0, оно не несёт смысла и мешает сверке
    return round(f, FLOAT_SIGNIFICANT_DIGITS - 1 - int(math.floor(math.log10(abs(f)))))


def _digest(payload: Any) -> str:
    """SHA-256 от УЖЕ нормализованных значений — округление применено до хеша,
    иначе дайджест ловил бы шум последнего бита, который сам же нормализатор
    и призван гасить."""
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def normalize(value: Any, _depth: int = 0) -> Any:
    """Привести объект к JSON-сериализуемому виду со стабильным порядком."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _num(value)
    if isinstance(value, np.ndarray):
        return [normalize(v, _depth + 1) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        cells = [[normalize(v, _depth + 1) for v in row] for row in value.values]
        head = {"__type__": "DataFrame", "shape": list(value.shape),
                "index_first_last": [str(value.index[0]), str(value.index[-1])] if len(value.index) else [],
                "columns": [str(c) for c in value.columns]}
        if len(value.index) > DIGEST_ROWS_THRESHOLD:
            head["__type__"] = "DataFrame:digest"
            head["values_sha256"] = _digest(cells)
            return head
        head["index"] = [str(i) for i in value.index]
        head["data"] = cells
        return head
    if isinstance(value, pd.Series):
        cells = [normalize(v, _depth + 1) for v in value.values]
        head = {"__type__": "Series", "length": int(len(value)),
                "index_first_last": [str(value.index[0]), str(value.index[-1])] if len(value.index) else []}
        if len(value) > DIGEST_ROWS_THRESHOLD:
            head["__type__"] = "Series:digest"
            head["values_sha256"] = _digest(cells)
            return head
        head["index"] = [str(i) for i in value.index]
        head["data"] = cells
        return head
    if isinstance(value, dict):
        return {str(k): normalize(v, _depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize(v, _depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    # Объекты вроде HistoryResult — снимаем публичные атрибуты.
    if hasattr(value, "__dict__"):
        return {
            "__type__": type(value).__name__,
            **{k: normalize(v, _depth + 1)
               for k, v in sorted(vars(value).items()) if not k.startswith("_")},
        }
    return f"<{type(value).__name__}>"


def fixture_json(results: dict | None = None) -> str:
    """Канонический JSON снимка: отсортированные ключи, фиксированный отступ."""
    res = results if results is not None else run_analyze_all()
    return json.dumps(normalize(res), ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def fixture_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
