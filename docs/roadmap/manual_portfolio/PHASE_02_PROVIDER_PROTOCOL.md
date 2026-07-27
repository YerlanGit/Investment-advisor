# Фаза 2 — Протокол провайдера цен + `TradernetProvider`

<!-- nav | area:roadmap | code:src/finance/price_providers.py,src/finance/investment_logic.py | read-before:перед добавлением любого источника ценовых данных -->

| | |
|---|---|
| **Соответствие заданию** | Спринт 1 (§5.1) |
| **Зависимости** | Фаза 1 |
| **Оценка** | 1.5 дня |
| **Флаг отката** | `PRICE_PROVIDER_CHAIN=off` (дефолт `off` до Фазы 9) |

---

## 1. Цель

Ввести шов, за которым источник цен становится сменным, **не изменив ни одного числа
в отчётах**. Фаза чисто рефакторинговая: единственный существующий провайдер —
обёртка над текущим `history.py` с поведением 1:1.

Архитектура согласована с уже утверждённым планом `docs/roadmap/ROADMAP_DATA_RESILIENCE.md §1`
(`PriceProvider` / `TradernetProvider` / `ChainProvider`) — эта фаза его исполняет,
а не заводит параллельную.

---

## 2. Контракт — `src/finance/price_providers.py` (новый файл)

```python
class PriceConvention(str, Enum):
    RAW            = "raw"
    SPLIT_ADJUSTED = "split_adjusted"   # текущая конвенция Tradernet
    TOTAL_RETURN   = "total_return"


class ProviderResult(NamedTuple):
    # ── Поля 1-4: ТОТ ЖЕ порядок и та же семантика, что у HistoryResult ──
    data:       pd.DataFrame                 # index=DatetimeIndex, columns=tickers
    loaded:     list[str]
    failed:     dict[str, str]               # ticker -> причина
    retried:    list[str]
    # ── Новое — строго в хвост (I-11) ──
    convention: dict[str, PriceConvention] = {}   # ticker -> конвенция, заполняется ВСЕГДА
    source:     dict[str, str]              = {}  # ticker -> имя провайдера (для CoVe)


class PriceProvider(Protocol):
    name:            str
    convention:      PriceConvention
    max_parallelism: int                     # свой у каждого провайдера (S-6)

    def fetch(self, tickers: list[str], *, days: int) -> ProviderResult: ...
```

### 2.1 Почему порядок полей критичен (I-11)

`HistoryResult` — `NamedTuple(data, loaded, failed, retried)`; объект расходится
по **шести** модулям и читается атрибутами:

| Модуль | Что читает |
|---|---|
| `pdf_payload.py:1110,1905` | `results["history_result"]` |
| `finance/data_lineage.py:99` | `.data` (свежесть последнего close) |
| `finance/scenario_report.py:43` | `.data` (полная ценовая матрица) |
| `finance/portfolio_series.py:65,154` | `.data` |
| `tg_bot.py:2291,2298` | `.failed`, `.retried` |
| `finance/investment_logic.py:1746` | `getattr(history_result, 'ohlc_data', None)` |

Из них `pdf_payload.py` заморожен I-2 — то есть `ProviderResult` обязан подставляться
на место `HistoryResult` **без единой правки на стороне потребителей**. Отсюда: новые
поля только в хвост, у обоих новых полей — дефолты.

**Побочная находка (F-3), зафиксировать, не чинить в этой фазе.** Поля `ohlc_data`
у `HistoryResult` нет и никогда не было, а `investment_logic.py:1746` читает его через
`getattr(..., None)`. Значит **ATR всегда считается по close-only fallback**, ветка
«True ATR (OHLC)» мертва с момента написания. Это не регресс фазы; занести строкой
в `docs/audit/AUDIT.md` и решить отдельно — реализовать OHLC в провайдерах или убрать
мёртвую ветку.

### 2.2 Требования к любой реализации

- пустой ответ по тикеру — это `failed[ticker]`, **не** исключение;
- ряд отсортирован по дате, без дублей индекса;
- только `logger`, никаких `print`;
- таймауты и backoff обязательны;
- `convention` заполняется для **каждого** загруженного тикера — «не знаю» не является
  допустимым значением (иначе I-7 непроверяем).

---

## 3. `TradernetProvider` — обёртка 1:1

```python
class TradernetProvider:
    name            = "tradernet"
    convention      = PriceConvention.SPLIT_ADJUSTED
    max_parallelism = 6            # ровно текущий ThreadPoolExecutor(max_workers=6)

    def fetch(self, tickers, *, days) -> ProviderResult:
        hr = get_history_frame(self._client, tickers, days=days,
                               max_workers=self.max_parallelism)
        return ProviderResult(hr.data, hr.loaded, hr.failed, hr.retried,
                              convention={t: self.convention for t in hr.loaded},
                              source={t: self.name for t in hr.loaded})
```

**Ни одной строки логики не переносится** — вся работа остаётся в `history.py`.
Обёртка только объявляет метаданные, которых у `HistoryResult` нет.

### 3.1 О конвенции Tradernet

Объявляется `SPLIT_ADJUSTED` — это осознанное **утверждение о поведении системы**,
а не наблюдение о сервере. Документация Tradernet конвенцию не фиксирует
(прямо сказано в шапке `history.py:8-19`), но модуль применяет `corr=1`, затем
`KNOWN_SPLITS`, затем эвристику `_detect_and_adjust_splits` — то есть **на выходе**
ряд скорректирован на сплиты независимо от того, что вернул сервер. Дивиденды
не корректируются нигде → не `TOTAL_RETURN`. Вывод устойчив.

---

## 4. Правка `investment_logic.py` (единственная в файле — I-2b)

Диапазон правки — строки 653–772, то есть `_get_tradernet_client` + `get_market_data`.

```python
def _get_price_provider(self):
    if self._price_provider is None:
        from finance.price_providers import TradernetProvider
        self._price_provider = TradernetProvider(self._get_tradernet_client())
    return self._price_provider

def get_market_data(self, tickers, period_days=None):
    ...
    result = self._get_price_provider().fetch(all_req, days=period_days)
    #        ↑ было: get_history_frame(client, all_req, days=period_days)
    ...
    return converted, result     # `result` duck-совместим с HistoryResult
```

Всё, что ниже (`math_firewall`, `_apply_fx_conversion`, возврат кортежа) — без изменений.
Имя переменной `history_result` в вызывающем коде сохраняется, чтобы диффы
`tg_bot.py`/`data_lineage.py` остались нулевыми.

---

## 5. Тесты

Дополняют `tests/test_phase35_price_providers.py`.

| Тест | Проверяет |
|---|---|
| `test_provider_result_field_order` | Первые 4 поля совпадают по имени и порядку с `HistoryResult` (I-11) |
| `test_provider_result_positional_construction` | `ProviderResult(df, loaded, failed, retried)` валиден без новых полей |
| `test_tradernet_provider_declares_convention` | `convention == SPLIT_ADJUSTED`, `max_parallelism == 6` |
| `test_convention_filled_for_every_loaded_ticker` | Ни один загруженный тикер не остался без конвенции |
| `test_failed_ticker_is_not_exception` | Пустой ответ → запись в `failed`, вызов не падает |
| `test_consumers_accept_provider_result` | `data_lineage.build_lineage`, `portfolio_series`, `scenario_report` работают на `ProviderResult` так же, как на `HistoryResult` |
| **`test_snapshot_numbers_unchanged`** | **Гейт фазы.** Фикстура портфеля + записанная ценовая матрица → `analyze_all` до и после ветки даёт побитово равные Risk Index, CVaR 95, Sharpe, топ-3 TRC, факторные беты (I-5) |
| `test_mixed_convention_rejected` | Матрица из тикеров с разными `convention` → отказ (I-7). Заготовка, полностью включается в Фазе 9 |

Снапшот хранится как JSON-фикстура в `tests/fixtures/phase35_snapshot_freedom.json`;
сравнение — точное (`==` по float), а не по допуску: провайдер обязан быть 1:1.

---

## 6. Гейт выхода

- [ ] `python -m pytest tests/ -q` → **793 + 8 = 801 passed, 1 xfailed**
- [ ] `test_snapshot_numbers_unchanged` зелёный на **точном** равенстве
- [ ] `git diff --stat` по восьми модулям I-2 = 0
- [ ] `git diff src/finance/investment_logic.py` не выходит за строки 653–772 (I-2b)
- [ ] `grep -rn "import tg_bot\|from tg_bot" src/finance/` → пусто

---

## 7. Риски

| Риск | Митигация |
|---|---|
| `ProviderResult` ломает потребителя, которого не нашли | `test_consumers_accept_provider_result` перебирает все 6 из §2.1 поимённо |
| Ленивый импорт `price_providers` тянет тяжёлые зависимости в юнит-тесты | Модуль — stdlib + pandas, как `leveraged.py`. Импорт `TradernetProvider` локальный, внутри метода |
| Незамеченный сдвиг чисел из-за порядка тикеров | `get_history_frame` уже собирает `df` в порядке входного списка (`history.py:196`); провайдер порядок не меняет — покрыто снапшотом |
