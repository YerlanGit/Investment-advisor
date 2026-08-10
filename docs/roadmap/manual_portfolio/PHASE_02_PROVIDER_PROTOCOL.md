# Фаза 2 — Протокол провайдера цен + `TradernetProvider`

<!-- nav | area:roadmap | code:src/finance/price_providers.py,src/finance/investment_logic.py | read-before:перед добавлением любого источника ценовых данных -->

> ⚠️ **Ссылки вида `investment_logic.py:NNN` в этом файле устарели с Арх-3.10**
> (2026-08-06): ядро разложено по пакету `finance/engine/`, а
> `finance/investment_logic.py` стал фасадом на 61 строку. Соответствие:
> `MAC3RiskEngine` → `engine/risk_engine.py`, `UniversalPortfolioManager` и
> семь стадий `analyze_all` → `engine/portfolio_manager.py`. Номера строк не
> перенумерованы намеренно — искать надо по ИМЕНИ функции: оно переезд
> пережило, а номер строки не переживает и обычной правки (`AUDIT §−69`).

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

## 4. Правка `investment_logic.py` (I-2b)

Правка затрагивает **четыре поимённо названные функции** — список в `PHASE_00 §2, I-2b`.
Прежняя редакция ограничивала правку строками 653–772; отброшена, потому что номера строк
едут от первой же вставки, а проброс источника (§4.1) до них не дотягивается.

```python
def _get_price_provider(self):
    """Провайдер выбирается ПО ИСТОЧНИКУ портфеля (I-12, юридическая граница)."""
    if self._price_provider is None:
        from finance.price_providers import provider_for_source
        self._price_provider = provider_for_source(self.price_source)
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

### 4.1 Проброс источника до движка

Провайдер выбирается по источнику портфеля (I-12), поэтому источник должен доехать
до `MAC3RiskEngine`. Путь — через конструкторы, по образцу уже существующих
`reporting_currency` и `risk_mandate` (`investment_logic.py:446-448`):

```
tg_bot.cb_confirm (source известен: freedom|manual|demo)
  → UniversalPortfolioManager(price_source=source)
  → MAC3RiskEngine(price_source=source)        # дефолт "freedom" — обратная совместимость
  → _get_price_provider() → provider_for_source(...)
```

**Дефолт `price_source="freedom"` обязателен**: все существующие вызовы
`UniversalPortfolioManager()` (в тестах и в `batch_reports.py`) остаются валидными
без правок — тот же механизм, которым держится I-5.

Читать источник из `df.attrs['_ramp_source']` **нельзя**: `prefetch_market_data`
вызывается на Шаге 1 (`tg_bot.py:2236`) от списка тикеров, DataFrame туда не передаётся.

### 4.2 `provider_for_source` — единственная точка выбора

```python
_MANUAL_FORBIDDEN = ("tradernet",)      # I-12: юридическая граница, не техническая

def provider_for_source(source: str) -> PriceProvider:
    if source == "manual":
        chain = _build_chain(os.getenv("PRICE_PROVIDER_MANUAL", "stooq"))
        assert not any(p.name in _MANUAL_FORBIDDEN for p in chain.providers), \
            "I-12: данные Tradernet недопустимы в ручном портфеле"
        return chain
    return _build_chain(os.getenv("PRICE_PROVIDER_FREEDOM", "tradernet"))
```

Проверка в рантайме, а не только в тесте: env-переменную можно выставить в проде,
и `PRICE_PROVIDER_MANUAL=tradernet` должен приводить к отказу сборки отчёта,
а не к тихому нарушению. Отказ — на старте, до сетевых вызовов и до списания токена.

**До Фазы 9 `StooqProvider` не существует**, поэтому в этой фазе `provider_for_source`
для `manual` поднимает `ProviderUnavailable` («ручной ввод ещё не запущен») —
и это ровно то поведение, которое требуется: `MANUAL_PORTFOLIO_ENABLED` в проде выключен.

---

## 4c. Предвыполненный анализ фазы (2026-07-29)

Проверка кода перед началом работ. Состояние изменилось с момента написания плана —
часть плюмбинга Фазы 2 уже построена при работе над витриной (раунд 31), и это меняет
объём.

### 4c.1 Что уже сделано и переиспользуется

| Пункт плана | Статус |
|---|---|
| `price_source` в `MAC3RiskEngine.__init__` с дефолтом `"freedom"` | ✅ есть (раунд 31) |
| `price_source` в `UniversalPortfolioManager.__init__` | ✅ есть |
| Проброс источника из бота (`_run_analysis_background(source=…)`) | ✅ есть |
| Ветвление `get_market_data` по источнику | ✅ есть для `demo`; для остальных — нет |
| `provider`/`convention` в кэше | ✅ Фаза 1 |

Осталось: сам протокол (`price_providers.py`), `TradernetProvider`, `provider_for_source`
и перевод ветки `freedom` на провайдера. Оценка снижается **1.5 дня → ~1 день**.

### 4c.2 🔴 Демо-ветка уже строит `HistoryResult` — её надо привести к протоколу

`get_market_data` для `demo` сейчас возвращает `HistoryResult(...)` напрямую. Как только
появится `ProviderResult`, в системе окажутся ДВА типа результата на одной воронке, и
`convention`/`source` у демо-ветки не будет — то есть строка CoVe (Фаза 6) не сможет
подписать витрину, а `test_single_source_per_matrix` (I-13) не найдёт, что проверять.

**Решение:** ввести `DemoProvider` (`name="demo"`, `convention=PriceConvention.SYNTHETIC`)
и свести демо-ветку к тому же `provider_for_source`. Это добавляет **четвёртое** значение
в `PriceConvention` — синтетические ряды витрины не являются ни `RAW`, ни
`SPLIT_ADJUSTED`, ни `TOTAL_RETURN`, и притворяться одним из них они не должны:
I-7 запрещает смешивать конвенции, а «витрина» обязана быть отличима от рыночных данных.

### 4c.3 Порядок полей `ProviderResult` — проверено на живых потребителях

Перепроверено, что `history_result` расходится по шести модулям и читается ТОЛЬКО
атрибутами (`.data`, `.failed`, `.retried`), позиционной распаковки нет нигде:
`pdf_payload.py:1110,1905` · `data_lineage.py:99` · `scenario_report.py:43` ·
`portfolio_series.py:65,154` · `tg_bot.py:2291,2298` · `investment_logic.py:1746`.
Значит duck-подстановка (I-11) безопасна при условии, что первые четыре поля сохраняют
имена и порядок.

### 4c.4 Ловушка: `_get_tradernet_client` нельзя удалять

План говорит «`_get_tradernet_client` → `_get_price_provider`». Буквальное переименование
сломает демо-тест `test_freedom_source_does_not_use_demo_matrix`
(`tests/test_phase35_demo_showcase.py`), который патчит `_get_tradernet_client` по имени,
и вообще уберёт единственную точку, где создаётся клиент. Правильно: **добавить**
`_get_price_provider`, оставив `_get_tradernet_client` как внутреннюю фабрику клиента,
которую использует `TradernetProvider`.

### 4c.5 Что перепроверено и осталось верным

- `EXTERNAL_DIVERSIFIERS` берут колонки из готового `all_data` — отдельной загрузки нет,
  второй воронки не появляется;
- `tool_plugins.py` — мёртвый код без вызывающих (и сам по себе сломан: `period="2y"`
  уходит в int-параметр, кортеж возврата распаковывается как одно значение). В объём
  фазы не входит, но при переводе `get_market_data` на провайдера он станет ещё более
  явно нерабочим — стоит удалить отдельным PR;
- ветка `ohlc_data` мертва (поля нет у `HistoryResult`), поэтому ATR всегда идёт по
  close-only fallback — `ProviderResult` не обязан его нести, но и не должен делать вид,
  что несёт.

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
| `test_default_source_is_freedom_tradernet` | `UniversalPortfolioManager()` без аргументов → Tradernet (обратная совместимость, §4.1) |
| **`test_manual_chain_has_no_tradernet`** | **I-12.** `provider_for_source("manual")` не содержит `TradernetProvider` ни в одном звене цепочки |
| `test_manual_forbidden_provider_raises` | `PRICE_PROVIDER_MANUAL=tradernet` → отказ на старте, до сетевых вызовов (§4.2) |
| `test_no_tradernet_client_constructed_for_manual` | За сборку ручного отчёта `TradernetClient.__init__` не вызывается ни разу (I-12, проверка на уровне поведения, а не конфигурации) |

🔴 **Сверено с кодом 2026-08-10** (`AUDIT §−76`) — таблица выше называла
будущие имена, и три из них не совпали с реализацией. Как есть на самом деле
(`grep -n 'def test_' tests/test_phase35_price_providers.py`, 40 тестов):

| Названо в плане | Что реально |
|---|---|
| `test_snapshot_numbers_unchanged` + фикстура `tests/fixtures/phase35_snapshot_freedom.json` | **файла нет.** Роль гейта «числа не изменились» исполняет golden-фикстура движка `tests/fixtures/results_golden.json` + `test_contracts_golden.py` — она шире и сравнивает так же точно |
| `test_manual_chain_has_no_tradernet` | есть под именем **`test_manual_never_returns_tradernet`** ✅ |
| `test_no_tradernet_client_constructed_for_manual` | 🔴 **отсутствует.** Есть только его демо-близнец `test_demo_never_builds_tradernet_client`. Поведение реализовано (`risk_engine._get_price_provider`: клиент не создаётся при `src in ('demo','manual')`), но для `manual` НЕ ЗАПИНЕНО. Заведено требованием ЧК-08.5-4 (`PHASE_08B §7`) |

Сравнение чисел — точное (`==` по float), а не по допуску: провайдер обязан
быть 1:1.

---

## 6. Гейт выхода

- [ ] `python -m pytest tests/ -q` → **793 + 12 = 805 passed, 1 xfailed**
- [ ] `test_snapshot_numbers_unchanged` зелёный на **точном** равенстве
- [ ] `test_manual_chain_has_no_tradernet` зелёный (I-12)
- [ ] `git diff --stat` по восьми модулям I-2 = 0
- [ ] `git diff src/finance/investment_logic.py` затрагивает только четыре функции
      из списка I-2b — проверяется ревью диффа, не номерами строк
- [ ] `grep -rn "import tg_bot\|from tg_bot" src/finance/` → пусто

---

## 7. Риски

| Риск | Митигация |
|---|---|
| `ProviderResult` ломает потребителя, которого не нашли | `test_consumers_accept_provider_result` перебирает все 6 из §2.1 поимённо |
| Ленивый импорт `price_providers` тянет тяжёлые зависимости в юнит-тесты | Модуль — stdlib + pandas, как `leveraged.py`. Импорт `TradernetProvider` локальный, внутри метода |
| Незамеченный сдвиг чисел из-за порядка тикеров | `get_history_frame` уже собирает `df` в порядке входного списка (`history.py:196`); провайдер порядок не меняет — покрыто снапшотом |
