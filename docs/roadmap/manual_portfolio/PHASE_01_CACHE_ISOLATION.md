# Фаза 1 — Изоляция кэша цен по провайдеру

<!-- nav | area:roadmap | code:src/freedom_portfolio/history.py | read-before:перед правкой кэша ценовых рядов -->

| | |
|---|---|
| **Соответствие заданию** | Спринт 0 (§5.2) |
| **Зависимости** | Фаза 0 |
| **Оценка** | 0.5 дня |
| **Флаг отката** | не требуется — правка обратно совместима по построению |

---

## 1. Дефект

`src/freedom_portfolio/history.py:553`:

```python
def _cache_path(ticker: str, days: int) -> Path:
    """Pickle-based cache (no extra deps; parquet would require pyarrow)."""
    safe = ticker.replace("/", "_").replace(".", "_")
    return _CACHE_DIR / f"{safe}_{days}d.pkl"          # провайдера в ключе нет
```

Ключ кэша — `(ticker, days)`. Как только появляется второй провайдер, читающий тот же
кэш, `AAPL_US_1825d.pkl`, записанный Tradernet, в течение TTL (`_CACHE_TTL_SECONDS = 3600`,
строка 41) отдаётся второму провайдеру как его собственные данные.

**Почему это хуже, чем выглядит.** Симптом плавающий: воспроизводится только внутри
часового окна и только если оба провайдера запрашивались в одном контейнере. Последствия —
неверный источник в CoVe (пользователю показывают не тот источник, из которого взяты числа)
и, что серьёзнее, **смешанные конвенции корректировок в одной ковариационной матрице**
(нарушение I-7), если конвенции провайдеров различаются.

Правка делается **первым коммитом ветки**, до появления второго провайдера, — тогда
дефект не успевает возникнуть ни разу.

---

## 2. Правка

```python
def _cache_path(ticker: str, days: int,
                provider: str = "tradernet",
                convention: str = "split_adjusted") -> Path:
    """Pickle-based cache (no extra deps; parquet would require pyarrow).

    Provider + convention are part of the key: two providers must never read
    each other's series (cross-provider cache poisoning), and one provider may
    legitimately serve several conventions (EODHD: close / adjusted_close /
    splitadjusted).
    """
    safe = ticker.replace("/", "_").replace(".", "_")
    return _CACHE_DIR / f"{provider}__{convention}__{safe}_{days}d.pkl"
```

`_read_cache` и `_write_cache` (строки 559 и 573) получают те же два параметра
с теми же дефолтами и пробрасывают их дальше. `get_candles` (78) — тоже,
чтобы `TradernetProvider` из Фазы 2 мог передать свои значения, не переписывая модуль.

**Дефолты обязательны.** Все существующие вызовы (`get_candles` → `_read_cache`/`_write_cache`)
остаются валидными без правки — это и есть механизм, которым I-5 держится в этой фазе.

**Про `convention` в ключе.** Формально провайдер однозначно задаёт конвенцию, и `provider`
достаточно. Параметр всё равно вносится: EODHD (Фаза 10) отдаёт три разных ряда на один
тикер (`close`, `adjusted_close`, `function=splitadjusted`) с одного эндпоинта, и без
конвенции в ключе они схлопнутся друг на друга — тот же дефект, только внутри одного провайдера.

---

## 3. Что происходит со старыми файлами

Имя файла меняется, поэтому записи предыдущей версии **не находятся** и перезаписываются
под новым именем. Потеря кэша безобидна: TTL всё равно час, а `/tmp/freedom_history_cache`
на Cloud Run живёт только внутри жизни контейнера.

**Нюанс, которого нет в задании.** Осиротевшие файлы **не удаляются** — вытеснения в модуле
нет вообще, ни до правки, ни после. На Cloud Run `/tmp` — это RAM при лимите 2 GiB,
поэтому в фазу добавляется однократная уборка при первой записи: удалить из `_CACHE_DIR`
файлы старше `2 × _CACHE_TTL_SECONDS`. Это не расширение объёма, а закрытие пункта
«рост `/tmp` измерен» из гейта §10 задания — измерять проще то, что ограничено.

---

## 4. Тесты

Новый файл `tests/test_phase35_price_providers.py` (первые кейсы; наполняется в Фазах 2 и 9).

| Тест | Проверяет |
|---|---|
| `test_cache_path_default_matches_tradernet` | Дефолты дают `tradernet__split_adjusted__AAPL_US_1825d.pkl` |
| `test_cache_isolated_between_providers` | Запись `provider="tradernet"` → чтение `provider="stooq"` возвращает `None`, а не чужой ряд |
| `test_cache_isolated_between_conventions` | Тот же провайдер, разные конвенции → разные файлы |
| `test_cache_ttl_still_applies` | TTL не сломан: файл старше 3600 с → `None` |
| `test_stale_files_evicted_on_write` | Уборка §3 удаляет файл старше `2×TTL` и **не трогает** свежий |
| `test_existing_callers_unchanged` | `get_candles` без новых аргументов пишет и читает ровно свой файл (регресс I-5) |

Тесты офлайн: `_fetch_hloc` мокается, в кэш кладётся синтетический `pd.Series`.

---

## 5. Гейт выхода

- [ ] `python -m pytest tests/ -q` → **787 + 6 = 793 passed, 1 xfailed**
- [ ] `git diff --stat` по восьми модулям I-2 = 0
- [ ] `git diff` затрагивает только `src/freedom_portfolio/history.py`
      и `tests/test_phase35_price_providers.py`
- [ ] Существующие тесты (`tests/test_freedom_history.py`) зелёные **без единой правки** —
      главный признак, что дефолты подобраны верно

---

## 6. Риски

| Риск | Митигация |
|---|---|
| Кто-то вызывает `_cache_path` позиционно с третьим аргументом | Проверено: вызовов вне модуля нет (`grep` по `src/` и `tests/` — 0) |
| Уборка удалит файл, который пишется параллельным потоком | Порог `2×TTL` заведомо больше времени жизни любой записи; `unlink(missing_ok=True)` под `try/except OSError` |
| Пользователь получит «медленный» первый отчёт после деплоя | Одноразово, в пределах одного TTL; на 2–4-минутном отчёте незаметно |
