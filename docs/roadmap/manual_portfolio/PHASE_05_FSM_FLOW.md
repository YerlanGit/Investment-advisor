# Фаза 5 — FSM-флоу, экран подтверждения, черновик

<!-- nav | area:roadmap | code:src/tg_bot.py,src/db_tokenomics.py | read-before:перед правкой FSM ручного ввода портфеля -->

| | |
|---|---|
| **Соответствие заданию** | Спринт 6 (§3.4) + черновик (§5.3) |
| **Зависимости** | Фаза 4 |
| **Оценка** | 2.5 дня |
| **Флаг отката** | `MANUAL_PORTFOLIO_ENABLED=off` — пункт меню не показывается, хендлеры не регистрируются |

---

## 1. Цель

Довести ручной ввод до пользователя: третий источник в меню подключения, ввод текстом,
экран подтверждения, отмена на каждом шаге, черновик, переживающий рестарт контейнера.

---

## 2. Точка входа

`kb_connect_choice()` (`tg_bot.py:1064-1068`) сегодня даёт две кнопки. Добавляется третья —
**только при `MANUAL_PORTFOLIO_ENABLED=on`** (I-9):

```python
[InlineKeyboardButton(text="📋 Демо-режим (Шаблон)", callback_data="connect:template")],
[InlineKeyboardButton(text="🔗 Freedom Broker API",  callback_data="connect:freedom")],
[InlineKeyboardButton(text="✍️ Ввести портфель вручную", callback_data="connect:manual")],  # ← новое
```

Ветка `mode == "manual"` в `cb_connect_choice` (1699) — по образцу существующих:
`save_connection_mode(user_id, "manual")` → `state.set_state(ManualPortfolio.Input)`.
Валидации значения режима в `save_connection_mode` нет (`db_tokenomics.py:480`),
строка `"manual"` сохраняется как есть.

### 2.1 🔴 Порядок веток в `_resolve_portfolio_source` (находка F-4)

Самая опасная точка фазы. Сейчас (`tg_bot.py:283-313`):

```python
stored = await get_connection_mode_explicit(user_id)
if stored == "freedom":
    return "freedom", stored
if await ...(_has_vault_keys_sync, user_id):     # ← ключи в vault ВСЕГДА побеждают
    await save_connection_mode(user_id, "freedom")
    return "freedom", stored
if stored == "template":
    return "demo", stored
return "undetermined", stored
```

Ветка `manual` обязана стоять **до** проверки vault:

```python
if stored == "freedom":
    return "freedom", stored
if stored == "manual":              # ← ЗДЕСЬ, не ниже
    return "manual", stored
if await ...(_has_vault_keys_sync, user_id):
    ...
```

Иначе пользователь, у которого когда-то были привязаны ключи брокера, **не сможет**
построить отчёт по ручному портфелю: само-лечение (введённое инцидентом 2026-07-14)
молча вернёт его в `freedom`. Ошибка воспроизводилась бы только у части пользователей —
ровно тот класс плавающих дефектов, который эта функция и была призвана убрать.

Само-лечение при этом **сохраняется в силе** для `template` и `None` — его смысл
(«ключи в vault = доказательство привязки») не нарушается: явный выбор `manual`
свежее, чем факт наличия ключей.

---

## 3. FSM

```python
class ManualPortfolio(StatesGroup):
    Input   = State()   # ждём текст (или файл — Фаза 7)
    Confirm = State()   # показан экран подтверждения
```

| Переход | Действие |
|---|---|
| `connect:manual` → `Input` | Показать формат, пример и кнопку «📄 Скачать шаблон» (шаблон — Фаза 7) |
| текст в `Input` | Разбор через `manual_portfolio.parse()`; успех → `Confirm`, полный провал → остаёмся в `Input` с перечнем ошибок |
| `manual:confirm` | Сохранить портфель в черновик, выйти в меню тиров (`_show_analysis_menu`) |
| `manual:edit` | Вернуться в `Input`, **предзаполнив** прошлым вводом |
| `manual:cancel` | `state.clear()`, черновик удалить, вернуться в меню подключения |

---

## 4. Экран подтверждения (§3.4 — обязателен)

Причина из задания: опечатка в количестве (лишний ноль) не видна в цифрах отчёта,
но полностью искажает веса, TRC и CVaR. **Доля позиции на экране делает её заметной глазом** —
поэтому доля обязательна, а не опциональна.

```
📋 Ваш портфель — проверьте перед расчётом

Тикер      Распознан    Кол-во      Цена      Доля
AAPL       AAPL.US      10          150.50    18.2%
SPY        SPY.US       25          480.00    72.6%
KSPI       KSPI.KZ      100         12.50      3.8%

💵 Кэш
USD                     5 000                  5.4%
KZT → USD               4 807.69¹              5.2%

Итого: $27 500

⚠️ Исключено 2 строки:
  • строка 4: BADTICKER — не найден ни у одного источника
  • строка 7: EUR — нет курса EUR→USD, позиция не учтена

¹ 2 500 000 KZT по курсу 520.0 на 2026-07-24

[✅ Рассчитать]  [✏️ Исправить]  [❌ Отмена]
```

Обязательные элементы: распознанный тикер после `resolve_tickers()` (пользователь должен
видеть, что `KSPI` стал `KSPI.KZ`, а нота AIX — прокси-ETF), доля каждой позиции,
кэш отдельным блоком со своей долей, суммарная стоимость, каждая исключённая строка
с номером и человеческой причиной, курс конвертации с датой.

---

## 5. Черновик (§5.3)

`MemoryStorage` в `Dispatcher` (`tg_bot.py:3035`) теряет состояние при рестарте
контейнера, а ввод портфеля на 20 позиций — это минуты работы. Черновик — в существующей
SQLite, без новой инфраструктуры:

```sql
CREATE TABLE IF NOT EXISTS manual_portfolio_draft (
    user_id      INTEGER PRIMARY KEY,
    payload_json TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL
);
```

- создаётся в `init_db()` (`db_tokenomics.py:174`) рядом с остальными — только
  `CREATE TABLE IF NOT EXISTS` (гейт «Откат» §10 задания);
- запись — UPSERT по образцу `save_connection_mode` (480-512), только параметризованные запросы;
- удаляется **после успешной доставки отчёта**, не раньше: если отчёт упал, ввод не потерян;
- хранится **сырой текст** пользователя, а не разобранный результат: правила разбора
  могут измениться между версиями, а исходный текст — нет;
- размер `payload_json` ограничен (≤ 64 КБ) до записи — 200 строк ×  ~40 символов
  с запасом; больше означает попытку положить в БД файл.

**Ограничение, о котором честно сказать пользователю:** SQLite лежит на gcsfuse
(`TOKENOMICS_DB_PATH=/mnt/state/tokenomics.db`, `cloudbuild.yaml:128`). Это надёжнее
`MemoryStorage`, но не бесплатно по задержке — запись черновика делается один раз,
на переходе `Input → Confirm`, а не на каждое сообщение.

---

## 6. Утечка single-flight слота (Т-9)

`_try_acquire_user_slot` / `_release_user_slot` защищают от параллельных отчётов.
В `cb_confirm` слот освобождается в **девяти** ветках отказа (`tg_bot.py:1939, 1954,
1976, 2011, 2039, 2074, 2085, 2100, 2125`) — каждая своим вызовом. Ручной флоу добавляет
ветки (отмена на экране подтверждения, ошибка разбора, `DataQualityBlocked` из Фазы 3),
и повторять этот приём означает почти наверняка забыть одну.

**Решение:** асинхронный контекст-менеджер

```python
@asynccontextmanager
async def user_slot(user_id: int):
    if not _try_acquire_user_slot(user_id):
        raise SlotBusy
    try:
        yield
    finally:
        _release_user_slot(user_id)
```

Новый код использует только его. Существующие девять веток `cb_confirm`
**в этой фазе не переписываются** — это отдельный рефакторинг с собственным риском,
а `cb_confirm` сейчас корректен. Тест `test_slot_released_in_every_branch`
перебирает все ветки ручного флоу и проверяет, что слот свободен после каждой.

---

## 7. Тесты

Дополняют `tests/test_phase35_manual_portfolio.py`.

| Тест | Проверяет |
|---|---|
| `test_manual_wins_over_vault_keys` | **F-4.** Пользователь с ключами в vault + `stored="manual"` → источник `manual`, а не `freedom` |
| `test_vault_selfheal_still_works_for_template` | Само-лечение не сломано для `template`/`None` (регресс M-1…M-5) |
| `test_manual_button_hidden_when_flag_off` | `MANUAL_PORTFOLIO_ENABLED=off` → в `kb_connect_choice` две кнопки, `connect:manual` не обрабатывается (I-9) |
| `test_confirm_screen_shows_weights` | Доля каждой позиции присутствует и суммируется в 100% ± 0.1пп |
| `test_confirm_screen_lists_excluded_rows` | Каждая исключённая строка — с номером и причиной |
| `test_confirm_screen_shows_fx_rate` | Курс и дата конвертации кэша видны |
| `test_cancel_at_every_step` | Отмена в `Input` и в `Confirm` → состояние очищено, черновик удалён |
| `test_slot_released_in_every_branch` | **Т-9.** Слот свободен после успеха, отмены, ошибки разбора и `DataQualityBlocked` |
| `test_draft_survives_restart` | **E-6.** Запись черновика → пересоздание `Dispatcher` с новым `MemoryStorage` → ввод восстановлен |
| `test_draft_deleted_after_report` | Черновик удаляется только после успешной доставки |
| `test_draft_payload_size_limit` | Payload > 64 КБ → отказ до записи в БД |
| `test_draft_sql_is_parameterised` | Ввод `'; DROP TABLE manual_portfolio_draft; --` сохраняется как данные, таблица цела |
| `test_report_time_within_20pct` | **Гейт §10.** Время отчёта на 25 ручных позициях не хуже `freedom` более чем на 20% (замер, не оценка) |

Ориентировочно 22 новых теста.

---

## 8. Гейт выхода

- [ ] `python -m pytest tests/ -q` → **877 + 22 = 899 passed, 1 xfailed**
- [ ] E2E: ручной ввод → подтверждение → отчёт; набор секций совпадает с `freedom` (I-1)
- [ ] E2E: `MANUAL_PORTFOLIO_ENABLED=off` → бот ведёт себя как сегодня (I-9)
- [ ] Слот освобождается во всех ветках ручного флоу
- [ ] `git diff --stat` по восьми модулям I-2 = 0
- [ ] Время отчёта на 25 позициях **измерено** и записано в `docs/audit/AUDIT.md`

---

## 9. Риски

| Риск | Митигация |
|---|---|
| Правка `_resolve_portfolio_source` ломает восстановление после инцидента 2026-07-14 | `test_vault_selfheal_still_works_for_template` + весь `tests/test_phase29_multiuser_connection.py` зелёный без правок |
| Telegram-сообщение с 200 позициями превысит лимит 4096 символов | Экран подтверждения показывает топ-20 по стоимости + строку «и ещё N позиций»; полный список — отдельным сообщением по кнопке |
| Черновик на gcsfuse тормозит флоу | Одна запись за флоу (§5). Задержка измеряется в том же замере, что и время отчёта |
| Пользователь ушёл на середине и вернулся через неделю | Черновик показывается с датой: «Найден черновик от 20.07 — продолжить или начать заново?» |
