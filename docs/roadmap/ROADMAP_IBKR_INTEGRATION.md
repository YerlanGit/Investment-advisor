# ROADMAP_IBKR_INTEGRATION.md — добавление Interactive Brokers по образцу Freedom API
<!-- nav | area:roadmap | code:src/finance/broker_api.py | read-before:план второго брокера (BrokerConnector, IBKR) -->

> **Статус:** стратегический анализ / план (не реализовано). **Дата:** 2026-07-16.
> **Автор:** аудит-раунд по запросу владельца («рассмотри добавление Interactive
> Brokers API аналогично FREEDOM API»).
> **Связанные доки:** `docs/audit/PRODUCTION_READINESS.md` (§3 Freedom API 6/10),
> `docs/bot/TELEGRAM_BOT.md` (флоу подключения), `docs/roadmap/ROADMAP_DATA_RESILIENCE.md`
> (второй источник котировок), `AUDIT.md §−21…−22` (multi-user connection_mode).

---

## 0. TL;DR — вердикт

- **Целесообразно, но НЕ «аналогично» по механике.** IBKR открывает второй пул
  клиентов (US/EU-ритейл и профи), но его API принципиально **stateful** — в
  отличие от stateless HMAC-ключей Freedom/Tradernet. Это главный архитектурный
  разрыв: у IBKR нужна **живущая аутентифицированная сессия/шлюз**, а не просто
  пара `api_key/secret` в vault.
- **Scope-паритет с Freedom = read-only.** Текущая интеграция Freedom — только
  чтение портфеля (Action Plan — советы, не исполнение). Для IBKR разумно начать
  с того же: чтение позиций/кэша → тот же MAC3-DataFrame → тот же отчёт.
- **Ключевой рефактор до IBKR:** ввести абстракцию **`BrokerConnector`** (протокол)
  и расширить `connection_mode` до мульти-брокерного (`freedom | ibkr | template`).
  Это ~2–3 дня и полезно само по себе (снимает хардкод Freedom в боте).
- **Оценка усилий (read-only MVP):** **3–5 недель** инженерных при выборе
  **OAuth Web API** (без per-user gateway-процесса). Вариант с Client Portal
  Gateway дешевле по коду, но дороже по эксплуатации (по процессу-шлюзу на юзера —
  несовместимо с `max-instances=1`).

---

## 1. Как устроена Freedom-интеграция сейчас (эталон для переноса)

Слои (снизу вверх):

| Слой | Файл | Что делает |
|---|---|---|
| Wire-клиент | `src/freedom_portfolio/client.py` `TradernetClient` | REST к `tradernet.com/api`, HMAC-SHA256 подпись (`auth.py`), Cloudflare-retry, парс в `models.Portfolio` |
| Модели | `src/freedom_portfolio/models.py` | Pydantic `Portfolio/Position/AccountBalance` |
| Адаптер | `src/finance/broker_api.py` `FreedomConnector` | `fetch_portfolio() -> pd.DataFrame` с контрактом `Ticker\|Quantity\|Purchase_Price\|Broker_Current_Price` + `df.attrs['_ramp_source'/'_ramp_is_mock'/'_ramp_is_fallback']` |
| Хранение ключей | `src/finance/security.py` `SecureVault` | Fernet-шифр `api_key/secret/login` по `user_id`; `has_user` (existence-check) |
| Оркестрация | `src/tg_bot.py` | `connection_mode` (`user_connection`), `_get_keys_sync`, `_resolve_portfolio_source`, `_fetch_portfolio_sync` → `FreedomConnector` |

**Точка абстракции, которую надо переиспользовать:** весь движок (MAC3, отчёт,
биллинг) зависит ТОЛЬКО от `pd.DataFrame` с указанным контрактом и `_ramp_*`
маркерами. Любой брокер, который умеет отдать такой DataFrame, «подключается»
без изменений в квант-ядре и рендере. Это и есть шов для IBKR.

---

## 2. Ландшафт API Interactive Brokers (на что можно опереться)

IBKR не имеет простого «REST + API-key», как Freedom. Есть три пути:

### 2.1. TWS API / IB Gateway (socket, `ibapi`/`ib_insync`)
- Сокетное соединение к запущенному **Trader Workstation** или **IB Gateway**
  (десктоп/headless Java-процесс), логин по паролю/2FA.
- ❌ **Непригодно для облачного мульти-юзер SaaS:** требует по процессу-шлюзу и
  интерактивной 2FA-сессии на каждого пользователя; нельзя держать чужие пароли.

### 2.2. Client Portal Web API (CP Web API)
- REST + WebSocket через локальный **Client Portal Gateway** (Java), сессия
  поднимается логином пользователя в браузере/2FA, живёт ~24 ч, требует keep-alive
  (`tickle`).
- ⚠️ Stateful: снова по gateway-сессии на юзера. Лучше, чем TWS, но всё ещё
  плохо ложится на `max-instances=1`.

### 2.3. OAuth Web API (институциональный / «Web API 1.0», OAuth 1.0a + RSA) — **рекомендуется**
- Пользователь единожды авторизует приложение (OAuth-флоу, consumer key + RSA-ключи
  приложения). Дальше сервер получает **read-only** доступ к аккаунту/портфелю
  **без хранения пароля пользователя и без per-user процесса**.
- Ближе всего к модели Freedom по эксплуатации: секреты приложения — в Secret
  Manager, per-user хранится только OAuth access token/secret (в тот же Fernet-vault).
- ✅ **Единственный вариант, совместимый с текущей инфраструктурой без per-user
  демона.** Требует регистрации приложения у IBKR и (для институционального
  OAuth) соответствующего типа доступа — это внешняя зависимость, а не код.

> Точный синтаксис эндпоинтов IBKR меняется — перед реализацией свериться с
> актуальной докой IBKR Web API. Здесь фиксируем архитектуру, не конкретные URL.

---

## 3. Целевая архитектура — абстракция `BrokerConnector`

### 3.1. Протокол (новый шов)

```python
# src/finance/brokers/base.py  (новый пакет)
from typing import Protocol
import pandas as pd

class BrokerConnector(Protocol):
    broker_id: str  # 'freedom' | 'ibkr'
    def fetch_portfolio(self) -> pd.DataFrame:
        """MAC3 contract: Ticker|Quantity|Purchase_Price|Broker_Current_Price
        + df.attrs['_ramp_source'|'_ramp_is_mock'|'_ramp_is_fallback']."""
    def fetch_balance(self) -> dict: ...
```

- `FreedomConnector` уже соответствует этому протоколу — просто объявить
  `broker_id='freedom'` и переместить под `finance/brokers/freedom.py` (тонкий
  ре-экспорт для обратной совместимости импортов).
- `IBKRConnector(finance/brokers/ibkr.py)` — новый: OAuth-клиент IBKR →
  нормализация позиций в тот же DataFrame + `_ramp_*` маркеры (включая
  `_ramp_is_fallback` при сбое сессии, чтобы MAC3-гейт и биллинг-логика
  «cost=0 только для demo» работали без изменений).

### 3.2. Фабрика по режиму

```python
def make_connector(mode: str, creds) -> BrokerConnector:
    if mode == "ibkr":    return IBKRConnector(**creds)
    if mode == "freedom": return FreedomConnector(**creds)
    return FreedomConnector(api_key="demo")   # template/demo
```

### 3.3. Нормализация тикеров (критично)

- Freedom отдаёт `AAPL.US`, `KSPI.KZ`; IBKR — `conid`/символ+exchange+currency.
- Движок и история (`freedom_portfolio.history`, Tradernet `getHloc`) завязаны на
  Tradernet-тикеры (`*.US`). **IBKR-позиции нужно маппить в тот же тикер-namespace**,
  иначе история цен (всё ещё через Tradernet) не подтянется. Это отдельная
  таблица соответствий `conid/symbol → tradernet_ticker` + фолбэк на символ+`.US`.
- ⚠️ Здесь же вскрывается существующий риск из `PRODUCTION_READINESS §2`: **цены —
  только Tradernet**. Для IBKR-клиентов без тикера в Tradernet история не
  загрузится → это усиливает приоритет второго источника котировок
  (`ROADMAP_DATA_RESILIENCE §1`, yfinance chain-fallback). **Второй источник цен
  становится пред-условием, а не «приятным дополнением», для IBKR-инструментов.**

---

## 4. Изменения по слоям (чек-лист)

| Слой | Изменение | Риск |
|---|---|---|
| `connection_mode` | значения `freedom \| ibkr \| template`; `get/save` уже UPSERT (M-3) — расширять не надо, только домен значений | низкий |
| Vault | тот же `SecureVault`, но per-broker запись: ключ `f"{broker}:{user_id}"` ИЛИ новая колонка `broker`; `has_user` → `has_user(user_id, broker)` | средний (миграция схемы vault) |
| `_resolve_portfolio_source` | вернуть `(source, broker, stored)`; «ключи всегда побеждают» — по конкретному брокеру | средний |
| Онбординг-бот | `kb_connect_choice`: 3 кнопки (Freedom / IBKR / Демо); для IBKR — OAuth-ссылка + callback, НЕ free-text ключи | средний (OAuth-редирект вне Telegram) |
| Адаптер | `IBKRConnector.fetch_portfolio()` + тикер-маппинг | высокий (внешний API) |
| История цен | второй источник котировок для не-Tradernet тикеров | высокий (см. §3.3) |
| Отчёт/движок | **без изменений** — контракт DataFrame тот же | — |
| Биллинг | **без изменений** — `_effective_cost`/charge-only-post-report брокер-агностичны | — |
| Тесты | `IBKRConnector` (моки OAuth), тикер-маппинг, фабрика, расширенный `_resolve_portfolio_source` | — |

---

## 5. Аутентификация: сравнение и хранение

| | Freedom (сейчас) | IBKR OAuth Web API |
|---|---|---|
| Секрет приложения | — (нет) | consumer key + RSA-ключи в Secret Manager |
| Per-user секрет | `api_key` + `secret` (HMAC) | OAuth access token + token secret |
| Ввод пользователем | free-text в чат (шифруется, удаляется) | **OAuth-редирект** (пароль IBKR НЕ проходит через бота — плюс к безопасности) |
| Живость | stateless (подпись на запрос) | токен долгоживущий; возможен refresh/expiry → нужен путь «переподключить» |
| Хранилище | `SecureVault` (Fernet) | тот же vault, поле `broker='ibkr'` |

**Плюс IBKR по безопасности:** пароль/2FA пользователя никогда не попадают в бота
(в отличие от Freedom, где `api_key/secret` идут текстом в чат и мы их удаляем).
**Минус:** OAuth-редирект в Telegram — это WebApp/внешняя ссылка + callback, а не
линейный FSM `Login→ApiKey→SecretKey`.

---

## 6. Поэтапный план

**Фаза 0 — рефактор-шов (2–3 дня, БЕЗ IBKR):**
`finance/brokers/` пакет, протокол `BrokerConnector`, `FreedomConnector` под него,
фабрика `make_connector`, `connection_mode` домен + `_resolve_portfolio_source`
возвращает брокера. Полностью покрыто существующими тестами Freedom + новые на
фабрику. Ценность: снимает хардкод Freedom, готовит почву.

**Фаза 1 — IBKR read-only MVP (2–3 нед):**
`IBKRConnector` на OAuth Web API (позиции + кэш → DataFrame), тикер-маппинг
`conid→tradernet`, OAuth-онбординг в боте, vault per-broker, `_ramp_is_fallback`
на сбой сессии. Гейт: DEEP-отчёт по реальному IBKR-счёту идентичен по качеству
Freedom-отчёту.

**Фаза 2 — устойчивость данных (1–2 нед, частично параллельно):**
второй источник котировок (yfinance chain-fallback) для не-Tradernet тикеров —
**блокер** полноценного IBKR-покрытия (иначе US-only тикеры без `.US`-прокси
слепнут). Корпоративные действия (сплиты/дивиденды) — от IBKR as-is для MVP.

**Фаза 3 — паритет и расширения (опц.):**
мульти-счёт IBKR, refresh-токенов по расписанию, вебсокет-ресинк позиций,
（далеко）исполнение ордеров — но это уже за рамками «advisor read-only» и требует
юридического пересмотра (`PRODUCTION_READINESS §8`).

---

## 7. Риски и открытые вопросы

1. **Доступ к API IBKR** — институциональный OAuth Web API требует одобрения типа
   доступа у IBKR; для чистого ритейла может понадобиться CP Web API (stateful).
   **Это внешняя блокирующая зависимость — проверить ДО кода.**
2. **Per-user gateway (если CP Web API)** несовместим с `max-instances=1`. Если
   OAuth Web API недоступен — IBKR откладывается до решения масштаба
   (`PRODUCTION_READINESS P0-инфра`).
3. **Ценовая история не-Tradernet тикеров** — второй источник обязателен (§3.3).
4. **Комплаенс** — расширение на US/EU-клиентов меняет регуляторный периметр
   (SEC/MiFID «investment advice»); дисклеймеры и юр.ревью (P1) становятся строже.
5. **Тикер-namespace** — маппинг `conid→tradernet` неполон по определению; нужна
   стратегия для инструментов без соответствия (дроп с раскрытием, как sparse-history).

---

## 8. Рекомендация

- **Сначала** закрыть P0 из `PRODUCTION_READINESS` (платежи, масштаб/очередь,
  алертинг) — они блокируют монетизацию любого брокера, включая уже готовый Freedom.
- **Параллельно** сделать Фазу 0 (рефактор-шов) — дёшево, полезно, снижает риск.
- **IBKR (Фазы 1–2)** — после подтверждения доступа к OAuth Web API у IBKR и
  наличия второго источника котировок. До этого — держать как спроектированный,
  но не начатый трек.

---
*Документ фиксирует архитектуру и последовательность; конкретные эндпоинты/поля
IBKR сверять с актуальной докой IBKR Web API на момент реализации.*
