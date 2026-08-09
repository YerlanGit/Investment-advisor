"""Асинхронный клиент Tradernet: батчинг, троттлинг, backoff, разбор getHloc.

Транспорт
─────────
    POST {base}/api/getHloc      Content-Type: application/json
    подпись — `services.auth.TradernetAuth` (APIv3, X-NtApi-*)

Параметры `getHloc` (сверено с `tradernet-sdk` 2.2.0, `client.py::get_candles`)::

    id           "AAPL.US,MSFT.US,…"   тикер или СПИСОК через запятую
    timeframe    1440                  минуты; 1440 = дневная свеча
    date_from    "01.08.2026 00:00"    формат DD.MM.YYYY HH:MM — не ISO!
    date_to      "09.08.2026 00:00"
    intervalMode "ClosedRay"
    count        -1                    «все свечи в интервале»

🔴 Формат дат. ISO (`YYYY-MM-DD`) сервер принимает, но отвечает ПУСТЫМ массивом
свечей вместо ошибки. Отладка такого молчания стоит часов: запрос успешен,
подпись верна, данных нет. Формат `DD.MM.YYYY HH:MM` обязателен.

Почему батчинг вообще возможен
──────────────────────────────
Ответ `getHloc` изначально имеет форму СЛОВАРЯ, ключ которого — тикер
(`{"hloc": {"AAPL.US": [...]}, "xSeries": {"AAPL.US": [...]}}`). Одиночный
запрос — вырожденный случай словаря из одного ключа. Именно поэтому список
через запятую в `id` возвращает столько ключей, сколько запрошено, и разбор
ниже устроен как «пройти по всем ключам», а не «взять свой».

Лимиты (ограничения ТЗ, зашиты в клиент, а не в вызывающий код)
───────────────────────────────────────────────────────────────
* ≤ 50 тикеров в батче;
* ≥ 2.0 с между ЛЮБЫМИ двумя запросами — троттлинг живёт внутри клиента,
  поэтому пауза не может «потеряться» при рефакторинге цикла снаружи;
* HTTP 429 → экспоненциальный backoff с уважением заголовка `Retry-After`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Iterator, Mapping, Sequence

import aiohttp

from services.auth import TradernetAuth

logger = logging.getLogger(__name__)

__all__ = ["TradernetClient", "RawCandle", "TradernetError",
           "TradernetAuthError", "RateLimitedError", "chunked"]

# Формат дат Tradernet. Вынесен в константу, потому что это САМАЯ дорогая
# опечатка модуля (см. 🔴 выше) — она не даёт ошибки, она даёт пустоту.
DATE_FORMAT = "%d.%m.%Y %H:%M"

# Дневная свеча = 1440 минут. `getHloc` меряет timeframe в МИНУТАХ
# (в SDK это `int(timeframe_seconds / 60)`), а не в секундах и не строкой "D".
TIMEFRAME_DAILY_MINUTES = 1440


# ── Ошибки ───────────────────────────────────────────────────────────────────


class TradernetError(RuntimeError):
    """Любой отказ Tradernet, который не лечится повтором."""


class TradernetAuthError(TradernetError):
    """Ключи/подпись отвергнуты (code 4 — подпись, code 12 — учётные данные)."""


class RateLimitedError(TradernetError):
    """HTTP 429. Поднимается наружу, только когда исчерпаны все попытки."""


# Коды Tradernet, при которых повтор бессмысленен: ключ не станет валиднее.
_FATAL_CODES = {4: "Invalid signature", 12: "Invalid credentials"}


# ── Доменный тип ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RawCandle:
    """Свеча как её отдал брокер: время — UNIX-секунды, цены — float.

    Приведение к торговой дате и к `Decimal` — задача `etl_processor`: здесь
    ещё неизвестен часовой пояс биржи, а округлять до попадания в
    `NUMERIC(12,4)` раньше времени значит потерять исходное значение.
    """

    ts: int
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


# ── Нарезка на батчи ─────────────────────────────────────────────────────────


def chunked(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Разрезать список на куски по `size` элементов.

    Отдельная функция, а не выражение по месту: размер батча — ограничение
    брокера, и его проверяет тест. `size <= 0` отвергается явно, иначе цикл
    вызывающего кода стал бы бесконечным на пустых кусках.
    """
    if size <= 0:
        raise ValueError(f"Размер батча должен быть положительным, получено {size}")
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


# ── Клиент ───────────────────────────────────────────────────────────────────


class TradernetClient:
    """Асинхронный клиент котировок поверх ОДНОЙ учётной записи Tradernet."""

    MAX_BATCH_SIZE = 50           # потолок из ТЗ
    MIN_REQUEST_INTERVAL = 2.0    # секунды; ниже опускаться нельзя

    def __init__(
        self,
        auth: TradernetAuth,
        *,
        base_url: str = "https://tradernet.com",
        batch_size: int = 50,
        batch_delay: float = 2.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        timeout: float = 60.0,
        interval_mode: str = "ClosedRay",
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        # Оба лимита зажимаются здесь, а не в конфиге: конфиг можно обойти
        # прямым вызовом клиента из скрипта, а этот конструктор — нет.
        self.batch_size = max(1, min(int(batch_size), self.MAX_BATCH_SIZE))
        self.batch_delay = max(float(batch_delay), self.MIN_REQUEST_INTERVAL)
        self.max_retries = max(1, int(max_retries))
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self.interval_mode = interval_mode
        self._timeout = aiohttp.ClientTimeout(total=float(timeout))

        self._session = session
        self._owns_session = session is None
        # Одна учётная запись = одна очередь. Лок сериализует запросы, а
        # `_next_allowed_at` держит между ними обязательную паузу — даже если
        # вызывающий код запустит батчи через `asyncio.gather`.
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    # ── Жизненный цикл ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "TradernetClient":
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
        self._session = None

    # ── Публичный интерфейс ──────────────────────────────────────────────────

    async def fetch_daily_candles(
        self,
        tickers: Sequence[str],
        date_from: datetime,
        date_to: datetime,
    ) -> dict[str, list[RawCandle]]:
        """Дневные свечи по списку тикеров ОДНИМ запросом (батч ≤ 50).

        Возвращает `{тикер: [свечи]}`. Тикеры, по которым брокер ничего не
        отдал (делистинг, опечатка, нет подписки на рынок), в словаре просто
        отсутствуют — молчаливо подставлять им пустой список нельзя: «нет
        данных» и «данных не запрашивали» должны различаться выше по стеку.
        """
        clean = [t.strip().upper() for t in tickers if t and t.strip()]
        if not clean:
            return {}
        if len(clean) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Батч из {len(clean)} тикеров превышает потолок "
                f"{self.MAX_BATCH_SIZE}; нарежьте вход через chunked()"
            )

        params = {
            "id":           ",".join(clean),
            "timeframe":    TIMEFRAME_DAILY_MINUTES,
            "date_from":    date_from.strftime(DATE_FORMAT),
            "date_to":      date_to.strftime(DATE_FORMAT),
            "intervalMode": self.interval_mode,
            "count":        -1,
        }
        raw = await self._post("getHloc", params)
        parsed = parse_hloc_response(raw, clean)

        missing = [t for t in clean if t not in parsed]
        if missing:
            logger.warning(
                "getHloc не вернул данных по %d из %d тикеров: %s",
                len(missing), len(clean), ", ".join(missing[:10]),
            )
        return parsed

    async def fetch_all(
        self,
        tickers: Sequence[str],
        date_from: datetime,
        date_to: datetime,
    ) -> dict[str, list[RawCandle]]:
        """Тот же запрос, но с автоматической нарезкой на батчи.

        Удобная обёртка для ручных прогонов и смоука. Основной конвейер ходит
        батчами сам (`etl_processor`), чтобы писать в БД по мере готовности, а
        не копить весь универсум в памяти.
        """
        out: dict[str, list[RawCandle]] = {}
        for batch in chunked(list(tickers), self.batch_size):
            out.update(await self.fetch_daily_candles(batch, date_from, date_to))
        return out

    # ── Транспорт ────────────────────────────────────────────────────────────

    async def _post(self, cmd: str, params: Mapping[str, Any]) -> dict:
        """POST с троттлингом, повторами и экспоненциальным backoff.

        Порядок именно такой: лок → пауза → попытка. Пауза считается ВНУТРИ
        лока, поэтому две конкурентные задачи не могут «сговориться» и уйти в
        сеть одновременно, отсчитав каждая свою паузу от общего прошлого.
        """
        if self._session is None:
            raise RuntimeError("TradernetClient используется вне `async with`")

        url = f"{self._base_url}/api/{cmd}"
        last_error: Exception | None = None

        async with self._lock:
            for attempt in range(1, self.max_retries + 1):
                await self._throttle()

                # Подпись включает timestamp, поэтому пересобирается на КАЖДОЙ
                # попытке: повтор со старым timestamp сервер отвергнет как
                # протухший, и backoff превратился бы в серию отказов подписи.
                signed = self._auth.sign(params)

                try:
                    async with self._session.post(
                        url, data=signed.body.encode("utf-8"),
                        headers=signed.headers,
                    ) as resp:
                        text = await resp.text()
                        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))

                        if resp.status == 429:
                            last_error = RateLimitedError(f"HTTP 429 от {url}")
                            await self._sleep_backoff(attempt, retry_after,
                                                      reason="HTTP 429")
                            continue

                        if resp.status >= 500 or _is_waf_challenge(resp.status, text):
                            last_error = TradernetError(
                                f"HTTP {resp.status} от {url}: {text[:200]}")
                            await self._sleep_backoff(attempt, retry_after,
                                                      reason=f"HTTP {resp.status}")
                            continue

                        if resp.status != 200:
                            # Прочие 4xx повтором не лечатся — падаем сразу,
                            # чтобы не жечь попытки и не маскировать причину.
                            raise TradernetError(
                                f"HTTP {resp.status} от {url}: {text[:200]}")

                        data = _decode_json(text, url)
                        _raise_on_api_error(data)
                        return data

                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    # Сетевые обрывы у Tradernet носят массовый характер при
                    # частых запросах (в основном приложении это видно как
                    # SSL EOF) — то же самое лечение, что и у 429.
                    last_error = exc
                    if attempt == self.max_retries:
                        break
                    await self._sleep_backoff(attempt, None,
                                              reason=type(exc).__name__)

        assert last_error is not None
        if isinstance(last_error, RateLimitedError):
            raise RateLimitedError(
                f"Лимит запросов не отпустил за {self.max_retries} попыток: {last_error}"
            ) from last_error
        raise TradernetError(
            f"Запрос {cmd} не удался за {self.max_retries} попыток: {last_error}"
        ) from last_error

    async def _throttle(self) -> None:
        """Выдержать обязательную паузу между запросами."""
        loop = asyncio.get_running_loop()
        wait = self._next_allowed_at - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        self._next_allowed_at = loop.time() + self.batch_delay

    async def _sleep_backoff(self, attempt: int, retry_after: float | None,
                             *, reason: str) -> None:
        """Экспоненциальная пауза `base * 2^(attempt-1)` с джиттером.

        `Retry-After` сильнее собственного расчёта: сервер знает, когда окно
        реально откроется, и игнорировать его — верный способ продлить бан.

        Джиттер (±25%) нужен на случай, когда сервис поднимут в нескольких
        репликах: без него все реплики, получив 429 одновременно, повторят
        тоже одновременно и воспроизведут ту же волну.
        """
        computed = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
        delay = max(retry_after or 0.0, computed)
        delay *= 1.0 + random.uniform(-0.25, 0.25)
        delay = min(max(delay, 0.0), self.backoff_max)
        logger.warning(
            "%s — попытка %d/%d, повтор через %.1f с",
            reason, attempt, self.max_retries, delay,
        )
        await asyncio.sleep(delay)


# ── Разбор ответа ────────────────────────────────────────────────────────────


def parse_hloc_response(raw: Mapping[str, Any],
                        requested: Sequence[str]) -> dict[str, list[RawCandle]]:
    """Разобрать ответ `getHloc` в `{тикер: [свечи]}`.

    Tradernet отдаёт свечи в трёх разных формах — какая именно придёт, зависит
    от региона и версии шлюза, отвечающего на запрос. Все три встречались в
    проде основного приложения (`src/freedom_portfolio/history.py`), поэтому
    поддержаны здесь целиком, а не «та, что видели последней».

    A. Параллельные массивы, ключ — тикер (форма официального SDK)::

           {"hloc":    {"AAPL.US": [[h, l, o, c], …]},
            "xSeries": {"AAPL.US": [ts, …]},
            "vl":      {"AAPL.US": [v, …]}}

       🔴 Порядок в строке — **high, low, open, close**, буквально «hloc».
       Прочитать его как OHLC — тихая катастрофа: `close` станет `low`, и
       весь ряд доходностей окажется смещённым, но правдоподобным.
       Подтверждено README SDK: `columns=["high","low","open","close"]`.

    B. Вложенные ряды под тикером (форма документации tradernet.kz)::

           {"hloc": {"AAPL.US": {"xSeries": [ts, …],
                                  "yPrices": [{o,h,l,c,v}, …]}}}

    C. Плоский список словарей (легаси)::

           {"hloc": [{t,o,h,l,c,v}, …]}

       Тикера внутри нет, поэтому форма принимается ТОЛЬКО когда запрошен
       ровно один тикер. В батче приписать такой список некому: угадывание
       здесь означало бы записать чужие цены под чужим именем.
    """
    body = raw.get("result", raw) if isinstance(raw, Mapping) else raw
    if not isinstance(body, Mapping):
        logger.warning("getHloc: неожиданный тип ответа %s", type(body).__name__)
        return {}

    hloc = body.get("hloc")
    out: dict[str, list[RawCandle]] = {}

    # ── Форма B ──────────────────────────────────────────────────────────────
    if isinstance(hloc, Mapping):
        for ticker, payload in hloc.items():
            if isinstance(payload, Mapping) and "yPrices" in payload:
                candles = _parse_nested_series(payload)
                if candles:
                    out[str(ticker).upper()] = candles
        if out:
            return out

    # ── Форма A ──────────────────────────────────────────────────────────────
    if isinstance(hloc, Mapping) and isinstance(body.get("xSeries"), Mapping):
        x_map = body["xSeries"]
        v_map = body.get("vl") if isinstance(body.get("vl"), Mapping) else {}
        for ticker, rows in hloc.items():
            key = str(ticker).upper()
            candles = _parse_parallel_arrays(
                rows, x_map.get(ticker), (v_map or {}).get(ticker))
            if candles:
                out[key] = candles
        if out:
            return out

    # ── Форма C ──────────────────────────────────────────────────────────────
    flat = hloc if isinstance(hloc, list) else (body if isinstance(body, list) else None)
    if isinstance(flat, list) and len(requested) == 1:
        candles = [c for c in (_parse_flat_row(r) for r in flat if isinstance(r, Mapping)) if c]
        if candles:
            return {requested[0].upper(): candles}

    if not out:
        logger.warning(
            "getHloc: не удалось разобрать ответ (ключи=%s, запрошено %d тикеров)",
            list(body.keys())[:10], len(requested),
        )
    return out


def _parse_parallel_arrays(rows: Any, timestamps: Any, volumes: Any) -> list[RawCandle]:
    """Форма A: `[[h,l,o,c], …]` + отдельные массивы времени и объёма."""
    if not isinstance(rows, list) or not isinstance(timestamps, list):
        return []
    vols = volumes if isinstance(volumes, list) else []
    out: list[RawCandle] = []
    for i in range(min(len(rows), len(timestamps))):
        row = rows[i]
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        high, low, open_, close = (_as_float(row[0]), _as_float(row[1]),
                                   _as_float(row[2]), _as_float(row[3]))
        out.append(RawCandle(
            ts=_normalize_ts(timestamps[i]),
            open=open_, high=high, low=low, close=close,
            volume=_as_float(vols[i]) if i < len(vols) else None,
        ))
    return out


def _parse_nested_series(payload: Mapping[str, Any]) -> list[RawCandle]:
    """Форма B: `{"xSeries": [...], "yPrices": [{o,h,l,c,v}, ...]}`."""
    timestamps = payload.get("xSeries")
    prices = payload.get("yPrices")
    if not isinstance(timestamps, list) or not isinstance(prices, list):
        return []
    out: list[RawCandle] = []
    for i in range(min(len(timestamps), len(prices))):
        point = prices[i]
        if not isinstance(point, Mapping):
            continue
        out.append(RawCandle(
            ts=_normalize_ts(timestamps[i]),
            open=_pick(point, "o", "open"),
            high=_pick(point, "h", "high"),
            low=_pick(point, "l", "low"),
            close=_pick(point, "c", "close"),
            volume=_pick(point, "v", "vol", "volume"),
        ))
    return out


def _parse_flat_row(row: Mapping[str, Any]) -> RawCandle | None:
    """Форма C: один словарь `{t,o,h,l,c,v}`."""
    ts = row.get("t") or row.get("ts") or row.get("time")
    if ts is None:
        return None
    return RawCandle(
        ts=_normalize_ts(ts),
        open=_pick(row, "o", "open"),
        high=_pick(row, "h", "high"),
        low=_pick(row, "l", "low"),
        close=_pick(row, "c", "close"),
        volume=_pick(row, "v", "vol", "volume"),
    )


# ── Мелкие помощники ─────────────────────────────────────────────────────────


def _pick(source: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in source:
            value = _as_float(source[name])
            if value is not None:
                return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN не сравним сам с собой и просочился бы в БД как «цена»,
    # ломая любые агрегаты ниже по конвейеру.
    return None if result != result else result


def _normalize_ts(value: Any) -> int:
    """Привести отметку времени к UNIX-СЕКУНДАМ.

    Шлюзы Tradernet отдают то секунды, то миллисекунды. Различить их можно
    только по порядку величины: 10**12 секунд — это год 33658, столько
    исторических котировок не бывает, значит это миллисекунды.
    """
    try:
        ts = int(float(value))
    except (TypeError, ValueError):
        return 0
    return ts // 1000 if ts > 10 ** 12 else ts


def _decode_json(text: str, url: str) -> dict:
    import json
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise TradernetError(f"Не-JSON ответ от {url}: {text[:200]}") from exc
    if not isinstance(data, dict):
        raise TradernetError(f"Ожидался объект JSON от {url}, получен {type(data).__name__}")
    return data


def _raise_on_api_error(data: Mapping[str, Any]) -> None:
    """Ошибка Tradernet приезжает в теле с HTTP 200 — проверять надо явно."""
    err = data.get("errMsg") or data.get("error") or data.get("err")
    code = data.get("code")
    if not err and (not isinstance(code, int) or code == 0):
        return
    message = str(err) if err else f"code={code}"
    if isinstance(code, int) and code in _FATAL_CODES:
        raise TradernetAuthError(f"{_FATAL_CODES[code]} (code={code}): {message}")
    raise TradernetError(f"Tradernet вернул ошибку (code={code}): {message}")


def _parse_retry_after(value: str | None) -> float | None:
    """`Retry-After` в секундах. HTTP-date форму игнорируем — брокер её не шлёт."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _is_waf_challenge(status: int, text: str) -> bool:
    """HTML-заглушка Cloudflare вместо JSON — повторяемая, а не фатальная.

    В основном приложении это регулярный сценарий: IP-диапазоны облака
    попадают в фильтр WAF, и запрос возвращает 403 с HTML. Отличается от
    настоящего 403 («доступ запрещён») по типу содержимого.
    """
    if status != 403:
        return False
    head = text[:500].lower()
    return "<!doctype" in head or "<html" in head or "cloudflare" in head
