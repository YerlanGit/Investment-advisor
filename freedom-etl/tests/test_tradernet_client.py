"""Клиент Tradernet: батчинг, лимиты, backoff и разбор трёх форм ответа.

Сеть не трогается: `TradernetClient` принимает готовую сессию, а `asyncio.sleep`
подменяется счётчиком. Подмена сна — не только про скорость: она превращает
паузы в НАБЛЮДАЕМОЕ значение, и тест может утверждать, что backoff растёт
экспоненциально, а не просто «что-то поспало».
"""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest import mock

from services.auth import TradernetAuth
from services.tradernet_client import (
    RateLimitedError,
    TradernetAuthError,
    TradernetClient,
    TradernetError,
    chunked,
    parse_hloc_response,
)

# Заведомо ненастоящие ключи — словарные намеренно, см. `test_auth.py`.
PUBLIC_KEY = "EXAMPLE-PUBLIC-KEY-FOR-UNIT-TESTS"
SECRET_KEY = "EXAMPLE-SECRET-KEY-FOR-UNIT-TESTS"

DATE_FROM = datetime(2026, 8, 1)
DATE_TO = datetime(2026, 8, 9)


# ── Тестовые двойники ────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status: int, text: str, headers: dict | None = None) -> None:
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None


class FakeSession:
    """Отдаёт заранее заданную очередь ответов и запоминает все запросы."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def post(self, url, *, data=None, headers=None):
        self.requests.append({"url": url, "data": data, "headers": headers})
        if not self._responses:
            raise AssertionError("Запросов больше, чем подготовлено ответов")
        return self._responses.pop(0)


def make_client(responses: list[FakeResponse], **kwargs) -> tuple[TradernetClient, FakeSession]:
    session = FakeSession(responses)
    client = TradernetClient(
        TradernetAuth(PUBLIC_KEY, SECRET_KEY),
        session=session,
        **kwargs,
    )
    return client, session


class SleepRecorder:
    """Замена `asyncio.sleep`, запоминающая длительности."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


# ── Нарезка на батчи ─────────────────────────────────────────────────────────


class ChunkingTest(unittest.TestCase):

    def test_splits_by_requested_size(self) -> None:
        self.assertEqual(
            list(chunked([f"T{i}.US" for i in range(5)], 2)),
            [["T0.US", "T1.US"], ["T2.US", "T3.US"], ["T4.US"]],
        )

    def test_empty_input_yields_nothing(self) -> None:
        self.assertEqual(list(chunked([], 50)), [])

    def test_zero_size_is_rejected(self) -> None:
        """Иначе цикл вызывающего кода крутил бы пустые куски бесконечно."""
        with self.assertRaises(ValueError):
            list(chunked(["A.US"], 0))

    def test_batch_size_is_capped_at_api_limit(self) -> None:
        """Потолок 50 зажат В КЛИЕНТЕ: конфиг можно обойти, конструктор — нет."""
        client, _ = make_client([], batch_size=500)
        self.assertEqual(client.batch_size, TradernetClient.MAX_BATCH_SIZE)

    def test_batch_delay_cannot_go_below_two_seconds(self) -> None:
        """«Ускорить выгрузку» — самый вероятный способ получить бан на ключ."""
        client, _ = make_client([], batch_delay=0.1)
        self.assertEqual(client.batch_delay, TradernetClient.MIN_REQUEST_INTERVAL)

    def test_oversized_batch_is_refused(self) -> None:
        client, _ = make_client([])
        with self.assertRaises(ValueError):
            _run(client.fetch_daily_candles(
                [f"T{i}.US" for i in range(51)], DATE_FROM, DATE_TO))


# ── Формирование запроса ─────────────────────────────────────────────────────


class RequestShapeTest(unittest.IsolatedAsyncioTestCase):

    async def test_tickers_go_as_comma_separated_id(self) -> None:
        """Батчинг ТЗ: один HTTP-запрос на 50 тикеров, а не 50 запросов."""
        import json

        client, session = make_client([FakeResponse(200, '{"hloc":{}}')])
        await client.fetch_daily_candles(["AAPL.US", "MSFT.US", "TSLA.US"],
                                         DATE_FROM, DATE_TO)

        self.assertEqual(len(session.requests), 1, "должен уйти РОВНО один запрос")
        params = json.loads(session.requests[0]["data"].decode())
        self.assertEqual(params["id"], "AAPL.US,MSFT.US,TSLA.US")

    async def test_daily_timeframe_is_1440_minutes(self) -> None:
        import json

        client, session = make_client([FakeResponse(200, '{"hloc":{}}')])
        await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        params = json.loads(session.requests[0]["data"].decode())
        self.assertEqual(params["timeframe"], 1440)

    async def test_dates_use_dd_mm_yyyy_not_iso(self) -> None:
        """🔴 ISO-формат сервер принимает, но отвечает ПУСТЫМ массивом свечей."""
        import json

        client, session = make_client([FakeResponse(200, '{"hloc":{}}')])
        await client.fetch_daily_candles(
            ["AAPL.US"], datetime(2026, 8, 1), datetime(2026, 8, 9))
        params = json.loads(session.requests[0]["data"].decode())
        self.assertEqual(params["date_from"], "01.08.2026 00:00")
        self.assertEqual(params["date_to"], "09.08.2026 00:00")

    async def test_url_targets_the_command_path(self) -> None:
        client, session = make_client([FakeResponse(200, '{"hloc":{}}')],
                                      base_url="https://tradernet.com")
        await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        self.assertEqual(session.requests[0]["url"], "https://tradernet.com/api/getHloc")

    async def test_request_carries_signed_headers(self) -> None:
        client, session = make_client([FakeResponse(200, '{"hloc":{}}')])
        await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        headers = session.requests[0]["headers"]
        self.assertEqual(headers["X-NtApi-PublicKey"], PUBLIC_KEY)
        self.assertIn("X-NtApi-Sig", headers)
        self.assertIn("X-NtApi-Timestamp", headers)

    async def test_tickers_are_normalised_and_blanks_dropped(self) -> None:
        import json

        client, session = make_client([FakeResponse(200, '{"hloc":{}}')])
        await client.fetch_daily_candles([" aapl.us ", "", "  "], DATE_FROM, DATE_TO)
        params = json.loads(session.requests[0]["data"].decode())
        self.assertEqual(params["id"], "AAPL.US")

    async def test_empty_ticker_list_makes_no_request(self) -> None:
        client, session = make_client([])
        self.assertEqual(await client.fetch_daily_candles([], DATE_FROM, DATE_TO), {})
        self.assertEqual(session.requests, [])


# ── Лимиты и повторы ─────────────────────────────────────────────────────────


class RateLimitTest(unittest.IsolatedAsyncioTestCase):

    async def test_429_is_retried_and_then_succeeds(self) -> None:
        client, session = make_client(
            [FakeResponse(429, "rate limited"),
             FakeResponse(429, "rate limited"),
             FakeResponse(200, '{"hloc":{}}')],
            backoff_base=2.0, max_retries=5,
        )
        sleeper = SleepRecorder()
        with mock.patch("asyncio.sleep", sleeper):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)

        self.assertEqual(len(session.requests), 3, "должно быть три попытки")

    async def test_backoff_grows_exponentially(self) -> None:
        """Пауза = base · 2^(n−1) с джиттером ±25% — каждая ступень выше прошлой."""
        client, _ = make_client(
            [FakeResponse(429, "x"), FakeResponse(429, "x"),
             FakeResponse(429, "x"), FakeResponse(200, '{"hloc":{}}')],
            backoff_base=10.0, batch_delay=2.0, max_retries=5,
        )
        sleeper = SleepRecorder()
        with mock.patch("asyncio.sleep", sleeper):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)

        # Первый сон — троттлинг (2 с), дальше три ступени backoff.
        backoffs = [d for d in sleeper.delays if d > 3.0]
        self.assertEqual(len(backoffs), 3)
        # Границы с учётом джиттера: 10·2^0=10 → [7.5, 12.5]; 20 → [15, 25]; 40 → [30, 50].
        for observed, base in zip(backoffs, (10.0, 20.0, 40.0)):
            self.assertGreaterEqual(observed, base * 0.75)
            self.assertLessEqual(observed, base * 1.25)
        self.assertLess(backoffs[0], backoffs[1])
        self.assertLess(backoffs[1], backoffs[2])

    async def test_backoff_is_capped(self) -> None:
        client, _ = make_client(
            [FakeResponse(429, "x")] * 4 + [FakeResponse(200, '{"hloc":{}}')],
            backoff_base=100.0, backoff_max=30.0, max_retries=6,
        )
        sleeper = SleepRecorder()
        with mock.patch("asyncio.sleep", sleeper):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        self.assertTrue(all(d <= 30.0 for d in sleeper.delays),
                        f"пауза превысила потолок: {sleeper.delays}")

    async def test_retry_after_header_wins_over_computed_backoff(self) -> None:
        """Сервер знает, когда окно откроется; игнорировать его — продлить бан."""
        client, _ = make_client(
            [FakeResponse(429, "x", {"Retry-After": "45"}),
             FakeResponse(200, '{"hloc":{}}')],
            backoff_base=2.0, backoff_max=120.0,
        )
        sleeper = SleepRecorder()
        with mock.patch("asyncio.sleep", sleeper):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)

        self.assertTrue(any(d >= 45 * 0.75 for d in sleeper.delays),
                        f"Retry-After проигнорирован: {sleeper.delays}")

    async def test_exhausted_retries_raise_rate_limited(self) -> None:
        client, _ = make_client([FakeResponse(429, "x")] * 3, max_retries=3)
        with mock.patch("asyncio.sleep", SleepRecorder()):
            with self.assertRaises(RateLimitedError):
                await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)

    async def test_signature_is_rebuilt_on_every_retry(self) -> None:
        """Повтор со старым timestamp сервер отвергнет как протухший."""
        client, session = make_client(
            [FakeResponse(429, "x"), FakeResponse(200, '{"hloc":{}}')],
            max_retries=3,
        )
        with mock.patch("asyncio.sleep", SleepRecorder()):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)

        first, second = session.requests[0]["headers"], session.requests[1]["headers"]
        # Подпись пересобрана: объекты заголовков — разные словари.
        self.assertIsNot(first, second)
        self.assertIn("X-NtApi-Sig", second)

    async def test_server_errors_are_retried(self) -> None:
        client, session = make_client(
            [FakeResponse(503, "bad gateway"), FakeResponse(200, '{"hloc":{}}')])
        with mock.patch("asyncio.sleep", SleepRecorder()):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        self.assertEqual(len(session.requests), 2)

    async def test_cloudflare_html_challenge_is_retried(self) -> None:
        """403 с HTML — заглушка WAF, а не отказ в доступе; лечится повтором."""
        client, session = make_client(
            [FakeResponse(403, "<!DOCTYPE html><html>cloudflare</html>"),
             FakeResponse(200, '{"hloc":{}}')])
        with mock.patch("asyncio.sleep", SleepRecorder()):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        self.assertEqual(len(session.requests), 2)

    async def test_plain_4xx_fails_immediately(self) -> None:
        """Обычные 4xx повтором не лечатся — не жжём попытки и не прячем причину."""
        client, session = make_client([FakeResponse(404, "not found")])
        with mock.patch("asyncio.sleep", SleepRecorder()):
            with self.assertRaises(TradernetError):
                await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        self.assertEqual(len(session.requests), 1)


class ApiErrorTest(unittest.IsolatedAsyncioTestCase):
    """Ошибки Tradernet приезжают в ТЕЛЕ с HTTP 200."""

    async def test_invalid_signature_code_is_fatal(self) -> None:
        client, session = make_client(
            [FakeResponse(200, '{"code":4,"errMsg":"Invalid signature"}')])
        with self.assertRaises(TradernetAuthError):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)
        self.assertEqual(len(session.requests), 1, "подпись повтором не чинится")

    async def test_invalid_credentials_code_is_fatal(self) -> None:
        client, _ = make_client(
            [FakeResponse(200, '{"code":12,"errMsg":"Invalid credentials"}')])
        with self.assertRaises(TradernetAuthError):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)

    async def test_non_json_body_is_reported(self) -> None:
        client, _ = make_client([FakeResponse(200, "<html>oops</html>")])
        with self.assertRaises(TradernetError):
            await client.fetch_daily_candles(["AAPL.US"], DATE_FROM, DATE_TO)


class ThrottleTest(unittest.IsolatedAsyncioTestCase):

    async def test_gap_is_kept_between_consecutive_batches(self) -> None:
        """Пауза живёт ВНУТРИ клиента и не может «потеряться» в цикле снаружи."""
        client, _ = make_client(
            [FakeResponse(200, '{"hloc":{}}'), FakeResponse(200, '{"hloc":{}}')],
            batch_delay=2.0,
        )
        sleeper = SleepRecorder()
        with mock.patch("asyncio.sleep", sleeper):
            await client.fetch_daily_candles(["A.US"], DATE_FROM, DATE_TO)
            await client.fetch_daily_candles(["B.US"], DATE_FROM, DATE_TO)

        self.assertTrue(any(d > 1.0 for d in sleeper.delays),
                        f"между батчами не было паузы: {sleeper.delays}")


# ── Разбор ответа ────────────────────────────────────────────────────────────


class HlocParsingTest(unittest.TestCase):

    def test_shape_a_parallel_arrays_column_order_is_hloc(self) -> None:
        """🔴 Строка — [high, low, open, close]. Прочитать как OHLC = сместить весь ряд."""
        raw = {
            "hloc":    {"AAPL.US": [[195.0, 190.0, 191.0, 194.0]]},
            "xSeries": {"AAPL.US": [1785960000]},
            "vl":      {"AAPL.US": [1_000_000]},
        }
        candles = parse_hloc_response(raw, ["AAPL.US"])["AAPL.US"]
        self.assertEqual(len(candles), 1)
        bar = candles[0]
        self.assertEqual(bar.high, 195.0)
        self.assertEqual(bar.low, 190.0)
        self.assertEqual(bar.open, 191.0)
        self.assertEqual(bar.close, 194.0)
        self.assertEqual(bar.volume, 1_000_000)

    def test_shape_a_returns_every_ticker_of_the_batch(self) -> None:
        """Ответ на батч — словарь по тикерам; берём ВСЕ ключи, а не первый."""
        raw = {
            "hloc": {
                "AAPL.US": [[1, 1, 1, 194.0]],
                "MSFT.US": [[2, 2, 2, 410.0]],
                "TSLA.US": [[3, 3, 3, 250.0]],
            },
            "xSeries": {
                "AAPL.US": [1785960000],
                "MSFT.US": [1785960000],
                "TSLA.US": [1785960000],
            },
        }
        parsed = parse_hloc_response(raw, ["AAPL.US", "MSFT.US", "TSLA.US"])
        self.assertEqual(set(parsed), {"AAPL.US", "MSFT.US", "TSLA.US"})
        self.assertEqual(parsed["MSFT.US"][0].close, 410.0)

    def test_shape_b_nested_series(self) -> None:
        raw = {"hloc": {"AAPL.US": {
            "xSeries": [1785960000],
            "yPrices": [{"o": 191.0, "h": 195.0, "l": 190.0, "c": 194.0, "v": 5}],
        }}}
        bar = parse_hloc_response(raw, ["AAPL.US"])["AAPL.US"][0]
        self.assertEqual((bar.open, bar.high, bar.low, bar.close), (191.0, 195.0, 190.0, 194.0))
        self.assertEqual(bar.volume, 5)

    def test_shape_c_flat_list_for_single_ticker(self) -> None:
        raw = {"hloc": [{"t": 1785960000, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10}]}
        bar = parse_hloc_response(raw, ["AAPL.US"])["AAPL.US"][0]
        self.assertEqual(bar.close, 1.5)

    def test_shape_c_is_refused_for_a_batch(self) -> None:
        """Тикера внутри нет — приписать список кому-то из батча значит соврать."""
        raw = {"hloc": [{"t": 1785960000, "c": 1.5}]}
        self.assertEqual(parse_hloc_response(raw, ["AAPL.US", "MSFT.US"]), {})

    def test_result_envelope_is_unwrapped(self) -> None:
        raw = {"result": {"hloc": {"AAPL.US": [[1, 1, 1, 194.0]]},
                          "xSeries": {"AAPL.US": [1785960000]}}}
        self.assertIn("AAPL.US", parse_hloc_response(raw, ["AAPL.US"]))

    def test_millisecond_timestamps_are_normalised(self) -> None:
        """Шлюзы отдают то секунды, то миллисекунды — различаем по порядку величины."""
        raw = {"hloc": {"AAPL.US": [[1, 1, 1, 194.0]]},
               "xSeries": {"AAPL.US": [1785960000000]}}
        self.assertEqual(parse_hloc_response(raw, ["AAPL.US"])["AAPL.US"][0].ts, 1785960000)

    def test_missing_ticker_is_absent_not_empty(self) -> None:
        """«Нет данных» и «не запрашивали» обязаны различаться выше по стеку."""
        raw = {"hloc": {"AAPL.US": [[1, 1, 1, 194.0]]},
               "xSeries": {"AAPL.US": [1785960000]}}
        parsed = parse_hloc_response(raw, ["AAPL.US", "DELISTED.US"])
        self.assertIn("AAPL.US", parsed)
        self.assertNotIn("DELISTED.US", parsed)

    def test_short_rows_are_skipped(self) -> None:
        raw = {"hloc": {"AAPL.US": [[1, 2], [195.0, 190.0, 191.0, 194.0]]},
               "xSeries": {"AAPL.US": [1785960000, 1786046400]}}
        self.assertEqual(len(parse_hloc_response(raw, ["AAPL.US"])["AAPL.US"]), 1)

    def test_unknown_shape_returns_empty(self) -> None:
        self.assertEqual(parse_hloc_response({"unexpected": 1}, ["AAPL.US"]), {})

    def test_nan_prices_become_none(self) -> None:
        """NaN не сравним сам с собой и просочился бы в БД как «цена»."""
        raw = {"hloc": {"AAPL.US": [[float("nan"), 190.0, 191.0, 194.0]]},
               "xSeries": {"AAPL.US": [1785960000]}}
        self.assertIsNone(parse_hloc_response(raw, ["AAPL.US"])["AAPL.US"][0].high)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
