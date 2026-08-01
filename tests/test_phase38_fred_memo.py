"""
Phase-38 · E-1: мемоизация FRED-серии в кредитном слое (2026-08-01).

Дефект (профилирование `analyze_all`): `_fred_hy_provider` вызывается ПО КАЖДОМУ
тикеру и каждый раз тянет ОДНУ И ТУ ЖЕ рыночную серию `BAMLH0A0HYM2` — это
macro-wide прокси качества «C», идентичный для всех бумаг по построению.
Портфель на 15 позиций делал 15 одинаковых HTTP-запросов; на это уходило 77%
времени счёта. 24-часовой SQLite-кэш модуля ключуется ПО ТИКЕРУ и не помогает.

Три свойства, которые обязаны выполняться одновременно (иначе лечение хуже
болезни):

  1. внутри одного отчёта — ОДИН запрос независимо от числа бумаг;
  2. кэш НЕ вечный: бот живёт долго, и `lru_cache` заморозил бы спред на
     значении при старте контейнера;
  3. ПРОВАЛ не кэшируется надолго: закэшировать пустой ответ на 12 часов значит
     ослепить кредитный пиллар на полсуток из-за одной сетевой ошибки.
"""

import time
import unittest
from unittest.mock import patch

import finance.cds_feed as cf


def _csv_response(rows=21):
    class _R:
        text = "DATE,BAMLH0A0HYM2\n" + "\n".join(
            f"2026-07-{d:02d},2.8{d % 10}" for d in range(10, 10 + rows))

        def raise_for_status(self):
            return None
    return _R()


class FredMemoTest(unittest.TestCase):

    def setUp(self):
        cf._clear_fred_memo()
        self.addCleanup(cf._clear_fred_memo)

    def _run(self, n_tickers, get_fn):
        calls = []

        def _get(url, **kw):
            calls.append(url)
            return get_fn(url)

        with patch.object(cf, "requests") as rq:
            rq.get = _get
            out = [cf._fred_observations(cf._FRED_HY_SERIES, days=14)
                   for _ in range(n_tickers)]
        return calls, out

    # ── свойство 1: один запрос на отчёт ────────────────────────────────
    def test_one_request_for_many_tickers(self):
        calls, out = self._run(15, lambda url: _csv_response())
        self.assertEqual(len(calls), 1,
                         "15 бумаг обязаны дать ОДИН запрос к общей серии")
        self.assertTrue(all(o == out[0] for o in out), "данные обязаны совпасть")
        self.assertEqual(len(out[0]), 21)

    def test_failure_is_not_retried_within_one_report(self):
        def _boom(url):
            raise RuntimeError("network down")
        calls, out = self._run(15, _boom)
        self.assertEqual(len(calls), 1, "провал не должен повторяться 15 раз")
        self.assertEqual(out[0], [], "при провале возвращается пустой список")

    # ── свойство 2: кэш не вечный ───────────────────────────────────────
    def test_success_ttl_is_bounded_and_daily(self):
        """Серия дневная (EOD) — TTL измеряется часами, не бесконечностью."""
        self.assertGreater(cf._FRED_MEMO_TTL_SEC, 3600)
        self.assertLessEqual(cf._FRED_MEMO_TTL_SEC, 24 * 3600)

    def test_expired_entry_is_refetched(self):
        calls, _ = self._run(1, lambda url: _csv_response())
        self.assertEqual(len(calls), 1)
        # состарить запись искусственно
        with cf._FRED_MEMO_LOCK:
            for k, (_, v) in list(cf._FRED_MEMO.items()):
                cf._FRED_MEMO[k] = (time.time() - cf._FRED_MEMO_TTL_SEC - 1, v)
        calls2, _ = self._run(1, lambda url: _csv_response())
        self.assertEqual(len(calls2), 1, "после истечения TTL нужен новый запрос")

    # ── свойство 3: провал забывается быстро ────────────────────────────
    def test_failure_ttl_is_much_shorter_than_success(self):
        self.assertLess(cf._FRED_MEMO_FAIL_TTL_SEC, cf._FRED_MEMO_TTL_SEC / 10,
                        "сетевая ошибка не должна ослеплять пиллар надолго")

    def test_next_report_retries_after_failure(self):
        def _boom(url):
            raise RuntimeError("down")
        self._run(3, _boom)
        with cf._FRED_MEMO_LOCK:
            for k, (_, v) in list(cf._FRED_MEMO.items()):
                cf._FRED_MEMO[k] = (time.time() - cf._FRED_MEMO_FAIL_TTL_SEC - 1, v)
        calls, out = self._run(1, lambda url: _csv_response())
        self.assertEqual(len(calls), 1, "после короткого TTL источник пробуется снова")
        self.assertEqual(len(out[0]), 21, "и данные восстанавливаются")

    # ── корректность кэша ───────────────────────────────────────────────
    def test_cache_returns_a_copy(self):
        """Вызывающий не должен уметь испортить общий кэш мутацией результата."""
        calls, out = self._run(1, lambda url: _csv_response())
        out[0].clear()
        _, out2 = self._run(1, lambda url: _csv_response())
        self.assertEqual(len(out2[0]), 21)

    def test_different_series_cached_separately(self):
        calls = []

        def _get(url, **kw):
            calls.append(url)
            return _csv_response()
        with patch.object(cf, "requests") as rq:
            rq.get = _get
            cf._fred_observations(cf._FRED_HY_SERIES, days=14)
            cf._fred_observations(cf._FRED_IG_SERIES, days=14)
            cf._fred_observations(cf._FRED_HY_SERIES, days=14)
        self.assertEqual(len(calls), 2, "две разные серии — два запроса, повтор — из кэша")

    def test_different_window_cached_separately(self):
        calls = []

        def _get(url, **kw):
            calls.append(url)
            return _csv_response()
        with patch.object(cf, "requests") as rq:
            rq.get = _get
            cf._fred_observations(cf._FRED_HY_SERIES, days=14)
            cf._fred_observations(cf._FRED_HY_SERIES, days=30)
        self.assertEqual(len(calls), 2, "окно входит в ключ кэша")

    # ── контракт не изменился ───────────────────────────────────────────
    def test_uncached_layer_still_exists_and_parses(self):
        """Сетевой слой остался отдельной функцией — мемо его лишь оборачивает."""
        self.assertTrue(callable(cf._fred_observations_uncached))
        with patch.object(cf, "requests") as rq:
            rq.get = lambda url, **kw: _csv_response()
            raw = cf._fred_observations_uncached(cf._FRED_HY_SERIES, days=14)
        self.assertEqual(len(raw), 21)
        self.assertEqual(raw, sorted(raw, key=lambda r: r[0]), "порядок по возрастанию")


if __name__ == "__main__":
    unittest.main()
