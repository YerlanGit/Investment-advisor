"""
Phase-35 · Фаза 1 — изоляция кэша цен по провайдеру (2026-07-28).

Дефект: ключ кэша был ``(ticker, days)``.  Как только появляется второй
провайдер, ``AAPL_US_1825d.pkl``, записанный Tradernet, в течение часового TTL
отдаётся другому провайдеру как его собственные данные.  Симптом плавающий
(только внутри окна TTL и только если оба провайдера запрашивались в одном
контейнере), последствия — неверный источник в CoVe и, что серьёзнее,
смешанные конвенции корректировок в одной ковариационной матрице.

Плюс две правки, найденные при предвыполненном анализе фазы:
  • запись в кэш была неатомарной (гонка между параллельными отчётами);
  • вытеснения не было вообще (``/tmp`` на Cloud Run — это RAM при лимите 2 GiB).
"""

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from freedom_portfolio import history as H


def _series(n: int = 5, start: float = 100.0) -> pd.Series:
    idx = pd.bdate_range(end="2026-07-24", periods=n)
    return pd.Series([start + i for i in range(n)], index=idx, dtype=float)


class _CacheDirMixin:
    """Каждый тест — на своём каталоге; общий /tmp-кэш не используется."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._dir_patch = patch.object(H, "_CACHE_DIR", Path(self._tmp.name))
        self._dir_patch.start()
        # Сброс троттлинга вытеснения — иначе порядок тестов влияет на результат.
        self._evict_reset = patch.object(H, "_last_evict_ts", 0.0)
        self._evict_reset.start()

    def tearDown(self):
        self._evict_reset.stop()
        self._dir_patch.stop()
        self._tmp.cleanup()


class CacheKeyTest(_CacheDirMixin, unittest.TestCase):
    """Ключ кэша обязан различать провайдера и конвенцию."""

    def test_defaults_describe_current_source(self):
        """Дефолты — текущий источник: существующие вызовы не меняют поведения."""
        self.assertEqual(H.DEFAULT_PROVIDER, "tradernet")
        self.assertEqual(H.DEFAULT_CONVENTION, "split_adjusted")
        self.assertEqual(H._cache_path("AAPL.US", 1825).name,
                         "tradernet__split_adjusted__AAPL_US_1825d.pkl")

    def test_provider_changes_path(self):
        self.assertNotEqual(H._cache_path("AAPL.US", 1825, "tradernet"),
                            H._cache_path("AAPL.US", 1825, "stooq"))

    def test_convention_changes_path(self):
        self.assertNotEqual(
            H._cache_path("AAPL.US", 1825, "eodhd", "split_adjusted"),
            H._cache_path("AAPL.US", 1825, "eodhd", "total_return"))

    def test_days_still_in_key(self):
        """Прежний компонент ключа не потерян."""
        self.assertNotEqual(H._cache_path("AAPL.US", 1825),
                            H._cache_path("AAPL.US", 730))

    def test_dots_and_slashes_sanitised(self):
        """Тикер не должен создавать вложенные каталоги."""
        path = H._cache_path("BRK/B.US", 1825)
        self.assertNotIn("/", path.name)
        self.assertEqual(path.parent, H._CACHE_DIR)


class CacheIsolationTest(_CacheDirMixin, unittest.TestCase):
    """Главный инвариант фазы: провайдеры не читают чужие ряды."""

    def test_cache_isolated_between_providers(self):
        """Запись Tradernet НЕ видна Stooq (межпровайдерное отравление)."""
        H._write_cache("AAPL.US", 1825, _series(), "tradernet")
        self.assertIsNotNone(H._read_cache("AAPL.US", 1825, "tradernet"))
        self.assertIsNone(H._read_cache("AAPL.US", 1825, "stooq"))

    def test_cache_isolated_between_conventions(self):
        """Один провайдер, разные конвенции — разные записи (случай EODHD)."""
        H._write_cache("AAPL.US", 1825, _series(), "eodhd", "split_adjusted")
        self.assertIsNotNone(
            H._read_cache("AAPL.US", 1825, "eodhd", "split_adjusted"))
        self.assertIsNone(
            H._read_cache("AAPL.US", 1825, "eodhd", "total_return"))

    def test_both_providers_coexist(self):
        """Оба ряда живут одновременно и не затирают друг друга."""
        H._write_cache("AAPL.US", 1825, _series(start=100.0), "tradernet")
        H._write_cache("AAPL.US", 1825, _series(start=900.0), "stooq")
        self.assertEqual(float(H._read_cache("AAPL.US", 1825, "tradernet").iloc[0]), 100.0)
        self.assertEqual(float(H._read_cache("AAPL.US", 1825, "stooq").iloc[0]), 900.0)

    def test_roundtrip_preserves_series(self):
        original = _series(7)
        H._write_cache("SPY.US", 1825, original)
        pd.testing.assert_series_equal(H._read_cache("SPY.US", 1825), original)


class CacheTtlTest(_CacheDirMixin, unittest.TestCase):
    """TTL не должен пострадать от смены схемы имён."""

    def test_fresh_entry_is_returned(self):
        H._write_cache("AAPL.US", 1825, _series())
        self.assertIsNotNone(H._read_cache("AAPL.US", 1825))

    def test_expired_entry_is_ignored(self):
        H._write_cache("AAPL.US", 1825, _series())
        path = H._cache_path("AAPL.US", 1825)
        stale = time.time() - H._CACHE_TTL_SECONDS - 60
        import os
        os.utime(path, (stale, stale))
        self.assertIsNone(H._read_cache("AAPL.US", 1825))

    def test_missing_entry_returns_none(self):
        self.assertIsNone(H._read_cache("NOPE.US", 1825))

    def test_corrupt_entry_degrades_to_none(self):
        """Битый файл → повторная загрузка, а не исключение."""
        H._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        H._cache_path("AAPL.US", 1825).write_bytes(b"not a pickle")
        self.assertIsNone(H._read_cache("AAPL.US", 1825))

    def test_empty_series_not_written(self):
        H._write_cache("AAPL.US", 1825, pd.Series(dtype=float))
        self.assertFalse(H._cache_path("AAPL.US", 1825).exists())


class AtomicWriteTest(_CacheDirMixin, unittest.TestCase):
    """Запись атомарна — иначе параллельные отчёты рвут файл."""

    def test_no_temp_files_left_behind(self):
        H._write_cache("AAPL.US", 1825, _series())
        leftovers = list(H._CACHE_DIR.glob("*.tmp"))
        self.assertEqual(leftovers, [], f"остались временные файлы: {leftovers}")

    def test_failed_write_leaves_no_partial_file(self):
        """Сбой на записи не оставляет ни временного файла, ни битого целевого."""
        with patch.object(pd.Series, "to_pickle", side_effect=OSError("disk full")):
            H._write_cache("AAPL.US", 1825, _series())
        self.assertFalse(H._cache_path("AAPL.US", 1825).exists())
        self.assertEqual(list(H._CACHE_DIR.glob("*.tmp")), [])

    def test_failed_write_keeps_previous_value(self):
        """Неудачная перезапись не портит уже лежащий валидный ряд.

        Это и есть смысл os.replace: читатель видит либо старое, либо новое.
        """
        H._write_cache("AAPL.US", 1825, _series(start=100.0))
        with patch.object(pd.Series, "to_pickle", side_effect=OSError("disk full")):
            H._write_cache("AAPL.US", 1825, _series(start=900.0))
        self.assertEqual(float(H._read_cache("AAPL.US", 1825).iloc[0]), 100.0)

    def test_concurrent_writers_produce_readable_file(self):
        """Восемь потоков пишут один ключ — файл остаётся читаемым."""
        from concurrent.futures import ThreadPoolExecutor

        def _write(i):
            H._write_cache("AAPL.US", 1825, _series(start=100.0 + i))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_write, range(8)))

        result = H._read_cache("AAPL.US", 1825)
        self.assertIsNotNone(result, "файл повреждён параллельной записью")
        self.assertEqual(len(result), 5)
        self.assertEqual(list(H._CACHE_DIR.glob("*.tmp")), [])


class CacheEvictionTest(_CacheDirMixin, unittest.TestCase):
    """Вытеснение: /tmp на Cloud Run — это RAM при лимите 2 GiB."""

    def _age(self, path: Path, seconds: float) -> None:
        import os
        old = time.time() - seconds
        os.utime(path, (old, old))

    def test_stale_files_evicted_on_write(self):
        H._write_cache("OLD.US", 1825, _series())
        stale_path = H._cache_path("OLD.US", 1825)
        self._age(stale_path, 3 * H._CACHE_TTL_SECONDS)

        H._last_evict_ts = 0.0
        H._write_cache("NEW.US", 1825, _series())

        self.assertFalse(stale_path.exists(), "просроченный файл не вытеснен")
        self.assertTrue(H._cache_path("NEW.US", 1825).exists(), "свежий файл удалён")

    def test_fresh_files_survive_eviction(self):
        H._write_cache("KEEP.US", 1825, _series())
        H._last_evict_ts = 0.0
        H._write_cache("OTHER.US", 1825, _series())
        self.assertTrue(H._cache_path("KEEP.US", 1825).exists())

    def test_eviction_is_throttled(self):
        """Скан каталога не делается на каждую запись (O(n) на тикер)."""
        H._write_cache("A.US", 1825, _series())
        first_ts = H._last_evict_ts
        self.assertGreater(first_ts, 0.0)
        H._write_cache("B.US", 1825, _series())
        self.assertEqual(H._last_evict_ts, first_ts, "уборка запустилась повторно")

    def test_eviction_ignores_temp_files(self):
        """Временный файл чужого потока трогать нельзя."""
        H._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        foreign = H._CACHE_DIR / "tradernet__split_adjusted__X_US_1825d.pkl.999.1.tmp"
        foreign.write_bytes(b"in flight")
        self._age(foreign, 5 * H._CACHE_TTL_SECONDS)

        H._last_evict_ts = 0.0
        H._write_cache("Z.US", 1825, _series())
        self.assertTrue(foreign.exists(), "вытеснение удалило чужой временный файл")


class BackwardCompatibilityTest(_CacheDirMixin, unittest.TestCase):
    """I-5: путь freedom не деградирует."""

    def test_get_candles_uses_default_key(self):
        """`get_candles` без новых аргументов пишет и читает свой файл."""
        with patch.object(H, "_fetch_hloc", return_value=[]), \
             patch.object(H, "_candles_to_series", return_value=_series()):
            H.get_candles(client=None, ticker="AAPL.US", days=1825)
        self.assertTrue(H._cache_path("AAPL.US", 1825).exists())

    def test_get_candles_cache_hit_skips_fetch(self):
        """Второй вызов берёт из кэша и не ходит на сервер."""
        H._write_cache("AAPL.US", 1825, _series())
        with patch.object(H, "_fetch_hloc") as fetch:
            result = H.get_candles(client=None, ticker="AAPL.US", days=1825)
            fetch.assert_not_called()
        self.assertEqual(len(result), 5)

    def test_get_candles_honours_provider_argument(self):
        """Провайдер пробрасывается до ключа — вход для Фазы 2."""
        H._write_cache("AAPL.US", 1825, _series(), "stooq")
        with patch.object(H, "_fetch_hloc") as fetch:
            H.get_candles(client=None, ticker="AAPL.US", days=1825, provider="stooq")
            fetch.assert_not_called()

    def test_tradernet_cache_invisible_to_stooq_via_get_candles(self):
        """Сквозной регресс дефекта на уровне публичного API."""
        H._write_cache("AAPL.US", 1825, _series(), "tradernet")
        with patch.object(H, "_fetch_hloc", return_value=[]) as fetch, \
             patch.object(H, "_candles_to_series", return_value=pd.Series(dtype=float)):
            H.get_candles(client=None, ticker="AAPL.US", days=1825, provider="stooq")
            fetch.assert_called_once()      # чужой кэш не подхвачен → пошёл на сервер


if __name__ == "__main__":
    unittest.main()


# ─── Фаза 2 — протокол провайдера ───────────────────────────────────────────


class ProviderResultContractTest(unittest.TestCase):
    """I-11: `ProviderResult` подставляется на место `HistoryResult`."""

    def test_first_four_fields_match_history_result(self):
        """Имена и ПОРЯДОК первых четырёх полей совпадают — иначе тихо ломаются
        шесть модулей-потребителей, читающих объект атрибутами."""
        from finance.price_providers import ProviderResult
        self.assertEqual(ProviderResult._fields[:4], H.HistoryResult._fields)

    def test_positional_construction_without_new_fields(self):
        from finance.price_providers import ProviderResult
        r = ProviderResult(pd.DataFrame(), ["A"], {"B": "нет"}, [])
        self.assertEqual(r.loaded, ["A"])
        self.assertEqual(r.convention, {})
        self.assertEqual(r.source, {})

    def test_single_source_and_convention_helpers(self):
        from finance.price_providers import PriceConvention, ProviderResult
        one = ProviderResult(pd.DataFrame(), ["A", "B"], {}, [],
                             convention={"A": PriceConvention.RAW,
                                         "B": PriceConvention.RAW},
                             source={"A": "x", "B": "x"})
        self.assertEqual(one.single_source(), "x")
        self.assertEqual(one.single_convention(), PriceConvention.RAW)

        mixed = ProviderResult(pd.DataFrame(), ["A", "B"], {}, [],
                               convention={"A": PriceConvention.RAW,
                                           "B": PriceConvention.TOTAL_RETURN},
                               source={"A": "x", "B": "y"})
        self.assertIsNone(mixed.single_source())      # I-13 нарушен → None
        self.assertIsNone(mixed.single_convention())  # I-7 нарушен → None


class ProviderMetadataTest(unittest.TestCase):
    """Каждый провайдер объявляет конвенцию и параллелизм ЯВНО."""

    def test_tradernet_declares_split_adjusted(self):
        from finance.price_providers import PriceConvention, TradernetProvider
        self.assertEqual(TradernetProvider.convention, PriceConvention.SPLIT_ADJUSTED)
        self.assertEqual(TradernetProvider.max_parallelism, 6)

    def test_demo_is_synthetic_not_a_market_convention(self):
        """Витрина не притворяется рыночной конвенцией — иначе I-7 «разрешил» бы
        смешать синтетику с реальными данными."""
        from finance.price_providers import DemoProvider, PriceConvention
        self.assertEqual(DemoProvider.convention, PriceConvention.SYNTHETIC)
        self.assertNotIn(DemoProvider.convention,
                         (PriceConvention.RAW, PriceConvention.SPLIT_ADJUSTED,
                          PriceConvention.TOTAL_RETURN))

    def test_demo_fetch_fills_source_and_convention(self):
        from finance.price_providers import DemoProvider
        res = DemoProvider().fetch(["AAPL.US", "НЕТ-ТАКОГО"], days=300)
        self.assertIn("AAPL.US", res.loaded)
        self.assertEqual(res.single_source(), "demo")
        self.assertIsNotNone(res.single_convention())
        self.assertIn("НЕТ-ТАКОГО", res.failed)   # пустой ответ ≠ исключение


class ProviderSelectionTest(unittest.TestCase):
    """`provider_for_source` — единственная точка выбора и юридический гард."""

    def test_freedom_selects_tradernet(self):
        from finance.price_providers import TradernetProvider, provider_for_source
        self.assertIsInstance(provider_for_source("freedom", client=None),
                              TradernetProvider)

    def test_default_source_selects_tradernet(self):
        """Пустой источник = живой фид (обратная совместимость, I-5)."""
        from finance.price_providers import TradernetProvider, provider_for_source
        self.assertIsInstance(provider_for_source("", client=None), TradernetProvider)

    def test_demo_selects_demo_provider(self):
        from finance.price_providers import DemoProvider, provider_for_source
        self.assertIsInstance(provider_for_source("demo"), DemoProvider)

    def test_manual_rejects_tradernet_at_runtime(self):
        """I-12: гард срабатывает в РАНТАЙМЕ, а не только в тесте — env можно
        выставить в проде, и тихое нарушение недопустимо."""
        from finance.price_providers import ProviderUnavailable, provider_for_source
        with patch.dict("os.environ", {"PRICE_PROVIDER_MANUAL": "tradernet"}):
            with self.assertRaises(ProviderUnavailable) as ctx:
                provider_for_source("manual", client=object())
            self.assertIn("I-12", str(ctx.exception))

    def test_manual_selects_stooq_provider(self):
        """MP-09: ручной портфель обслуживает Stooq — и больше никто.

        Тест-предшественник (`test_manual_stooq_not_implemented_yet`) пинил
        ГРАНИЦУ ФАЗЫ: до Фазы 09 `manual` обязан был честно отказать. Граница
        сдвинулась, поэтому тест заменён на свою противоположность, а не удалён:
        утверждение «ручной ввод обслуживается ровно одним провайдером»
        продолжает проверяться.
        """
        from finance.price_providers import provider_for_source
        from finance.stooq_provider import StooqProvider
        with patch.dict("os.environ", {"PRICE_PROVIDER_MANUAL": "stooq"}):
            self.assertIsInstance(provider_for_source("manual"), StooqProvider)

    def test_manual_unknown_provider_still_refuses(self):
        """Незнакомое имя провайдера — отказ, а не подстановка чего попало."""
        from finance.price_providers import ProviderUnavailable, provider_for_source
        with patch.dict("os.environ", {"PRICE_PROVIDER_MANUAL": "eodhd"}):
            with self.assertRaises(ProviderUnavailable):
                provider_for_source("manual")

    def test_manual_never_returns_tradernet(self):
        """I-12 сквозной: ни при каком значении флага manual не получит брокера."""
        from finance.price_providers import (ProviderUnavailable,
                                             TradernetProvider, provider_for_source)
        for flag in ("tradernet", "stooq", "eodhd", ""):
            with patch.dict("os.environ", {"PRICE_PROVIDER_MANUAL": flag}):
                try:
                    got = provider_for_source("manual", client=object())
                except ProviderUnavailable:
                    continue
                self.assertNotIsInstance(got, TradernetProvider)


class EngineProviderWiringTest(unittest.TestCase):
    """Движок ходит за ценами через провайдера."""

    def test_engine_selects_by_price_source(self):
        from finance.investment_logic import MAC3RiskEngine
        self.assertEqual(MAC3RiskEngine(price_source="demo")._get_price_provider().name,
                         "demo")
        self.assertEqual(MAC3RiskEngine(price_source="freedom")._get_price_provider().name,
                         "tradernet")

    def test_demo_never_builds_tradernet_client(self):
        """I-12 на уровне поведения: витрина не создаёт клиента вообще."""
        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine(price_source="demo")
        with patch.object(eng, "_get_tradernet_client") as factory:
            eng._get_price_provider()
            factory.assert_not_called()

    def test_provider_is_cached_per_engine(self):
        from finance.investment_logic import MAC3RiskEngine
        eng = MAC3RiskEngine(price_source="demo")
        self.assertIs(eng._get_price_provider(), eng._get_price_provider())
