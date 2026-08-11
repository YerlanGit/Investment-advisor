"""
Паритет отчётов: `manual` не имеет права отличаться от `freedom` НИЧЕМ, кроме
того, что обязано отличаться (MP-06).

Зачем этот файл существует
──────────────────────────
Требование владельца 2026-08-11: «логика и отчёт, который формируется за счёт
Manual Portfolio, не должен отличаться от отчёта, который уже работает через
Freedom API».  Это требование ПРОВЕРЯЕМО, и проверять его надо целиком, а не
по выборочным метрикам: два прогона `analyze_all` на ОДНИХ И ТЕХ ЖЕ ценах
обязаны дать один и тот же контракт `results` — все 36 ключей.

Метод
─────
Оба прогона получают цены из одной и той же базы Stooq; отличается только
ПОДПИСЬ провайдера.  Так изолируется ровно то, что проверяется: разница в
логике конвейера, а не разница в данных.  Сравнивать реальные Tradernet и
Stooq здесь бессмысленно — они разойдутся в четвёртом знаке цены, и тест
превратился бы в замер источников (это делает сверка `PHASE_09 §4.1`,
и делает её человек).

🔴 Список РАЗРЕШЁННЫХ расхождений закрыт и обоснован построчно.  Тест падает
не только когда появилось новое расхождение, но и когда разрешённое ИСЧЕЗЛО:
исчезновение подписи источника означало бы, что отчёт перестал называть
происхождение цен, а это I-12 на слое текста (F-6).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden_support as gs                                  # noqa: E402
from finance import stooq_ingest as si                       # noqa: E402
from finance.data_checks import BENCHMARK_ETFS, FACTOR_ETFS  # noqa: E402
from finance.stooq_provider import StooqProvider             # noqa: E402

HEADER = ("<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,"
          "<VOL>,<OPENINT>")

#: Ключи `results`, которым РАЗРЕШЕНО различаться, и причина у каждого.
#:
#: Словарь, а не список: причина обязана лежать рядом с исключением, иначе
#: через два раунда никто не вспомнит, почему ключ здесь, и список начнёт
#: расти «за компанию».
ALLOWED_DIFFERENCES = {
    "history_result":
        "подпись источника (`stooq` против `tradernet`) — она ОБЯЗАНА "
        "различаться: отчёт называет происхождение цен (I-13, F-6)",
    "data_quality":
        "профиль чекеров: `strict` у ручного ввода против `legacy` у "
        "брокерского (I-10). Отличается не СОДЕРЖАНИЕ отчёта, а порог отказа: "
        "на плохих данных ручной ввод блокирует там, где брокерский деградирует",
}


class ManualFreedomParityTest(unittest.TestCase):

    HISTORY_DAYS = 1300          # ≈5 лет: меньше — и C-4 заблокирует отчёт

    def setUp(self) -> None:                                # noqa: N802
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.db = self.tmp / "prices.sqlite"
        self._build_store()

    # ── фикстура ─────────────────────────────────────────────────────────

    @staticmethod
    def _business_days_ending(end: dt.date, count: int) -> list[int]:
        out: list[int] = []
        cursor = end
        while len(out) < count:
            if cursor.weekday() < 5:
                out.append(int(cursor.strftime("%Y%m%d")))
            cursor -= dt.timedelta(days=1)
        return sorted(out)

    def _build_store(self) -> None:
        archive = self.tmp / "archive"
        archive.mkdir()
        days = self._business_days_ending(dt.date.today(), self.HISTORY_DAYS)
        random.seed(11)                       # ряд детерминирован
        universe = (list(FACTOR_ETFS) + list(BENCHMARK_ETFS)
                    + ["AAPL.US", "MSFT.US", "TLT.US"])
        for offset, symbol in enumerate(universe):
            price = 100.0 + offset
            rows = []
            for day in days:
                price *= 1 + random.gauss(0.0004, 0.011)
                rows.append(f"{symbol},D,{day},000000,{price:.4f},"
                            f"{price * 1.01:.4f},{price * 0.99:.4f},"
                            f"{price:.4f},1000,0")
            (archive / f"{symbol.lower()}.txt").write_text(
                "\r\n".join([HEADER, *rows]) + "\r\n", encoding="utf-8")
        conn = si.connect(self.db)
        si.ensure_schema(conn)
        si.bootstrap(conn, archive, window_days=1825)
        conn.close()

    def _portfolio(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Ticker": "AAPL.US", "Quantity": 50, "Purchase_Price": 200.0,
             "Currency": "USD"},
            {"Ticker": "MSFT.US", "Quantity": 30, "Purchase_Price": 400.0,
             "Currency": "USD"},
            {"Ticker": "TLT.US", "Quantity": 100, "Purchase_Price": 90.0,
             "Currency": "USD"},
            {"Ticker": "USD", "Quantity": 5000, "Purchase_Price": 1.0,
             "Currency": "USD"},
        ])

    def _run(self, source: str, *, extra=None) -> dict:
        from finance.investment_logic import UniversalPortfolioManager

        env = {"STOOQ_DB_PATH": str(self.db), "HISTORY_LOOKBACK_DAYS": "1825",
               "REPORTING_CURRENCY": "USD", "RFR_USD": "0.045",
               "FRED_API_KEY": "", "FACTOR_ORTHOGONALIZE": "1"}
        layers = [
            mock.patch.dict("os.environ", env),
            mock.patch("finance.sec_edgar.batch_fundamental_scan",
                       return_value=pd.DataFrame()),
            mock.patch("finance.cds_feed.make_lookup",
                       return_value=lambda _t: {}),
            mock.patch("finance.smart_money.build_insider_signals",
                       return_value={}),
        ]
        if extra is not None:
            layers.append(extra)
        with contextlib.ExitStack() as stack:
            for layer in layers:
                stack.enter_context(layer)
            upm = UniversalPortfolioManager()
            upm.engine.price_source = source
            upm.engine.current_rfr_annual = 0.045
            upm.engine.rfr_source = "fixture[USD]=0.045"
            return upm.analyze_all(source=self._portfolio(),
                                   profile_benchmark="SPY.US",
                                   risk_mandate="MODERATE")

    def _both_runs(self) -> tuple[dict, dict]:
        """Ручной и брокерский прогоны на ОДНИХ ценах."""
        class _SignedAsBroker(StooqProvider):
            """Те же данные, другая подпись — изолирует логику от данных."""

            name = "tradernet"

            def __init__(self, client=None) -> None:
                super().__init__()

        manual = self._run("manual")
        freedom = self._run("freedom", extra=mock.patch(
            "finance.price_providers.TradernetProvider", _SignedAsBroker))
        return gs.normalize(manual), gs.normalize(freedom)

    # ── собственно проверки ──────────────────────────────────────────────

    def test_reports_differ_only_where_they_must(self) -> None:
        """🔴 Главный тест требования владельца.

        Тридцать шесть ключей контракта, два разрешённых расхождения, и оба
        названы поимённо с причиной. Новое расхождение = регресс: значит
        ручной отчёт начал считать что-то иначе, чем брокерский.
        """
        manual, freedom = self._both_runs()
        self.assertEqual(set(manual), set(freedom),
                         "контракт results обязан совпадать по составу ключей")
        differing = {k for k in manual if manual[k] != freedom[k]}
        unexpected = differing - set(ALLOWED_DIFFERENCES)
        self.assertEqual(unexpected, set(),
                         "ручной отчёт разошёлся с брокерским там, где не должен")

    def test_every_allowed_difference_still_actually_happens(self) -> None:
        """Разрешённое расхождение, которое ИСЧЕЗЛО, — тоже регресс.

        Если пропала подпись источника, отчёт перестал называть происхождение
        цен (F-6, I-12). Если пропал профиль — ручной ввод потерял свой более
        строгий гейт (I-10). Список исключений обязан оставаться живым, а не
        накапливать мёртвые записи.
        """
        manual, freedom = self._both_runs()
        for key, reason in ALLOWED_DIFFERENCES.items():
            self.assertNotEqual(manual[key], freedom[key],
                                f"ожидалось расхождение по '{key}': {reason}")

    def test_price_matrix_itself_is_identical(self) -> None:
        """Цены одинаковы — значит расхождения выше НЕ про данные.

        Без этой проверки тест выше доказывал бы лишь «на разных числах вышло
        по-разному», то есть не доказывал бы ничего.
        """
        manual, freedom = self._both_runs()
        a, b = manual["history_result"], freedom["history_result"]
        frame_a = next(x for x in a if isinstance(x, dict)
                       and str(x.get("__type__", "")).startswith("DataFrame"))
        frame_b = next(x for x in b if isinstance(x, dict)
                       and str(x.get("__type__", "")).startswith("DataFrame"))
        self.assertEqual(frame_a, frame_b,
                         "ценовые матрицы обязаны совпадать побитово")

    def test_manual_profile_is_the_stricter_one(self) -> None:
        """Направление отличия проверяется, а не только его факт.

        `strict` мягче `legacy` быть не может: ручной ввод — самый вероятный
        источник кривых данных, и ослабление гейта здесь прошло бы тест на
        «расхождение есть» незамеченным.
        """
        manual, freedom = self._both_runs()
        self.assertEqual(manual["data_quality"]["profile"], "strict")
        self.assertEqual(freedom["data_quality"]["profile"], "legacy")


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
