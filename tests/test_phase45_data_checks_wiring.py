"""Ф-3 · подключение `data_checks.py` к прод-пути.

Фаза 3 была доставлена как **библиотека без потребителя**: 526 строк, 56 тестов,
13 чекеров и НОЛЬ вызовов в `src/` (`AUDIT §−39`, `PHASE_03 §6a`).

Главная опасность подключения — не написать вызов, а не уронить им живые
отчёты (`PHASE_03 §3a.5`). Замерено ДО правки: наивное подключение
заблокировало БЫ КАЖДОГО freedom-пользователя — `values` приходят ключами
ПОРТФЕЛЯ (`AAPL`, `TLT`), а `data.columns` — РАЗРЕШЁННЫМИ (`AAPL.US`,
`TLT.US`), и C-8 докладывал «не загружены цены по 7 из 9 бумаг (70% стоимости)»
на демо-портфеле, который строит нормальный отчёт.

Здесь зафиксировано:
  1. Мост пространств имён (`resolve_tickers`) — без него всё блокируется.
  2. Калибровочный гейт: LEGACY не блокирует книги, которые строятся сегодня,
     включая плечо с ОТРИЦАТЕЛЬНЫМ кэшем и проксированную неликвидную ноту.
  3. BLOCK действительно отказывает — и отказывает БЕЗ списания токена.
  4. DEGRADE доезжает до CoVe, а не тонет в логе.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance import data_checks as dc
from finance.investment_logic import UniversalPortfolioManager


def _frame(engine, extra=(), periods=1300, seed=5, young=()):
    idx = pd.bdate_range(end="2026-07-31", periods=periods)
    rng = np.random.default_rng(seed)
    cols = list(dict.fromkeys(
        list(engine.factor_tickers.values())
        + ["QQQ.US", "AGG.US", "URTH.US", "VWOB.US"] + list(extra)))
    f = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.standard_normal(periods) * 0.01))
         for c in cols}, index=idx)
    for t, keep in young:
        if t in f.columns:
            f.loc[f.index[:-keep], t] = np.nan
    return f


def _cols_frame(cols, rows=400):
    """Матрица нужной ФОРМЫ: пустой фрейм роняет C-2 («0 дней») и мешает
    изолированно проверять покрытие."""
    idx = pd.bdate_range(end="2026-07-31", periods=rows)
    return pd.DataFrame({c: np.linspace(100.0, 120.0, rows) for c in cols}, index=idx)


def _run(mgr, frame, portfolio):
    hr = type("HR", (), {"data": frame, "loaded": list(frame.columns),
                         "failed": {}, "retried": []})()
    with patch.object(mgr.engine, "get_market_data", return_value=(frame, hr)):
        return mgr.analyze_all(portfolio)


# ──────────────────────────────────────────────────────────────────────────
# 1. Подключение состоялось
# ──────────────────────────────────────────────────────────────────────────
class WiredIntoProductionTest(unittest.TestCase):
    """Раньше единственным импортёром был собственный тест модуля."""

    def test_engine_calls_the_checkers(self):
        src = Path("src/finance/investment_logic.py").read_text(encoding="utf-8")
        self.assertIn("data_checks", src)
        self.assertIn("check_portfolio_sufficiency", src)
        self.assertIn("check_price_matrix", src)

    def test_bot_handles_the_refusal(self):
        src = Path("src/tg_bot.py").read_text(encoding="utf-8")
        self.assertIn("DataQualityBlocked", src)

    def test_refusal_is_caught_before_generic_runtimeerror(self):
        """`DataQualityBlocked` — подкласс RuntimeError.

        Если ветка окажется ПОСЛЕ `except RuntimeError`, отказ утонет в общем
        обработчике и пользователь получит «сбой генерации» вместо причины.
        """
        import ast
        tree = ast.parse(Path("src/tg_bot.py").read_text(encoding="utf-8"))
        checked = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            names = [getattr(h.type, "id", None) for h in node.handlers]
            if "DataQualityBlocked" in names and "RuntimeError" in names:
                self.assertLess(names.index("DataQualityBlocked"),
                                names.index("RuntimeError"))
                checked = True
        self.assertTrue(checked, "ветка DataQualityBlocked не найдена")

    def test_lineage_exposes_degrade_findings(self):
        src = Path("src/finance/data_lineage.py").read_text(encoding="utf-8")
        self.assertIn("_data_quality_status", src)


# ──────────────────────────────────────────────────────────────────────────
# 2. 🔴 Мост пространств имён
# ──────────────────────────────────────────────────────────────────────────
class TickerNamespaceBridgeTest(unittest.TestCase):
    """Без моста C-8 считает покрытие между разными системами имён."""

    def test_portfolio_and_matrix_namespaces_really_differ(self):
        eng = UniversalPortfolioManager(price_source="demo").engine
        self.assertEqual(eng.resolve_tickers(["AAPL"]), ["AAPL.US"])
        self.assertEqual(eng.resolve_tickers(["KSPI"]), ["KSPI.KZ"])

    def test_naive_wiring_would_block_a_healthy_book(self):
        """Замер дефекта: сырые ключи портфеля против колонок матрицы."""
        data = _cols_frame(["AAPL.US", "MSFT.US", "SPY.US"])
        report = dc.check_portfolio_sufficiency(
            data=data,
            weights={"AAPL": 0.5, "MSFT": 0.5},
            values={"AAPL": 5000.0, "MSFT": 5000.0},
            profile=dc.CheckProfile.LEGACY)
        self.assertTrue(report.blocked)
        self.assertIn("C-8", [f.id for f in report.blocking])

    def test_bridged_keys_pass(self):
        data = _cols_frame(["AAPL.US", "MSFT.US", "SPY.US"])
        report = dc.check_portfolio_sufficiency(
            data=data,
            weights={"AAPL.US": 0.5, "MSFT.US": 0.5},
            values={"AAPL.US": 5000.0, "MSFT.US": 5000.0},
            profile=dc.CheckProfile.LEGACY)
        self.assertFalse(report.blocked)

    def test_proxied_note_counts_as_covered(self):
        """AIX-нота проксирована на `VWOB.US` (§−38) — она В модели, значит покрыта."""
        eng = UniversalPortfolioManager(price_source="demo").engine
        self.assertEqual(eng.resolve_tickers(["FFSPC6.1028.AIX"]), ["VWOB.US"])


# ──────────────────────────────────────────────────────────────────────────
# 3. Калибровочный гейт (PHASE_03 §3a.5)
# ──────────────────────────────────────────────────────────────────────────
class LegacyNeverBlocksCurrentReportsTest(unittest.TestCase):
    """Обязательный гейт фазы: LEGACY не отказывает живым книгам."""

    def _live_like(self):
        """Книга из отчёта 02.08: плечо, ОТРИЦАТЕЛЬНЫЙ кэш, проксированная нота."""
        tick = ["AAPL", "GOOGL", "META", "MSFT", "NVDA", "NXPI", "ORCL",
                "GLD", "SLV", "TLT", "FFSPC6.1028.AIX"]
        qty = [12, 8, 6, 10, 9, 30, 25, 20, 50, 10, 1000]
        pp = [190.0, 150.0, 480.0, 380.0, 110.0, 210.0, 120.0,
              190.0, 22.0, 92.0, 100.0]
        return pd.DataFrame({
            "Ticker": tick + ["USD"],
            "Quantity": qty + [-1050.0],
            "Purchase_Price": pp + [1.0],
            "Broker_Current_Price": [np.nan] * 10 + [103.5, 1.0],
        })

    def _analyse(self, source="freedom", young=()):
        mgr = UniversalPortfolioManager(price_source=source)
        port = self._live_like()
        extra = [(mgr.engine.resolve_tickers([t]) or [t])[0]
                 for t in port["Ticker"] if t != "USD"]
        frame = _frame(mgr.engine, extra=extra, young=young)
        return _run(mgr, frame, port)

    def test_leveraged_book_with_negative_cash_is_not_blocked(self):
        """C-10 проверяет СУММУ весов: при маржинальном займе она всё равно 1.0.

        Наивное «все веса ≥ 0» забраковало бы совершенно корректную книгу.
        """
        res = self._analyse()
        self.assertGreater(res["total_value"], 0)
        levels = {f["level"] for f in res["data_quality"]["findings"]}
        self.assertNotIn("block", levels)

    def test_proxied_illiquid_note_does_not_trip_coverage(self):
        res = self._analyse()
        ids = {f["id"] for f in res["data_quality"]["findings"]}
        self.assertNotIn("C-8", ids)

    def test_young_listing_degrades_but_does_not_block(self):
        """Молодой листинг режет общее окно — это DEGRADE, не отказ."""
        mgr = UniversalPortfolioManager(price_source="freedom")
        port = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "SPCX", "USD"],
                             "Quantity": [10, 10, 20, 500.0],
                             "Purchase_Price": [190.0, 380.0, 14.0, 1.0]})
        frame = _frame(mgr.engine, extra=["AAPL.US", "MSFT.US", "SPCX.US"],
                       young=[("SPCX.US", 200)])
        res = _run(mgr, frame, port)
        ids = {f["id"] for f in res["data_quality"]["findings"]}
        self.assertIn("C-3", ids)
        self.assertNotIn("block", {f["level"] for f in res["data_quality"]["findings"]})

    def test_demo_showcase_never_blocks(self):
        from finance.demo_portfolio import build_demo_portfolio
        mgr = UniversalPortfolioManager(price_source="demo")
        res = mgr.analyze_all(build_demo_portfolio())
        self.assertEqual(res["data_quality"]["profile"], "demo")
        self.assertNotIn("block",
                         {f["level"] for f in res["data_quality"]["findings"]})

    def test_profile_follows_the_portfolio_source(self):
        """Строгость задаёт источник ПОРТФЕЛЯ, а не имя отработавшего фида."""
        self.assertIs(dc.profile_for_source("freedom"), dc.CheckProfile.LEGACY)
        self.assertIs(dc.profile_for_source("manual"), dc.CheckProfile.STRICT)
        self.assertIs(dc.profile_for_source("demo"), dc.CheckProfile.DEMO)
        self.assertIs(dc.profile_for_source(None), dc.CheckProfile.LEGACY)


# ──────────────────────────────────────────────────────────────────────────
# 4. Отказ работает и стоит НОЛЬ токенов
# ──────────────────────────────────────────────────────────────────────────
class BlockingPathTest(unittest.TestCase):

    def test_unpriced_majority_of_value_blocks(self):
        """Ловушка Т-6: по числу тикеров покрытие 67%, по СТОИМОСТИ — 4%."""
        mgr = UniversalPortfolioManager(price_source="freedom")
        frame = _frame(mgr.engine, extra=["AAPL.US", "MSFT.US"])
        port = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "ILLIQ.XX"],
            "Quantity": [10, 10, 600],
            "Purchase_Price": [190.0, 380.0, 100.0],
            "Broker_Current_Price": [np.nan, np.nan, 100.0],
        })
        with self.assertRaises(dc.DataQualityBlocked) as ctx:
            _run(mgr, frame, port)
        self.assertIn("C-8", [f.id for f in ctx.exception.report.blocking])

    def test_refusal_message_is_human(self):
        """Без кодов проверок и без str(exc) — требование PHASE_03 §4.1."""
        report = dc.DataQualityReport(profile=dc.CheckProfile.LEGACY)
        report.findings.append(dc.CheckFinding(
            "C-8", dc.CheckLevel.BLOCK,
            "Не удалось загрузить цены по 2 из 8 бумаг (54% стоимости портфеля)."))
        msg = report.user_message()
        self.assertIn("токен не списан", msg)
        self.assertNotIn("C-8", msg)
        self.assertNotIn("Traceback", msg)

    def test_deduction_happens_after_delivery_so_a_raise_costs_nothing(self):
        """Гарантия «токен не списан» — структурная, не декларативная.

        Списание живёт ПОСЛЕ CHECKPOINT 3; исключение из движка до него
        физически не может дойти до `deduct_tokens`.
        """
        src = Path("src/tg_bot.py").read_text(encoding="utf-8")
        cp3 = src.index("CHECKPOINT 3")
        self.assertLess(cp3, src.index("await deduct_tokens("))


# ──────────────────────────────────────────────────────────────────────────
# 5. Кэш — из SSOT, а не седьмой копией
# ──────────────────────────────────────────────────────────────────────────
class CashClassificationFromSsotTest(unittest.TestCase):
    """A-5: перечень валют жил здесь СЕДЬМОЙ копией — без GBP/CHF/CNY/JPY/RUR."""

    def test_no_local_currency_list_left(self):
        src = Path("src/finance/data_checks.py").read_text(encoding="utf-8")
        # допустим только фолбэк внутри except-ветки
        self.assertLessEqual(src.count('("USD", "EUR", "KZT", "RUB", "CASH")'), 1)

    def test_gbp_cash_is_cash_not_a_risky_asset(self):
        """После −37 строка кэша в неба́зовой валюте реальна."""
        data = _cols_frame(["AAPL.US", "MSFT.US", "SPY.US"])
        report = dc.check_portfolio_sufficiency(
            data=data,
            weights={"AAPL.US": 0.5, "MSFT.US": 0.4, "GBP": 0.1},
            values={"AAPL.US": 5000.0, "MSFT.US": 4000.0, "GBP": 1000.0},
            profile=dc.CheckProfile.LEGACY)
        self.assertFalse(report.blocked, [f.message for f in report.blocking])

    def test_cash_in_the_matrix_is_still_a_code_bug(self):
        """C-12 блокирует во ВСЕХ профилях — это про баг, а не про данные."""
        data = _cols_frame(["AAPL.US", "USD"])
        report = dc.check_portfolio_sufficiency(
            data=data, weights={"AAPL.US": 0.9, "USD": 0.1},
            values={"AAPL.US": 9000.0, "USD": 1000.0},
            profile=dc.CheckProfile.DEMO)
        self.assertIn("C-12", [f.id for f in report.blocking])


# ──────────────────────────────────────────────────────────────────────────
# 6. DEGRADE доезжает до читателя
# ──────────────────────────────────────────────────────────────────────────
class LineageSurfaceTest(unittest.TestCase):

    def test_row_absent_when_nothing_to_report(self):
        """Реестр CoVe сжат до 16 строк — всегда-включённую добавлять нельзя."""
        from finance.data_lineage import _data_quality_status
        self.assertIsNone(_data_quality_status({}))
        self.assertIsNone(_data_quality_status(
            {"data_quality": {"profile": "legacy", "findings": []}}))

    def test_block_level_never_reaches_the_row(self):
        """BLOCK не доходит до CoVe по построению — отчёта в этом случае нет."""
        from finance.data_lineage import _data_quality_status
        self.assertIsNone(_data_quality_status({"data_quality": {
            "profile": "legacy",
            "findings": [{"id": "C-8", "level": "block", "message": "x"}]}}))

    def test_row_speaks_human_not_check_codes(self):
        """2026-08-03: строка НЕ имеет права печатать идентификаторы проверок.

        `PHASE_03 §4.1` запрещает «коды ошибок и имена внутренних проверок» в
        пользовательском тексте. Первая редакция это правило нарушила — в живом
        отчёте 03.08 читатель увидел «сработали B-6, C-11, C-13, C-3, C-5».
        """
        import re
        from finance.data_lineage import _data_quality_status
        row = _data_quality_status({"data_quality": {
            "profile": "legacy",
            "findings": [
                {"id": "C-3", "level": "degrade", "message": "Окно короче года."},
                {"id": "C-13", "level": "degrade", "message": "Мало наблюдений."}]}})
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "degraded")
        # тема — словами, а не кодом
        self.assertIn("достаточность истории", row["method"])
        # сообщения чекеров сохранены
        self.assertIn("Окно короче года", row["note"])
        # …и ни одного идентификатора во всей строке
        self.assertEqual(re.findall(r"\b[ABC]-\d+\b", str(row)), [])

    def test_unconvertible_currency_reaches_c11(self):
        """−37 только писал в лог; теперь факт доезжает до чекера."""
        src = Path("src/finance/investment_logic.py").read_text(encoding="utf-8")
        self.assertIn("_fx_unconvertible", src)
        self.assertIn("unconvertible_currencies=_fx_unconvertible", src)

    def test_lineage_row_appears_in_a_real_run(self):
        from finance.data_lineage import build_lineage
        mgr = UniversalPortfolioManager(price_source="freedom")
        port = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "SPCX", "USD"],
                             "Quantity": [10, 10, 20, 500.0],
                             "Purchase_Price": [190.0, 380.0, 14.0, 1.0]})
        frame = _frame(mgr.engine, extra=["AAPL.US", "MSFT.US", "SPCX.US"],
                       young=[("SPCX.US", 200)])
        res = _run(mgr, frame, port)
        names = [r["name"] for r in build_lineage(res, {})]
        self.assertTrue(any("Качество данных" in n for n in names), names)


if __name__ == "__main__":
    unittest.main()
