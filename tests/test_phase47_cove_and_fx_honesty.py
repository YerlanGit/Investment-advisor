"""Аудит живого отчёта 03.08 — CoVe и честность FX (D-1…D-4).

Первый живой прогон с подключёнными чекерами (`§−49`) показал ЧЕТЫРЕ дефекта,
и три из них — в моей же вчерашней проводке:

  D-1 🔴 «Наблюдений (37) мало для 13 активов» при том, что движок построил
      ПОЛНУЮ модель (покрытие факторами 100%, MaxDD −41.3%, Sharpe 0.49).
      37 — история самой молодой бумаги книги (SPCX, исключена движком как
      разреженная): чекер брал СТРОГОЕ пересечение
      всех колонок, а движок сначала выбрасывает разреженные бумаги
      (`MIN_OVERLAP_TDAYS`). Три ложных предупреждения сразу: C-3, C-5, C-13.
      Это повторение ошибки F-15/F-21, которую движок уже прошёл.

  D-2 🔴 Тенговая строка кэша печаталась как «1.00 USD». `fx_converted`
      вычислялся из НЕСОВПАДЕНИЯ валют, а не из факта конверсии: когда курса
      нет, строка остаётся в своей валюте (§−45), но карточка заявляла
      «пересчитано» и вешала ярлык базовой валюты на тенге.

  D-3 CoVe печатал «сработали B-6, C-11, C-13, C-3, C-5» — идентификаторы
      внутренних проверок в пользовательском тексте, прямо запрещённые
      `PHASE_03 §4.1`.

  D-4 B-6 говорил «аномальные дневные скачки в 0.24% наблюдений» — доля от
      числа наблюдений не читается, а формулировка звучит как поломка данных,
      хотя проверка не отличает реальный обвал на отчётности от сплита.
"""

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance import data_checks as dc
from finance.investment_logic import UniversalPortfolioManager


def _panel(engine, young: dict, periods=1260, seed=7):
    """Панель цен, где `young` = {тикер: сколько дней истории}."""
    idx = pd.bdate_range(end="2026-08-01", periods=periods)
    rng = np.random.default_rng(seed)
    cols = list(dict.fromkeys(
        list(engine.factor_tickers.values())
        + ["QQQ.US", "AGG.US", "URTH.US", "AAPL.US", "MSFT.US", "NVDA.US"]
        + list(young)))
    f = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.standard_normal(periods) * 0.01))
         for c in cols}, index=idx)
    for t, keep in young.items():
        f.loc[f.index[:-keep], t] = np.nan
    return f


# ──────────────────────────────────────────────────────────────────────────
# D-1 · окно чекера обязано быть окном движка
# ──────────────────────────────────────────────────────────────────────────
class EffectiveWindowMirrorsEngineTest(unittest.TestCase):

    def test_one_young_listing_no_longer_collapses_the_window(self):
        """🔴 Замер живого дефекта: 1260 строк панели → 37 у чекера."""
        eng = UniversalPortfolioManager(price_source="demo").engine
        f = _panel(eng, {"SPCX.US": 200, "AAOI.US": 37})
        naive = len(f.dropna(axis=0, how="any"))
        self.assertLessEqual(naive, 40, "фикстура должна воспроизводить схлопывание")
        self.assertGreater(dc._effective_window(f), 150,
                           "разреженная бумага всё ещё определяет общее окно")

    def test_matches_the_engine_regression_window(self):
        """Чекер и движок обязаны сообщать ОДНО число наблюдений."""
        mgr = UniversalPortfolioManager(price_source="freedom")
        f = _panel(mgr.engine, {"SPCX.US": 200, "AAOI.US": 37})
        port = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "NVDA", "SPCX", "AAOI", "USD"],
            "Quantity": [10, 10, 10, 20, 40, 500.0],
            "Purchase_Price": [190., 380., 110., 14., 18., 1.]})
        hr = type("HR", (), {"data": f, "loaded": list(f.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(f, hr)):
            mgr.analyze_all(port)
        self.assertEqual(dc._effective_window(f),
                         int(mgr.engine._last_regression_nobs))

    def test_no_false_c5_on_a_healthy_book(self):
        """C-5 «наблюдений мало для N активов» больше не срабатывает ложно."""
        mgr = UniversalPortfolioManager(price_source="freedom")
        f = _panel(mgr.engine, {"AAOI.US": 37})
        port = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "NVDA", "AAOI", "USD"],
            "Quantity": [10, 10, 10, 40, 500.0],
            "Purchase_Price": [190., 380., 110., 18., 1.]})
        hr = type("HR", (), {"data": f, "loaded": list(f.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(f, hr)):
            res = mgr.analyze_all(port)
        ids = {x["id"] for x in res["data_quality"]["findings"]}
        self.assertNotIn("C-5", ids)
        self.assertNotIn("C-3", ids)

    def test_genuinely_short_history_still_warns(self):
        """Гард не ослаблен: ПРАВДА короткое окно по-прежнему замечается."""
        eng = UniversalPortfolioManager(price_source="demo").engine
        f = _panel(eng, {}, periods=120)
        self.assertLess(dc._effective_window(f), 252)

    def test_threshold_comes_from_the_engine_module(self):
        """Порог разреженности — из того же места, что импортирует движок."""
        src = Path("src/finance/data_checks.py").read_text(encoding="utf-8")
        self.assertIn("from finance.period_returns import MIN_OVERLAP_TDAYS", src)

    def test_empty_and_all_sparse_are_safe(self):
        self.assertEqual(dc._effective_window(None), 0)
        self.assertEqual(dc._effective_window(pd.DataFrame()), 0)
        idx = pd.bdate_range(end="2026-08-01", periods=10)
        self.assertEqual(dc._effective_window(
            pd.DataFrame({"X.US": np.arange(10.0)}, index=idx)), 0)


# ──────────────────────────────────────────────────────────────────────────
# D-2 · валюта строки: флаг обязан отражать ФАКТ конверсии
# ──────────────────────────────────────────────────────────────────────────
class FxHonestyTest(unittest.TestCase):

    def test_unconverted_row_keeps_its_own_currency(self):
        """🔴 Живой отчёт печатал тенговую строку как «1.00 USD»."""
        from pdf_payload import _fmt_purchase_price
        out = _fmt_purchase_price(1.00, "KZT", "USD", [], converted=False)
        self.assertIn("₸", out)
        self.assertNotIn("USD", out)

    def test_converted_row_shows_both(self):
        from pdf_payload import _fmt_purchase_price
        out = _fmt_purchase_price(23.08, "KZT", "USD",
                                  [{"ticker": "X", "currency": "KZT",
                                    "rate": 0.00192308}], converted=True)
        self.assertIn("23.08 USD", out)
        self.assertIn("₸", out)

    def test_base_currency_row_untouched(self):
        from pdf_payload import _fmt_purchase_price
        self.assertEqual(
            _fmt_purchase_price(206.0, "USD", "USD", [], converted=True),
            "206.00 USD")

    def test_flag_reads_the_conversion_registry_not_the_mismatch(self):
        src = Path("src/pdf_payload.py").read_text(encoding="utf-8")
        self.assertIn("_fx_done_tickers", src)
        self.assertNotIn(
            '"fx_converted":  bool(_row_ccy and _row_ccy != _rep_ccy_payload)',
            src)


# ──────────────────────────────────────────────────────────────────────────
# D-3 · CoVe говорит по-человечески
# ──────────────────────────────────────────────────────────────────────────
class CoveWordingTest(unittest.TestCase):

    #: Реальные тексты чекеров — они СВОИХ идентификаторов не содержат.
    _MSG = {
        "B-6":  "AAOI.US: 3 дн. с движением цены более 49%",
        "C-11": "Нет курса для валют: KZT",
        "C-13": "Наблюдений (200) мало для устойчивой оценки хвостового риска",
        "C-3":  "Общая история короче года (200 дней)",
        "C-5":  "Наблюдений (200) мало для 13 активов",
    }

    def _row(self, ids):
        from finance.data_lineage import _data_quality_status
        return _data_quality_status({"data_quality": {"profile": "legacy",
            "findings": [{"id": i, "level": "degrade",
                          "message": self._MSG.get(i, "замечание")}
                         for i in ids]}})

    def test_no_check_identifiers_anywhere_in_the_row(self):
        row = self._row(["B-6", "C-11", "C-13", "C-3", "C-5"])
        self.assertEqual(re.findall(r"\b[ABC]-\d+\b", str(row)), [])

    def test_topics_are_named_in_words(self):
        row = self._row(["B-6", "C-3"])
        self.assertIn("качество ценового ряда", row["method"])
        self.assertIn("достаточность истории", row["method"])

    def test_count_is_declined_correctly(self):
        self.assertIn("1 замечание", self._row(["C-3"])["name"])
        self.assertIn("2 замечания", self._row(["C-3", "C-5"])["name"])
        self.assertIn("5 замечаний",
                      self._row(["C-3", "C-5", "C-13", "B-6", "C-11"])["name"])


# ──────────────────────────────────────────────────────────────────────────
# D-4 · B-6 называет факт, а не долю в процентах
# ──────────────────────────────────────────────────────────────────────────
class AnomalyWordingTest(unittest.TestCase):

    def _series(self, n_jumps, periods=1260):
        v = np.full(periods, 100.0)
        for i in range(n_jumps):
            v[100 + i * 50] *= 1.7
        return pd.Series(np.abs(v))

    def test_message_states_days_and_magnitude(self):
        msg = dc._b_anomalies(self._series(6), "AAOI.US")
        self.assertIsNotNone(msg)
        self.assertIn("дн.", msg)
        self.assertIn("%", msg)
        self.assertNotIn("наблюдений", msg)

    def test_message_does_not_claim_broken_data(self):
        """Проверка эвристическая — реальный обвал она от сплита не отличает."""
        msg = dc._b_anomalies(self._series(6), "AAOI.US")
        self.assertIn("отчётность", msg)
        self.assertNotIn("аномальные", msg)

    def test_quiet_series_says_nothing(self):
        self.assertIsNone(dc._b_anomalies(self._series(0), "MSFT.US"))


# ──────────────────────────────────────────────────────────────────────────
# Правило плеча (задание §3): термины скрыты, если кэш НЕ отрицательный
# ──────────────────────────────────────────────────────────────────────────
class LeverageDisclosureGateTest(unittest.TestCase):

    #: Термины ПЛЕЧА, запрещённые на книге с неотрицательным кэшем.
    #:
    #: Осознанно НЕ включена «маржинальность»: это ОПЕРАЦИОННАЯ МАРЖА компании
    #: из SEC («маржинальность 30%», `premium_payload._fund_verdict`) — обычная
    #: метрика прибыльности, к заёмным средствам отношения не имеющая.
    #: Первая редакция теста искала подстроку «маржинальн» и ловила её —
    #: локально этого не было видно (SEC недоступен в песочнице, поле пустое),
    #: а CI с живым фундаменталом упал. Проверять надо ФРАЗЫ, а не корни слов.
    _LEVERAGE_PHRASES = (
        r"плеч[оаеу]", r"маржинальн\w*\s+долг\w*", r"заёмн\w*\s+средств\w*",
        r"куплены\s+в\s+долг", r"\bleverage\b",
    )

    @staticmethod
    def _visible_text(obj) -> str:
        """Только ТЕКСТ, который увидит пользователь — без имён ключей.

        Прошлая редакция дампила словарь целиком и ловила ключ контракта
        `leverage` (`design_data["leverage"]` — числовой блок, JSX гейтит его
        по `is_leveraged`). Имя поля пользователю не показывается; проверять
        надо строковые ЗНАЧЕНИЯ.
        """
        out = []
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, str):
                out.append(cur)
            elif isinstance(cur, dict):
                stack.extend(cur.values())          # ключи НЕ берём
            elif isinstance(cur, (list, tuple)):
                stack.extend(cur)
        return " \n ".join(out)

    def test_non_levered_book_shows_no_leverage_wording(self):
        import pdf_payload
        import premium_payload
        from finance.demo_portfolio import build_demo_portfolio
        mgr = UniversalPortfolioManager(price_source="demo")
        res = mgr.analyze_all(build_demo_portfolio())
        self.assertFalse(res["leverage_metrics"]["is_leveraged"])
        for tier in ("base", "deep"):
            d = premium_payload.build_design_data(
                pdf_payload.build_payload(res, tier=tier), tier=tier)
            txt = self._visible_text(d)
            for phrase in self._LEVERAGE_PHRASES:
                hits = re.findall(phrase, txt, re.I)
                self.assertEqual(hits, [],
                                 f"{tier}: «{phrase}» на книге БЕЗ плеча → {hits[:3]}")

    def test_levered_book_DOES_disclose(self):
        """Обратная проверка: гард не должен глушить РЕАЛЬНОЕ плечо."""
        import pdf_payload
        import premium_payload
        f = _panel(UniversalPortfolioManager(price_source="demo").engine, {})
        mgr = UniversalPortfolioManager(price_source="freedom")
        port = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "USD"],
                             "Quantity": [40, 20, -3000.0],
                             "Purchase_Price": [190., 380., 1.]})
        hr = type("HR", (), {"data": f, "loaded": list(f.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(f, hr)):
            res = mgr.analyze_all(port)
        self.assertTrue(res["leverage_metrics"]["is_leveraged"])
        txt = self._visible_text(premium_payload.build_design_data(
            pdf_payload.build_payload(res, tier="deep"), tier="deep"))
        self.assertTrue(any(re.search(p, txt, re.I) for p in self._LEVERAGE_PHRASES),
                        "плечо есть, а в отчёте о нём ни слова")

    def test_profitability_wording_is_not_suppressed(self):
        """Обратная сторона: «маржинальность» — законный термин, глушить нельзя."""
        from premium_payload import _fund_verdict
        v = _fund_verdict({"roe": "25%", "margin": "30%", "growth": "15%",
                           "debt": "0.２0"})
        self.assertIn("маржинальность", v)

    def test_mandate_card_gates_the_margin_row(self):
        js = (Path("src/premium_assets") / "deep-components.js").read_text(encoding="utf-8")
        self.assertIn("m.leveraged", js)


if __name__ == "__main__":
    unittest.main()


# ──────────────────────────────────────────────────────────────────────────
# D-5…D-7 · вторая волна: покрытие, жаргон плана (аудит 03.08, раунд 2)
# ──────────────────────────────────────────────────────────────────────────
class FactorCoverageHonestyTest(unittest.TestCase):
    """🔴 «покрытие 100%» при позиции весом 5.3% БЕЗ бет."""

    def test_uncovered_share_is_exposed(self):
        mgr = UniversalPortfolioManager(price_source="freedom")
        f = _panel(mgr.engine, {"SPCX.US": 37})
        port = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "SPCX", "USD"],
                             "Quantity": [10, 10, 20, 500.0],
                             "Purchase_Price": [190., 380., 14., 1.]})
        hr = type("HR", (), {"data": f, "loaded": list(f.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(f, hr)):
            res = mgr.analyze_all(port)
        mu = res["model_uncovered"]
        self.assertIn("SPCX", mu["names"])
        self.assertGreater(mu["weight_pct"], 0)

    def test_report_separates_series_load_from_portfolio_coverage(self):
        import pdf_payload
        import premium_payload
        mgr = UniversalPortfolioManager(price_source="freedom")
        f = _panel(mgr.engine, {"SPCX.US": 37})
        port = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "SPCX", "USD"],
                             "Quantity": [10, 10, 20, 500.0],
                             "Purchase_Price": [190., 380., 14., 1.]})
        hr = type("HR", (), {"data": f, "loaded": list(f.columns),
                             "failed": {}, "retried": []})()
        with patch.object(mgr.engine, "get_market_data", return_value=(f, hr)):
            res = mgr.analyze_all(port)
        d = premium_payload.build_design_data(
            pdf_payload.build_payload(res, tier="deep"), tier="deep")
        self.assertEqual(d["factorCoverage"], 100.0)   # фактор-серии загружены
        self.assertGreater(d["uncoveredPct"], 0)       # …но книга покрыта НЕ вся
        self.assertIn("SPCX", d["uncoveredNames"])

    def test_label_no_longer_says_bare_coverage(self):
        js = (Path("src/premium_assets") / "deep-components.js").read_text(encoding="utf-8")
        self.assertIn("фактор-серий загружено", js)
        self.assertIn("вне факторной модели", js)

    def test_clean_book_shows_nothing(self):
        """Блок условный: когда исключений нет, строка не появляется."""
        from finance.demo_portfolio import build_demo_portfolio
        mgr = UniversalPortfolioManager(price_source="demo")
        res = mgr.analyze_all(build_demo_portfolio())
        self.assertEqual(res["model_uncovered"]["weight_pct"], 0.0)


class ActionPlanWordingTest(unittest.TestCase):
    """Внутренний жаргон в тексте плана для пользователя."""

    def test_no_bl_abbreviation(self):
        src = Path("src/finance/action_plan.py").read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn('f"BL расходится с сигналом', code)
        self.assertIn("оптимизатор предлагает обратное", code)

    def test_no_english_deferred(self):
        src = Path("src/finance/action_plan.py").read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn("deferred (turnover cap)", code)
        self.assertIn("лимит оборота за период", code)

    def test_divergence_note_explains_the_missing_quantity(self):
        """Строка обязана объяснять, ПОЧЕМУ в количестве стоит «—»."""
        src = Path("src/finance/action_plan.py").read_text(encoding="utf-8")
        self.assertIn("количество не указываем", src)
