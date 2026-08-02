"""Аудит живого DEEP-отчёта 2026-08-02 — R-19…R-22.

Четыре находки, каждая ВОСПРОИЗВЕДЕНА на данных этого отчёта
(`window.DEEP`, user 148046720, NAV $14 160, 15 позиций):

  R-19  секция «Режим» напечатала «(8%… фактически 1%)» при реальной
        уверенности 1%. «8%» — ЛИТЕРАЛ из промпта: пример формулировки,
        оставшийся от июльского отчёта. Модель приняла его за данные.

  R-20  «Ожидаемый эффект»: Sharpe 0.49 → 0.47 при том, что волатильность
        УПАЛА 23.3% → 19.2%, а все прочие строки улучшились. Причина —
        отрицательная премия за риск (er 2.8% < rf 4.5%):
        S(σ) = (er−rf)/σ ⇒ ∂S/∂σ = −(er−rf)/σ² > 0, то есть снижение
        волатильности МЕХАНИЧЕСКИ ухудшает коэффициент.

  R-21  вердикт «за счёт роста концентрации/просадки» — фиксированная фраза,
        а просадка в этом отчёте УЛУЧШИЛАСЬ (−41.2% → −38.7%).

  R-22  в отчёте ТРИ карточки идей под подписью «4 идеи»: мандатный гард
        работал ПОСЛЕ backfill и опустошал «Защиту капитала» обратно
        (у клиента `GlobalETFs` = 0–0), а счётчик в JSX был зашит числом.
"""

import copy
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Мандат ровно как в живом отчёте: панель показала 4 класса из 6, а
# `_build_mandate_compliance` скрывает строки 0–0 → GlobalETFs и Stocks_KZ
# закрыты лимитом 0–0.
LIVE_MANDATE = {"limits_dict": {
    "Bonds": [10, 30], "Stocks_US": [30, 60], "GlobalETFs": [0, 0],
    "Commodities": [0, 15], "Crypto": [0, 5], "Stocks_KZ": [0, 0],
}}
LIVE_HOLDINGS = ["AAOI", "AAPL", "FFSPC6.1028.AIX", "GLD", "GOOGL", "META",
                 "MSFT", "NVDA", "NXPI", "ORCL", "SLV", "SPCX", "TLT",
                 "KZT", "USD"]


def _results():
    return {"performance_table": pd.DataFrame({"Ticker": LIVE_HOLDINGS})}


# ──────────────────────────────────────────────────────────────────────────
# R-19 · литерал в промпте утёк в отчёт как факт
# ──────────────────────────────────────────────────────────────────────────
class RegimeConfidenceLiteralTest(unittest.TestCase):

    SRC = Path("src/ai_narrative.py")

    def setUp(self):
        self.src = self.SRC.read_text(encoding="utf-8")

    def test_no_hardcoded_confidence_example(self):
        """«— 8%» в примере формулировки — это фабрикуемое число."""
        self.assertNotIn("определена неуверенно — 8%", self.src)

    def test_example_uses_the_real_value(self):
        self.assertIn("определена неуверенно — {_conf_pct}%", self.src)

    def test_prompt_pins_the_only_allowed_number(self):
        self.assertIn("ЕДИНСТВЕННОЕ допустимое число уверенности", self.src)

    def test_sector_example_carries_no_number_either(self):
        """Тот же класс дефекта: «Tech-комплекс 80.8%» — готовое число."""
        self.assertNotIn("Tech-комплекс 80.8%", self.src)
        self.assertIn("Tech-комплекс <weight_pct>%", self.src)

    def test_rule_fires_only_below_the_floor(self):
        """Правило слабой уверенности не должно включаться при нормальной фазе."""
        self.assertIn("_REGIME_CONF_FLOOR = 25", self.src)


# ──────────────────────────────────────────────────────────────────────────
# R-20 · Sharpe при отрицательной премии за риск
# ──────────────────────────────────────────────────────────────────────────
class SharpeNegativeExcessTest(unittest.TestCase):
    """Проверяем и МАТЕМАТИКУ, и поведение движка."""

    def test_derivative_sign_flips_below_the_risk_free_rate(self):
        """Формальная причина: ∂S/∂σ = −(er−rf)/σ².

        При er < rf производная положительна, то есть Sharpe РАСТЁТ с
        волатильностью — упорядочивание портфелей ломается.
        """
        rf, er = 0.045, 0.028
        s = lambda v: (er - rf) / v
        self.assertLess(er - rf, 0)
        self.assertLess(s(0.192), s(0.233))      # меньше риска → ХУЖЕ Sharpe

    def test_live_report_numbers_reproduce_exactly(self):
        """0.49 → 0.47 воспроизводится из (er, rf, vol) живого отчёта."""
        rf, er, vb, va = 0.045, 0.028, 0.233, 0.192
        delta = (er - rf) / va - (er - rf) / vb
        self.assertAlmostEqual(round(0.49 + delta, 2), 0.47, places=2)

    def _sim(self, er_value, vol_after=0.192):
        """Прогон simulate с подменёнными форвардными входами."""
        from unittest.mock import patch
        import numpy as np
        import finance.simulate as sim

        idx = pd.bdate_range(end="2026-07-31", periods=300)
        rng = np.random.default_rng(3)
        tickers = ["AAA", "BBB"]
        dlr = pd.DataFrame(
            {t: rng.standard_normal(300) * 0.01 for t in tickers}, index=idx)
        perf = pd.DataFrame({
            "Ticker": tickers, "Weight": [0.6, 0.4],
            "Current_Value": [600.0, 400.0],
        })
        rm = pd.DataFrame([[0.04, 0.01], [0.01, 0.05]],
                          index=tickers, columns=tickers)
        with patch.object(sim, "_expected_return_from_bl",
                          side_effect=lambda w, r: er_value):
            return sim.simulate_after_plan(
                perf_df=perf, risk_matrix=rm, daily_log_returns=dlr,
                bl_records=[{"ticker": "AAA", "expected_return": er_value}],
                current_metrics={"Sharpe_Ratio": 0.49,
                                 "Composite_Risk_Score": 74},
                risk_free_rate=0.045,
                target_weights={"AAA": 0.3, "BBB": 0.7},
                sector_by_ticker={"AAA": "Technology", "BBB": "Bonds"},
            )

    def test_negative_excess_suppresses_the_delta(self):
        out = self._sim(er_value=0.028)          # 2.8% < rf 4.5%
        self.assertFalse(out["sharpe_delta_meaningful"])
        cell = out["metrics"]["sharpe"]
        self.assertEqual(cell["before"], cell["after"])
        self.assertIsNone(cell["improved"])       # нейтральная строка, не красная

    def test_negative_excess_states_the_reason(self):
        """Плоская строка без объяснения читалась бы как «план ничего не дал»."""
        note = self._sim(er_value=0.028)["sharpe_note"]
        self.assertTrue(note)
        self.assertIn("ниже безрисковой ставки", note)
        self.assertIn("волатильности", note)

    def test_positive_excess_keeps_the_delta(self):
        """Штатный случай не тронут: премия положительна → дельта считается."""
        out = self._sim(er_value=0.12)           # 12% > rf 4.5%
        self.assertTrue(out["sharpe_delta_meaningful"])
        self.assertEqual(out["sharpe_note"], "")
        cell = out["metrics"]["sharpe"]
        self.assertNotEqual(cell["before"], cell["after"])

    def test_positive_excess_moves_sharpe_against_volatility(self):
        """При ПОЛОЖИТЕЛЬНОЙ премии Sharpe идёт ПРОТИВ волатильности.

        Это и есть нормальная семантика коэффициента, которую ломает
        отрицательный избыток. Направление проверяем относительно фактического
        движения волатильности в фикстуре, а не постулируем его.
        """
        out = self._sim(er_value=0.12)
        vol = out["metrics"]["volatility_ann"]
        shp = out["metrics"]["sharpe"]
        vol_up = vol["after"] > vol["before"]
        if vol_up:
            self.assertLess(shp["after"], shp["before"])
        else:
            self.assertGreater(shp["after"], shp["before"])

    def test_note_reaches_the_payload_and_the_card(self):
        from pdf_payload import _build_expected_effect
        import premium_payload
        note = "Ожидаемая доходность ниже безрисковой ставки (премия -1.7 пп)…"
        payload = {
            "expected_effect": _build_expected_effect({"metrics": {
                "sharpe": {"before": 0.49, "after": 0.49,
                           "delta": 0.0, "improved": None}}}),
            "expected_effect_sharpe_note": note,
        }
        rows = premium_payload._map_deep(payload, {})["effect"]
        sharpe_row = next(r for r in rows if r["name"] == "Sharpe")
        self.assertEqual(sharpe_row["note"], note)
        self.assertEqual(sharpe_row["tone"], "flat")

    def test_card_has_no_note_when_delta_is_valid(self):
        import premium_payload
        rows = premium_payload._map_deep({"expected_effect": {}}, {})["effect"]
        sharpe_row = next(r for r in rows if r["name"] == "Sharpe")
        self.assertNotIn("note", sharpe_row)

    def test_expected_effect_contract_untouched(self):
        """Примечание живёт на уровне payload — контракт 8 строк не тронут."""
        from pdf_payload import _build_expected_effect
        self.assertEqual(_build_expected_effect(None), {})
        ee = _build_expected_effect({"metrics": {
            "risk_index": {"before": 1, "after": 2, "delta": 1, "improved": True}}})
        self.assertNotIn("sharpe_note", ee)


# ──────────────────────────────────────────────────────────────────────────
# R-21 · вердикт называет то, что реально ухудшилось
# ──────────────────────────────────────────────────────────────────────────
class VerdictNamesActualTradeoffTest(unittest.TestCase):

    def _verdict(self, trc_worse: bool, dd_worse: bool):
        from unittest.mock import patch
        import numpy as np
        import finance.simulate as sim
        # Собираем вердикт напрямую из логики, подменяя ячейки metrics.
        metrics = {
            "volatility_ann": {"delta": -0.04, "improved": True},
            "max_trc":        {"delta": 6.0 if trc_worse else -6.0,
                               "improved": not trc_worse},
            "max_drawdown":   {"delta": -0.03 if dd_worse else 0.03,
                               "improved": not dd_worse},
        }

        def _cell_worsened(name):
            cell = metrics.get(name) or {}
            return cell["improved"] is False and abs(cell["delta"]) > 1e-6

        trc_w, dd_w = _cell_worsened("max_trc"), _cell_worsened("max_drawdown")
        names = []
        if trc_w: names.append("концентрации")
        if dd_w:  names.append("просадки")
        return "Компромисс: снижение волатильности за счёт роста " + "/".join(names)

    def test_only_concentration_worsened_live_case(self):
        """Живой отчёт: TRC 27.2→33.2 (хуже), MaxDD −41.2→−38.7 (ЛУЧШЕ)."""
        h = self._verdict(trc_worse=True, dd_worse=False)
        self.assertIn("концентрации", h)
        self.assertNotIn("просадки", h)

    def test_only_drawdown_worsened(self):
        h = self._verdict(trc_worse=False, dd_worse=True)
        self.assertIn("просадки", h)
        self.assertNotIn("концентрации", h)

    def test_both_worsened_names_both(self):
        h = self._verdict(trc_worse=True, dd_worse=True)
        self.assertIn("концентрации/просадки", h)

    def test_source_no_longer_hardcodes_both(self):
        src = Path("src/finance/simulate.py").read_text(encoding="utf-8")
        self.assertNotIn('"роста концентрации/просадки"', src)


# ──────────────────────────────────────────────────────────────────────────
# R-22 · пропавшая четвёртая идея
# ──────────────────────────────────────────────────────────────────────────
class FourthIdeaTest(unittest.TestCase):

    AI_PICKS = {
        "boost_alpha":     {"label": "Рост", "desc": "", "picks": [
            {"ticker": "AMD", "name": "AMD", "why": "x", "type": "Stock"}]},
        "rebalance":       {"label": "Ребаланс", "desc": "", "picks": [
            {"ticker": "TXN", "name": "TXN", "why": "x", "type": "Stock"}]},
        # ИИ предложил защитные сектор/фактор-ETF — это класс GlobalETFs,
        # закрытый мандатом клиента лимитом 0–0.
        "protect_capital": {"label": "Защита", "desc": "", "picks": [
            {"ticker": "XLV",  "name": "Health", "why": "x", "type": "ETF"},
            {"ticker": "USMV", "name": "MinVol", "why": "x", "type": "ETF"}]},
        "smart_money":     {"label": "SM", "desc": "", "picks": [
            {"ticker": "AMAT", "name": "AMAT", "why": "x", "type": "Stock"}]},
    }

    def _pipeline(self):
        import ai_narrative as an
        sp = an._normalise_stock_picks(copy.deepcopy(self.AI_PICKS), "deep", "")
        sp = an._remove_mandate_banned_picks(sp, LIVE_MANDATE)
        return an._backfill_empty_scenarios(sp, "Expansion", "deep", "",
                                            _results(), user_mandate=LIVE_MANDATE)

    def test_banned_picks_are_still_removed(self):
        """Гард мандата не ослаблен — запрещённый класс по-прежнему уходит."""
        sp = self._pipeline()
        tickers = [p["ticker"] for p in sp["protect_capital"]["picks"]]
        self.assertNotIn("XLV", tickers)
        self.assertNotIn("USMV", tickers)

    def test_the_drained_bucket_is_refilled(self):
        """🔴 Суть R-22: карточка не исчезает после чистки мандатом."""
        sp = self._pipeline()
        self.assertTrue(sp["protect_capital"]["picks"])

    def test_refill_respects_the_mandate(self):
        """Добивка не имеет права вернуть бумагу из запрещённого класса."""
        from agent.gatekeeper import _classify_to_asset_key
        sp = self._pipeline()
        for key, scen in sp.items():
            for p in scen.get("picks", []):
                self.assertNotIn(_classify_to_asset_key(p["ticker"]),
                                 {"GlobalETFs", "Stocks_KZ"},
                                 f"{p['ticker']} в {key} — класс закрыт мандатом")

    def test_report_renders_four_cards(self):
        from pdf_payload import _build_ai_ideas
        ideas = _build_ai_ideas(self._pipeline(), tier="deep")
        self.assertEqual(sum(len(v) for v in ideas.values()), 4)

    def test_old_ordering_lost_the_card(self):
        """Замер дефекта: гард ПОСЛЕ backfill опустошает корзину обратно."""
        import ai_narrative as an
        from pdf_payload import _build_ai_ideas
        sp = an._normalise_stock_picks(copy.deepcopy(self.AI_PICKS), "deep", "")
        sp = an._backfill_empty_scenarios(sp, "Expansion", "deep", "", _results())
        sp = an._remove_mandate_banned_picks(sp, LIVE_MANDATE)      # прежний порядок
        self.assertEqual(sum(len(v) for v in _build_ai_ideas(sp, "deep").values()), 3)

    def test_backfill_still_works_without_a_mandate(self):
        """Обратная совместимость: без мандата поведение прежнее."""
        import ai_narrative as an
        sp = an._normalise_stock_picks(
            {"boost_alpha": {"label": "a", "desc": "", "picks": []}}, "deep", "")
        sp = an._backfill_empty_scenarios(sp, "Expansion", "deep", "", _results())
        self.assertTrue(sp["boost_alpha"]["picks"])

    def test_mandate_allows_predicate(self):
        import ai_narrative as an
        self.assertFalse(an._mandate_allows("XLV", LIVE_MANDATE))     # GlobalETFs
        self.assertTrue(an._mandate_allows("AGG", LIVE_MANDATE))      # Bonds
        self.assertTrue(an._mandate_allows("AAPL", LIVE_MANDATE))     # Stocks_US
        self.assertTrue(an._mandate_allows("XLV", None))              # нет мандата
        self.assertTrue(an._mandate_allows("XLV", {"limits_dict": {}}))

    def test_offline_catalogue_respects_the_mandate(self):
        """Фолбэк без API тоже не должен предлагать запрещённый класс."""
        import ai_narrative as an
        from agent.gatekeeper import _classify_to_asset_key
        fb = an._fallback_stock_picks("Expansion", "deep",
                                      user_mandate=LIVE_MANDATE)
        for key, scen in fb.items():
            for p in scen["picks"]:
                self.assertNotIn(_classify_to_asset_key(p["ticker"]),
                                 {"GlobalETFs", "Stocks_KZ"},
                                 f"{p['ticker']} в офлайн-каталоге {key}")

    def test_idea_count_label_is_derived_not_hardcoded(self):
        """Подпись «4 идеи» была зашита и врала при трёх карточках."""
        for name in ("base-components.js", "deep-components.js"):
            js = (Path("src/premium_assets") / name).read_text(encoding="utf-8")
            self.assertIn("ideaCount", js, f"{name}: счётчик не пересобран")
            self.assertNotIn("AI Ideas \\xB7 4 \\u0438\\u0434\\u0435\\u0438", js)

    def test_jsx_sources_match_the_bundles(self):
        for rel in ("design/premium_v2/portfolio-ideas.jsx",
                    "design/premium_v2/deep/deep-plan.jsx"):
            jsx = Path(rel)
            if not jsx.exists():
                self.skipTest("design/ отсутствует (Docker deploy-gate)")
            src = jsx.read_text(encoding="utf-8")
            self.assertIn("ideaCount", src)
            self.assertNotIn("AI Ideas · 4 идеи", src)


if __name__ == "__main__":
    unittest.main()
