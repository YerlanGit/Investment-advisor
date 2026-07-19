"""Раунд 29 (2026-07-19, live-репро отчёта 13:03): качество реинвеста Effect.

L-17 — CONVICTION GATE: Effect исполняет ПЛАН, а не изобретает покупки.
    Live: FFSPC6.1028.AIX (неликвидная AIX-нота, 4-Pillar HOLD со всеми
    пилларами 0.0 — данных нет) получала «Купить +12пп», потому что её
    сглаженная цена (corr 0.97 c TLT) нравится оптимизатору.  Теперь held-имя
    — кандидат реинвеста ТОЛЬКО при план-рейтинге Buy/Strong Buy.

L-18 — EXTERNAL GLOBAL-ETF SLEEVE: высвобожденный вес, который некуда деть
    внутри книги, идёт в ликвидные глобальные ETF из ФАКТОРНОЙ панели движка
    (IEF/EEM/EMB — история скачана каждым прогоном), в порядке мандата,
    ≤8пп на имя; остаток — честно в Кэш.  Симуляция расширяет ковариацию
    sample-блоком, чтобы покупка не считалась «бесплатным кэшем».

L-19 — ПРОМПТ-КАЧЕСТВО: расшифровка CVaR без «худший день из 20», правило
    структуры 2–3 предложений, numbers_rule (анти-фабрикация чисел) в ОБОИХ
    тирах, rebalance_actions в данных промпта.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("RAMP_BOT_TOKEN", "test-token-collection")

import numpy as np
import pandas as pd


# ── L-17 · conviction gate ───────────────────────────────────────────────────

class ConvictionGateTest(unittest.TestCase):
    """Live-репро 19.07 13:03: FFSPC (план HOLD, пиллары 0.0) не должна
    покупаться, даже когда blocklist её не ловит (нота ЕСТЬ в факторной
    модели — corr 0.97 с TLT — и потому не sparse/broker-priced)."""

    def _book(self):
        cur = {"ORCL": 0.09, "MSFT": 0.17, "FFSPC6.1028.AIX": 0.046,
               "TLT": 0.078, "GLD": 0.052}
        sector = {"ORCL": "Technology", "MSFT": "Technology",
                  "FFSPC6.1028.AIX": "EM_Kazakhstan", "TLT": "Bonds",
                  "GLD": "Gold"}
        rows = [{"ticker": "ORCL", "action": "Sell", "delta_w_pp": -8.65},
                {"ticker": "MSFT", "action": "Trim", "delta_w_pp": -3.37},
                {"ticker": "FFSPC6.1028.AIX", "action": "Hold", "delta_w_pp": 0.0},
                {"ticker": "TLT",  "action": "Trim", "delta_w_pp": -1.0}]
        bl = [{"ticker": "FFSPC6.1028.AIX", "delta_w_pp": 9.0},
              {"ticker": "GLD", "delta_w_pp": 1.5}]
        return cur, sector, rows, bl

    def test_hold_rated_name_not_bought_even_without_blocklist(self):
        from finance.simulate import high_priority_target_weights
        cur, sector, rows, bl = self._book()
        _t, _tk, actions = high_priority_target_weights(
            cur, rows, bl, sector_by_ticker=sector)   # blocklist ПУСТОЙ
        buys = {a["ticker"] for a in actions
                if a["side"] == "buy" and not a.get("is_cash")}
        self.assertNotIn("FFSPC6.1028.AIX", buys)
        # GLD: BL Δw>0, но план говорит Trim/нет Buy-рейтинга → тоже не покупаем.
        self.assertNotIn("GLD", buys)
        self.assertTrue(any(a.get("is_cash") for a in actions))

    def test_plan_buy_rated_name_still_bought(self):
        from finance.simulate import high_priority_target_weights
        cur, sector, rows, bl = self._book()
        rows2 = rows + [{"ticker": "GLD", "action": "Buy", "delta_w_pp": 0.0}]
        _t, _tk, actions = high_priority_target_weights(
            cur, rows2, bl, sector_by_ticker=sector)
        buys = {a["ticker"] for a in actions
                if a["side"] == "buy" and not a.get("is_cash")}
        self.assertIn("GLD", buys)
        self.assertNotIn("FFSPC6.1028.AIX", buys)


# ── L-18 · external global-ETF sleeve ────────────────────────────────────────

class ExternalSleeveTest(unittest.TestCase):
    def _sold_book(self):
        cur = {"ORCL": 0.3, "MSFT": 0.3, "NVDA": 0.4}
        sector = {"ORCL": "Technology", "MSFT": "Technology",
                  "NVDA": "Semiconductors"}
        rows = [{"ticker": "ORCL", "action": "Sell", "delta_w_pp": -20.0}]
        bl = [{"ticker": "MSFT", "delta_w_pp": 5.0}]   # tech → заблокирован
        return cur, sector, rows, bl

    def test_externals_absorb_freed_weight_with_cap(self):
        from finance.simulate import (high_priority_target_weights,
                                      external_diversifier_candidates,
                                      _EXTERNAL_PER_BUY_CAP)
        cur, sector, rows, bl = self._sold_book()
        cands = external_diversifier_candidates("MODERATE")
        target, _tk, actions = high_priority_target_weights(
            cur, rows, bl, sector_by_ticker=sector, external_candidates=cands)
        ext = [a for a in actions if a.get("is_external")]
        self.assertTrue(ext, "external sleeve must engage")
        for a in ext:
            self.assertLessEqual(a["delta_pp"], _EXTERNAL_PER_BUY_CAP * 100 + 1e-6)
            self.assertTrue(a.get("name"), "external buy must carry a name")
            self.assertEqual(a["side"], "buy")
        # порядок мандата MODERATE: первым идёт IEF
        self.assertEqual(ext[0]["ticker"], "IEF")
        # 20пп продаж: IEF 8 + EEM 8 + EMB 4 → кэша не остаётся (или мелочь)
        total_ext = sum(a["delta_pp"] for a in ext)
        cash = sum(a["delta_pp"] for a in actions if a.get("is_cash"))
        self.assertAlmostEqual(total_ext + cash, 20.0, delta=0.1)
        for a in ext:
            self.assertGreater(target.get(a["ticker"], 0.0), 0.0)

    def test_held_candidate_not_duplicated(self):
        """Если IEF уже держат — внешняя рука его не предлагает (докуп held
        решается conviction-гейтом, не sleeve'ом)."""
        from finance.simulate import (high_priority_target_weights,
                                      external_diversifier_candidates)
        cur, sector, rows, bl = self._sold_book()
        cur = dict(cur); cur["IEF"] = 0.05
        cands = external_diversifier_candidates("MODERATE")
        _t, _tk, actions = high_priority_target_weights(
            cur, rows, bl, sector_by_ticker=sector, external_candidates=cands)
        ext = {a["ticker"] for a in actions if a.get("is_external")}
        self.assertNotIn("IEF", ext)

    def test_no_candidates_falls_back_to_cash(self):
        from finance.simulate import high_priority_target_weights
        cur, sector, rows, bl = self._sold_book()
        _t, _tk, actions = high_priority_target_weights(
            cur, rows, bl, sector_by_ticker=sector, external_candidates=None)
        self.assertTrue(any(a.get("is_cash") for a in actions))
        self.assertFalse(any(a.get("is_external") for a in actions))

    def test_mandate_ordering(self):
        from finance.simulate import external_diversifier_candidates
        self.assertEqual(external_diversifier_candidates("CONSERVATIVE")[0]["ticker"], "IEF")
        self.assertEqual(external_diversifier_candidates("AGGRESSIVE")[0]["ticker"], "EEM")
        # неизвестный мандат → MODERATE
        self.assertEqual(external_diversifier_candidates("???")[0]["ticker"], "IEF")

    def test_candidates_are_factor_panel_tickers(self):
        """Каждый кандидат обязан быть факторным ETF движка — иначе «загрузить
        сразу» перестаёт быть бесплатным (и тест это поймает при правке реестра)."""
        from finance.simulate import EXTERNAL_DIVERSIFIERS
        from finance.investment_logic import MAC3RiskEngine
        panel = set(MAC3RiskEngine().factor_tickers.values())
        for cands in EXTERNAL_DIVERSIFIERS.values():
            for c in cands:
                self.assertIn(c["panel"], panel, c)

    def test_candidates_serve_mandate_classes(self):
        """Классификатор мандат-панели согласен с asset_key кандидатов
        (IEF/EMB → Bonds, EEM → GlobalETFs) — покупка видна лимитам."""
        from finance.simulate import EXTERNAL_DIVERSIFIERS
        from agent.gatekeeper import _classify_to_asset_key
        for cands in EXTERNAL_DIVERSIFIERS.values():
            for c in cands:
                self.assertEqual(_classify_to_asset_key(c["ticker"]),
                                 c["asset_key"], c)


class ExtendedCovSimulationTest(unittest.TestCase):
    """L-18: simulate_after_plan расширяет ковариацию sample-блоком для внешних
    покупок — метрики «после» видят реальный риск ETF, а не фантомный кэш."""

    def _inputs(self):
        rng = np.random.default_rng(7)
        n = 500
        dates = pd.bdate_range("2024-01-02", periods=n)
        tech = rng.normal(0.0008, 0.02, n)
        ief  = rng.normal(0.0002, 0.004, n)
        daily = pd.DataFrame({"TECH": tech, "IEF": ief}, index=dates)
        cov = pd.DataFrame([[float(np.var(tech) * 252)]],
                           index=["TECH"], columns=["TECH"])
        perf = pd.DataFrame({"Ticker": ["TECH"], "Current_Value": [100.0],
                             "Fundamental_Sector": ["Technology"]})
        return perf, cov, daily

    def test_external_buy_carries_real_risk(self):
        from finance.simulate import simulate_after_plan
        perf, cov, daily = self._inputs()
        res = simulate_after_plan(
            perf_df=perf, risk_matrix=cov, daily_log_returns=daily,
            bl_records=None, current_metrics={}, risk_free_rate=0.04,
            target_weights={"TECH": 0.7, "IEF": 0.3})
        self.assertIsNotNone(res)
        vol_b = res["metrics"]["volatility_ann"]["before"]
        vol_a = res["metrics"]["volatility_ann"]["after"]
        # «до» — нетронутый структурный headline (только TECH)
        self.assertAlmostEqual(vol_b, float(np.sqrt(cov.iloc[0, 0])), places=6)
        # «после» ниже (диверсификация в низковольный IEF), но НЕ равно
        # чистому кэш-сценарию 0.7×σ_TECH — IEF несёт собственный риск.
        cash_only = 0.7 * vol_b
        self.assertLess(vol_a, vol_b)
        self.assertGreater(vol_a, cash_only)

    def test_short_overlap_never_grants_free_lunch(self):
        """< 60 дней перекрытия → расширения нет; вес выпадает (эквивалент
        кэша), но НЕ появляется актив с нулевым риском в матрице."""
        from finance.simulate import simulate_after_plan
        perf, cov, daily = self._inputs()
        daily_short = daily.iloc[-30:]
        res = simulate_after_plan(
            perf_df=perf, risk_matrix=cov, daily_log_returns=daily_short,
            bl_records=None, current_metrics={}, risk_free_rate=0.04,
            target_weights={"TECH": 0.7, "IEF": 0.3})
        self.assertIsNotNone(res)
        m = res["metrics"]["volatility_ann"]
        self.assertAlmostEqual(m["after"], 0.7 * m["before"], places=6)


# ── L-19 · качество комментариев ИИ ─────────────────────────────────────────

class PromptQualityTest(unittest.TestCase):
    def _prompts(self):
        from ai_narrative import _user_prompt
        summary = {
            "verdict": "x", "portfolio_metrics": {}, "assets": [],
            "reporting": {"currency": "USD"},
            "regime": {"label": "Expansion", "confidence": 30},
        }
        return (_user_prompt(summary, tier="base"),
                _user_prompt(summary, tier="deep"))

    def test_old_cvar_template_gone(self):
        base, deep = self._prompts()
        for p in (base, deep):
            self.assertNotIn("худший день из 20", p)

    def test_new_cvar_decoding_present(self):
        base, deep = self._prompts()
        for p in (base, deep):
            self.assertIn("редкий плохой день", p)

    def test_numbers_rule_in_both_tiers(self):
        base, deep = self._prompts()
        for p in (base, deep):
            self.assertIn("ЧИСЛА — СТРОГО ИЗ ДАННЫХ", p)
            self.assertIn("Не пере-округляй", p)

    def test_structure_rule_present(self):
        base, deep = self._prompts()
        for p in (base, deep):
            self.assertIn("СТРУКТУРА комментария", p)

    def test_deep_effect_spec_names_destinations(self):
        _b, deep = self._prompts()
        self.assertIn("rebalance_actions", deep)
        self.assertIn("КУДА уходит высвобожденный вес", deep)

    def test_rebalance_actions_reach_prompt_data(self):
        from ai_narrative import _summarise_for_prompt
        results = {
            "portfolio_metrics": {}, "assets": [],
            "expected_effect": {
                "verdict": {"kind": "improvement", "headline": "ok"},
                "high_priority_actions": [
                    {"ticker": "ORCL", "side": "sell", "delta_pp": -8.65},
                    {"ticker": "IEF", "side": "buy", "delta_pp": 8.0,
                     "is_external": True,
                     "name": "гособлигации США 7–10 лет"},
                ],
            },
        }
        s = _summarise_for_prompt(results)
        acts = s.get("rebalance_actions")
        self.assertTrue(acts and len(acts) == 2)
        self.assertTrue(acts[1]["external"])
        self.assertIn("гособлигации", acts[1]["name"])


class EffectActionsNamePropagationTest(unittest.TestCase):
    """L-18: имя внешнего ETF доезжает payload → premium effectActions."""

    def test_name_mapped_through_payload_layers(self):
        from pdf_payload import _build_expected_effect
        raw = {
            "verdict": {"kind": "improvement", "headline": "ok"},
            "high_priority_tickers": ["ORCL", "IEF"],
            "high_priority_actions": [
                {"ticker": "ORCL", "action": "Sell", "side": "sell",
                 "delta_pp": -8.65},
                {"ticker": "IEF", "action": "Buy", "side": "buy",
                 "delta_pp": 8.0, "is_external": True,
                 "name": "гособлигации США 7–10 лет"},
            ],
        }
        out = _build_expected_effect(raw)
        acts = out["high_priority_actions"]
        ief = [a for a in acts if a["ticker"] == "IEF"][0]
        self.assertEqual(ief["side"], "Купить")
        self.assertTrue(ief["is_external"])
        self.assertIn("гособлигации", ief["name"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
