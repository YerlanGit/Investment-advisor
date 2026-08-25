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
# формат 'id:secret' обязателен: `tg_bot` валидирует токен на ИМПОРТЕ,
# и `setdefault` с невалидным значением ломает импорт в любом файле,
# который запустится ПОСЛЕ этого (`AUDIT §−85`).
os.environ.setdefault("OMBRI_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")

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


class EffectPanelClosesArithmeticTest(unittest.TestCase):
    """§−102 (live DEEP 24.08) · у высвобожденного веса ВСЕГДА есть адресат.

    🔴 Дефект. Лестница реинвеста и честная строка «Кэш» стояли за БИНАРНЫМ
    гейтом «в плане НЕТ ни одной покупки» (`not have_buy`). План живого отчёта
    нёс ОДНУ символическую покупку — GOOGL +0.66 пп, — и этого хватило, чтобы
    пропустить весь блок: панель показала −20.61 пп продаж против +0.66 пп
    покупок, а 19.95 пп (это ~20% NAV) исчезли из отчёта БЕЗ АДРЕСАТА.

    🔴 Цена. Дыру заполнила модель: `ai_effect_comment` и `ai_action_comment`
    написали «высвобожденный вес движок направляет в качественные
    диверсификаторы вне техно», чего движок не делал — вес уходил в кэш.
    Это ровно `§−90` A-5: запрет в промпте («не выдумывай адресат») неисполним,
    пока состояние движка до модели не доехало.

    Инвариант, который здесь пинится: Σ delta_pp по ВСЕМ строкам панели = 0.
    Продажи, покупки и кэш обязаны сходиться — панель, где 20% NAV просто
    испаряются, арифметически неполна независимо от того, кто её читает.
    """

    #: Книга и план — с живого DEEP 24.08.2026, числа не подогнаны.
    _BASE = {"MSTR": .079, "TLT": .069, "ORCL": .046, "AAPL": .100, "CRCL": .029,
             "SLV": .028, "USAR": .047, "META": .108, "GOOGL": .045,
             "MSFT": .188, "NVDA": .134}
    _SECTOR = {"MSTR": "Technology", "GOOGL": "Technology", "MSFT": "Technology",
               "NVDA": "Semiconductors", "META": "Technology", "AAPL": "Technology",
               "ORCL": "Technology", "CRCL": "Other", "USAR": "Other",
               "TLT": "Bonds", "SLV": "Gold"}
    _ROWS = [{"ticker": "MSTR",  "action": "Sell", "delta_w_pp": -6.62},
             {"ticker": "TLT",   "action": "Trim", "delta_w_pp": -5.78},
             {"ticker": "ORCL",  "action": "Sell", "delta_w_pp": -3.84},
             {"ticker": "AAPL",  "action": "Trim", "delta_w_pp": -1.39},
             {"ticker": "CRCL",  "action": "Sell", "delta_w_pp": -1.25},
             {"ticker": "SLV",   "action": "Trim", "delta_w_pp": -1.18},
             {"ticker": "USAR",  "action": "Sell", "delta_w_pp": -0.41},
             {"ticker": "META",  "action": "Trim", "delta_w_pp": -0.14},
             {"ticker": "GOOGL", "action": "Buy",  "delta_w_pp":  0.66}]

    def _actions(self):
        from finance.simulate import high_priority_target_weights
        bl = [{"ticker": t, "delta_w_pp": 1.0, "action": "Buy", "target_w": w}
              for t, w in self._BASE.items()]
        _t, _tk, actions = high_priority_target_weights(
            self._BASE, self._ROWS, bl, sector_by_ticker=self._SECTOR)
        return actions

    def test_freed_weight_has_a_destination_even_when_the_plan_has_a_buy(self):
        actions = self._actions()
        self.assertTrue(
            any(a.get("is_cash") for a in actions),
            "план с ОДНОЙ покупкой снова спрятал адресат высвобожденного веса: "
            "строки «Кэш» нет, и 19.95 пп NAV исчезли из панели")

    def test_sum_of_all_moves_is_zero(self):
        total = sum(a["delta_pp"] for a in self._actions())
        self.assertAlmostEqual(
            total, 0.0, places=1,
            msg=f"панель не сходится: Σ delta_pp = {total:+.2f} пп. "
                "Продажи, покупки и кэш обязаны давать ноль")

    def test_cash_row_carries_the_whole_unabsorbed_remainder(self):
        actions = self._actions()
        cash = [a for a in actions if a.get("is_cash")][0]
        sells = -sum(a["delta_pp"] for a in actions if a["delta_pp"] < 0)
        buys  = sum(a["delta_pp"] for a in actions
                    if a["delta_pp"] > 0 and not a.get("is_cash"))
        self.assertAlmostEqual(cash["delta_pp"], sells - buys, places=1)

    def test_net_buying_plan_is_funded_from_cash_and_says_so(self):
        """Зеркальный случай: план покупает больше, чем продаёт.

        Деньги приходят ИЗ кэша, и это тоже адресат — панель обязана сойтись
        и здесь, иначе инвариант держится только на одной ветке.
        """
        from finance.simulate import high_priority_target_weights
        rows = [{"ticker": "MSTR",  "action": "Sell", "delta_w_pp": -1.0},
                {"ticker": "GOOGL", "action": "Buy",  "delta_w_pp":  4.0}]
        bl = [{"ticker": t, "delta_w_pp": 1.0, "action": "Buy", "target_w": w}
              for t, w in self._BASE.items()]
        _t, _tk, actions = high_priority_target_weights(
            self._BASE, rows, bl, sector_by_ticker=self._SECTOR)
        cash = [a for a in actions if a.get("is_cash")]
        self.assertTrue(cash, "финансирование покупок не названо")
        self.assertLess(cash[0]["delta_pp"], 0, "приход ИЗ кэша обязан быть отрицательным")
        self.assertAlmostEqual(sum(a["delta_pp"] for a in actions), 0.0, places=1)

    def test_residual_still_reaches_real_diversifiers_not_only_cash(self):
        """Гейт стал ОСТАТОЧНЫМ, а не «есть ли покупка вообще».

        Иначе лечение вырождается в противоположную крайность: план с одной
        покупкой всегда сваливал бы ВЕСЬ остаток в кэш, хотя честный
        диверсификатор с рейтингом Buy в книге есть. Реинвест обязан
        отработать на ОСТАТКЕ — и только непоглощённый хвост идёт в кэш.
        """
        from finance.simulate import high_priority_target_weights
        # GLD: план рейтингует Buy, но |Δw| = 0 → он НЕ среди торгуемых строк,
        # значит кандидат реинвеста; сектор Gold вне заблокированного техно.
        rows = self._ROWS + [{"ticker": "GLD", "action": "Buy", "delta_w_pp": 0.0}]
        base = dict(self._BASE, GLD=.052)
        sector = dict(self._SECTOR, GLD="Gold")
        bl = [{"ticker": "GLD", "delta_w_pp": 9.0, "action": "Buy"}]
        _t, _tk, actions = high_priority_target_weights(
            base, rows, bl, sector_by_ticker=sector)
        bought = {a["ticker"] for a in actions
                  if a["delta_pp"] > 0 and not a.get("is_cash")}
        self.assertIn(
            "GLD", bought,
            "остаток не дошёл до диверсификатора: гейт снова читает «есть ли "
            "покупка», а не «сколько веса осталось непоглощённым»")
        self.assertAlmostEqual(sum(a["delta_pp"] for a in actions), 0.0, places=1)

    def test_residual_is_split_proportionally_not_by_gross_sells(self):
        """Лестница делит ОСТАТОК, а не всю выручку от продаж.

        Если долю каждого кандидата считать от ВАЛОВЫХ продаж, первый же
        кандидат выбирает весь остаток, и второй диверсификатор получает ноль —
        реинвест концентрируется вместо того, чтобы разносить.
        """
        from finance.simulate import high_priority_target_weights
        base   = {"MSTR": .30, "MSFT": .30, "GLD": .20, "AGGX": .20}
        sector = {"MSTR": "Technology", "MSFT": "Technology",
                  "GLD": "Gold", "AGGX": "Bonds"}
        rows = [{"ticker": "MSTR", "action": "Sell", "delta_w_pp": -20.0},
                {"ticker": "MSFT", "action": "Buy",  "delta_w_pp":  10.0},
                {"ticker": "GLD",  "action": "Buy",  "delta_w_pp":   0.0},
                {"ticker": "AGGX", "action": "Buy",  "delta_w_pp":   0.0}]
        bl = [{"ticker": "GLD", "delta_w_pp": 5.0, "action": "Buy"},
              {"ticker": "AGGX", "delta_w_pp": 5.0, "action": "Buy"}]
        _t, _tk, actions = high_priority_target_weights(
            base, rows, bl, sector_by_ticker=sector)
        got = {a["ticker"]: a["delta_pp"] for a in actions if a["delta_pp"] > 0}
        for t in ("GLD", "AGGX"):
            self.assertGreater(
                got.get(t, 0.0), 0.0,
                f"{t} остался без веса: доля кандидата считается от валовых "
                f"продаж, а не от остатка — получили {got}")
        self.assertAlmostEqual(sum(a["delta_pp"] for a in actions), 0.0, places=1)

    def test_cash_row_is_labelled_by_direction_in_the_payload(self):
        """«В кэш» и «Из кэша» — разные факты; один ярлык на оба лжёт."""
        from pdf_payload import _build_expected_effect
        out_in = _build_expected_effect({"high_priority_actions": [
            {"ticker": "Кэш", "action": "Cash", "side": "buy",
             "delta_pp": 19.95, "is_cash": True}]})
        out_from = _build_expected_effect({"high_priority_actions": [
            {"ticker": "Кэш", "action": "Cash", "side": "sell",
             "delta_pp": -3.0, "is_cash": True}]})
        self.assertEqual(out_in["high_priority_actions"][0]["side"], "В кэш")
        self.assertEqual(out_from["high_priority_actions"][0]["side"], "Из кэша")

    def test_prompt_receives_cash_as_a_named_destination(self):
        """R-5/§−102: `reinvest_destination` — ГОТОВАЯ строка для модели.

        Пока кэша в ней не было, модель дописывала адресат сама.
        """
        from ai_narrative import _reinvest_destination
        dest = _reinvest_destination({"expected_effect": {"high_priority_actions": [
            {"ticker": "Кэш",   "side": "buy", "delta_pp": 19.95, "is_cash": True},
            {"ticker": "GOOGL", "side": "buy", "delta_pp": 0.66},
        ]}})
        self.assertIn("Кэш", dest)
        self.assertIn("19.9", dest)
        self.assertIn("GOOGL", dest)

    def test_prompt_permits_naming_cash_instead_of_inventing_buys(self):
        """`§−97` E-7: правила, которого нет в промпте, модель не исполняет.

        Прежняя редакция утверждала «движок покупает конкретные
        диверсификаторы» — и УЧИЛА не называть кэш даже тогда, когда кэш и
        есть настоящий адресат.
        """
        import ai_narrative
        spec = ai_narrative._deep_prompt_spec() if hasattr(
            ai_narrative, "_deep_prompt_spec") else None
        if spec is None:
            import inspect
            spec = inspect.getsource(ai_narrative)
        self.assertIn("«Кэш» — ПОЛНОПРАВНЫЙ адресат", spec)
        self.assertNotIn("движок покупает конкретные", spec)


class RiskIndexDeltaPluralTest(unittest.TestCase):
    """§−102 · «−8 пункта» в живом отчёте 24.08.

    Форма числительного — функция ЧИСЛА, а не константа строки: карточка
    печатала родительный падеж литералом и была права ровно для 2–4.
    """

    def test_plural_forms(self):
        from premium_payload import _eff_delta
        cases = {-8: "пунктов", -5: "пунктов", -2: "пункта", -1: "пункт",
                 1: "пункт", 2: "пункта", 4: "пункта", 5: "пунктов",
                 11: "пунктов", 12: "пунктов", 21: "пункт", 22: "пункта"}
        for n, want in cases.items():
            with self.subTest(n=n):
                self.assertTrue(
                    _eff_delta("risk_index", n).endswith(want),
                    f"{n} → {_eff_delta('risk_index', n)!r}, ожидалось «…{want}»")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
