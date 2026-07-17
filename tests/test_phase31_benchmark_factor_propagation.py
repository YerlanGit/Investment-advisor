"""
Phase 31 — сквозной проброс мандатного бенчмарка в «Факторное разложение»
+ /mandate-меню без биллинга (ТЗ B1, 2026-07-17).

Regression for the 2026-07-17 P1 report bug: клиент сменил бенчмарк
S&P 500 → NASDAQ через /mandate, но секция «Факторное разложение (β)» в DEEP
по-прежнему сравнивала портфель с S&P 500 — хардкод в ТРЁХ слоях сразу:

  A. фронт  — подписи «S&P 500» в deep-factors.jsx / deep-components.js;
  B. payload — `_BENCH_FACTOR_BETAS` = константа (Market=1, rest=0), профиль
     клиента вообще не читался (`tg_bot._build_factor_betas_table`);
  C. движок — ось Market жёстко SPY.US (это НЕ баг — каноническая ось модели;
     ADR Вариант A: бенчмарк — отдельная сущность, модель не трогаем).

Fix: движок считает РЕАЛЬНЫЕ факторные беты выбранного бенчмарка тем же
Ridge-пайплайном (`MAC3RiskEngine.fit_factor_betas` →
results["benchmark_factor_profile"]); столбец/подписи секции — динамические
(payload.benchmark_name → premium benchmarkName); фолбэк — прежняя
S&P-константа С прежней подписью (подпись всегда совпадает с числами).

Плюс /mandate: смена бенчмарка за 2 тапа без повторной анкеты и БЕЗ единого
вызова биллинга (guardrail §5.3 ТЗ).
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("RAMP_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")

from finance.investment_logic import MAC3RiskEngine  # noqa: E402


# ── Synthetic factor panel ───────────────────────────────────────────────────

_FACTOR_ETFS = {
    "Market": "SPY.US", "Momentum": "MTUM.US", "Value": "VLUE.US",
    "Quality": "QUAL.US", "Size": "IWM.US", "Volatility": "SPLV.US",
    "Commodities": "DBC.US", "Rates": "IEF.US",
    "EM_Equity": "EEM.US", "EM_Bond": "EMB.US",
}


def _make_price_panel(n: int = 420, seed: int = 7) -> pd.DataFrame:
    """Независимые факторы + QQQ с ИЗВЕСТНЫМИ наклонами (1.05·SPY + 0.25·MTUM)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    rets = {tk: rng.normal(0.0003, 0.01, n) for tk in _FACTOR_ETFS.values()}
    rets["QQQ.US"] = (1.05 * rets["SPY.US"] + 0.25 * rets["MTUM.US"]
                      + rng.normal(0.0, 0.002, n))
    return pd.DataFrame(
        {tk: 100.0 * np.exp(np.cumsum(r)) for tk, r in rets.items()}, index=idx)


class EngineBenchmarkBetasTest(unittest.TestCase):
    """Слой C — `fit_factor_betas`: тот же Ridge-пайплайн, что и для активов."""

    @classmethod
    def setUpClass(cls):
        cls.engine = MAC3RiskEngine()
        cls.prices = _make_price_panel()

    def test_qqq_betas_are_real_not_constant(self):
        betas = self.engine.fit_factor_betas(self.prices, "QQQ.US")
        self.assertIsNotNone(betas)
        # Market-наклон NASDAQ в разумном диапазоне (введён как 1.05).
        self.assertGreaterEqual(betas["Market"], 0.9)
        self.assertLessEqual(betas["Market"], 1.2)
        # Стилевые беты НЕ все нули — Momentum введён как +0.25.
        self.assertGreater(betas["Momentum"], 0.05)
        style_abs = [abs(v) for k, v in betas.items() if k != "Market"]
        self.assertGreater(max(style_abs), 0.05)

    def test_spy_recovers_the_snp_constant_by_construction(self):
        """Обратная совместимость: для SPY беты ≈ (Market=1, стили≈0)."""
        betas = self.engine.fit_factor_betas(self.prices, "SPY.US")
        self.assertIsNotNone(betas)
        self.assertAlmostEqual(betas["Market"], 1.0, delta=0.03)
        for axis, v in betas.items():
            if axis != "Market":
                self.assertAlmostEqual(v, 0.0, delta=0.05,
                                       msg=f"style axis {axis} drifted: {v}")

    def test_missing_ticker_returns_none(self):
        self.assertIsNone(self.engine.fit_factor_betas(self.prices, "XXXX.US"))

    def test_short_history_returns_none(self):
        self.assertIsNone(
            self.engine.fit_factor_betas(self.prices.head(15), "QQQ.US"))

    def test_empty_panel_returns_none(self):
        self.assertIsNone(self.engine.fit_factor_betas(None, "QQQ.US"))
        self.assertIsNone(
            self.engine.fit_factor_betas(pd.DataFrame(), "QQQ.US"))


# ── Слой B — таблица факторов в tg_bot ───────────────────────────────────────

def _import_tg_bot():
    os.environ.setdefault("RAMP_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")
    try:
        import tg_bot
        return tg_bot
    except Exception:
        return None


def _perf_results(bench_profile=None) -> dict:
    perf = pd.DataFrame({
        "Ticker":        ["AAA", "BBB"],
        "Current_Value": [600.0, 400.0],
        "Beta_Market":   [1.30, 0.90],
        "Beta_Momentum": [0.50, 0.10],
        "Beta_Value":    [-0.20, 0.00],
    })
    results = {"performance_table": perf, "total_value": 1000.0}
    if bench_profile is not None:
        results["benchmark_factor_profile"] = bench_profile
    return results


class FactorTableBenchmarkColumnTest(unittest.TestCase):
    """`_build_factor_betas_table` читает профиль бенчмарка с фолбэком."""

    @classmethod
    def setUpClass(cls):
        cls.tg = _import_tg_bot()

    def setUp(self):
        if self.tg is None:
            self.skipTest("tg_bot import unavailable")

    def _row(self, rows, axis):
        return next(r for r in rows if r["axis"] == axis)

    def test_backward_compat_no_profile_uses_snp_constant(self):
        rows = self.tg._build_factor_betas_table(_perf_results())
        self.assertTrue(rows)
        self.assertEqual(self._row(rows, "Market")["bench"], 1.0)
        self.assertEqual(self._row(rows, "Momentum")["bench"], 0.0)
        mkt = self._row(rows, "Market")
        self.assertAlmostEqual(mkt["delta"], round(mkt["beta"] - 1.0, 2))

    def test_nasdaq_profile_drives_bench_column_and_delta(self):
        rows = self.tg._build_factor_betas_table(_perf_results({
            "ticker": "QQQ.US", "name": "Nasdaq 100",
            "betas": {"Market": 1.05, "Momentum": 0.18, "Value": -0.10},
        }))
        mkt = self._row(rows, "Market")
        mom = self._row(rows, "Momentum")
        self.assertEqual(mkt["bench"], 1.05)
        self.assertEqual(mom["bench"], 0.18)
        self.assertAlmostEqual(mkt["delta"], round(mkt["beta"] - 1.05, 2))
        self.assertAlmostEqual(mom["delta"], round(mom["beta"] - 0.18, 2))
        # Оси, которых нет в бетах бенчмарка → 0.0 (не S&P-константа).
        self.assertEqual(self._row(rows, "Rates")["bench"], 0.0)

    def test_empty_betas_fall_back_to_constant(self):
        rows = self.tg._build_factor_betas_table(_perf_results({
            "ticker": "ODD.US", "name": "Odd", "betas": {},
        }))
        self.assertEqual(self._row(rows, "Market")["bench"], 1.0)


# ── Payload → Premium design data ────────────────────────────────────────────

def _minimal_results(bench_profile=None) -> dict:
    perf = pd.DataFrame({
        "Ticker":        ["AAPL", "AGG"],
        "Quantity":      [10, 20],
        "Purchase_Price": [150.0, 95.0],
        "Current_Price": [180.0, 97.0],
        "Current_Value": [1800.0, 1940.0],
        "Total_Cost":    [1500.0, 1900.0],
        "PnL":           [300.0, 40.0],
        "Return_Pct":    [0.2, 0.021],
    })
    results = {
        "performance_table": perf,
        "total_value": 3740.0,
        "portfolio_metrics": {
            "CVaR_95_Daily": -0.05, "Sharpe_Ratio": 1.1,
            "Sortino_Ratio": 1.3, "VaR_95_Daily": -0.02,
            "Max_Drawdown": -0.12, "Total_Volatility_Ann": 0.14,
            "Composite_Risk_Score": 55, "Max_Euler_Risk_Pct": 20.0,
            "CVaR_95_Bootstrap": {"point": -0.05, "lo95": -0.07, "hi95": -0.04},
        },
        "sector_exposure": {"Technology": 0.5, "Fixed_Income": 0.5},
        "benchmark_comparison": {},
        "asset_scores": {},
    }
    if bench_profile is not None:
        results["benchmark_factor_profile"] = bench_profile
    return results


_QQQ_PROFILE = {"ticker": "QQQ.US", "name": "Nasdaq 100",
                "betas": {"Market": 1.05, "Momentum": 0.18}}


class PayloadBenchmarkIdentityTest(unittest.TestCase):
    """benchmark_name/ticker в payload и benchmarkName в design-data."""

    def _payload(self, results, tier="deep"):
        from pdf_payload import build_payload
        return build_payload(results, tier=tier, ai_summary={
            "verdict": "v", "plain_summary": "s", "bullets": ["b"],
            "stock_picks": {}, "used_rag": False, "model_used": "test",
        })

    def test_nasdaq_profile_sets_dynamic_identity(self):
        payload = self._payload(_minimal_results(_QQQ_PROFILE))
        self.assertEqual(payload["benchmark_name"], "Nasdaq 100")
        self.assertEqual(payload["benchmark_ticker"], "QQQ.US")

    def test_fallback_keeps_snp_identity(self):
        """Фолбэк: подпись обязана совпадать с числами S&P-константы —
        «Nasdaq» над S&P-профилем (отклонённый Вариант C) недопустим."""
        payload = self._payload(_minimal_results())
        self.assertEqual(payload["benchmark_name"], "S&P 500")
        self.assertEqual(payload["benchmark_ticker"], "SPY.US")
        # Профиль без бет (graceful None) → тоже S&P-подпись.
        payload2 = self._payload(_minimal_results(
            {"ticker": "QQQ.US", "name": "Nasdaq 100", "betas": None}))
        self.assertEqual(payload2["benchmark_name"], "S&P 500")

    def test_design_data_carries_benchmark_name(self):
        from premium_payload import build_design_data
        payload = self._payload(_minimal_results(_QQQ_PROFILE))
        data = build_design_data(payload, tier="deep", user_id=1)
        self.assertEqual(data["benchmarkName"], "Nasdaq 100")
        self.assertEqual(data["benchmarkTicker"], "QQQ.US")

    def test_design_data_default_for_legacy_payload(self):
        from premium_payload import build_design_data
        data = build_design_data({}, tier="deep", user_id=1)
        self.assertEqual(data["benchmarkName"], "S&P 500")
        self.assertEqual(data["benchmarkTicker"], "SPY.US")


# ── Слой A — скомпилированные ассеты и шаблоны ───────────────────────────────

class CompiledAssetsTest(unittest.TestCase):
    """Тест-греп: в FactorTable/легенде нет захардкоженного «S&P 500»."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = (ROOT / "src" / "premium_assets" /
                      "deep-components.js").read_text(encoding="utf-8")
        cls.jsx = (ROOT / "design" / "premium_v2" / "deep" /
                   "deep-factors.jsx").read_text(encoding="utf-8")

    def test_bundle_reads_dynamic_benchmark_name(self):
        self.assertIn("benchmarkName", self.bundle)

    def test_bundle_has_no_hardcoded_th_or_legend_literal(self):
        # Старые компилированные литералы: заголовок столбца и легенда радара.
        self.assertNotIn('}, "S&P 500")', self.bundle)
        self.assertNotIn("Рынок (S&P 500)", self.bundle)
        self.assertNotIn("Почему у S&P 500 ненулевая только Market", self.bundle)

    def test_jsx_source_did_not_diverge_from_artifact(self):
        """Источник (.jsx) и артефакт (.js) обязаны меняться синхронно."""
        self.assertIn("benchmarkName", self.jsx)
        self.assertNotIn("Рынок (S&P 500)", self.jsx)
        design_bundle = (ROOT / "design" / "premium_v2" /
                         "deep-components.js").read_text(encoding="utf-8")
        self.assertEqual(design_bundle, self.bundle,
                         "design/ и src/premium_assets/ бандлы разошлись — "
                         "прогоните design/premium_v2/build.sh")

    def test_sample_data_carries_benchmark_keys(self):
        import json
        for p in (ROOT / "design" / "premium_v2" / "deep-data.sample.json",
                  ROOT / "src" / "premium_assets" / "deep-data.sample.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            self.assertIn("benchmarkName", d, p)
            self.assertIn("benchmarkTicker", d, p)

    def test_v3_fallback_template_is_dynamic_too(self):
        tpl = (ROOT / "src" / "templates" /
               "report_deep_v3.html").read_text(encoding="utf-8")
        self.assertIn("data.benchmark_name", tpl)


# ── ИИ-слой: имя бенчмарка + мандатный гард идей ─────────────────────────────

class AiMandateAwarenessTest(unittest.TestCase):

    def test_benchmark_profile_for_prompt_prefers_engine_profile(self):
        from ai_narrative import _benchmark_profile_for_prompt
        out = _benchmark_profile_for_prompt(
            {"benchmark_factor_profile": _QQQ_PROFILE})
        self.assertEqual(out["name"], "Nasdaq 100")
        self.assertEqual(out["betas"]["Market"], 1.05)

    def test_benchmark_profile_falls_back_to_ticker_name(self):
        from ai_narrative import _benchmark_profile_for_prompt
        out = _benchmark_profile_for_prompt(
            {"profile_benchmark_ticker": "QQQ.US"})
        self.assertEqual(out["name"], "Nasdaq 100")
        self.assertIsNone(out["betas"])
        self.assertIsNone(_benchmark_profile_for_prompt({}))

    def test_prompt_names_the_users_benchmark(self):
        from ai_narrative import _summarise_for_prompt, _user_prompt
        summary = _summarise_for_prompt(
            {"benchmark_factor_profile": _QQQ_PROFILE})
        for tier in ("base", "deep"):
            prompt = _user_prompt(summary, tier=tier)
            self.assertIn("Nasdaq 100", prompt)

    def test_prompt_carries_banned_asset_classes(self):
        from ai_narrative import _summarise_for_prompt, _user_prompt
        summary = _summarise_for_prompt({})
        prompt = _user_prompt(summary, tier="deep", user_mandate={
            "profile_name": "Консервативный",
            "limits_dict": {"Crypto": [0, 0], "Stocks_US": [0, 20]},
        })
        self.assertIn("ЗАПРЕЩЁННЫЕ", prompt)
        self.assertIn("Крипто", prompt)
        self.assertNotIn("Акции США: 0", prompt)  # ненулевой лимит — не забанен

    def test_mandate_guard_removes_banned_class_picks(self):
        from ai_narrative import _remove_mandate_banned_picks
        picks = {"boost_alpha": {"label": "x", "picks": [
            {"ticker": "BTC-USD", "name": "Bitcoin"},
            {"ticker": "NVDA", "name": "NVIDIA"},
        ]}}
        out = _remove_mandate_banned_picks(
            picks, {"limits_dict": {"Crypto": [0, 0]}})
        tickers = [p["ticker"] for p in out["boost_alpha"]["picks"]]
        self.assertEqual(tickers, ["NVDA"])

    def test_mandate_guard_noop_without_mandate(self):
        from ai_narrative import _remove_mandate_banned_picks
        picks = {"boost_alpha": {"picks": [{"ticker": "BTC-USD"}]}}
        self.assertEqual(_remove_mandate_banned_picks(picks, None), picks)

    def test_risk_mandate_reaches_the_prompt_summary(self):
        """§6.1-1: results["risk_mandate"] доезжает до ИИ-контекста."""
        from ai_narrative import _summarise_for_prompt
        summary = _summarise_for_prompt({"risk_mandate": "CONSERVATIVE"})
        self.assertEqual(summary["risk_mandate"], "CONSERVATIVE")


# ── Бот: /mandate меняет БД без биллинга ─────────────────────────────────────

class _FakeState:
    def __init__(self, data=None):
        self._data = dict(data or {})
        self._state = None

    async def get_data(self):
        return dict(self._data)

    async def update_data(self, **kw):
        self._data.update(kw)

    async def set_state(self, s):
        self._state = s

    async def get_state(self):
        return self._state

    async def clear(self):
        self._data = {}
        self._state = None


class _FakeMessage:
    def __init__(self):
        self.answers: list[str] = []
        self.edits: list[str] = []
        self.message_id = 100

    async def answer(self, text, **kw):
        self.answers.append(text)
        return SimpleNamespace(message_id=101)

    async def edit_text(self, text, **kw):
        self.edits.append(text)

    async def edit_reply_markup(self, **kw):
        pass


class _FakeCallback:
    """НЕ isinstance(CallbackQuery) → `_edit_or_answer` идёт по Message-ветке
    `target.answer(...)` — фейк записывает текст и отдаёт message_id."""

    def __init__(self, data, user_id=777, message=None):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.message = message or _FakeMessage()

    async def answer(self, *args, **kw):
        if args:                       # текстовый answer (не ack-у колбэка)
            self.message.answers.append(args[0])
        return SimpleNamespace(message_id=102)


_STORED_PROFILE = {
    "telegram_id": 777, "score": 14,
    "profile_name": "Умеренно-агрессивный",
    "target_volatility": 0.14, "target_te": 0.06,
    "selected_assets": ["Stocks_US", "GlobalETFs"],
    "limits_dict": {"Stocks_US": [30, 60], "GlobalETFs": [10, 40],
                    "Bonds": [10, 30], "Crypto": [0, 0]},
    "benchmark_ticker": "SPY.US", "mandate_approved": True,
}


class MandateEditNoBillingTest(unittest.TestCase):
    """§7-5: mandate:edit:* меняет БД без сброса баллов и БЕЗ deduct_tokens."""

    @classmethod
    def setUpClass(cls):
        cls.tg = _import_tg_bot()

    def setUp(self):
        if self.tg is None:
            self.skipTest("tg_bot import unavailable")
        self.calls = {"save_bench": [], "save_profile": [],
                      "approve": [], "deduct": []}
        self._orig = (self.tg.get_profile, self.tg.save_benchmark_ticker,
                      self.tg.save_profile, self.tg.approve_mandate,
                      self.tg.deduct_tokens)

        async def fake_get_profile(uid):
            return dict(_STORED_PROFILE)

        async def fake_save_bench(uid, ticker):
            self.calls["save_bench"].append((uid, ticker))

        async def fake_save_profile(**kw):
            self.calls["save_profile"].append(kw)

        async def fake_approve(uid):
            self.calls["approve"].append(uid)

        async def fake_deduct(*a, **kw):
            self.calls["deduct"].append((a, kw))

        self.tg.get_profile           = fake_get_profile
        self.tg.save_benchmark_ticker = fake_save_bench
        self.tg.save_profile          = fake_save_profile
        self.tg.approve_mandate       = fake_approve
        self.tg.deduct_tokens         = fake_deduct

    def tearDown(self):
        (self.tg.get_profile, self.tg.save_benchmark_ticker,
         self.tg.save_profile, self.tg.approve_mandate,
         self.tg.deduct_tokens) = self._orig

    def test_benchmark_edit_two_taps_no_requiz_no_billing(self):
        """Антидот первопричины: смена бенчмарка без анкеты и списаний."""
        state = _FakeState({"edit_mode": True, "benchmark_ticker": "QQQ.US",
                            "ob_message_id": 5})
        cb = _FakeCallback("ob:bench:confirm")
        asyncio.run(self.tg.cb_benchmark_confirm(cb, state))
        self.assertEqual(self.calls["save_bench"], [(777, "QQQ.US")])
        self.assertEqual(self.calls["save_profile"], [])   # баллы не тронуты
        self.assertEqual(self.calls["deduct"], [])         # биллинга нет
        blob = " ".join(cb.message.edits + cb.message.answers)
        self.assertIn("Nasdaq 100", blob)

    def test_universe_edit_keeps_score_and_benchmark(self):
        state = _FakeState({"edit_mode": True,
                            "universe": ["Stocks_US", "Bonds"],
                            "ob_message_id": 5})
        cb = _FakeCallback("ob:uni:confirm")
        asyncio.run(self.tg.cb_universe_confirm(cb, state))
        self.assertEqual(len(self.calls["save_profile"]), 1)
        kw = self.calls["save_profile"][0]
        self.assertEqual(kw["score"], 14)                       # не сброшен
        self.assertEqual(kw["benchmark_ticker"], "SPY.US")      # сохранён
        self.assertEqual(kw["selected_assets"], ["Stocks_US", "Bonds"])
        self.assertEqual(kw["limits_dict"]["GlobalETFs"], [0, 0])  # исключён
        self.assertNotEqual(kw["limits_dict"]["Stocks_US"], [0, 0])
        self.assertEqual(self.calls["approve"], [777])  # мандат жив
        self.assertEqual(self.calls["deduct"], [])

    def test_profile_edit_keeps_users_benchmark(self):
        state = _FakeState({"edit_mode": True, "ob_message_id": 5})
        cb = _FakeCallback("mandate:profile:17")
        asyncio.run(self.tg.cb_mandate_action(cb, state))
        self.assertEqual(len(self.calls["save_profile"]), 1)
        kw = self.calls["save_profile"][0]
        self.assertEqual(kw["profile_name"], "Агрессивный")
        self.assertEqual(kw["score"], 17)
        # Выбранный пользователем бенчмарк НЕ перетёрт дефолтом профиля (QQQ).
        self.assertEqual(kw["benchmark_ticker"], "SPY.US")
        self.assertEqual(self.calls["deduct"], [])

    def test_onboarding_flow_unchanged_without_edit_mode(self):
        """Без edit_mode confirm-обработчик идёт по прежнему онбордингу."""
        state = _FakeState({"benchmark_ticker": "QQQ.US",
                            "profile_data": {"name": "Умеренный",
                                             "target_vol": 0.10,
                                             "target_te": 0.04,
                                             "score": 10,
                                             "limits": {"Bonds": [30, 50]}},
                            "universe": ["Bonds"], "ob_message_id": 5})
        cb = _FakeCallback("ob:bench:confirm")
        asyncio.run(self.tg.cb_benchmark_confirm(cb, state))
        self.assertEqual(self.calls["save_bench"], [])  # ничего не сохранялось
        self.assertEqual(self.calls["deduct"], [])


# ── Сквозняк: QQQ-профиль → обе секции согласованы ───────────────────────────

class EndToEndConsistencyTest(unittest.TestCase):
    """§7-6: с QQQ и факторная секция, и подпись согласованы (Nasdaq 100)."""

    def test_factor_rows_and_identity_agree(self):
        tg = _import_tg_bot()
        if tg is None:
            self.skipTest("tg_bot import unavailable")
        from pdf_payload import build_payload
        results = _minimal_results(_QQQ_PROFILE)
        results["performance_table"]["Beta_Market"] = [1.3, 0.4]
        results["performance_table"]["Beta_Momentum"] = [0.5, 0.0]
        payload = build_payload(results, tier="deep", ai_summary={
            "verdict": "v", "plain_summary": "s", "bullets": ["b"],
            "stock_picks": {}, "used_rag": False, "model_used": "t"})
        payload["factor_betas"] = tg._build_factor_betas_table(results)
        self.assertEqual(payload["benchmark_name"], "Nasdaq 100")
        mkt = next(r for r in payload["factor_betas"] if r["axis"] == "Market")
        self.assertEqual(mkt["bench"], 1.05)  # реальная бета QQQ, не 1.0-константа

        from premium_payload import build_design_data
        data = build_design_data(payload, tier="deep", user_id=1)
        self.assertEqual(data["benchmarkName"], "Nasdaq 100")
        mkt_row = next(f for f in data["factors"] if f["name"] == "Market")
        self.assertEqual(mkt_row["mkt"], 1.05)


if __name__ == "__main__":
    unittest.main()
