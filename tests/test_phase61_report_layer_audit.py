"""`§−121` — аудит слоя отчёта: три находки, у каждой гейт.

1. Правило §−13 для ПРОЗЫ: плечо и долг видны только при кэше < 0.
2. VaR 95% считался и форматировался, но не показывался нигде.
3. `ai_action_impact` просил модель СЧИТАТЬ — поле выведено из работы.
4. Зона входа Buy терялась в Premium-плане (Jinja её показывал).

── 1 ──

Вёрстка правило соблюдала давно (баннер, бейдж «Долг», строка маржи — всё под
`is_leveraged`). Дыра была в тексте модели:

* поле `ai_leverage_warning` проходило насквозь — «пусто на книге без маржи»
  держалось на послушании модели, а Jinja-баннер проверял лишь непустоту;
* промпт отдавал модели `leverage_ratio 1.0 / валовая 94%` и на книге без
  маржи — повод написать «портфель без кредитного плеча»;
* запрета в промпте не было вовсе (урок `§−97` E-7: правила вне промпта
  модель не исполняет).

Тест подменяет модель на НЕПОСЛУШНУЮ: она заполняет предупреждение и пишет о
марже там, где долга нет, — и проверяет, что до читателя это не доходит.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd


def _results(levered: bool) -> dict:
    lev = ({"is_leveraged": True, "gross_exposure": 1.25, "long_weight": 1.25,
            "net_exposure": 1.0, "cash_weight": -0.25, "leverage_ratio": 1.25}
           if levered else
           {"is_leveraged": False, "gross_exposure": 0.94, "long_weight": 1.0,
            "net_exposure": 1.0, "cash_weight": 0.06, "leverage_ratio": 1.0})
    return {
        "portfolio_metrics": {"Composite_Risk_Score": 48, "Sharpe_Ratio": 0.7,
                              "CVaR_95_Daily": -0.02, "Max_Drawdown": -0.12},
        "total_value": 100_000.0,
        "regime": {"regime": "Expansion", "confidence": 0.5},
        "performance_table": pd.DataFrame([{"Ticker": "AAPL"}, {"Ticker": "CONL"}]),
        "leverage_metrics": lev,
    }


_MARGIN = "При просадке возможен Margin Call по маржинальному долгу."
_ETF = "CONL — плечевой ETF 2× на Coinbase, его просадки вдвое глубже."
_PROFIT = "Маржинальность Apple 30% поддерживает оценку."


def _misbehaving_model(**_kw):
    block = SimpleNamespace(type="tool_use", name="emit_report", input={
        "verdict": f"Портфель сбалансирован. {_MARGIN}",
        "bullets": [_MARGIN, _ETF, _PROFIT, "Портфель без кредитного плеча."],
        "ai_leverage_warning": "⚠ Заёмные средства: риск Margin Call.",
        "ai_risk_comment": f"Риск умеренный. {_MARGIN}",
    })
    usage = SimpleNamespace(input_tokens=1, output_tokens=1,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=[block], usage=usage, stop_reason="tool_use")


def _run(levered: bool) -> dict:
    from ai_narrative import generate_narrative
    fake = SimpleNamespace(messages=SimpleNamespace(create=_misbehaving_model))
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
         mock.patch("anthropic.Anthropic", return_value=fake):
        return generate_narrative(_results(levered), tier="base")


class ProseScrubTest(unittest.TestCase):

    def test_unlevered_book_loses_margin_sentences_only(self) -> None:
        from ai_narrative import strip_account_leverage
        text = f"{_PROFIT} {_MARGIN} {_ETF}"
        out = strip_account_leverage(text, leveraged=False)
        self.assertNotIn("Margin Call", out)
        self.assertIn("Маржинальность Apple", out)      # рентабельность — законно
        self.assertIn("плечевой ETF", out)              # свойство инструмента

    def test_instrument_leverage_wording_survives(self) -> None:
        """Плечевой ETF — свойство ИНСТРУМЕНТА, а не займ счёта: не режется."""
        from ai_narrative import strip_account_leverage
        for keep in ("Портфель с плечевыми ETF несёт повышенный риск.",
                     "CONL — ETF с кредитным плечом 2× на Coinbase.",
                     "Маржинальность Apple 30% поддерживает оценку."):
            with self.subTest(text=keep):
                self.assertEqual(strip_account_leverage(keep, False), keep)
        for drop in ("Кредитное плечо портфеля 1.2x.",
                     "Портфель с плечом усиливает убытки.",
                     "Возможен margin call."):
            with self.subTest(text=drop):
                self.assertEqual(strip_account_leverage(drop, False), "")

    def test_levered_book_is_untouched(self) -> None:
        from ai_narrative import strip_account_leverage
        text = f"{_PROFIT} {_MARGIN}"
        self.assertEqual(strip_account_leverage(text, leveraged=True), text)

    def test_negation_is_also_a_mention(self) -> None:
        """«Без плеча» — всё равно разговор о плече там, где его быть не должно."""
        from ai_narrative import strip_account_leverage
        self.assertEqual(
            strip_account_leverage("Портфель без кредитного плеча.", False), "")


class PromptFactsTest(unittest.TestCase):

    def test_unlevered_prompt_gets_no_leverage_numbers(self) -> None:
        from ai_narrative import _leverage_for_prompt
        self.assertEqual(_leverage_for_prompt(_results(False)["leverage_metrics"]),
                         {"is_leveraged": False})

    def test_levered_prompt_keeps_the_numbers(self) -> None:
        from ai_narrative import _leverage_for_prompt
        lv = _leverage_for_prompt(_results(True)["leverage_metrics"])
        self.assertTrue(lv["is_leveraged"])
        self.assertEqual(lv["margin_debt_pct"], 25.0)
        self.assertEqual(lv["leverage_ratio"], 1.25)


class MisbehavingModelTest(unittest.TestCase):

    def test_unlevered_report_carries_no_leverage_wording(self) -> None:
        out = _run(levered=False)
        # Прогон обязан пройти ВЕТКОЙ МОДЕЛИ: фолбэк о марже не пишет вовсе,
        # и тест на нём был бы пуст.
        self.assertNotEqual(out.get("model_used"), "fallback")
        self.assertEqual(out["ai_leverage_warning"], "")
        blob = " ".join([out.get("verdict", ""), out.get("ai_risk_comment", ""),
                         *out.get("bullets", [])])
        self.assertNotIn("Margin Call", blob)
        self.assertNotIn("кредитного плеча", blob)
        self.assertIn("плечевой ETF", blob)
        self.assertIn("Маржинальность Apple", blob)

    def test_levered_report_keeps_the_warning(self) -> None:
        out = _run(levered=True)
        self.assertNotEqual(out.get("model_used"), "fallback")
        self.assertIn("Margin Call", out["ai_leverage_warning"])


class TemplateGateTest(unittest.TestCase):
    """Jinja-баннер обязан спрашивать ДВИЖОК, а не только непустоту текста."""

    def test_banner_needs_the_engine_flag(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent / "src" / "templates"
        for name in ("report_basic_v3.html", "report_deep_v3.html"):
            src = (root / name).read_text(encoding="utf-8")
            with self.subTest(template=name):
                self.assertNotIn("{% if data.ai_leverage_warning %}", src)
                self.assertIn("data.ai_leverage_warning and data.leverage_metrics "
                              "and data.leverage_metrics.is_leveraged", src)



# ═══════════════ 2 · VaR 95% доезжает до глаз ═══════════════

def _golden_payload(tier: str) -> dict:
    import sys
    from pathlib import Path
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import golden_support as gs
    from pdf_payload import build_payload
    return build_payload(gs.run_analyze_all("base"), tier)


class VarIsShownTest(unittest.TestCase):

    def test_premium_cvar_card_carries_var(self) -> None:
        from premium_payload import build_design_data
        for tier in ("base", "deep"):
            p = _golden_payload(tier)
            with self.subTest(tier=tier):
                dd = build_design_data(p, tier)
                cvar = next(k for k in dd["kpis"] if k["key"] == "cvar")
                self.assertIn(f"VaR 95% {p['var_95_daily']}", cvar["sub"])
                self.assertIn(p["var_dollar"], cvar["sub"])
                self.assertIn(p["cvar_dollar"], cvar["sub"])

    def test_missing_dollar_leaves_no_dash_in_brackets(self) -> None:
        from premium_payload import _cvar_sub
        sub = _cvar_sub({"cvar_dollar": "—", "var_95_daily": "-0.7%",
                         "var_dollar": "—"})
        self.assertEqual(sub, "VaR 95% -0.7%")

    def test_jinja_fallback_shows_var_too(self) -> None:
        """Починка обязана накрывать фолбэк (`§−91` B-1)."""
        import importlib
        import html_renderer
        for tier in ("base", "deep"):
            p = _golden_payload(tier)
            with self.subTest(tier=tier), \
                 mock.patch.dict(os.environ, {"PREMIUM_REPORT_ENABLED": "false"}):
                hr = importlib.reload(html_renderer)
                html = hr.render_report_html(p, 148046720, tier=tier)
                self.assertIn(f"VaR 95% {p['var_95_daily']}", html)
        importlib.reload(html_renderer)

    def test_premium_dom_shows_var(self) -> None:
        """Не только в контракте — в DOM после монтирования React."""
        try:
            import layout_probe
        except Exception:                                  # pragma: no cover
            self.skipTest("layout_probe недоступен")
        if layout_probe.chromium_path() is None:
            self.skipTest("Chromium недоступен")
        try:
            import playwright  # noqa: F401
        except Exception:
            self.skipTest("playwright не установлен (инструмент разработки)")
        import tempfile
        from pathlib import Path
        import html_renderer
        for tier in ("base", "deep"):
            p = _golden_payload(tier)
            html = html_renderer.render_report_html(p, 148046720, tier=tier)
            with tempfile.TemporaryDirectory() as tmp, self.subTest(tier=tier):
                f = Path(tmp) / f"{tier}.html"
                f.write_text(html, encoding="utf-8")
                self.assertIn(f"var 95% {p['var_95_daily']}".casefold(),
                              layout_probe.dom_text(str(f)).casefold())


# ═══════════════ 3 · модель не считает эффект плана ═══════════════

class ActionImpactRetiredTest(unittest.TestCase):

    def test_prompt_does_not_ask_the_model_for_post_plan_numbers(self) -> None:
        from ai_narrative import _summarise_for_prompt, _user_prompt
        text = _user_prompt(_summarise_for_prompt(_results(False)), tier="deep")
        self.assertNotIn("ai_action_impact", text)
        self.assertNotIn("количественный прогноз", text)
        # Тот же прогон — промпт без маржи несёт запрет на разговор о плече.
        self.assertIn("ПЛЕЧО СЧЁТА", text)

    def test_tool_schema_no_longer_declares_the_field(self) -> None:
        from ai_narrative import REPORT_TOOL
        self.assertNotIn("ai_action_impact",
                         REPORT_TOOL["input_schema"]["properties"])

    def test_misbehaving_model_value_never_reaches_the_payload(self) -> None:
        out = _run(levered=False)
        self.assertEqual(out["ai_action_impact"], "")



# ═══════════════ 4 · зона входа Buy видна в Premium ═══════════════

class BuyEntryZoneShownTest(unittest.TestCase):

    def test_buy_row_carries_its_entry_zone(self) -> None:
        from premium_payload import build_design_data
        p = _golden_payload("deep")
        src = {r["ticker"]: r for r in p["action_plan"]}
        plan = {r["t"]: r for r in build_design_data(p, "deep")["actionPlan"]}
        buys = [t for t, r in src.items() if str(r.get("action", "")).startswith(("Buy", "Strong"))]
        self.assertTrue(buys, "в эталоне нет ни одной строки Buy — тест пуст")
        for t in buys:
            with self.subTest(ticker=t):
                self.assertEqual(plan[t]["entry"], src[t]["buy_zone"])
        for t, r in plan.items():
            if t not in buys:
                self.assertEqual(r["entry"], "", t)

    def test_entry_zone_reaches_the_dom(self) -> None:
        try:
            import layout_probe
            import playwright  # noqa: F401
        except Exception:
            self.skipTest("playwright не установлен (инструмент разработки)")
        if layout_probe.chromium_path() is None:
            self.skipTest("Chromium недоступен")
        import tempfile
        from pathlib import Path
        import html_renderer
        p = _golden_payload("deep")
        zone = next(r["buy_zone"] for r in p["action_plan"]
                    if str(r.get("action", "")).startswith(("Buy", "Strong")))
        html = html_renderer.render_report_html(p, 148046720, tier="deep")
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "deep.html"
            f.write_text(html, encoding="utf-8")
            self.assertIn(f"вход {zone}", layout_probe.dom_text(str(f)))


if __name__ == "__main__":
    unittest.main()
