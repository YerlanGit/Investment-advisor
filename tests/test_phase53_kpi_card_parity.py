"""Раунд §−97: KPI-карточка BASE и DEEP — одна форма и одна вёрстка.

Владелец прислал два скриншота одной и той же полосы показателей: в DEEP на
карточке есть график динамики и пояснение ИИ, в BASE — только число и подпись.
Разбор показал, что дело НЕ в нехватке данных:

  🔴 **Данные для полной карточки лежали в payload ОБОИХ тиров.**
     `kpi_sparklines` собирается в `tg_bot` для base и deep одинаково, а
     `ai_cvar_note` / `ai_sharpe_note` / `ai_mdd_note` запрашивает и BASE-промпт.
     Терялось это в МАППЕРЕ: BASE имел собственный `_kpi_obj`, который отдавал
     `{label, value, unit, delta, deltaText, frame, tone}` — без заметки, без
     ряда точек и без SVG. Тот же класс, что мёртвый `kpis` (`§−90` A-2) и
     непрокинутый `holdingsAI`: число считается и выбрасывается по дороге.

  🔴 **Вёрстка была скопирована, а не общая.** Две реализации одной карточки
     разошлись ровно так же, как четыре копии имени эмитента (`§−95`).
     Теперь карточка одна — `design/premium_v2/shared-kpi.jsx`, включаемый в
     оба бандла.

Гейты ниже держат три утверждения: маппер отдаёт ОДНУ форму на два тира,
поставляемые бандлы читают ОДИН компонент, а счётчик срезов спарклайна
считается по данным, а не пишется словом.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import premium_payload as pp                                    # noqa: E402
from finance.portfolio_series import compute_kpi_trend_series   # noqa: E402
from report_charts import build_kpi_sparklines                  # noqa: E402

_ASSETS = ROOT / "src" / "premium_assets"

#: Ключи, которые обязана нести КАЖДАЯ карточка обоих тиров.  Список полный, а
#: не «не меньше чем»: карточка, потерявшая `ai` или `pts`, обязана валить гейт,
#: а не проходить его как «расширенная».
CARD_KEYS = {"key", "name", "value", "status", "color", "sub", "ai", "pts", "svg"}


def _payload(**over) -> dict:
    """Минимальный payload — ровно те ключи, которые читает `_kpi`."""
    p = {
        "cvar": "-3.7%", "sharpe": "0.57", "max_drawdown": "-45.5%",
        "volatility": "21.8%", "sortino": "0.67",
        "cvar_dollar": "$570", "mdd_dollar": "$7,014",
        "kpi_status": {"cvar": "warn", "sharpe": "warn", "dd": "bad", "vol": "bad"},
        "kpi_sparklines": {
            "cvar_pts": [-2.8, -3.4, -3.7], "cvar_svg": "<svg/>",
            "sharpe_pts": [0.4, 0.5, 0.57], "sharpe_svg": "<svg/>",
            "mdd_pts": [-40.0, -43.2, -45.5], "mdd_svg": "<svg/>",
            "vol_pts": [19.4, 20.8, 21.8], "vol_svg": "<svg/>",
        },
        "ai_cvar_note": "заметка cvar", "ai_sharpe_note": "заметка sharpe",
        "ai_mdd_note": "заметка mdd", "ai_vol_note": "заметка vol",
        "assets": [], "sectors": [], "risk_waterfall": {},
    }
    p.update(over)
    return p


def _cards(tier: str, payload: dict) -> list:
    kpis = pp.build_design_data(payload, tier)["kpis"]
    assert isinstance(kpis, list), f"{tier}: kpis обязан быть списком карточек"
    return kpis


class KpiCardShapeIsSharedTest(unittest.TestCase):
    """Одна форма карточки на два тира — иначе вёрстка разойдётся снова."""

    def test_both_tiers_emit_the_same_card_keys(self) -> None:
        p = _payload()
        for tier in ("base", "deep"):
            for card in _cards(tier, p):
                with self.subTest(tier=tier, key=card.get("key")):
                    self.assertEqual(
                        set(card), CARD_KEYS,
                        f"{tier}/{card.get('key')}: форма карточки разошлась. "
                        "Лишние ключи никто не читает, недостающие — теряют "
                        "уже посчитанные данные (§−90 A-2).")

    def test_base_carries_the_ai_note_and_the_series(self) -> None:
        """🔴 Прямое воспроизведение дефекта со скриншота владельца."""
        for card in _cards("base", _payload()):
            with self.subTest(key=card["key"]):
                self.assertTrue(card["ai"], f"{card['key']}: заметка ИИ потеряна маппером")
                self.assertGreaterEqual(
                    len(card["pts"]), 2,
                    f"{card['key']}: ряд спарклайна потерян маппером")

    def test_base_shows_volatility_and_deep_does_not(self) -> None:
        """Состав полосы РАЗНЫЙ осознанно и потому пинится.

        BASE показывает волатильность четвёртой карточкой; в DEEP её роль
        играет отдельная риск-панель, и дублировать её на обложке незачем.
        """
        self.assertEqual([c["key"] for c in _cards("base", _payload())],
                         ["cvar", "sharpe", "dd", "vol"])
        self.assertEqual([c["key"] for c in _cards("deep", _payload())],
                         ["cvar", "sharpe", "dd"])

    def test_missing_note_is_empty_not_a_dash(self) -> None:
        """Проза без значения — ПУСТО, а не прочерк.

        `_txt` ставит `–` любому отсутствующему полю, и карточка печатала
        рамку «комментарий ИИ» с одним прочерком внутри. На no-API пути
        (в фолбэк-нарративе заметок KPI нет вовсе) так выглядела КАЖДАЯ
        карточка DEEP: рамка есть, содержания нет.
        """
        bare = _payload(ai_cvar_note="", ai_sharpe_note="", ai_mdd_note="", ai_vol_note="")
        for tier in ("base", "deep"):
            for card in _cards(tier, bare):
                with self.subTest(tier=tier, key=card["key"]):
                    self.assertEqual(card["ai"], "")

    def test_tone_is_a_verdict_not_a_literal(self) -> None:
        """Тон обеих полос читается из движка, а не из аргумента-литерала."""
        p = _payload(kpi_status={"cvar": "ok", "sharpe": "ok", "dd": "ok", "vol": "ok"})
        for tier in ("base", "deep"):
            for card in _cards(tier, p):
                with self.subTest(tier=tier, key=card["key"]):
                    self.assertEqual(card["status"], "ok")
                    self.assertEqual(card["color"], "#5d7c5c")

    def test_unknown_tone_never_reads_as_ok(self) -> None:
        p = _payload(kpi_status={})
        for tier in ("base", "deep"):
            for card in _cards(tier, p):
                with self.subTest(tier=tier, key=card["key"]):
                    self.assertNotEqual(
                        card["status"], "ok",
                        "неизвестный вход дал «всё хорошо» — ровно та ложь, "
                        "из-за которой правило и появилось (§−90 A-1)")


class VolatilitySeriesExistsTest(unittest.TestCase):
    """Четвёртая карточка BASE обязана иметь СВОЙ ряд, а не «нет истории»."""

    @staticmethod
    def _results(n: int = 300) -> dict:
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(7)
        return {"port_log_returns": pd.Series(rng.normal(0.0004, 0.011, n))}

    def test_trend_series_carries_vol(self) -> None:
        s = compute_kpi_trend_series(self._results())
        self.assertIsNotNone(s)
        self.assertIn("vol_pts", s)
        self.assertEqual(len(s["vol_pts"]), len(s["sharpe_pts"]),
                         "ряд волатильности обязан идти по ТЕМ ЖЕ срезам")
        self.assertTrue(all(v > 0 for v in s["vol_pts"]),
                        "годовая σ не может быть отрицательной")

    def test_vol_is_the_sharpe_denominator(self) -> None:
        """Одно окно — одно σ.

        Волатильность в карточке и знаменатель Sharpe — одна и та же величина;
        если развести их по двум формулам, отчёт получит два разных ответа на
        один вопрос.
        """
        r = self._results()
        s = compute_kpi_trend_series(r)
        import numpy as np
        import pandas as pd
        lr = pd.Series(r["port_log_returns"])
        if len(lr) > 252:
            lr = lr.iloc[-252:]
        n = len(lr)
        step = max(1, n // 12)
        idx = [step * (i + 1) - 1 for i in range(12) if step * (i + 1) - 1 < n]
        expect = []
        for end in idx:
            win = lr.iloc[max(0, end - 60):end + 1]
            if len(win) < 30:
                continue
            expect.append(float(win.std()) * (252 ** 0.5))
        self.assertEqual(len(expect), len(s["vol_pts"]))
        for a, b in zip(expect, s["vol_pts"]):
            self.assertAlmostEqual(a, b, places=12)

    def test_sparkline_builder_renders_the_vol_cell(self) -> None:
        out = build_kpi_sparklines(self._results())
        self.assertIsNotNone(out)
        self.assertIn("vol_svg", out)
        self.assertTrue(out["vol_svg"].startswith("<svg"))
        self.assertIn("vol_pts", out)


class BundlesShareOneCardTest(unittest.TestCase):
    """Гейт читает ПОСТАВЛЯЕМЫЙ артефакт, а не исходник (`§−90` A-2)."""

    @staticmethod
    def _js(tier: str) -> str:
        """Код бандла без комментариев — см. `tests/bundle_text.py`."""
        from bundle_text import code_only
        return code_only((_ASSETS / f"{tier}-components.js").read_text(encoding="utf-8"))

    def test_both_bundles_carry_the_same_card_markers(self) -> None:
        markers = ("Динамика · окно 60 дн.",
                   "число выше — по полному окну истории",
                   "_sparkNote")
        for tier in ("base", "deep"):
            js = self._js(tier)
            for m in markers:
                with self.subTest(tier=tier, marker=m):
                    self.assertIn(
                        m, js,
                        f"бандл {tier} потерял общий компонент карточки — "
                        "значит вёрстка снова разошлась по копиям")

    def test_slice_count_is_computed_not_written(self) -> None:
        """🔴 Подпись обещала «12 срезов», движок отдаёт 11.

        При окне 252 дня шаг равен 21, и первый срез не набирает минимальных
        30 наблюдений — он отбрасывается ВСЕГДА. Тот же класс, что счётчики
        позиций и стресс-сценариев (`§−90` A-8, `§−91` B-4).
        """
        for tier in ("base", "deep"):
            js = self._js(tier)
            with self.subTest(tier=tier):
                self.assertNotIn("12 срезов", js)
                self.assertNotIn("12 \\u0441\\u0440\\u0435\\u0437\\u043e\\u0432", js)
                self.assertRegex(
                    js, r"srezov|срезов за год",
                    "подпись про окно спарклайна исчезла из бандла целиком")

    def test_ai_box_is_guarded(self) -> None:
        """Пустая заметка не рисует пустую рамку."""
        for tier in ("base", "deep"):
            js = self._js(tier)
            with self.subTest(tier=tier):
                self.assertRegex(
                    js, r"const ai = \(k\.ai \|\| ''\)\.trim\(\)",
                    "карточка перестала отличать «нет заметки» от «есть заметка»")


class JinjaFallbackKeepsParityTest(unittest.TestCase):
    """Фолбэк — тот же отчёт, а не его эскиз (`§−91` B-1)."""

    @staticmethod
    def _tpl(name: str) -> str:
        return (ROOT / "src" / "templates" / name).read_text(encoding="utf-8")

    def test_base_fallback_shows_all_four_metrics(self) -> None:
        html = self._tpl("report_basic_v3.html")
        self.assertIn('<div class="kpi-name">Волатильность</div>', html,
                      "отключение Premium-флага убирало из BASE целую риск-метрику")
        self.assertIn("data.kpi_sparklines.vol_svg", html)
        self.assertIn("data.ai_vol_note", html)

    def test_no_literal_tone_fallback_reads_as_good(self) -> None:
        """Неизвестный тон в шаблоне обязан давать «внимание».

        `§−90` A-1 починил Premium и оставил в обоих Jinja-шаблонах
        `else 'good'` / `else 'normal'` / `else 'watch'` — то есть отсутствие
        вердикта продолжало окрашивать карточку по вкусу макета.
        """
        for name in ("report_basic_v3.html", "report_deep_v3.html"):
            html = self._tpl(name)
            with self.subTest(tpl=name):
                for literal in ("else 'good' }}", "else 'normal' }}", "else 'watch' }}"):
                    self.assertNotIn(literal, html)

    def test_stale_sparkline_rationale_is_gone(self) -> None:
        """Устаревшее обоснование опаснее отсутствующего (`§−64`, `§−91` B-2).

        Комментарий у КАЖДОЙ ячейки утверждал, что движок «ещё не отдаёт
        поштучные ряды» и ждал будущего хелпера `_build_kpi_sparkline_series`.
        Хелпер приехал и работает; текст остался и убеждал не проверять.
        """
        for name in ("report_basic_v3.html", "report_deep_v3.html"):
            html = self._tpl(name)
            with self.subTest(tpl=name):
                self.assertNotIn("_build_kpi_sparkline_series", html)
                self.assertNotIn("the engine doesn't yet", html)


class SampleMatchesTheContractTest(unittest.TestCase):
    """Образец — ФОЛБЭК РЕНДЕРА, а не картинка (`§−91` B-3)."""

    def test_base_sample_uses_the_card_contract(self) -> None:
        sample = json.loads((_ASSETS / "base-data.sample.json").read_text(encoding="utf-8"))
        kpis = sample["kpis"]
        self.assertIsInstance(kpis, list)
        self.assertEqual([c["key"] for c in kpis], ["cvar", "sharpe", "dd", "vol"])
        for card in kpis:
            with self.subTest(key=card["key"]):
                self.assertEqual(set(card), CARD_KEYS)
                self.assertTrue(card["ai"])
                self.assertGreaterEqual(len(card["pts"]), 2)

    def test_both_samples_agree_on_the_card_shape(self) -> None:
        base = json.loads((_ASSETS / "base-data.sample.json").read_text(encoding="utf-8"))
        deep = json.loads((_ASSETS / "deep-data.sample.json").read_text(encoding="utf-8"))
        self.assertEqual(set(base["kpis"][0]), set(deep["kpis"][0]))


if __name__ == "__main__":
    unittest.main()
