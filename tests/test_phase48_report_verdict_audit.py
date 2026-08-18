"""Аудит живых отчётов 12.08 (BASE + DEEP, источник Freedom API) — раунд §−90.

Восемь находок, и общий у них ровно один класс: **отчёт уверенно печатал то,
чего не считал**. Числа движка при этом верны — все дефекты жили в слое показа
и в промпте.

  A-1 🔴 Статус KPI-карточек DEEP приезжал ЛИТЕРАЛОМ дизайн-макета:
      `"good"` у Sharpe, `"normal"` у CVaR, `"watch"` у просадки — независимо
      от значения. Живой отчёт: Sharpe 0.56 нёс золотой бейдж «good», а
      ИИ-заметка на ТОЙ ЖЕ карточке говорила «отдача на единицу риска слабая».
      Бейдж спорил с текстом в двух сантиметрах от себя. Тот же класс, что
      литерал «8%» (`§−48`) и mock-дата (`R-2`).

  A-2 🔴 BASE не рендерил `kpis` ВООБЩЕ. Ключ с CVaR/Sharpe/просадкой/
      волатильностью собирался маппером с самого начала, но в бандле
      `base-components.js` слова `kpis` не было ни разу — как и
      `verdict.expReturn`/`expSharpe`. Платный тир не показывал свои главные
      риск-метрики.

  A-3 🔴 Валютный слой CoVe печатал «конверсия не требуется (все активы в
      USD)» со статусом ok, когда записей о конверсии не было. Но пустой
      список — ДВА разных факта: «конвертировать нечего» и «курса не дали».
      Живой отчёт был вторым случаем: книга держала строку в KZT, чекер C-11
      рядом сообщал «Нет курса для валют: KZT». Отсутствие доказательства
      печаталось как доказательство отсутствия.

  A-4 🔴 Тон драйверов режима считался по СВЕЖЕСТИ ряда, а не по смыслу
      движения: «Real GDP growth ▼ −2.13 пп за 3кв» светился зелёным, потому
      что ряд свежий, — а пунктом ниже стоял ✗ «ВВП замедляется — против фазы
      роста». Зелёными были ТРИ драйвера из шести при ухудшающемся тренде.

  A-5 🔴 Промпт `ai_effect_comment` требовал назвать Sharpe даже когда движок
      дельту ПОГАСИЛ (er < rf). Модель исполняла: «отдача на риск улучшается»
      рядом с карточкой «дельта не показана — смотрите на волатильность,
      CVaR и просадку». Промпт сам провоцировал ложь.

  A-6 `it_share` считалась от NAV, а панель секторов и текст ИИ — от
      инвестированного капитала. На маржинальной книге одна величина
      печаталась двумя числами: «Доля IT 57.0%» и «Tech-комплекс 55.9%».

  A-7 Нулевое количество в плане («сделка меньше одной акции») выглядело как
      честное «количество не определено» — тот же прочерк, но БЕЗ причины,
      тогда как у всех прочих прочерков причина была.

  A-8 Счётчик секции состава был литералом «9 позиций» — печатался на любой
      книге (в живом отчёте позиций 17).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import premium_payload as pp
from finance import data_lineage as dl
from finance.regime import MACRO_DRIVER_DIRECTION, macro_trend_stance
from finance.scoring import kpi_status
from finance.simulate import _it_share

_ASSETS = Path(__file__).resolve().parents[1] / "src" / "premium_assets"


# ──────────────────────────────────────────────────────────────────────────
# A-1 · вердикт KPI считается по значению
# ──────────────────────────────────────────────────────────────────────────

class KpiStatusTest(unittest.TestCase):
    """Пороги якорены на матрицу мандата, а не выдуманы в шаблоне."""

    def test_sharpe_ladder(self) -> None:
        self.assertEqual(kpi_status("sharpe", 1.4), "ok")
        self.assertEqual(kpi_status("sharpe", 0.56), "warn")   # живой отчёт
        self.assertEqual(kpi_status("sharpe", 0.2), "bad")
        self.assertEqual(kpi_status("sharpe", -1.0), "bad")

    def test_cvar_scales_with_mandate(self) -> None:
        """Один и тот же хвост судится РАЗНО у разных мандатов — как и гаудж."""
        tail = -0.035
        self.assertEqual(kpi_status("cvar", tail, mandate="CONSERVATIVE"), "bad")
        self.assertEqual(kpi_status("cvar", tail, mandate="MODERATE"), "warn")
        self.assertEqual(kpi_status("cvar", tail, mandate="AGGRESSIVE"), "ok")

    def test_drawdown_uses_gauge_anchors(self) -> None:
        self.assertEqual(kpi_status("dd", -0.15), "ok")
        self.assertEqual(kpi_status("dd", -0.30), "warn")
        self.assertEqual(kpi_status("dd", -0.403), "bad")      # живой отчёт

    def test_vol_judged_against_mandate_target(self) -> None:
        self.assertEqual(kpi_status("vol", 0.13, target_vol=0.14), "ok")
        self.assertEqual(kpi_status("vol", 0.16, target_vol=0.14), "warn")
        self.assertEqual(kpi_status("vol", 0.205, target_vol=0.14), "bad")

    def test_unknown_input_never_claims_ok(self) -> None:
        """Молчаливое «всё хорошо» на мусоре — ровно та ложь, из-за которой
        правило и появилось. Неизвестность обязана читаться как «внимание»."""
        for bad_input in (float("nan"), None, "—", float("inf")):
            self.assertEqual(kpi_status("sharpe", bad_input), "warn")
        self.assertEqual(kpi_status("неизвестная_метрика", 1.0), "warn")


class KpiToneReachesDesignDataTest(unittest.TestCase):
    """Посчитанный тон обязан ДОЕХАТЬ до обоих тиров, а не остаться в payload."""

    PAYLOAD = {
        "cvar": "-3.4", "sharpe": "0.56", "max_drawdown": "-40.3",
        "volatility": "20.5", "sortino": "0.68",
        "kpi_status": {"cvar": "warn", "sharpe": "warn", "dd": "bad", "vol": "bad"},
    }

    def test_deep_card_takes_computed_tone_not_mock_literal(self) -> None:
        kpis = {k["key"]: k for k in pp.build_design_data(self.PAYLOAD, "deep")["kpis"]}
        self.assertEqual(kpis["sharpe"]["status"], "warn")
        self.assertEqual(kpis["dd"]["status"], "bad")
        # Живой дефект: Sharpe 0.56 показывался как «good».
        self.assertNotEqual(kpis["sharpe"]["status"], "good")
        # Цвет обязан следовать за тоном, иначе бейдж и рамка разойдутся.
        self.assertEqual(kpis["dd"]["color"], "#c47358")

    def test_base_carries_tone_too(self) -> None:
        # `§−97`: BASE отдаёт карточки в ТОЙ ЖЕ форме, что DEEP (`status`, а не
        # собственный `tone`) — одна форма на два тира, потому что двух копий
        # карточки не стало.  Утверждение теста прежнее: тон ДОЕХАЛ.
        kpis = {k["key"]: k for k in pp.build_design_data(self.PAYLOAD, "base")["kpis"]}
        self.assertEqual(kpis["sharpe"]["status"], "warn")
        self.assertEqual(kpis["vol"]["status"], "bad")

    def test_payload_without_status_does_not_invent_ok(self) -> None:
        """Старый payload не должен получить зелёный «всё хорошо» задаром."""
        thin = {k: v for k, v in self.PAYLOAD.items() if k != "kpi_status"}
        base = pp.build_design_data(thin, "base")["kpis"]
        self.assertNotIn("ok", {c["status"] for c in base})


# ──────────────────────────────────────────────────────────────────────────
# A-2 · BASE обязан ПОКАЗЫВАТЬ свои риск-метрики
# ──────────────────────────────────────────────────────────────────────────

class BaseRendersItsKpisTest(unittest.TestCase):
    """Регресс-гард против класса «ключ есть в payload, но его никто не рендерит».

    Читаем ПОСТАВЛЯЕМЫЙ артефакт (`src/premium_assets/`), а не исходник
    `design/*.jsx`: именно бандл едет в образ, и именно его отсутствие делало
    метрики невидимыми. Заодно тест исполняется в зеркале деплой-образа, где
    каталога `design/` нет.
    """

    def test_bundle_consumes_the_kpis_key(self) -> None:
        bundle = (_ASSETS / "base-components.js").read_text(encoding="utf-8")
        self.assertIn("kpis", bundle,
                      "BASE-бандл снова не читает `kpis` — риск-метрики "
                      "считаются и не показываются (регресс A-2)")
        self.assertIn("RiskKpiStrip", bundle)

    def test_bundle_shows_forward_estimates(self) -> None:
        bundle = (_ASSETS / "base-components.js").read_text(encoding="utf-8")
        for key in ("expReturn", "expSharpe"):
            self.assertIn(key, bundle,
                          f"`verdict.{key}` снова не рендерится в BASE")

    def test_mapper_still_emits_all_four_metrics(self) -> None:
        kpis = pp.build_design_data({"cvar": "-3.4", "sharpe": "0.56",
                                     "max_drawdown": "-40.3",
                                     "volatility": "20.5"}, "base")["kpis"]
        self.assertEqual({c["key"] for c in kpis}, {"cvar", "sharpe", "dd", "vol"})


class HoldingsCounterIsComputedTest(unittest.TestCase):
    """A-8: счётчик состава — из данных, а не литерал макета."""

    def test_no_hardcoded_nine_positions(self) -> None:
        bundle = (_ASSETS / "base-components.js").read_text(encoding="utf-8")
        self.assertNotIn("9 \\u043f\\u043e\\u0437\\u0438\\u0446\\u0438\\u0439",
                         bundle)
        self.assertNotIn("9 позиций", bundle,
                         "счётчик состава снова зашит литералом (регресс A-8)")


# ──────────────────────────────────────────────────────────────────────────
# A-3 · валютный слой не выдаёт «нет курса» за «конверсия не нужна»
# ──────────────────────────────────────────────────────────────────────────

class FxLineageHonestyTest(unittest.TestCase):

    @staticmethod
    def _results(findings):
        return {
            "portfolio_metrics": {"reporting_currency": "USD",
                                  "risk_free_rate_annual": 0.045,
                                  "risk_free_rate_source": "default[USD]=0.045",
                                  "fx_conversion": []},
            "data_quality": {"profile": "moderate", "findings": findings},
        }

    def test_missing_rate_is_not_reported_as_nothing_to_convert(self) -> None:
        row = dl._fx_status(self._results([
            {"id": "C-11", "level": "degrade",
             "message": "Нет курса для валют: KZT.", "ticker": None},
        ]))
        self.assertEqual(row["status"], "warn")
        self.assertNotIn("не требуется", row["method"])
        self.assertIn("KZT", row["note"])

    def test_genuinely_single_currency_book_stays_green(self) -> None:
        row = dl._fx_status(self._results([]))
        self.assertEqual(row["status"], "ok")
        self.assertIn("не требуется", row["method"])

    def test_other_findings_do_not_trip_the_warning(self) -> None:
        """Гард обязан реагировать на C-11, а не на любое замечание подряд."""
        row = dl._fx_status(self._results([
            {"id": "C-13", "level": "degrade", "message": "Наблюдений мало."},
        ]))
        self.assertEqual(row["status"], "ok")


# ──────────────────────────────────────────────────────────────────────────
# A-4 · тон драйвера — по смыслу движения, а не по свежести ряда
# ──────────────────────────────────────────────────────────────────────────

class MacroDriverStanceTest(unittest.TestCase):

    def test_signs_match_the_engines_own_convention(self) -> None:
        """Знаки взяты из `_macro_nudges`, а не назначены заново.

        🔴 §−92: имя ряда обязано быть НАСТОЯЩИМ. Прежняя редакция этого теста
        сверяла таблицу сама с собой и пропустила выдуманный ключ `hy_oas`
        (каталог зовёт ряд `hy_credit_spread`) — драйвер кредитного спреда
        молча оставался нейтральным. Привязка к каталогу — в `test_phase50`.
        """
        self.assertEqual(MACRO_DRIVER_DIRECTION["gdp_growth"], +1)
        self.assertEqual(MACRO_DRIVER_DIRECTION["unemployment"], -1)
        self.assertEqual(MACRO_DRIVER_DIRECTION["breakeven_inflation"], -1)
        self.assertEqual(MACRO_DRIVER_DIRECTION["hy_credit_spread"], -1)
        self.assertEqual(MACRO_DRIVER_DIRECTION["vix"], -1)
        self.assertEqual(MACRO_DRIVER_DIRECTION["yield_curve_10y2y"], +1)

    def test_live_report_case_gdp_slowing_is_negative(self) -> None:
        """Именно эта строка светилась зелёным в отчёте 12.08."""
        self.assertEqual(macro_trend_stance("gdp_growth", -2.13), "neg")
        self.assertEqual(macro_trend_stance("gdp_growth", +1.0), "pos")

    def test_widening_spreads_and_rising_fear_are_negative(self) -> None:
        self.assertEqual(macro_trend_stance("hy_credit_spread", +6.30), "neg")
        self.assertEqual(macro_trend_stance("hy_credit_spread", -0.37), "pos")
        self.assertEqual(macro_trend_stance("vix", -1.12), "pos")
        self.assertEqual(macro_trend_stance("breakeven_inflation", +0.02), "neg")
        self.assertEqual(macro_trend_stance("unemployment", -0.21), "pos")

    def test_unknown_series_or_no_trend_stays_flat(self) -> None:
        self.assertEqual(macro_trend_stance("нечто_неизвестное", 5.0), "flat")
        self.assertEqual(macro_trend_stance("vix", None), "flat")
        self.assertEqual(macro_trend_stance("vix", 0.0), "flat")

    def test_premium_mapper_separates_freshness_from_direction(self) -> None:
        payload = {"macro_drivers": {"series": [{
            "key": "gdp_growth", "name": "Real GDP growth (SAAR)",
            "value": "1.50%", "status": "ok", "trend_label": "▼ -2.13 пп за 3кв",
            "trend_stance": "neg",
        }]}}
        drv = pp.build_design_data(payload, "deep")["regime"]["drivers"][0]
        self.assertEqual(drv["tone"], "pos")        # ряд свежий
        self.assertEqual(drv["trendTone"], "neg")   # но движение против режима

    def test_deep_bundle_paints_value_by_direction(self) -> None:
        bundle = (_ASSETS / "deep-components.js").read_text(encoding="utf-8")
        self.assertIn("trendTone", bundle,
                      "панель драйверов снова красит значение свежестью "
                      "(регресс A-4)")


# ──────────────────────────────────────────────────────────────────────────
# A-5 · промпт не требует Sharpe, когда движок его погасил
# ──────────────────────────────────────────────────────────────────────────

class SharpeSuppressionReachesPromptTest(unittest.TestCase):

    def test_fact_is_passed_to_the_model(self) -> None:
        from ai_narrative import _summarise_for_prompt
        summary = _summarise_for_prompt({
            "expected_effect": {"sharpe_note": "Ожидаемая доходность ниже "
                                               "безрисковой ставки (премия -2.2 пп)"},
            "portfolio_metrics": {},
        })
        self.assertTrue(summary["sharpe_delta_suppressed"])
        self.assertIn("безрисковой", summary["sharpe_suppressed_reason"])

    def test_absent_note_means_not_suppressed(self) -> None:
        from ai_narrative import _summarise_for_prompt
        summary = _summarise_for_prompt({"expected_effect": {}, "portfolio_metrics": {}})
        self.assertFalse(summary["sharpe_delta_suppressed"])

    def test_prompt_forbids_the_claim(self) -> None:
        """Без явного запрета модель добросовестно исполняла «назови Sharpe»."""
        import ai_narrative
        src = Path(ai_narrative.__file__).read_text(encoding="utf-8")
        self.assertIn("sharpe_delta_suppressed=true", src)
        self.assertIn("НЕ ПИШИ ВООБЩЕ", src)


# ──────────────────────────────────────────────────────────────────────────
# A-6 · доля IT считается на том же базисе, что панель секторов
# ──────────────────────────────────────────────────────────────────────────

class ItShareBasisTest(unittest.TestCase):

    def test_leveraged_book_matches_share_of_invested_capital(self) -> None:
        """Книга с маржой: длинные 101.86%, кэш −1.86%. Панель секторов и ИИ
        считают от инвестированного капитала — доля IT обязана делать то же."""
        tickers = ["MSFT", "NVDA", "TLT", "CASH"]
        weights = [0.40, 0.20, 0.4186, -0.0186]
        sectors = {"MSFT": "Technology", "NVDA": "Semiconductors",
                   "TLT": "Bonds", "CASH": ""}
        share = _it_share(tickers, weights, sectors)
        self.assertAlmostEqual(share, 0.60 / 1.0186, places=6)
        # Наивная доля от NAV дала бы 0.60 — ровно то расхождение (57.0 vs 55.9),
        # которое отчёт показывал двумя числами.
        self.assertLess(share, 0.60)

    def test_unlevered_book_is_unchanged(self) -> None:
        """На книге без плеча знаменатель равен 1.0 — число прежнее."""
        share = _it_share(["MSFT", "TLT"], [0.6, 0.4],
                          {"MSFT": "Technology", "TLT": "Bonds"})
        self.assertAlmostEqual(share, 0.6, places=9)

    def test_empty_book_does_not_divide_by_zero(self) -> None:
        self.assertEqual(_it_share([], [], {}), 0.0)


# ──────────────────────────────────────────────────────────────────────────
# A-7 · нулевое количество объясняется, а не молчит
# ──────────────────────────────────────────────────────────────────────────

class SubShareTradeIsExplainedTest(unittest.TestCase):

    def test_reason_names_the_cause(self) -> None:
        import finance.action_plan as ap
        src = Path(ap.__file__).read_text(encoding="utf-8")
        self.assertIn("сделка меньше одной акции", src)

    def test_every_qtyless_row_carries_a_reason(self) -> None:
        """Смысл правки: прочерк без объяснения читается как ошибка движка."""
        import finance.action_plan as ap
        src = Path(ap.__file__).read_text(encoding="utf-8")
        # Обе ветки прочерка обязаны дописывать причину.
        self.assertIn("оптимизатор предлагает обратное", src)
        self.assertIn("количество не указываем", src)


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
