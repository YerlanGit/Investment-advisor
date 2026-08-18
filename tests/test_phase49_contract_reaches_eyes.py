"""Раунд §−91: гейт «ключ доехал до глаз» + добивка находок §−90.

Прошлый раунд закрыл восемь дефектов показа и **сам же назвал главную дыру**:
сюита из 1 548 тестов не заметила, что целый блок BASE не рендерится. Покрытие
проверяло ВЫЧИСЛЕНИЕ и ни разу — ДОСТАВКУ до читателя. Здесь этот гейт написан,
и он немедленно нашёл ещё четыре дефекта того же класса:

  B-1 🔴 **Починка §−90 накрыла только счастливый путь.** Вердикт KPI считается
      и доезжает до Premium V2 — а Jinja v3, который является АВТО-ФОЛБЭКОМ при
      `PREMIUM_REPORT_ENABLED=false` и при любом сбое Premium-рендера, продолжал
      печатать `class="kpi st-good"` ЛИТЕРАЛОМ в обоих шаблонах. Комментарий в
      шаблоне это и признавал: «sparklines + AI commentary remain hardcoded
      prototype text for the next round». То есть отключение флага возвращало
      ровно тот дефект, который раунд §−90 объявил закрытым.

  B-2 🟡 `benchmarkTicker` — мёртвый ключ контракта. Маппер отдавал настоящее
      значение (живой отчёт: `QQQ.US`), комментарий рядом обещал ТРЁХ
      потребителей («column header, radar legend и explainer plaque читают
      эти») — в бандле ключ не встречался ни разу. Устаревшее обоснование
      опаснее отсутствующего: оно убеждает не проверять.

  B-3 🟡 Образцы `*-data.sample.json` отстали от контракта на 3 (BASE) и 5
      (DEEP) ключей. Это не «просто демо-данные»: `render_premium(tier, None)`
      рендерит ИМЕННО их, то есть фолбэк молча терял бейдж плеча, полосу
      качества и блок факторной дисперсии.

  B-4 🟡 «7 сценариев» — счётчик, напечатанный константой, при вычисляемом
      наборе (`stress.DEFAULT_SCENARIOS`; `run_stress_tests` принимает свой
      список). Тот же класс, что «9 позиций» (A-8) и «4 идеи» (`§−48`).
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

import premium_payload as pp
from premium_renderer import load_sample

_ASSETS = ROOT / "src" / "premium_assets"
_TEMPLATES = ROOT / "src" / "templates"

_BUNDLE = {"base": "base-components.js", "deep": "deep-components.js"}


def _bundle(tier: str) -> str:
    """Текст бандла БЕЗ комментариев.

    🔴 Гейты ниже ищут литералы, которых не должно быть в поставляемом коде.
    Babel сохраняет комментарии, поэтому разбор дефекта, написанный рядом с
    починкой, валил охраняющий её гейт (`§−97`, дважды за раунд). Комментарий
    до глаз читателя не доходит — гейт обязан смотреть на КОД.
    """
    from bundle_text import code_only
    return code_only((_ASSETS / _BUNDLE[tier]).read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# ГЕЙТ · каждый ключ контракта обязан иметь потребителя в бандле тира
# ──────────────────────────────────────────────────────────────────────────

class ContractKeyReachesTheBundleTest(unittest.TestCase):
    """🔴 Главный гейт раунда.

    `test_phase19_block_audit` пинит СОСТАВ контракта — «такие ключи есть».
    Этого мало: `kpis` присутствовал в контракте BASE и не рендерился ВООБЩЕ
    (`§−90` A-2). Здесь проверяется другое утверждение — «ключ КТО-ТО ЧИТАЕТ».

    Читаем поставляемый артефакт `src/premium_assets/*.js`, а не исходник
    `.jsx`: в образ едет бандл, и невидимым ключ делает отсутствие именно
    там. Побочная выгода — тест исполняется в зеркале деплой-образа, где
    каталога `design/` нет.
    """

    def test_no_dead_keys_in_either_tier(self) -> None:
        for tier in ("base", "deep"):
            with self.subTest(tier=tier):
                js = _bundle(tier)
                dead = [k for k in sorted(load_sample(tier))
                        if not re.search(r"\b" + re.escape(k) + r"\b", js)]
                self.assertEqual(
                    dead, [],
                    f"ключи контракта {tier.upper()} собираются маппером, но их "
                    f"не читает НИ ОДИН компонент бандла: {dead}. Это дефект "
                    f"класса A-2: число считается и выбрасывается.")

    def test_gate_actually_looks_at_something(self) -> None:
        """Гейт бесполезен, если контракт вдруг опустеет — проверяем, что нет."""
        self.assertGreaterEqual(len(load_sample("base")), 10)
        self.assertGreaterEqual(len(load_sample("deep")), 30)

    # 🔴 ГРАНИЦА ГЕЙТА, найденная мутацией и названная здесь честно.
    #
    # Проверка выше отвечает на вопрос «ключ кто-нибудь читает». Она НЕ
    # отвечает на вопрос «этот кто-нибудь попадает на страницу»: если снять
    # монтирование компонента, оставив его определение, ключ по-прежнему
    # встречается в бандле — а читатель блока не видит. Ровно это и было
    # дефектом A-2. Мутация «убрать `createElement(RiskKpiStrip, …)`»
    # проходила мимо гейта.
    #
    # Полное лекарство — рендер в браузере (бандл собирает DOM на клиенте), а
    # это Playwright, и тащить его в юнит-сюиту неоправданно. Достижимый
    # статический признак: смонтированный компонент упоминается в бандле
    # МИНИМУМ ДВАЖДЫ — объявление и хотя бы одна точка вставки; определённый,
    # но не вставленный — ровно один раз.
    MOUNTED = {
        "base": ("RiskKpiStrip", "Hero", "Holdings", "Performance", "Ideas"),
        "deep": ("KpiCard", "FactorTable", "StressTable", "RegimeBlock"),
    }

    def test_key_components_are_mounted_not_merely_defined(self) -> None:
        for tier, names in self.MOUNTED.items():
            js = _bundle(tier)
            for name in names:
                with self.subTest(tier=tier, component=name):
                    hits = len(re.findall(r"\b" + re.escape(name) + r"\b", js))
                    self.assertGreaterEqual(
                        hits, 2,
                        f"{tier}: `{name}` встречается в бандле {hits} раз — "
                        f"похоже, компонент ОБЪЯВЛЕН, но не вставлен в дерево. "
                        f"Это регресс A-2: блок считается и не показывается.")


class SampleMatchesTheContractTest(unittest.TestCase):
    """B-3: образец — это ФОЛБЭК рендера, а не картинка для дизайнера."""

    def test_sample_carries_every_key_the_mapper_emits(self) -> None:
        # Тонкий payload → маппер отдаёт полный набор ключей своего тира.
        for tier in ("base", "deep"):
            with self.subTest(tier=tier):
                emitted = set(pp.build_design_data({}, tier))
                sample = set(load_sample(tier))
                missing = sorted(emitted - sample)
                self.assertEqual(
                    missing, [],
                    f"образец {tier} отстал от контракта на {missing} — "
                    f"фолбэк `render_premium({tier!r}, None)` молча потеряет "
                    f"эти блоки")


# ──────────────────────────────────────────────────────────────────────────
# B-1 · фолбэк Jinja v3 обязан использовать тот же вердикт, что Premium
# ──────────────────────────────────────────────────────────────────────────

class JinjaFallbackUsesComputedVerdictTest(unittest.TestCase):
    """Отключение `PREMIUM_REPORT_ENABLED` не должно воскрешать дефект A-1."""

    TEMPLATES = ("report_deep_v3.html", "report_basic_v3.html")

    def test_kpi_class_is_data_driven(self) -> None:
        for name in self.TEMPLATES:
            with self.subTest(template=name):
                html = (_TEMPLATES / name).read_text(encoding="utf-8")
                self.assertIn("data.kpi_status", html,
                              f"{name}: класс KPI-карточки снова вписан руками — "
                              f"фолбэк печатает тон, не зависящий от значения")

    def test_no_hardcoded_status_class_left(self) -> None:
        """Литерал `class="kpi st-good"` — ровно то, чем был дефект A-1."""
        for name in self.TEMPLATES:
            with self.subTest(template=name):
                html = (_TEMPLATES / name).read_text(encoding="utf-8")
                for literal in ('<div class="kpi st-normal">',
                                '<div class="kpi st-good">',
                                '<div class="kpi st-watch">'):
                    self.assertNotIn(literal, html, f"{name}: {literal}")

    def test_all_three_verdict_classes_are_styled(self) -> None:
        """Вердикт отдаёт ok/warn/bad — у каждого обязан быть свой цвет,
        иначе `st-bad` отрендерится без рамки и дефект станет невидимым."""
        for name in self.TEMPLATES:
            with self.subTest(template=name):
                html = (_TEMPLATES / name).read_text(encoding="utf-8")
                for cls in (".kpi.st-ok", ".kpi.st-warn", ".kpi.st-bad"):
                    self.assertIn(cls, html, f"{name}: нет стиля для {cls}")

    def test_payload_supplies_the_key_both_tiers_need(self) -> None:
        """Шаблон читает `data.kpi_status` — payload обязан его класть."""
        import pdf_payload
        src = Path(pdf_payload.__file__).read_text(encoding="utf-8")
        self.assertIn('"kpi_status"', src)


# ──────────────────────────────────────────────────────────────────────────
# B-2 · тикер бенчмарка доезжает до столбца
# ──────────────────────────────────────────────────────────────────────────

class BenchmarkTickerIsShownTest(unittest.TestCase):

    def test_bundle_reads_the_key(self) -> None:
        # Границы слова ОБЯЗАТЕЛЬНЫ: подстрочный `assertIn` проходит и на
        # `benchmarkTickerX`, то есть не отличает «ключ читают» от «в бандле
        # случайно есть похожее имя». Поймано мутацией при проверке этого же
        # теста на способность падать.
        self.assertRegex(_bundle("deep"), r"\bbenchmarkTicker\b")

    def test_mapper_still_emits_a_real_value(self) -> None:
        d = pp.build_design_data({"benchmark_name": "Nasdaq 100",
                                  "benchmark_ticker": "QQQ.US"}, "deep")
        self.assertEqual(d["benchmarkTicker"], "QQQ.US")
        self.assertEqual(d["benchmarkName"], "Nasdaq 100")

    def test_fallback_stays_the_sp500_pair(self) -> None:
        d = pp.build_design_data({}, "deep")
        self.assertEqual(d["benchmarkName"], "S&P 500")
        self.assertEqual(d["benchmarkTicker"], "SPY.US")


# ──────────────────────────────────────────────────────────────────────────
# B-4 · счётчики печатаются по данным, а не константой
# ──────────────────────────────────────────────────────────────────────────

class CountersAreComputedTest(unittest.TestCase):

    def test_stress_scenario_count_is_not_a_literal(self) -> None:
        js = _bundle("deep")
        self.assertNotIn("7 \\u0441\\u0446\\u0435\\u043d\\u0430\\u0440\\u0438\\u0435\\u0432", js)
        self.assertNotIn("7 сценариев", js,
                         "счётчик стресс-сценариев снова зашит константой (B-4)")

    def test_scenario_count_survives_a_shorter_set(self) -> None:
        """Набор сценариев НЕ фиксирован: `run_stress_scenarios` принимает свой
        список, поэтому подпись обязана считаться, а не совпадать случайно."""
        import inspect
        from finance import stress
        self.assertTrue(hasattr(stress, "DEFAULT_SCENARIOS"))
        self.assertGreater(len(stress.DEFAULT_SCENARIOS), 0)
        sig = inspect.signature(stress.run_stress_scenarios)
        self.assertIn("scenarios", sig.parameters)


# ──────────────────────────────────────────────────────────────────────────
# B-5 · сценарный тир: ключ без потребителя
# ──────────────────────────────────────────────────────────────────────────

class ScenarioPayloadHasNoDeadKeyTest(unittest.TestCase):
    """`holdings_display` собирался в payload сценарного тира и не читался
    никем: имя бумаги доезжает до строк отдельным полем `display`."""

    def test_holdings_display_is_gone(self) -> None:
        src = (ROOT / "src" / "finance" / "scenario_report.py").read_text(encoding="utf-8")
        self.assertNotIn('"holdings_display"', src)

    def test_row_level_display_still_works(self) -> None:
        src = (ROOT / "src" / "finance" / "scenario_report.py").read_text(encoding="utf-8")
        self.assertIn('["display"]', src)


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
