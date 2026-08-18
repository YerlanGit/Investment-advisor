"""Раунд §−97: R-5 (мобильный замер в CI) и R-7 (ключ доехал ДО ГЛАЗ).

Обе задачи годами стояли на доске как «нужен рендер в браузере», и обе здесь
закрываются одним стендом (`tests/layout_probe.py`).

  **R-5.** Утверждение «0 px переполнения» устаревало молча и было опровергнуто
  замером (`§−90` §3). Без гейта оно устареет снова. Здесь замер гоняется на
  320/360/390/414 по ОБОИМ тирам.

  🔴 **Прежняя метрика не могла дать ненулевой ответ.** `custom.css` ставит
  `html, body { overflow-x: hidden }` на телефонных ширинах, поэтому
  `document.scrollWidth` не превышает экран НИКОГДА. Контрольный опыт
  2026-08-18: заведомо широкий блок, вставленный в страницу, старым детектором
  не обнаруживался. Новый детектор считает предков только до `body` — и сразу
  нашёл реальное переполнение в DEEP на 320 px (бейдж «Hotspot» уезжал на 4 px
  из-за `min-width: auto` у `fr`-трека).

  **R-7.** Гейт `§−91` ловит «ключ никто не читает», но компонент, ОБЪЯВЛЕННЫЙ
  и не вставленный в дерево, проходит мимо: смягчение «упоминается ≥ 2 раз»
  это признавало прямо. Полное лекарство названо там же — рендер в браузере.
  Здесь он и сделан: значение ключа ищется в ВИДИМОМ ТЕКСТЕ страницы.

Тесты пропускаются, когда нет playwright/Chromium: они инструменты разработки,
в поставляемый образ не входят, а деплой-гейт гоняет всю сюиту (IB-F1).
"""

from __future__ import annotations

import atexit
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import layout_probe                                       # noqa: E402


def _requires_browser(case: unittest.TestCase) -> None:
    try:
        import playwright  # noqa: F401
    except ImportError:
        case.skipTest("playwright не установлен (инструмент разработки)")
    if layout_probe.chromium_path() is None:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as pw:
                pw.chromium.executable_path
        except Exception:
            case.skipTest("Chromium недоступен")


class _RenderedReports:
    """Оба тира рендерятся ОДИН раз на МОДУЛЬ — рендер дорогой.

    Каталог тоже модульный, а не классовый: кэш путей переживал `tearDownClass`
    первого класса, а вместе с ним удалялся и файл — второй класс получал
    `FileNotFoundError`. Время жизни кэша обязано совпадать со временем жизни
    того, что он кэширует.
    """

    _cache: dict[str, str] = {}
    _dir: "tempfile.TemporaryDirectory | None" = None

    @classmethod
    def dir(cls) -> Path:
        if cls._dir is None:
            cls._dir = tempfile.TemporaryDirectory(prefix="ramp-layout-")
            atexit.register(cls._dir.cleanup)
        return Path(cls._dir.name)

    @classmethod
    def path(cls, tier: str) -> str:
        if tier not in cls._cache:
            import html_renderer
            html = html_renderer.render_report_html(None, 148046720, tier=tier)
            out = cls.dir() / f"{tier}.html"
            out.write_text(html, encoding="utf-8")
            cls._cache[tier] = str(out)
        return cls._cache[tier]


class MobileLayoutIsMeasuredTest(unittest.TestCase):
    """R-5 · вёрстка чиста на телефонных ширинах — по ЗАМЕРУ, а не на глаз."""

    def test_no_content_runs_off_screen(self) -> None:
        _requires_browser(self)
        for tier in ("base", "deep"):
            path = _RenderedReports.path(tier)
            for width, res in layout_probe.measure(path).items():
                with self.subTest(tier=tier, width=width):
                    self.assertGreater(
                        res["nodes"], 200,
                        "дерево не смонтировалось — «ноль переполнений» "
                        "здесь означал бы пустую страницу")
                    self.assertEqual(
                        res["offenders"], [],
                        f"{tier} на {width}px: содержимое уезжает за экран "
                        f"{[o['cls'] or o['tag'] for o in res['offenders']]}")

    def test_the_detector_itself_catches_a_known_overflow(self) -> None:
        """🔴 Инструмент проверки сам обязан быть проверен.

        Прежний детектор мерил `scrollWidth`, обнулённый страничным
        `overflow-x: hidden`, и молчал даже на заведомо широком блоке. Без
        этого контрольного опыта «0 переполнений» не значит ничего.
        """
        _requires_browser(self)
        src = Path(_RenderedReports.path("base")).read_text(encoding="utf-8")
        mutant = _RenderedReports.dir() / "mutant.html"
        mutant.write_text(
            src.replace("</body>",
                        '<div style="width:900px;height:20px">ЗОНД</div></body>', 1),
            encoding="utf-8")
        res = layout_probe.measure(str(mutant), widths=(320,))[320]
        self.assertTrue(
            any("ЗОНД" in o["text"] for o in res["offenders"]),
            "детектор не увидел заведомое переполнение — значит его «ноль» "
            "на настоящем отчёте ничего не доказывает")


class ContractKeyReachesTheEyesTest(unittest.TestCase):
    """R-7 · значение ключа обязано оказаться в ВИДИМОМ тексте страницы.

    Гейт `§−91` смотрит на текст бандла и потому не отличает объявленный
    компонент от смонтированного. Здесь проверка идёт по DOM после рендера —
    то самое «полное лекарство», которое `§−91` назвал и отложил.
    """

    #: Что обязано быть ВИДНО в каждом тире. Не «ключ есть в контракте», а
    #: «читатель это прочитает»: подпись окна спарклайна, вердикт, полоса KPI.
    _MUST_SHOW = {
        "base": ("Динамика", "число выше — по полному окну истории",
                 "CVaR 95%", "Sharpe Ratio", "Max Drawdown", "Волатильность"),
        "deep": ("Динамика", "число выше — по полному окну истории",
                 "CVaR 95%", "Sharpe Ratio", "Max Drawdown"),
    }

    def test_declared_blocks_are_actually_mounted(self) -> None:
        _requires_browser(self)
        for tier, needles in self._MUST_SHOW.items():
            # 🔴 Сравнение РЕГИСТРОНЕЗАВИСИМОЕ: `innerText` отдаёт текст таким,
            # каким его ВИДИТ читатель, а подписи карточек идут через
            # `text-transform: uppercase`. Поиск «Динамика» в «ДИНАМИКА · ОКНО
            # 60 ДН.» не находил ничего — гейт падал бы на исправном отчёте.
            text = layout_probe.dom_text(_RenderedReports.path(tier)).casefold()
            for needle in needles:
                with self.subTest(tier=tier, needle=needle):
                    self.assertIn(
                        needle.casefold(), text,
                        f"{tier}: блок объявлен, но в дерево не вставлен — "
                        "ровно та дыра, которую текстовый гейт не видит")

    def test_kpi_values_from_the_payload_are_visible(self) -> None:
        """Значения KPI доезжают до глаз, а не только до контракта."""
        _requires_browser(self)
        import premium_renderer
        for tier in ("base", "deep"):
            sample = premium_renderer.load_sample(tier)
            text = layout_probe.dom_text(_RenderedReports.path(tier)).casefold()
            for card in sample["kpis"]:
                with self.subTest(tier=tier, key=card["key"]):
                    self.assertIn(
                        str(card["name"]).casefold(), text,
                        f"{tier}/{card['key']}: карточка не отрисована")

    def test_the_dom_gate_itself_sees_a_removal(self) -> None:
        """Контрольный опыт: гейт обязан падать, когда блок ИСЧЕЗ."""
        _requires_browser(self)
        src = Path(_RenderedReports.path("base")).read_text(encoding="utf-8")
        text = layout_probe.dom_text(_RenderedReports.path("base")).casefold()
        self.assertIn("волатильность", text)
        self.assertNotIn("этого-слова-нет-в-отчёте", text,
                         "гейт находит то, чего в отчёте нет — он бесполезен")
        self.assertGreater(len(src), 10_000)


if __name__ == "__main__":
    unittest.main()
