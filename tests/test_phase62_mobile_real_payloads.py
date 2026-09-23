"""`§−121` — мобильный замер на РЕАЛЬНЫХ payload'ах, во всех трёх рендерах.

Гейт `test_phase55` меряет смоук-рендер: мок-payload и только Premium. Замер
раунда `§−121` на эталонных книгах (`golden_support`) нашёл то, что мок не
содержит и потому не мог показать:

* **Premium DEEP, 320 px** — чип «факторных двойников» с длинным тикером AIX
  («MSFT.US ↔ FFSPC6.1028.AIX») уходил на 6 px за экран;
* **Jinja-фолбэк** (его видит пользователь, когда Premium падает) — 13–15
  нарушителей на КАЖДОЙ ширине: подпись KPI с `white-space: nowrap` задавала
  дорожке `1fr` = minmax(auto, 1fr) ширину 506 px на экране 320, а чипы QC
  тянулись до 792 px. Замер на `HEAD` до правки — те же 15: дефект старый,
  просто этот рендер не мерил никто;
* **сценарный тир** — не измерялся вовсе: стенд ждал >200 узлов React, а у
  серверного Jinja их ~160.

Метрики те же две, что и у `test_phase55` (уход за экран и обрезка внутри
колонки), ширины — те же 320/360/390/414. Лечит РАСКЛАДКА, не кегль.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import layout_probe  # noqa: E402


def _requires_browser(case: unittest.TestCase) -> None:
    try:
        import playwright  # noqa: F401
    except Exception:
        case.skipTest("playwright не установлен (инструмент разработки)")
    if layout_probe.chromium_path() is None:
        case.skipTest("Chromium недоступен")


class _Renders:
    """Один прогон движка на класс: эталон дорогой, рендеры дешёвые."""

    _dir: "tempfile.TemporaryDirectory | None" = None
    _paths: dict[str, str] = {}

    @classmethod
    def build(cls) -> dict[str, str]:
        if cls._paths:
            return cls._paths
        import golden_support as gs
        import html_renderer
        from pdf_payload import build_payload
        cls._dir = tempfile.TemporaryDirectory(prefix="ramp-mobile-real-")
        out = Path(cls._dir.name)
        # «base» — книга с длинным тикером AIX (двойники DEEP); «leveraged_fx» —
        # маржа, плечевой ETP и тенге: баннеры плеча и чипы валютной конверсии.
        plan = [("base", "deep", True), ("base", "base", True),
                ("leveraged_fx", "deep", True), ("leveraged_fx", "base", True),
                ("leveraged_fx", "deep", False), ("leveraged_fx", "base", False)]
        results = {sc: gs.run_analyze_all(sc) for sc in {p[0] for p in plan}}
        for sc, tier, premium in plan:
            flag = "true" if premium else "false"
            with mock.patch.dict(os.environ, {"PREMIUM_REPORT_ENABLED": flag}):
                hr = importlib.reload(html_renderer)
                html = hr.render_report_html(build_payload(results[sc], tier),
                                             148046720, tier=tier)
            name = f"{'premium' if premium else 'jinja'}-{sc}-{tier}"
            (out / f"{name}.html").write_text(html, encoding="utf-8")
            cls._paths[name] = str(out / f"{name}.html")
        importlib.reload(html_renderer)
        with mock.patch.dict(os.environ, {"PREMIUM_REPORT_ENABLED": "true"}):
            hr = importlib.reload(html_renderer)
            (out / "scenario.html").write_text(
                hr.render_report_html(None, 148046720, tier="scenario"),
                encoding="utf-8")
        importlib.reload(html_renderer)
        cls._paths["jinja-scenario"] = str(out / "scenario.html")
        return cls._paths


def _min_nodes(name: str) -> int:
    return 100 if name == "jinja-scenario" else layout_probe.REACT_MIN_NODES


class RealPayloadMobileTest(unittest.TestCase):

    def test_nothing_runs_off_screen(self) -> None:
        _requires_browser(self)
        for name, path in _Renders.build().items():
            res = layout_probe.measure(path, min_nodes=_min_nodes(name))
            for w, r in res.items():
                with self.subTest(render=name, width=w):
                    self.assertEqual(r["offenders"], [],
                                     f"{name} @ {w}px: {r['offenders'][:3]}")

    def test_no_text_is_clipped_beyond_reach(self) -> None:
        _requires_browser(self)
        for name, path in _Renders.build().items():
            res = layout_probe.measure_clipped(path, min_nodes=_min_nodes(name))
            for w, leaves in res.items():
                with self.subTest(render=name, width=w):
                    self.assertEqual(leaves, [], f"{name} @ {w}px: {leaves[:3]}")

    def test_the_detector_sees_the_old_kpi_blowout(self) -> None:
        """Контрольный опыт: вернуть `1fr` + nowrap — и гейт обязан упасть.

        Без него «0 нарушителей» доказывало бы лишь, что детектор слеп
        (`§−97` E-6: метрика, обнулённая тем, что она измеряет, — не метрика).
        """
        _requires_browser(self)
        src = Path(_Renders.build()["jinja-leveraged_fx-base"]).read_text(encoding="utf-8")
        broken = src.replace("grid-template-columns: minmax(0, 1fr) !important;",
                             "grid-template-columns: 1fr !important;")
        broken = broken.replace(".kpi-sub, .qc-chip { white-space: normal !important;",
                                ".kpi-sub-off, .qc-chip-off { white-space: normal !important;")
        broken = broken.replace(".kpi, .kpi > *, [class*=\"-grid\"] > * { min-width: 0 !important; }", "")
        self.assertNotEqual(src, broken, "правка шаблона не найдена — опыт пуст")
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "broken.html"
            f.write_text(broken, encoding="utf-8")
            res = layout_probe.measure(str(f), widths=(360,))
        self.assertGreater(len(res[360]["offenders"]), 0)


if __name__ == "__main__":
    unittest.main()
