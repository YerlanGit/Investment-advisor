"""Раунд 30 (2026-07-19, live-инцидент 20:27): честность при сбое брокера.

M-6 — БРОКЕР УПАЛ ≠ «УСПЕШНО ПОЛУЧЕН».  При падении Freedom API коннектор
    возвращает fallback-мок (шаблонную книгу BTC-USD/AAPL/KSPI) с маркером
    `_ramp_is_fallback` — движковый гейт (`RealPortfolioRequired`) его ловит,
    но БОТ показывал превью «✅ Портфель успешно получен» с ДЕМО-позициями
    ДО анализа (юзер увидел чужую книгу как свою), а Шаг 1 затем падал с
    вводящей в заблуждение диагностикой «нет подписки Market Data».
    Фикс: гейт в cb_confirm ДО превью + честная формулировка Шага 1.

CI-1 — workflow-файл `.github/workflows/python-ci.yml` был обёрнут в «…»
    (невалидный YAML) с merge PR #64 (25.06) — GitHub CI молча падал с нулём
    джобов почти месяц.  Тест парсит все workflow-файлы как YAML.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("RAMP_BOT_TOKEN", "test-token-collection")

SRC = Path(__file__).resolve().parent.parent / "src"
REPO = SRC.parent


# ── M-6 · маркеры fallback-мока на стороне коннектора ────────────────────────

class FallbackMockMarkersTest(unittest.TestCase):
    """Любой сбой Tradernet-вызова обязан вернуть мок С ОБОИМИ маркерами —
    на них смотрят и бот-гейт (новый), и движковый гейт (RealPortfolioRequired)."""

    def _connector_with_failing_client(self, exc):
        import finance.broker_api as ba

        class _Boom:
            def __init__(self, *a, **k): pass
            def get_portfolio(self):
                raise exc
        orig = ba.TradernetClient
        ba.TradernetClient = _Boom
        self.addCleanup(setattr, ba, "TradernetClient", orig)
        return ba.FreedomConnector("real-key", "real-secret", "login")

    def test_api_error_returns_marked_mock(self):
        from freedom_portfolio.client import BrokerAPIError
        conn = self._connector_with_failing_client(BrokerAPIError("SSL EOF"))
        df = conn.fetch_portfolio()
        self.assertTrue(df.attrs.get("_ramp_is_mock"))
        self.assertTrue(df.attrs.get("_ramp_is_fallback"))

    def test_unexpected_error_returns_marked_mock(self):
        conn = self._connector_with_failing_client(ValueError("malformed json"))
        df = conn.fetch_portfolio()
        self.assertTrue(df.attrs.get("_ramp_is_mock"))
        self.assertTrue(df.attrs.get("_ramp_is_fallback"))

    def test_explicit_demo_is_not_fallback(self):
        from finance.broker_api import FreedomConnector
        df = FreedomConnector("demo").fetch_portfolio()
        self.assertTrue(df.attrs.get("_ramp_is_mock"))
        self.assertFalse(df.attrs.get("_ramp_is_fallback", False))
        self.assertEqual(df.attrs.get("_ramp_source"), "demo")


# ── M-6 · бот-гейт стоит ДО превью (source-level, phase-29 style) ────────────

class BotFallbackGateOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (SRC / "tg_bot.py").read_text(encoding="utf-8")

    def test_gate_checks_fallback_marker_before_preview(self):
        gate = "_ramp_is_fallback"
        preview = "✅ *Портфель успешно получен и принят в обработку.*"
        self.assertIn(gate, self.src)
        self.assertIn(preview, self.src)
        self.assertLess(self.src.index(gate), self.src.index(preview),
                        "гейт fallback-мока обязан стоять ДО превью «успешно получен»")

    def test_gate_halts_without_charging(self):
        block = self.src.split("Freedom Broker сейчас недоступен", 1)[1][:600]
        self.assertIn("Токен *не списан*", block)
        self.assertIn("return", block)

    def test_gate_also_catches_unexpected_mock_for_live_source(self):
        """Мок без fallback-флага (потерянные attrs и т.п.) при source=freedom
        тоже не должен уйти в превью как «ваш портфель»."""
        gate_block = self.src.split("M-6 (2026-07-19)", 1)[1][:900]
        self.assertIn('_ramp_is_mock', gate_block)
        self.assertIn('source != "demo"', gate_block)

    def test_step1_copy_leads_with_outage(self):
        """Диагностика Шага 1 начинается с самой частой причины (сбой API),
        а не с «нет подписки» — иначе юзер идёт покупать ненужную подписку."""
        step1 = self.src.split("Freedom API не вернул ни одной серии цен", 1)[1][:700]
        self.assertIn("Временный сбой", step1)
        self.assertLess(step1.index("Временный сбой"), step1.index("Market Data"))


# ── CI-1 · все workflow-файлы — валидный YAML ────────────────────────────────

class WorkflowYamlValidityTest(unittest.TestCase):
    """Ловит класс аварий «CI молча мёртв»: python-ci.yml месяц лежал обёрнутым
    в «…» — GitHub парсил файл в ошибку с нулём джобов на каждом пуше."""

    def test_workflow_files_parse(self):
        wf_dir = REPO / ".github" / "workflows"
        if not wf_dir.is_dir():
            self.skipTest("no .github/workflows in this tree (Docker image)")
        try:
            import yaml
        except ImportError:                             # pragma: no cover
            self.skipTest("pyyaml unavailable")
        files = sorted(wf_dir.glob("*.y*ml"))
        self.assertTrue(files, "workflows dir exists but is empty")
        for f in files:
            with self.subTest(workflow=f.name):
                text = f.read_text(encoding="utf-8")
                self.assertFalse(text.lstrip().startswith("«"),
                                 f"{f.name} wrapped in guillemets")
                data = yaml.safe_load(text)
                self.assertIsInstance(data, dict, f.name)
                self.assertIn("jobs", data, f.name)
                self.assertTrue(data["jobs"], f.name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
