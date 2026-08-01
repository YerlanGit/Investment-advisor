"""
Phase-37 · R-18: ИИ-комментарии в секции «Что держите» отчёта BASE (2026-08-01).

Дефект из живого отчёта: под фундаменталом каждой раскрытой карточки рендерилась
золотая иконка ✨ и ПУСТОЙ абзац.  Три причины сразу:

  1. `_map_base` не отдавал `fundNote` — связный вывод по фундаменталу
     (`_fund_verdict`) производил ТОЛЬКО `_map_deep`;
  2. BASE-строка читала `note` из `assets[].note`, а он `""` у всех позиций;
  3. `portfolio-holdings.jsx` рендерил блок БЕЗ условия, тогда как DEEP-аналог
     защищён `{(h.fundNote || h.note) && (…)}`.

Вдобавок в payload BASE не было НИ ОДНОГО ИИ-ключа, хотя BASE-промпт
`ai_holdings_comment` ЗАПРАШИВАЕТ («≤170 знаков — hotspot-позиции с наибольшим
вкладом в риск… через бету») и экстрактор его сохраняет — секционная сводка
терялась в маппере.
"""

import json
import pathlib
import unittest

from premium_payload import _map_base

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_JSX = _ROOT / "design" / "premium_v2" / "portfolio-holdings.jsx"
_BUNDLE = _ROOT / "src" / "premium_assets" / "base-components.js"


def _payload(**over):
    p = {
        "assets": [
            {"ticker": "AAOI", "asset_class": "Акции США", "weight_pct_num": 1.3,
             "beta_num": 2.19, "euler_risk_pct": 2.8, "pnl_pct_num": -7.8,
             "atr_pct": "9.44%", "action": "Sell"},
            # ETF без покрытия SEC — вывод по фундаменталу невозможен
            {"ticker": "GLD", "asset_class": "Сырьё", "weight_pct_num": 5.0,
             "beta_num": 0.6, "euler_risk_pct": 2.0, "pnl_pct_num": -10.0,
             "action": "Trim"},
        ],
        "ai_holdings_comment": "CRCL даёт 17.5% риска при доле 7.2% — бета 2.73.",
        "fundamental_layer": [
            {"ticker": "AAOI", "roe": "-5.2%", "margin": "-12.0%",
             "debt": "37.2%", "growth": "82.8%", "z": "1.09"},
        ],
    }
    p.update(over)
    return p


class BaseSectionCommentTest(unittest.TestCase):
    """Секционная сводка ИИ обязана доезжать до BASE, а не только до DEEP."""

    def test_holdings_ai_reaches_base(self):
        out = _map_base(_payload(), {"id": "x"})
        self.assertEqual(out["holdingsAI"],
                         "CRCL даёт 17.5% риска при доле 7.2% — бета 2.73.")

    def test_base_payload_is_not_ai_free(self):
        """Живой отчёт 01.08: у BASE было 0 ИИ-ключей против 7 у DEEP."""
        out = _map_base(_payload(), {"id": "x"})
        ai_keys = [k for k in out if k.endswith("AI")]
        self.assertTrue(ai_keys, "в BASE-payload нет ни одного ИИ-ключа")

    def test_missing_comment_degrades_to_empty_not_dash(self):
        """Фолбэк без ИИ: пустая строка (блок скроется), а не «—» на экране."""
        out = _map_base(_payload(ai_holdings_comment=""), {"id": "x"})
        self.assertEqual(out["holdingsAI"], "")


class BasePerHoldingVerdictTest(unittest.TestCase):
    """Вывод по фундаменталу на карточке позиции — паритет с DEEP."""

    def setUp(self):
        self.out = _map_base(_payload(), {"id": "x"})
        self.by_t = {h["t"]: h for h in self.out["holdings"]}

    def test_fund_note_present_for_sec_covered_name(self):
        note = self.by_t["AAOI"]["fundNote"]
        self.assertTrue(note, "у бумаги с данными SEC вывод обязан быть")
        self.assertIn("ROE", note)

    def test_fund_note_empty_for_etf_without_sec(self):
        """ETF/кэш/EM-прокси не имеют отчётности — пустая строка, не выдумка."""
        self.assertEqual(self.by_t["GLD"]["fundNote"], "")

    def test_base_and_deep_use_the_same_key(self):
        """Иначе два тира разойдутся по смыслу одного блока."""
        from premium_payload import _map_deep
        self.assertIn("fundNote", self.by_t["AAOI"])
        src = pathlib.Path(_ROOT / "src" / "premium_payload.py").read_text()
        self.assertEqual(src.count('"fundNote"'), 2,
                         "fundNote должен отдаваться обоими мапперами")

    def test_legacy_note_still_wins_when_no_sec_verdict(self):
        p = _payload(fundamental_layer=[])
        p["assets"][0]["note"] = "заметка движка"
        out = _map_base(p, {"id": "x"})
        self.assertEqual(out["holdings"][0]["fundNote"], "заметка движка")


class BaseJsxGuardsEmptyBlockTest(unittest.TestCase):
    """🔴 Корень визуального дефекта: блок рендерился без условия."""

    def test_jsx_guards_the_sparkle_block(self):
        src = _JSX.read_text()
        self.assertIn("(h.fundNote || h.note) &&", src,
                      "блок ✨ обязан скрываться, когда текста нет")

    def test_jsx_prefers_fund_verdict(self):
        self.assertIn("h.fundNote || h.note", _JSX.read_text())

    def test_jsx_renders_section_summary(self):
        src = _JSX.read_text()
        self.assertIn("holdingsAI", src)
        self.assertIn("AI · сводка по составу", src)

    def test_compiled_bundle_is_in_sync(self):
        """`build.sh` обязан быть прогнан — иначе прод отдаёт старый JSX."""
        bundle = _BUNDLE.read_text()
        self.assertIn("holdingsAI", bundle,
                      "base-components.js устарел — запусти design/premium_v2/build.sh")
        self.assertIn("fundNote", bundle)


if __name__ == "__main__":
    unittest.main()
