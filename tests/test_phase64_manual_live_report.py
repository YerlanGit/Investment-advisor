"""`§−127` — первый живой ручной отчёт DEEP (02.10): пять дефектов доставки и плана.

Каждый тест — один дефект, найденный по живому артефакту, а не по формуле:

1. **Старая ссылка показывала новый отчёт.** Ключ объекта в GCS был
   `r/<user>/<дата>/<тир>.html` — один файл на тир в сутки. Второй DEEP за
   день перезаписывал первый, и ссылка на DEEP #1 (ещё живая 48 ч) открывала
   DEEP #2. Источник портфеля тут ни при чём — так было для всех.
2. **Время подписи — UTC с биркой «UTC+5».** На Cloud Run `datetime.now()` —
   UTC: отчёт, собранный в 17:01 по Алматы, подписан «12:01 UTC+5».
3. **Кэш стоял на продажу.** HY-прокси давал строке `USD` кредитный пиллар
   −2 → «Sell»; модель читает `asset_scores`/`action_plan` движка и писала
   «продать XLU, FTNT, PEP, USD».
4. **План зависел от порядка, в котором набраны тикеры.** Лимит оборота
   раздаёт бюджет сверху вниз, а строки внутри «Sell» шли в порядке ввода:
   hotspot PGR (31.8% риска) стоял последним и ушёл в «отложено».
5. **Нарушение мандата было невидимо.** Класс, закрытый лимитом 0–0, панель
   пропускала, даже когда он в портфеле есть: XLU 18.3% в GlobalETFs 0–0.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ──────────────────────────────────────────────────────────────────────────
# 1 · одна ссылка — один отчёт
# ──────────────────────────────────────────────────────────────────────────
class ReportObjectIsUniqueTest(unittest.TestCase):

    def test_same_day_same_tier_gets_two_objects(self):
        from services.report_storage import _object_path
        now = datetime(2026, 10, 2, 12, 1, 45, tzinfo=timezone.utc)
        a = _object_path(148046720, "deep", now=now)
        b = _object_path(148046720, "deep", now=now)
        self.assertNotEqual(a, b, "даже в одну секунду объекты обязаны различаться")
        for key in (a, b):
            self.assertTrue(key.startswith("r/148046720/2026-10-02/deep-120145-"), key)
            self.assertTrue(key.endswith(".html"), key)

    def test_upload_never_reuses_an_object_name(self):
        import tempfile
        from services import report_storage as rs
        seen: list[str] = []

        def _fake_upload(_path, _bucket, object_name, _ttl):
            seen.append(object_name)
            return f"https://example.invalid/{object_name}"

        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
            fh.write("<html></html>")
        try:
            with patch.object(rs, "_upload_to_gcs", side_effect=_fake_upload):
                u1 = rs.upload_report(fh.name, 1, "deep", bucket_name="b")
                u2 = rs.upload_report(fh.name, 1, "deep", bucket_name="b")
        finally:
            Path(fh.name).unlink(missing_ok=True)
        self.assertEqual(len(set(seen)), 2, seen)
        self.assertNotEqual(u1, u2)


# ──────────────────────────────────────────────────────────────────────────
# 2 · подпись времени — в UTC+5, а не в поясе сервера
# ──────────────────────────────────────────────────────────────────────────
class ReportTimestampTest(unittest.TestCase):

    def test_live_case_utc_noon_is_almaty_five_pm(self):
        from html_renderer import report_timestamp
        utc = datetime(2026, 10, 2, 12, 1, tzinfo=timezone.utc)
        self.assertEqual(report_timestamp(utc), "02.10.2026 17:01 UTC+5")

    def test_naive_now_is_treated_as_utc_and_rolls_the_date(self):
        from html_renderer import report_timestamp
        self.assertEqual(report_timestamp(datetime(2026, 10, 2, 20, 30)),
                         "03.10.2026 01:30 UTC+5")

    def test_renderer_has_no_naive_now_label(self):
        src = (SRC / "html_renderer.py").read_text(encoding="utf-8")
        self.assertEqual(
            src.count('datetime.now().strftime("%d.%m.%Y %H:%M UTC+5")'), 0,
            "подпись «UTC+5» от наивного now() — это время сервера (UTC)")


# ──────────────────────────────────────────────────────────────────────────
# 3 · кэш — не эмитент: кредитный пиллар к нему неприменим
# ──────────────────────────────────────────────────────────────────────────
class CashIsNeverSellTest(unittest.TestCase):

    @staticmethod
    def _hy_proxy(_ticker):
        # Живое значение 02.10: HY OAS 312 bp — рыночный прокси на ВСЕ тикеры.
        return {"bps": 312.0, "change_7d": 0.01, "source": "FRED:HY", "quality": "C"}

    def test_cash_rows_hold_while_stock_still_gets_credit(self):
        from finance.scoring_orchestrator import score_portfolio
        perf = pd.DataFrame([
            {"Ticker": "PEP", "Euler_Risk_Contribution_Pct": 11.0},
            {"Ticker": "USD", "Euler_Risk_Contribution_Pct": 0.0},
            {"Ticker": "KZT", "Euler_Risk_Contribution_Pct": 0.0},
        ])
        scores = score_portfolio(perf, {}, cds_lookup=self._hy_proxy)
        for cash in ("USD", "KZT"):
            sc = scores[cash]
            self.assertFalse(sc.credit_applicable, cash)
            self.assertEqual(sc.credit, 0.0, cash)
            self.assertEqual(sc.action, "Hold", cash)
        # Акция по-прежнему проходит кредитный пиллар — guard узкий.
        self.assertTrue(scores["PEP"].credit_applicable)
        # C-1 (`§−129`): рыночный индекс HY в оценку эмитента больше не входит,
        # поэтому без данных SEC у PEP нейтральный C, а не прежний −2.
        self.assertEqual(scores["PEP"].credit, 0.0)


# ──────────────────────────────────────────────────────────────────────────
# 4 · лимит оборота раздаётся по риску, а не по порядку ввода
# ──────────────────────────────────────────────────────────────────────────
class TurnoverCapFollowsRiskTest(unittest.TestCase):
    """Числа — из живого DEEP 02.10 (вес, TRC, |Δw| оптимизатора)."""

    ROWS = {
        #        price,  TRC,  hotspot, |Δw| пп
        "XLU":  (39.44,  13.7, False,  8.3),
        "FTNT": (178.76, 17.5, False,  9.0),
        "PEP":  (126.72, 11.0, False,  6.7),
        "PGR":  (207.31, 31.8, True,  10.0),
    }

    def _plan(self, order):
        from finance.action_plan import build_action_plan
        perf = pd.DataFrame([
            {"Ticker": t, "Current_Price": self.ROWS[t][0], "Quantity": 10,
             "ATR_Absolute": self.ROWS[t][0] * 0.01,
             "Euler_Risk_Contribution_Pct": self.ROWS[t][1]}
            for t in order])
        scores = {t: {"action": "Sell", "total": -2.5, "hotspot": self.ROWS[t][2]}
                  for t in order}
        bl = [{"ticker": t, "delta_w_pp": -self.ROWS[t][3]} for t in order]
        return build_action_plan(perf_table=perf, asset_scores=scores,
                                 technicals_map={}, bl_records=bl,
                                 portfolio_value=8621.0)

    def test_hotspot_typed_last_is_not_deferred(self):
        rows = self._plan(["XLU", "FTNT", "PEP", "PGR"])   # порядок живого ввода
        pgr = next(r for r in rows if r.ticker == "PGR")
        self.assertEqual(pgr.action, "Sell", pgr.reason)
        self.assertNotIn("отложено", pgr.reason)
        self.assertEqual(rows[0].ticker, "PGR")

    def test_plan_does_not_depend_on_input_order(self):
        a = self._plan(["XLU", "FTNT", "PEP", "PGR"])
        b = self._plan(["PGR", "PEP", "FTNT", "XLU"])
        self.assertEqual([(r.ticker, r.action, r.delta_w_pp) for r in a],
                         [(r.ticker, r.action, r.delta_w_pp) for r in b])

    def test_non_hotspot_sells_ranked_by_risk_share(self):
        rows = self._plan(["PEP", "XLU", "FTNT", "PGR"])
        self.assertEqual([r.ticker for r in rows], ["PGR", "FTNT", "XLU", "PEP"])


# ──────────────────────────────────────────────────────────────────────────
# 5 · закрытый мандатом класс в портфеле — нарушение, а не «пропуск»
# ──────────────────────────────────────────────────────────────────────────
class ClosedMandateClassIsVisibleTest(unittest.TestCase):

    PROFILE = {"profile_name": "Умеренно-агрессивный", "target_volatility": 0.14,
               "target_te": 0.06,
               "limits_dict": {"Bonds": [10, 30], "Stocks_US": [30, 60],
                               "GlobalETFs": [0, 0], "Commodities": [0, 15],
                               "Crypto": [0, 5], "Stocks_KZ": [0, 0]}}

    def _mc(self):
        from pdf_payload import _build_mandate_compliance
        # Живой состав 02.10 (NAV $8 621).
        perf = pd.DataFrame([
            {"Ticker": "XLU",  "Current_Value": 1577.7},
            {"Ticker": "FTNT", "Current_Value": 893.8},
            {"Ticker": "EMB",  "Current_Value": 908.4},
            {"Ticker": "TXN",  "Current_Value": 1400.5},
            {"Ticker": "USD",  "Current_Value": 500.0},
            {"Ticker": "PEP",  "Current_Value": 1267.2},
            {"Ticker": "PGR",  "Current_Value": 2073.1},
        ])
        return _build_mandate_compliance(perf, 8621.0, self.PROFILE)

    def test_held_closed_class_is_a_breach(self):
        mc = self._mc()
        row = next((r for r in mc["rows"] if r["key"] == "GlobalETFs"), None)
        self.assertIsNotNone(row, "XLU 18.3% в закрытом классе спрятан")
        self.assertAlmostEqual(row["actual"], 18.3, places=1)
        self.assertEqual(row["status"], "over")
        self.assertEqual(mc["breaches"], 2)     # Stocks_US 65.4% > 60 и GlobalETFs
        self.assertFalse(mc["compliant"])

    def test_empty_closed_class_stays_hidden(self):
        keys = {r["key"] for r in self._mc()["rows"]}
        self.assertNotIn("Stocks_KZ", keys)


if __name__ == "__main__":
    unittest.main()
