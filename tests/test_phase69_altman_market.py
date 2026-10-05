"""`§−132` — H-4: Altman Z — классическая формула с РЫНОЧНОЙ капитализацией.

Решение владельца 05.10: «H-4 — классическая формула с капитализацией».

Было: `sec_edgar` брал коэффициенты и пороги КЛАССИЧЕСКОЙ формулы (1968:
Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5, зоны 1.81 / 2.99), но в X4
подставлял БАЛАНСОВЫЙ капитал / обязательства вместо рыночной капитализации.
У компаний с крупными выкупами балансовый капитал мал или отрицателен — Z
падал в «Distress» при капитализации, многократно превышающей долги (FTNT в
живом отчёте 02.10: Z = 1.43, C-пиллар −1).

Сверка — с ИСТИНОЙ: Z, посчитанный руками по определению Altman из отчётных
строк и цены, а не формула модуля.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

_TESTS = Path(__file__).resolve().parent
SRC = _TESTS.parent / "src"
for p in (str(SRC), str(_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _fact(value: float, end: str = "2025-12-31") -> dict:
    return {"val": value, "end": end, "form": "10-K", "fp": "FY",
            "filed": "2026-02-01"}


def _company(*, ta, tl, ca, cl, re_, ebit, rev, shares, ni=1.0e9, eq=None) -> dict:
    """Минимальный CompanyFacts: ровно те теги, что читает `sec_edgar`."""
    usd = {
        "Assets": ta, "Liabilities": tl, "AssetsCurrent": ca,
        "LiabilitiesCurrent": cl, "RetainedEarningsAccumulatedDeficit": re_,
        "OperatingIncomeLoss": ebit, "Revenues": rev, "NetIncomeLoss": ni,
    }
    if eq is not None:
        usd["StockholdersEquity"] = eq
    g = {k: {"units": {"USD": [_fact(v)]}} for k, v in usd.items()}
    g["CommonStockSharesOutstanding"] = {"units": {"shares": [_fact(shares)]}}
    return {"facts": {"us-gaap": g}}


#: Компания с крупными выкупами (профиль FTNT, круглые числа): капитал по
#: балансу почти нулевой, рыночная капитализация — в разы больше долгов.
BUYBACK = dict(ta=10.0e9, tl=9.6e9, ca=6.0e9, cl=4.5e9, re_=-0.5e9,
               ebit=1.8e9, rev=6.0e9, shares=0.76e9, eq=0.4e9)
PRICE = 80.0                                   # → капитализация $60.8 млрд


def _truth_z(c: dict, price: float) -> float:
    """Altman (1968) по определению, без кода модуля."""
    x1 = (c["ca"] - c["cl"]) / c["ta"]
    x2 = c["re_"] / c["ta"]
    x3 = c["ebit"] / c["ta"]
    x4 = price * c["shares"] / c["tl"]
    x5 = c["rev"] / c["ta"]
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


class ClassicZIsMarketValueTest(unittest.TestCase):

    def _ext(self, c):
        from finance import sec_edgar as se
        se.get_extended_fundamentals.cache_clear()
        with mock.patch.object(se, "_fetch_company_facts", return_value=_company(**c)):
            return se.get_extended_fundamentals("TEST")

    def test_matches_altman_by_definition(self) -> None:
        from finance.sec_edgar import altman_z_classic
        ext = self._ext(BUYBACK)
        z = altman_z_classic(ext["altman_ex_x4"], PRICE * BUYBACK["shares"],
                             ext["total_liabilities"])
        self.assertAlmostEqual(z, _truth_z(BUYBACK, PRICE), places=10)

    def test_buyback_company_is_not_distress(self) -> None:
        """Было: X4 = 0.4/9.6 → Z ≈ 1.33 «Distress»; по рынку X4 ≈ 6.3 → Z ≈ 5.1 «Safe»."""
        from finance.sec_edgar import altman_z_classic, altman_zone_of
        ext = self._ext(BUYBACK)
        book_z = ext["altman_ex_x4"] + 0.6 * (BUYBACK["ta"] - BUYBACK["tl"]) / BUYBACK["tl"]
        self.assertEqual(altman_zone_of(book_z), "Distress")      # прежний ответ
        z = altman_z_classic(ext["altman_ex_x4"], PRICE * BUYBACK["shares"],
                             ext["total_liabilities"])
        self.assertEqual(altman_zone_of(z), "Safe")

    def test_module_no_longer_prints_a_book_zone(self) -> None:
        """Без цены зоны нет — модуль отдаёт части формулы, а не балансовую Z."""
        ext = self._ext(BUYBACK)
        self.assertNotIn("altman_z", ext)
        self.assertNotIn("altman_zone", ext)
        self.assertIn("altman_ex_x4", ext)

    def test_zones_are_the_classic_thresholds(self) -> None:
        from finance.sec_edgar import altman_zone_of
        self.assertEqual(altman_zone_of(3.0), "Safe")
        self.assertEqual(altman_zone_of(2.99), "Grey")
        self.assertEqual(altman_zone_of(1.81), "Grey")
        self.assertEqual(altman_zone_of(1.80), "Distress")

    def test_invalid_inputs_give_nothing(self) -> None:
        from finance.sec_edgar import altman_z_classic
        for args in ((1.0, None, 5.0), (1.0, 10.0, 0.0), (None, 10.0, 5.0),
                     (1.0, -1.0, 5.0), (1.0, float("nan"), 5.0)):
            self.assertIsNone(altman_z_classic(*args), args)


class EngineFillsTheZoneTest(unittest.TestCase):
    """`apply_market_altman` — там, где цена уже есть (стадия риск-модели)."""

    def _frame(self, **over):
        row = {"Current_Price": PRICE, "SEC_Shares_Outstanding": BUYBACK["shares"],
               "SEC_Altman_Ex_X4": 1.5, "SEC_Total_Liabilities": BUYBACK["tl"],
               "Fundamental_Sector": "Technology"}
        row.update(over)
        return pd.DataFrame([row], index=["FTNT"])

    def test_usd_report(self) -> None:
        from finance.sec_edgar import apply_market_altman
        df = apply_market_altman(self._frame(), usd_per_price_unit=1.0)
        z = 1.5 + 0.6 * PRICE * BUYBACK["shares"] / BUYBACK["tl"]
        self.assertAlmostEqual(df.at["FTNT", "SEC_Altman_Z"], z, places=10)
        self.assertEqual(df.at["FTNT", "SEC_Altman_Zone"], "Safe")

    def test_kzt_report_converts_the_price_back(self) -> None:
        """Цена в отчётной валюте (₸), отчётность — в USD: капитализацию
        переводим обратно, иначе X4 вырос бы в ~500 раз."""
        from finance.sec_edgar import apply_market_altman
        rate = 500.0                                   # ₸ за $1
        usd = apply_market_altman(self._frame(), usd_per_price_unit=1.0)
        kzt = apply_market_altman(self._frame(Current_Price=PRICE * rate),
                                  usd_per_price_unit=1.0 / rate)
        self.assertAlmostEqual(kzt.at["FTNT", "SEC_Altman_Z"],
                               usd.at["FTNT", "SEC_Altman_Z"], places=10)

    def test_no_price_basis_no_zone(self) -> None:
        """Нет капитализации — нет Z: балансовый суррогат был ДРУГОЙ моделью."""
        from finance.sec_edgar import apply_market_altman
        for over, rate in (({"SEC_Shares_Outstanding": None}, 1.0),
                           ({"Current_Price": None}, 1.0),
                           ({}, None)):
            with self.subTest(over=over, rate=rate):
                df = apply_market_altman(self._frame(**over), usd_per_price_unit=rate)
                self.assertTrue(pd.isna(df.at["FTNT", "SEC_Altman_Z"]))
                self.assertIsNone(df.at["FTNT", "SEC_Altman_Zone"])

    def test_banks_are_not_scored(self) -> None:
        """Обязательства банка — депозиты: Altman к нему неприменим (у банка
        с любой капитализацией Z ≈ 0.5 → ложное «Distress»)."""
        from finance.sec_edgar import apply_market_altman
        df = apply_market_altman(self._frame(Fundamental_Sector="Finance"),
                                 usd_per_price_unit=1.0)
        self.assertIsNone(df.at["FTNT", "SEC_Altman_Zone"])

    def test_frame_without_sec_columns_is_untouched(self) -> None:
        from finance.sec_edgar import apply_market_altman
        df = pd.DataFrame([{"Current_Price": 10.0}], index=["X"])
        out = apply_market_altman(df.copy(), usd_per_price_unit=1.0)
        self.assertListEqual(list(out.columns), list(df.columns))


class CreditPillarReadsTheMarketZoneTest(unittest.TestCase):
    """Конец в конец: SEC-строка через `analyze_all` → зона по рынку → C-пиллар."""

    def test_buyback_name_end_to_end(self) -> None:
        import golden_support as gs
        from finance.sec_edgar import altman_zone_of
        sec = pd.DataFrame([{
            "Ticker": "AAPL.US", "Fundamental_Score": 50,
            "Fundamental_Sector": "Technology",
            "SEC_Altman_Ex_X4": 1.5, "SEC_Total_Liabilities": 9.6e9,
            "SEC_Shares_Outstanding": 0.76e9,
        }]).set_index("Ticker")
        r = gs.run_analyze_all("base", sec_frame=sec)
        perf = r["performance_table"]
        perf = perf.set_index("Ticker") if "Ticker" in perf.columns else perf
        price = float(perf.at["AAPL.US", "Current_Price"])
        z = 1.5 + 0.6 * price * 0.76e9 / 9.6e9
        self.assertAlmostEqual(float(perf.at["AAPL.US", "SEC_Altman_Z"]), z, places=8)
        self.assertEqual(perf.at["AAPL.US", "SEC_Altman_Zone"], altman_zone_of(z))


if __name__ == "__main__":
    unittest.main()
