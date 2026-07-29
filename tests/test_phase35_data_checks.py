"""
Phase-35 · Фаза 3 — чекеры качества данных (2026-07-29).

Три блока (A — целостность выгрузки, B — корректность ряда, C — достаточность
для математики) и три профиля строгости.

Главный риск фазы — НЕ написать чекеры, а не уронить ими живые отчёты.
Поэтому центральный тест здесь — `LegacyNeverBlocksLiveBookTest`: он гоняет
профиль `LEGACY` на реплике РЕАЛЬНОГО портфеля из аудита 28.07 (15 позиций,
плечо 1.07×, отрицательный денежный остаток −7.45%, структурная нота AIX без
ценовой истории, молодой листинг SPCX) и требует, чтобы ни один чекер не
заблокировал отчёт, который сегодня успешно строится.
"""

import unittest

import numpy as np
import pandas as pd

from finance.data_checks import (BENCHMARK_ETFS, FACTOR_ETFS, CheckLevel,
                                 CheckProfile, DataQualityBlocked,
                                 DataQualityReport, check_portfolio_sufficiency,
                                 check_price_matrix, check_response_body,
                                 merge_reports, profile_for_source)

_CASH = ("USD", "EUR", "KZT", "RUB", "CASH")


def _prices(tickers, *, days: int = 1300, seed: int = 7,
            end: str = "2026-07-24") -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=days)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.standard_normal(days) * 0.01))
         for t in tickers}, index=idx)


class ProfileSelectionTest(unittest.TestCase):
    """Профиль задаёт ИСТОЧНИК ПОРТФЕЛЯ, а не отработавший провайдер."""

    def test_known_sources(self):
        self.assertIs(profile_for_source("manual"), CheckProfile.STRICT)
        self.assertIs(profile_for_source("demo"), CheckProfile.DEMO)
        self.assertIs(profile_for_source("freedom"), CheckProfile.LEGACY)

    def test_unknown_source_defaults_to_legacy(self):
        """Забытый вызов не должен молча ужесточать поведение (I-10)."""
        self.assertIs(profile_for_source(None), CheckProfile.LEGACY)
        self.assertIs(profile_for_source(""), CheckProfile.LEGACY)
        self.assertIs(profile_for_source("что-то новое"), CheckProfile.LEGACY)


class BlockAResponseTest(unittest.TestCase):
    """A: отказ источника приходит как HTTP 200 с текстом — это надо ловить."""

    HEADER = "Date,Open,High,Low,Close,Volume"

    def _ok_body(self) -> str:
        return self.HEADER + "\n2026-07-24,1,2,0.5,1.5,100\n"

    def test_valid_csv_passes(self):
        self.assertIsNone(check_response_body(
            self._ok_body(), expected_header=self.HEADER, max_bytes=5_000_000))

    def test_quota_message_blocks(self):
        """Тело «Exceeded the daily hits limit» при коде 200."""
        f = check_response_body("Exceeded the daily hits limit",
                                expected_header=self.HEADER, max_bytes=5_000_000)
        self.assertIsNotNone(f)
        self.assertIs(f.level, CheckLevel.BLOCK)
        self.assertEqual(f.id, "A-1")

    def test_apikey_page_blocks(self):
        f = check_response_body("Please get your apikey at stooq.com",
                                expected_header=self.HEADER, max_bytes=5_000_000)
        self.assertEqual(f.id, "A-1")

    def test_html_body_blocks(self):
        f = check_response_body("<html><body>Sorry</body></html>",
                                expected_header=self.HEADER, max_bytes=5_000_000)
        self.assertEqual(f.id, "A-1")

    def test_redirect_blocks(self):
        f = check_response_body(self._ok_body(), expected_header=self.HEADER,
                                max_bytes=5_000_000, redirected=True)
        self.assertEqual(f.id, "A-2")

    def test_non_200_blocks(self):
        f = check_response_body(self._ok_body(), expected_header=self.HEADER,
                                max_bytes=5_000_000, status=429)
        self.assertEqual(f.id, "A-2")

    def test_oversize_body_blocks(self):
        f = check_response_body(self._ok_body(), expected_header=self.HEADER,
                                max_bytes=10)
        self.assertEqual(f.id, "A-3")

    def test_header_only_blocks(self):
        f = check_response_body(self.HEADER, expected_header=self.HEADER,
                                max_bytes=5_000_000)
        self.assertEqual(f.id, "A-4")

    def test_wrong_header_blocks(self):
        f = check_response_body("a,b,c\n1,2,3\n", expected_header=self.HEADER,
                                max_bytes=5_000_000)
        self.assertEqual(f.id, "A-6")

    def test_bom_is_tolerated(self):
        """A-5: BOM не должен считаться повреждением."""
        self.assertIsNone(check_response_body(
            "﻿" + self._ok_body(), expected_header=self.HEADER,
            max_bytes=5_000_000))

    def test_message_carries_no_secret(self):
        """I-8: причина отказа не раскрывает ключ."""
        f = check_response_body("apikey=SECRET123 invalid",
                                expected_header=self.HEADER, max_bytes=5_000_000)
        self.assertNotIn("SECRET123", f.message)


class BlockBSeriesTest(unittest.TestCase):
    """B: корректность ряда. Проверки идут на СЫРОМ ряде, до math_firewall."""

    def _report(self, frame, profile=CheckProfile.LEGACY):
        return check_price_matrix(frame, profile=profile, as_of="2026-07-24")

    def _ids(self, report, level=None):
        return {f.id for f in report.findings
                if level is None or f.level is level}

    def test_clean_matrix_passes(self):
        r = self._report(_prices(["AAPL.US", "SPY.US"]))
        self.assertFalse(r.blocked)
        self.assertEqual(r.findings, [])

    def test_empty_matrix_blocks(self):
        self.assertTrue(self._report(pd.DataFrame()).blocked)

    def test_empty_matrix_only_degrades_on_demo(self):
        """Витрина не блокируется даже без данных (§3.4)."""
        r = self._report(pd.DataFrame(), profile=CheckProfile.DEMO)
        self.assertFalse(r.blocked)

    def test_duplicate_dates_block(self):
        idx = pd.DatetimeIndex(["2026-07-22", "2026-07-22", "2026-07-23"])
        frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
        self.assertIn("B-1", self._ids(self._report(frame)))

    def test_unsorted_index_blocks(self):
        idx = pd.DatetimeIndex(["2026-07-23", "2026-07-21", "2026-07-22"])
        frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
        self.assertIn("B-1", self._ids(self._report(frame)))

    def test_non_positive_price_blocks(self):
        frame = _prices(["A"], days=30)
        frame.iloc[5, 0] = 0.0
        self.assertIn("B-2", self._ids(self._report(frame)))

    def test_constant_series_blocks(self):
        """Постоянный ряд вырождает ковариацию."""
        idx = pd.bdate_range(end="2026-07-24", periods=30)
        frame = pd.DataFrame({"A": [100.0] * 30}, index=idx)
        self.assertIn("B-3", self._ids(self._report(frame)))

    def test_stale_series_degrades_not_blocks(self):
        frame = _prices(["A"], days=60, end="2026-06-01")
        r = self._report(frame)
        self.assertFalse(r.blocked)
        self.assertIn("B-4", self._ids(r, CheckLevel.DEGRADE))

    def test_gap_degrades(self):
        frame = _prices(["A"], days=60)
        frame = frame.drop(frame.index[20:40])
        self.assertIn("B-5", self._ids(self._report(frame), CheckLevel.DEGRADE))

    def test_anomaly_degrades(self):
        frame = _prices(["A"], days=300)
        frame.iloc[150, 0] *= 8.0          # скачок ×8
        self.assertIn("B-6", self._ids(self._report(frame), CheckLevel.DEGRADE))


class BlockCSufficiencyTest(unittest.TestCase):
    """C: хватает ли данных, чтобы математика была осмысленной."""

    def _full_matrix(self, extra=(), days=1300):
        return _prices(list(FACTOR_ETFS) + list(BENCHMARK_ETFS) + list(extra),
                       days=days)

    def _book(self, tickers, cash=0.0):
        """Равновесная книга + денежная строка."""
        n = len(tickers)
        risky_w = (1.0 - cash) / n
        weights = {t: risky_w for t in tickers}
        weights["USD"] = cash
        values = {t: risky_w * 100_000 for t in tickers}
        values["USD"] = cash * 100_000
        return weights, values

    def _run(self, matrix, weights, values, profile=CheckProfile.LEGACY, **kw):
        return check_portfolio_sufficiency(
            data=matrix, weights=weights, values=values, profile=profile, **kw)

    def _ids(self, report, level=None):
        return {f.id for f in report.findings
                if level is None or f.level is level}

    def test_healthy_book_passes(self):
        tk = ["AAPL.US", "MSFT.US", "TLT.US"]
        w, v = self._book(tk, cash=0.1)
        r = self._run(self._full_matrix(tk), w, v)
        self.assertFalse(r.blocked, [f.message for f in r.blocking])

    # ── C-8: покрытие ПО СТОИМОСТИ (ловушка Т-6) ────────────────────────────

    def test_c8_counts_value_not_tickers(self):
        """10 бумаг, не загрузилась ОДНА весом 55% → BLOCK.

        По числу тикеров это 90% покрытия и всё «хорошо»; по стоимости — 45%.
        Именно ради этого различия C-8 считает деньги.
        """
        loaded = [f"OK{i}.US" for i in range(9)]
        heavy = "HEAVY.US"
        weights = {t: 0.05 for t in loaded}
        weights[heavy] = 0.55
        values = {t: 5_000 for t in loaded}
        values[heavy] = 55_000
        r = self._run(self._full_matrix(loaded), weights, values)
        self.assertTrue(r.blocked)
        self.assertIn("C-8", self._ids(r, CheckLevel.BLOCK))

    def test_c8_message_speaks_money_not_codes(self):
        loaded = [f"OK{i}.US" for i in range(9)]
        weights = {t: 0.05 for t in loaded}; weights["HEAVY.US"] = 0.55
        values = {t: 5_000 for t in loaded}; values["HEAVY.US"] = 55_000
        r = self._run(self._full_matrix(loaded), weights, values)
        msg = r.user_message()
        self.assertIn("стоимости портфеля", msg)
        self.assertIn("токен не списан", msg)
        self.assertNotIn("C-8", msg)          # код проверки пользователю не нужен

    def test_c9_partial_coverage_degrades(self):
        """Покрытие в коридоре 90–98% → DEGRADE, отчёт строится."""
        loaded = [f"OK{i}.US" for i in range(9)]
        values = {t: 10_000 for t in loaded}
        values["SMALL.US"] = 4_000            # ⇒ покрытие 90 000 / 94 000 ≈ 95.7%
        total = sum(values.values())
        weights = {t: v / total for t, v in values.items()}
        r = self._run(self._full_matrix(loaded), weights, values)
        self.assertFalse(r.blocked)
        self.assertIn("C-9", self._ids(r, CheckLevel.DEGRADE))

    # ── C-1: факторы против бенчмарков ──────────────────────────────────────

    def test_c1_market_factor_loss_blocks_even_in_legacy(self):
        present = [t for t in FACTOR_ETFS if t != "SPY.US"]
        matrix = _prices(present + list(BENCHMARK_ETFS) + ["AAPL.US", "MSFT.US"])
        w, v = self._book(["AAPL.US", "MSFT.US"])
        r = self._run(matrix, w, v)
        self.assertIn("C-1", self._ids(r, CheckLevel.BLOCK))

    def test_c1_one_factor_loss_degrades_in_legacy(self):
        """Движок переживает потерю ЧАСТИ факторов — LEGACY повторяет его."""
        present = [t for t in FACTOR_ETFS if t != "VLUE.US"]
        matrix = _prices(present + list(BENCHMARK_ETFS) + ["AAPL.US", "MSFT.US"])
        w, v = self._book(["AAPL.US", "MSFT.US"])
        r = self._run(matrix, w, v)
        self.assertFalse(r.blocked)
        self.assertIn("C-1", self._ids(r, CheckLevel.DEGRADE))

    def test_c1_one_factor_loss_blocks_in_strict(self):
        """У ручного портфеля источник один — дыра значит отказ источника."""
        present = [t for t in FACTOR_ETFS if t != "VLUE.US"]
        matrix = _prices(present + list(BENCHMARK_ETFS) + ["AAPL.US", "MSFT.US"])
        w, v = self._book(["AAPL.US", "MSFT.US"])
        r = self._run(matrix, w, v, profile=CheckProfile.STRICT)
        self.assertIn("C-1", self._ids(r, CheckLevel.BLOCK))

    def test_c1_benchmark_loss_only_degrades(self):
        """Бенчмарки не факторы: потеря URTH портит сравнение, не модель."""
        matrix = _prices(list(FACTOR_ETFS) + ["QQQ.US", "AGG.US", "AAPL.US", "MSFT.US"])
        w, v = self._book(["AAPL.US", "MSFT.US"])
        r = self._run(matrix, w, v, profile=CheckProfile.STRICT)
        self.assertFalse(r.blocked)
        self.assertIn("C-1", self._ids(r, CheckLevel.DEGRADE))

    # ── C-2 / C-3: окна зеркалят движок ─────────────────────────────────────

    def test_c2_below_engine_floor_blocks(self):
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk, days=20), *self._book(tk))
        self.assertIn("C-2", self._ids(r, CheckLevel.BLOCK))

    def test_c3_short_window_degrades_not_blocks(self):
        """Окно 100 дней: движок честно считает реализованные метрики."""
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk, days=100), *self._book(tk))
        self.assertFalse(r.blocked)
        self.assertIn("C-3", self._ids(r, CheckLevel.DEGRADE))

    def test_c3_same_in_strict(self):
        """STRICT НЕ строже по окну — иначе книга с молодой бумагой (SKHY)
        перестала бы строиться на ровном месте."""
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk, days=100), *self._book(tk),
                      profile=CheckProfile.STRICT)
        self.assertFalse(r.blocked)

    # ── C-4: усечение источника vs молодой листинг ──────────────────────────

    def test_c4_truncation_detected_by_reference_length(self):
        """Признак ПАНЕЛЬНЫЙ: коротко даже у SPY/AGG ⇒ усечён источник."""
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk, days=400), *self._book(tk),
                      requested_days=1825, profile=CheckProfile.STRICT)
        self.assertIn("C-4", self._ids(r, CheckLevel.BLOCK))

    def test_c4_only_degrades_in_legacy(self):
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk, days=400), *self._book(tk),
                      requested_days=1825)
        self.assertFalse(r.blocked)
        self.assertIn("C-4", self._ids(r, CheckLevel.DEGRADE))

    def test_c4_quiet_when_references_are_long(self):
        """Молодой листинг НЕ должен выглядеть усечением."""
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk, days=1300), *self._book(tk),
                      requested_days=1825)
        self.assertNotIn("C-4", self._ids(r))

    # ── C-5 / C-7 ───────────────────────────────────────────────────────────

    def test_c5_underdetermined_blocks_strict_degrades_legacy(self):
        tk = [f"N{i}.US" for i in range(40)]
        matrix = self._full_matrix(tk, days=90)
        w, v = self._book(tk)
        self.assertIn("C-5", self._ids(self._run(matrix, w, v,
                                                 profile=CheckProfile.STRICT),
                                       CheckLevel.BLOCK))
        legacy = self._run(matrix, w, v)
        self.assertIn("C-5", self._ids(legacy, CheckLevel.DEGRADE))

    def test_c7_single_risky_position_blocks(self):
        tk = ["AAPL.US"]
        r = self._run(self._full_matrix(tk), *self._book(tk))
        self.assertIn("C-7", self._ids(r, CheckLevel.BLOCK))

    # ── C-10 / C-12: внутренняя согласованность, блокируют ВСЕГДА ───────────

    def test_c10_broken_weight_sum_blocks(self):
        tk = ["AAPL.US", "MSFT.US"]
        w, v = self._book(tk)
        w["AAPL.US"] = 5.0                  # арифметика поехала
        r = self._run(self._full_matrix(tk), w, v)
        self.assertIn("C-10", self._ids(r, CheckLevel.BLOCK))

    def test_c10_tolerates_negative_cash(self):
        """КЛЮЧЕВОЙ КЕЙС: маржинальный заём — норма, а не дефект.

        Реплика живого портфеля: денежная компонента ОТРИЦАТЕЛЬНА (−7.45%),
        сумма при этом ровно 1.0.  Наивная проверка «все веса ≥ 0» забраковала
        бы совершенно корректную книгу.
        """
        tk = ["AAPL.US", "MSFT.US", "NVDA.US"]
        weights = {"AAPL.US": 0.36, "MSFT.US": 0.36, "NVDA.US": 0.3545,
                   "USD": -0.0745}
        values = {t: abs(x) * 13_773 for t, x in weights.items()}
        r = self._run(self._full_matrix(tk), weights, values)
        self.assertNotIn("C-10", self._ids(r))

    def test_c10_and_c12_block_even_on_demo(self):
        """Витрина «не блокируется» — КРОМЕ признаков бага в коде."""
        tk = ["AAPL.US", "MSFT.US"]
        w, v = self._book(tk)
        w["AAPL.US"] = 5.0
        r = self._run(self._full_matrix(tk), w, v, profile=CheckProfile.DEMO)
        self.assertTrue(r.blocked)
        self.assertIn("C-10", self._ids(r, CheckLevel.BLOCK))

    def test_c12_cash_in_factor_model_blocks(self):
        tk = ["AAPL.US", "MSFT.US"]
        matrix = self._full_matrix(tk)
        matrix["USD"] = 1.0                 # кэш попал в ценовую матрицу
        w, v = self._book(tk, cash=0.1)
        r = self._run(matrix, w, v)
        self.assertIn("C-12", self._ids(r, CheckLevel.BLOCK))

    # ── C-11 / C-13 ─────────────────────────────────────────────────────────

    def test_c11_unconvertible_currency_degrades(self):
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk), *self._book(tk),
                      unconvertible_currencies=["EUR"])
        self.assertFalse(r.blocked)
        self.assertIn("C-11", self._ids(r, CheckLevel.DEGRADE))

    def test_c13_short_window_widens_cvar_ci(self):
        tk = ["AAPL.US", "MSFT.US"]
        r = self._run(self._full_matrix(tk, days=300), *self._book(tk))
        self.assertIn("C-13", self._ids(r, CheckLevel.DEGRADE))


class DemoProfileTest(unittest.TestCase):
    """Витрина обязана рендериться всегда."""

    def test_data_quality_never_blocks_demo(self):
        tk = ["AAPL.US"]
        matrix = _prices(["AAPL.US"], days=40)   # и мало данных, и одна позиция
        r = check_portfolio_sufficiency(
            data=matrix, weights={"AAPL.US": 1.0}, values={"AAPL.US": 1000},
            profile=CheckProfile.DEMO)
        self.assertFalse(r.blocked, [f.id for f in r.blocking])
        self.assertTrue(r.degraded)          # но факты зафиксированы


class LegacyNeverBlocksLiveBookTest(unittest.TestCase):
    """I-10 — ГЛАВНЫЙ ГЕЙТ ФАЗЫ.

    Профиль LEGACY не смеет заблокировать отчёт, который сегодня строится.
    Фикстура — реплика живого портфеля из аудита 28.07: 15 позиций, плечо
    1.07×, отрицательный денежный остаток, структурная нота AIX и молодой
    листинг SPCX (обе без собственной ценовой истории в матрице).
    """

    #: веса из фактического отчёта (сумма ровно 1.0 при отрицательном USD)
    LIVE_WEIGHTS = {
        "AAOI": 0.011996, "AAPL": 0.122915, "CRCL": 0.073536,
        "FFSPC6.1028.AIX": 0.046461, "GLD": 0.053578, "GOOGL": 0.119202,
        "META": 0.086193, "MSFT": 0.173419, "NVDA": 0.141263, "ORCL": 0.084456,
        "SLV": 0.026408, "SPCX": 0.055892, "TLT": 0.079154,
        "KZT": 0.000052, "USD": -0.074524,
    }
    NAV = 13_773.0

    def _matrix(self):
        # Все рисковые тикеры, КРОМЕ AIX-ноты и SPCX — они и в живом отчёте шли
        # без собственного ряда (нота broker-priced, SPCX — молодой листинг).
        priced = [t for t in self.LIVE_WEIGHTS
                  if t not in ("USD", "KZT", "FFSPC6.1028.AIX", "SPCX")]
        return _prices(list(FACTOR_ETFS) + list(BENCHMARK_ETFS) + priced)

    def _values(self):
        return {t: abs(w) * self.NAV for t, w in self.LIVE_WEIGHTS.items()}

    def test_legacy_does_not_block_live_book(self):
        r = check_portfolio_sufficiency(
            data=self._matrix(), weights=self.LIVE_WEIGHTS, values=self._values(),
            profile=CheckProfile.LEGACY, requested_days=1825)
        self.assertFalse(r.blocked,
                         "LEGACY заблокировал живой отчёт: "
                         + "; ".join(f"{f.id}: {f.message}" for f in r.blocking))

    def test_weight_sum_invariant_holds_on_live_book(self):
        self.assertAlmostEqual(sum(self.LIVE_WEIGHTS.values()), 1.0, places=5)

    def test_coverage_by_value_is_acceptable(self):
        """Две бумаги без ряда весят ~10% — покрытие по стоимости ~90%."""
        uncovered = abs(self.LIVE_WEIGHTS["FFSPC6.1028.AIX"]) + \
            abs(self.LIVE_WEIGHTS["SPCX"])
        self.assertLess(uncovered, 0.11)

    def test_series_checks_pass_on_live_matrix(self):
        r = check_price_matrix(self._matrix(), profile=CheckProfile.LEGACY,
                               as_of="2026-07-24")
        self.assertFalse(r.blocked)


class ReportPlumbingTest(unittest.TestCase):
    """Отчёт доносит результат до пользователя и до CoVe."""

    def test_blocked_raises_with_human_message(self):
        rep = DataQualityReport(profile=CheckProfile.LEGACY)
        from finance.data_checks import CheckFinding
        rep.findings.append(CheckFinding("C-8", CheckLevel.BLOCK,
                                         "Не удалось загрузить цены по 2 из 8 бумаг."))
        exc = DataQualityBlocked(rep)
        self.assertIn("токен не списан", str(exc))
        self.assertIs(exc.report, rep)

    def test_degrade_rows_reach_lineage(self):
        rep = DataQualityReport(profile=CheckProfile.LEGACY)
        from finance.data_checks import CheckFinding
        rep.findings.append(CheckFinding("B-4", CheckLevel.DEGRADE, "устарело"))
        rows = rep.lineage_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "warn")

    def test_merge_preserves_profile_and_findings(self):
        from finance.data_checks import CheckFinding
        a = DataQualityReport(profile=CheckProfile.STRICT)
        a.findings.append(CheckFinding("B-4", CheckLevel.DEGRADE, "x"))
        b = DataQualityReport(profile=CheckProfile.STRICT)
        b.findings.append(CheckFinding("C-9", CheckLevel.DEGRADE, "y"))
        m = merge_reports(a, b)
        self.assertIs(m.profile, CheckProfile.STRICT)
        self.assertEqual(len(m.findings), 2)

    def test_no_message_when_not_blocked(self):
        self.assertEqual(DataQualityReport(profile=CheckProfile.LEGACY).user_message(), "")


class RegistryShapeTest(unittest.TestCase):
    """Чекеры — данные, а не разбросанные `if` (гейт §10 задания)."""

    def test_series_checks_are_a_registry(self):
        from finance.data_checks import _SERIES_CHECKS
        self.assertTrue(_SERIES_CHECKS)
        for spec in _SERIES_CHECKS:
            self.assertTrue(spec.id.startswith("B-"))
            self.assertTrue(callable(spec.fn))
            for profile in CheckProfile:
                self.assertIsInstance(spec.level_for(profile), CheckLevel)

    def test_module_does_not_import_tg_bot(self):
        import inspect

        from finance import data_checks
        self.assertNotIn("tg_bot", inspect.getsource(data_checks))


if __name__ == "__main__":
    unittest.main()
