"""DQ-8 — одиночный битый тик: скачок с возвратом на следующий день → `degrade` с тикером и датой.

Проверка аномалий B-6 его не видит по построению: один день из ≈ 1 250 — 0.08%, ниже
порога доли 0.1%, а скачок на 30–40% — ниже порога величины ±49%. Но σ и корреляции он
раздувает: живой отчёт сайта OMBRI дал σ₁₂ ≈ 43% при доходности −0.2% (план миграции §1.1).

B-7 смотрит на ФОРМУ: цена ушла и назавтра вернулась туда, откуда ушла. Реальный шаг
(цена ушла и осталась) и обычный шум — не находка.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from finance import data_checks as dc
from finance.data_checks import CheckLevel, CheckProfile, check_price_matrix


def _walk(days: int = 1250, seed: int = 8) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-09", periods=days)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, days))), index=idx)


class SpikeReversalTest(unittest.TestCase):

    def test_single_bad_tick_is_named_with_date(self) -> None:
        s = _walk()
        s.iloc[600] *= 1.4                       # +40% на один день и обратно
        self.assertIsNone(dc._b_anomalies(s, "AAA.US"))      # B-6 молчит — дефект DQ-8
        msg = dc._b_spike_reversal(s, "AAA.US")
        self.assertIsNotNone(msg)
        self.assertIn(s.index[600].date().isoformat(), msg)
        self.assertIn("→", msg)

    def test_finding_is_degrade_with_ticker(self) -> None:
        s = _walk()
        s.iloc[900] *= 0.7                       # провал −30% на один день
        report = check_price_matrix(pd.DataFrame({"AAA.US": s}), profile=CheckProfile.STRICT)
        b7 = [f for f in report.findings if f.id == "B-7"]
        self.assertEqual(len(b7), 1)
        self.assertIs(b7[0].level, CheckLevel.DEGRADE)
        self.assertEqual(b7[0].ticker, "AAA.US")
        self.assertIn(s.index[900].date().isoformat(), b7[0].message)
        self.assertFalse(report.blocked)

    def test_noise_and_real_step_are_not_findings(self) -> None:
        self.assertIsNone(dc._b_spike_reversal(_walk(), "AAA.US"))
        stepped = _walk()
        stepped.iloc[700:] *= 1.35               # ушла и осталась — не битый тик
        self.assertIsNone(dc._b_spike_reversal(stepped, "AAA.US"))

    def test_partial_reversal_is_not_a_tick(self) -> None:
        s = _walk()
        s.iloc[500] *= 1.4
        s.iloc[501:] *= 1.25                     # назавтра вернулась лишь наполовину
        self.assertIsNone(dc._b_spike_reversal(s, "AAA.US"))

    def test_many_ticks_are_counted_not_all_listed(self) -> None:
        s = _walk()
        for i in (100, 300, 500, 700, 900):
            s.iloc[i] *= 1.5
        msg = dc._b_spike_reversal(s, "AAA.US")
        self.assertIn("и ещё 2", msg)


if __name__ == "__main__":
    unittest.main()
