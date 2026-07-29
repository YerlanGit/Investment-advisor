"""
Phase-36 · Исправления по аудиту живых отчётов DEEP 28.07 / BASE 29.07 (2026-07-29).

Три дефекта, найденные сквозной проверкой фактических отчётов:

**R-1 — знак Δw в Action Plan противоречил Effect-панели ТОГО ЖЕ отчёта.**
Инвариант «направление сделок — из 4-Pillar ACTION, НЕ из знака BL Δw» был
применён только к Effect (`simulate.high_priority_target_weights`, раунд 26),
а строка плана несла сырой BL-знак.  В живом отчёте: Action Plan «MSFT ·
Сократить · +3.0 пп» против Effect «MSFT · Продать · −2.97 пп» (та же величина,
обратный знак; то же у GLD и META).

**R-2/R-3 — панель и промпт ИИ считали темп макро-ряда РАЗНЫМИ окнами.**
Панель: окно по частоте публикации (дневной ряд → 21 наблюдение, «за 1м»).
Промпт: жёстко зашитый `lag=3`.  Один индикатор — две оценки: пользователь
видел «VIX ▲ +1.61 за 1м», модель получала +1.79.  На инфляционных ожиданиях
окна разошлись по ЗНАКУ («растут» в панели, «снижаются» в тексте).  Модель
ничего не выдумывала — расходились два расчёта движка.
"""

import unittest


class ActionSignedDeltaTest(unittest.TestCase):
    """R-1: знак Δw — из ACTION, величина — от Black-Litterman."""

    def setUp(self):
        from finance.action_plan import _action_signed_delta
        self.f = _action_signed_delta

    def test_sell_side_forces_negative(self):
        """Sell/Trim с ПОЛОЖИТЕЛЬНЫМ BL Δw → отрицательная дельта (корень R-1)."""
        self.assertEqual(self.f("Trim", 3.0), -3.0)
        self.assertEqual(self.f("Sell", 2.0), -2.0)

    def test_sell_side_keeps_agreeing_sign(self):
        """Когда BL и так согласен — величина не трогается."""
        self.assertEqual(self.f("Trim", -3.0), -3.0)

    def test_buy_side_forces_positive(self):
        self.assertEqual(self.f("Buy", -1.5), 1.5)
        self.assertEqual(self.f("Strong Buy", 1.5), 1.5)

    def test_hold_and_unknown_pass_through(self):
        """Hold и незнакомое действие сохраняют знак BL — направления нет."""
        self.assertEqual(self.f("Hold", 0.7), 0.7)
        self.assertEqual(self.f("", 2.2), 2.2)

    def test_magnitude_is_preserved(self):
        """Меняется ТОЛЬКО знак: |Δw| остаётся оптимизаторским."""
        for action in ("Trim", "Sell", "Buy", "Strong Buy"):
            self.assertAlmostEqual(abs(self.f(action, 4.37)), 4.37, places=9)

    def test_idempotent(self):
        """Повторное применение ничего не меняет — Effect вызывает её поверх."""
        for action, value in (("Trim", 3.0), ("Buy", -1.5), ("Hold", 0.7)):
            once = self.f(action, value)
            self.assertEqual(self.f(action, once), once)

    def test_non_numeric_is_safe(self):
        self.assertEqual(self.f("Trim", None), 0.0)
        self.assertEqual(self.f("Trim", "мусор"), 0.0)


class ActionPlanEffectAgreementTest(unittest.TestCase):
    """R-1 сквозной регресс: две панели одного отчёта не расходятся в знаке."""

    def _plan_rows(self):
        """Реплика живого кейса: TRIM-бумага с положительным BL Δw."""
        from finance.action_plan import _action_signed_delta
        raw = [("MSFT", "Trim", 3.0), ("GLD", "Trim", 1.8),
               ("META", "Trim", 1.2), ("ORCL", "Sell", -7.7)]
        return [{"ticker": t, "action": a, "delta_w_pp": _action_signed_delta(a, d)}
                for t, a, d in raw]

    def test_plan_row_sign_matches_action(self):
        for row in self._plan_rows():
            if row["action"] in ("Trim", "Sell"):
                self.assertLessEqual(row["delta_w_pp"], 0.0,
                                     f"{row['ticker']}: {row['action']} с положительной Δw")

    def test_effect_side_matches_plan_sign(self):
        """Effect выводит сторону из того же знака → «Продать» у всех Trim/Sell."""
        from finance.simulate import high_priority_target_weights
        rows = self._plan_rows()
        base = {"MSFT": 0.17, "GLD": 0.05, "META": 0.09, "ORCL": 0.08}
        _target, _hp, actions = high_priority_target_weights(
            action_plan_rows=rows, current_weights=base, bl_records=[])
        by_t = {a["ticker"]: a for a in actions}
        for row in rows:
            act = by_t.get(row["ticker"])
            if act is None:
                continue
            self.assertEqual(act["side"], "sell",
                             f"{row['ticker']}: план говорит {row['action']}, "
                             f"Effect — {act['side']}")
            self.assertLessEqual(act["delta_pp"], 0.0)


class MacroTrendWindowTest(unittest.TestCase):
    """R-2/R-3: окно темпа — единый источник для панели и промпта."""

    def test_lookback_is_cadence_aware(self):
        from finance.regime import macro_trend_lookback
        self.assertEqual(macro_trend_lookback("daily"), (21, "1м"))
        self.assertEqual(macro_trend_lookback("monthly"), (3, "3м"))
        self.assertEqual(macro_trend_lookback("quarterly"), (3, "3кв"))

    def test_unknown_cadence_has_fallback(self):
        from finance.regime import macro_trend_lookback
        lag, _win = macro_trend_lookback("непонятно")
        self.assertGreater(lag, 0)

    def test_panel_uses_shared_lookback(self):
        """`pdf_payload` больше не держит свою копию таблицы окон."""
        import pdf_payload
        from finance.regime import macro_trend_lookback
        self.assertIs(pdf_payload._macro_trend_lookback, macro_trend_lookback)
        self.assertFalse(hasattr(pdf_payload, "_MACRO_TREND_LOOKBACK"),
                         "осталась локальная копия таблицы — источник снова раздвоился")

    def test_prompt_and_panel_agree_on_daily_series(self):
        """Ключевой регресс: на одном ряде оба потребителя дают ОДНО число.

        Раньше панель считала по 21 наблюдению, а промпт — по 3, и на дневном
        ряде числа расходились (VIX +1.61 против +1.79).
        """
        import pdf_payload
        from finance.regime import macro_trend_lookback, series_trend

        # Ряд с разным поведением на коротком и длинном окне: сначала растёт,
        # в последние дни разворачивается вниз.  Именно такой профиль и давал
        # «панель: растёт / текст: снижается».
        values = [2.00 + 0.01 * i for i in range(25)] + [2.24, 2.22, 2.20]
        row = {"history_30d": [{"value": v} for v in values],
               "publish_cadence": "daily", "value": values[-1],
               "series_id": "T10YIE", "unit": "%"}

        lag, _win = macro_trend_lookback("daily")
        expected, _slope, _n = series_trend(values, lag)

        panel = pdf_payload._macro_series_trend(row)
        self.assertIsNotNone(panel)
        self.assertAlmostEqual(panel["delta"], round(expected, 2), places=2)

        short, _s, _n = series_trend(values, 3)
        self.assertNotAlmostEqual(short, expected, places=2,
                                  msg="фикстура не воспроизводит расхождение окон")

    def test_ai_macro_entry_carries_window_label(self):
        """Модели передаётся подпись окна, чтобы писать «+X за 1м», а не голую дельту."""
        import ai_narrative
        values = [18.0 + 0.05 * i for i in range(25)]
        results = {"macro_drivers": {"vix": {
            "value": values[-1], "publish_cadence": "daily", "unit": "",
            "status": "ok", "as_of": "2026-07-28",
            "history_30d": [{"value": v} for v in values]}}}
        summary = ai_narrative._summarise_for_prompt(results)
        entry = (summary.get("macro") or {}).get("vix") or {}
        self.assertIn("trend", entry)
        self.assertEqual(entry.get("trend_window"), "1м")

    def test_ai_trend_equals_panel_trend(self):
        """Сквозной регресс R-3: модель и панель получают ОДНО И ТО ЖЕ число."""
        import ai_narrative
        import pdf_payload

        values = [18.0 + 0.05 * i for i in range(25)] + [19.4, 19.2, 19.0]
        row = {"value": values[-1], "publish_cadence": "daily", "unit": "",
               "status": "ok", "as_of": "2026-07-28",
               "series_id": "VIXCLS",
               "history_30d": [{"value": v} for v in values]}

        panel = pdf_payload._macro_series_trend(dict(row))
        summary = ai_narrative._summarise_for_prompt({"macro_drivers": {"vix": dict(row)}})
        ai_trend = (summary.get("macro") or {}).get("vix", {}).get("trend")

        self.assertIsNotNone(panel)
        self.assertIsNotNone(ai_trend)
        self.assertAlmostEqual(panel["delta"], ai_trend, places=2,
                               msg="панель и промпт снова разошлись в темпе")


class NumbersRuleTest(unittest.TestCase):
    """R-2…R-5: правило «ИИ описывает, а не считает» присутствует в промпте."""

    def _rule(self) -> str:
        import ai_narrative
        import inspect
        src = inspect.getsource(ai_narrative)
        start = src.index("numbers_rule = (")
        return src[start:start + 2200]

    def test_forbids_computing(self):
        rule = self._rule()
        self.assertIn("ТЫ НЕ СЧИТАЕШЬ", rule)

    def test_requires_verbatim_trend_with_window(self):
        rule = self._rule()
        self.assertIn("trend_window", rule)
        self.assertIn("ВЕРБАТИМ", rule)

    def test_forbids_inventing_actions(self):
        """«Продать KZT, USD» — инструменты не из плана, плюс заём не продают."""
        rule = self._rule()
        self.assertIn("action_plan", rule)
        self.assertIn("не «продают»", rule)


if __name__ == "__main__":
    unittest.main()
