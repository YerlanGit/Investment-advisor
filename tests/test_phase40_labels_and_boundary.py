"""
Phase-40 · R-3, R-5, A-3 — ярлыки величин, адресат реинвеста, граница I-12.

R-3. «Ожид. доходность» существует в отчёте ДВАЖДЫ и это РАЗНЫЕ величины:
     обложка = форвард факторной модели (Σwᵢ·E[rᵢ]), панель Effect = Σw·μ
     постериора Black-Litterman либо реализованная оценка-фолбэк.  В живом
     отчёте 30.07 — 14.9% против 2.4% под одним ярлыком.

R-5. `ai_effect_comment` написал «высвобожденный вес идёт в кэш и снижение
     маржинального долга», тогда как движок покупал IEF/EEM/EMB.  Модель
     ПОЛУЧАЛА `rebalance_actions` — значит проблема не в данных, а в том, что
     формулировка адресата оставалась на её усмотрение.  Адресат теперь
     собирается детерминированно, модель его цитирует.

A-3. `provider_for_source` охранял I-12 fail-OPEN: гард был привязан к имени
     `== "manual"`, а всё прочее падало в `return TradernetProvider(...)` —
     `('stooq')` и `('eodhd')` молча отдавали брокерский фид.
"""

import unittest


class ExpectedReturnLabelNamesItsSourceTest(unittest.TestCase):
    """R-3: одинаковый ярлык на разных величинах — это противоречие."""

    def _labels(self, uses_bl):
        from premium_payload import build_design_data
        d = build_design_data({"expected_effect_uses_bl": uses_bl,
                               "expected_effect": {
            "expected_return": {"before": "2.4%", "after": "2.2%"}}}, "deep")
        return [e["name"] for e in d["effect"]]

    def test_bl_source_is_named(self):
        lbl = [x for x in self._labels(True) if "оходность" in x]
        self.assertEqual(lbl, ["Ожид. доходность (год., BL)"])

    def test_realised_fallback_is_named(self):
        lbl = [x for x in self._labels(False) if "оходность" in x]
        self.assertEqual(lbl, ["Ожид. доходность (год., реализ.)"])

    def test_horizon_is_always_stated(self):
        """Классический v3 подписывает «(год.)»; Premium терял горизонт."""
        for flag in (True, False):
            lbl = next(x for x in self._labels(flag) if "оходность" in x)
            self.assertIn("год.", lbl)

    def test_label_differs_from_cover_metric(self):
        """Ярлык панели обязан ОТЛИЧАТЬСЯ от обложечного, иначе смысл тот же."""
        lbl = next(x for x in self._labels(True) if "оходность" in x)
        self.assertNotEqual(lbl, "Ожид. доходность")

    def test_flag_survives_the_payload_adapter(self):
        from pdf_payload import build_payload
        pl = build_payload({"expected_effect": {"uses_bl_returns": True},
                            "portfolio_metrics": {}}, "deep", {})
        self.assertTrue(pl["expected_effect_uses_bl"])

    def test_metric_dict_contract_is_untouched(self):
        """`expected_effect` обязан нести ровно 8 строк-метрик и {} на пустом."""
        from pdf_payload import _build_expected_effect
        self.assertEqual(_build_expected_effect({"metrics": {}}), {})


class ReinvestDestinationIsDeterministicTest(unittest.TestCase):
    """R-5: адресат высвобожденного веса — факт движка, не выбор модели."""

    def _dest(self, actions):
        from ai_narrative import _reinvest_destination
        return _reinvest_destination(
            {"expected_effect": {"high_priority_actions": actions}})

    def test_names_the_actual_etfs(self):
        """Репро живого отчёта: движок покупал IEF/EEM/EMB."""
        d = self._dest([
            {"ticker": "IEF", "side": "buy", "delta_pp": 8.0,
             "name": "гособлигации США 7–10 лет", "is_external": True},
            {"ticker": "EEM", "side": "buy", "delta_pp": 8.0,
             "name": "акции развивающихся рынков", "is_external": True},
            {"ticker": "ORCL", "side": "sell", "delta_pp": -6.28},
        ])
        self.assertIn("IEF", d)
        self.assertIn("EEM", d)
        self.assertNotIn("ORCL", d, "продажи — не адресат реинвеста")
        self.assertIn("+8.0 пп", d)

    def test_cash_destination_is_labelled_cash(self):
        d = self._dest([{"ticker": "CASH", "side": "buy",
                         "delta_pp": 12.0, "is_cash": True}])
        self.assertTrue(d.startswith("Кэш"))

    def test_no_reinvest_yields_empty_string(self):
        """Пустая строка → промпт запрещает называть адресат вообще."""
        self.assertEqual(self._dest([]), "")
        self.assertEqual(self._dest([{"ticker": "X", "side": "sell",
                                      "delta_pp": -3.0}]), "")

    def test_sorted_by_size(self):
        d = self._dest([
            {"ticker": "SMALL", "side": "buy", "delta_pp": 1.0},
            {"ticker": "BIG", "side": "buy", "delta_pp": 9.0},
        ])
        self.assertLess(d.index("BIG"), d.index("SMALL"))

    def test_malformed_delta_does_not_raise(self):
        self.assertEqual(self._dest([{"ticker": "X", "side": "buy",
                                      "delta_pp": "мусор"}]), "")

    def test_prompt_forbids_inventing_a_destination(self):
        """Инструкция обязана явно запрещать подмену адресата."""
        import inspect
        import ai_narrative
        src = inspect.getsource(ai_narrative)
        self.assertIn("reinvest_destination", src)
        self.assertIn("ЗАПРЕЩЕНО", src)
        self.assertIn("маржинального долга", src,
                      "конкретная фраза из живого дефекта должна быть под запретом")

    def test_field_reaches_the_prompt_payload(self):
        import inspect
        import ai_narrative
        self.assertIn('"reinvest_destination": _reinvest_destination(results)',
                      inspect.getsource(ai_narrative))


class ProviderAllowlistFailsClosedTest(unittest.TestCase):
    """A-3: неизвестный источник = отказ, а не тихая подстановка брокера."""

    def _resolve(self, source):
        from finance.price_providers import provider_for_source
        return provider_for_source(source, client=object())

    def test_stooq_no_longer_returns_tradernet(self):
        from finance.price_providers import ProviderUnavailable
        with self.assertRaises(ProviderUnavailable) as cm:
            self._resolve("stooq")
        self.assertIn("I-12", str(cm.exception))

    def test_eodhd_no_longer_returns_tradernet(self):
        from finance.price_providers import ProviderUnavailable
        with self.assertRaises(ProviderUnavailable):
            self._resolve("eodhd")

    def test_typo_in_connection_mode_is_refused(self):
        from finance.price_providers import ProviderUnavailable
        with self.assertRaises(ProviderUnavailable):
            self._resolve("freedomm")

    def test_known_sources_still_work(self):
        self.assertEqual(type(self._resolve("freedom")).__name__, "TradernetProvider")
        self.assertEqual(type(self._resolve("demo")).__name__, "DemoProvider")

    def test_manual_is_served_by_its_own_branch_never_the_broker(self):
        """Ручной ввод идёт СВОЕЙ веткой и получает не брокера (I-12).

        Прежняя редакция проверяла отказ «провайдер не реализован» — то есть
        состояние ДО MP-09, а не правило. Провайдер написан; правило прежнее.
        """
        got = self._resolve("manual")
        self.assertEqual(type(got).__name__, "StooqProvider")

    def test_manual_refuses_an_unknown_provider_name(self):
        """Ветка `manual` остаётся fail-closed для незнакомого имени."""
        from unittest.mock import patch

        from finance.price_providers import ProviderUnavailable
        with patch.dict("os.environ", {"PRICE_PROVIDER_MANUAL": "eodhd"}):
            with self.assertRaises(ProviderUnavailable):
                self._resolve("manual")

    def test_empty_source_defaults_to_freedom(self):
        """Пустое значение — исторический дефолт, не «неизвестный источник»."""
        self.assertEqual(type(self._resolve(None)).__name__, "TradernetProvider")
        self.assertEqual(type(self._resolve("")).__name__, "TradernetProvider")

    def test_allowlist_is_explicit(self):
        from finance.price_providers import _KNOWN_SOURCES
        self.assertEqual(set(_KNOWN_SOURCES), {"freedom", "demo", "manual"})


if __name__ == "__main__":
    unittest.main()
