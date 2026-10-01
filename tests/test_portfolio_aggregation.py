"""Гибридный портфель: источники, гейт I-15, агрегатор, правки (пакет `portfolio_aggregation`).

Задание: `pico_prompt_hybrid_architecture_fallback_v2` (D-1…D-12), инвариант I-15
в `docs/roadmap/manual_portfolio/PHASE_00_CONTRACT.md`.

Что охраняется
--------------
* **S-1** — fallback-мок брокера не доезжает ни до слияния, ни до движка:
  `FreedomSource` превращает его в `ok=False` с ПУСТЫМ фреймом, а гейт
  агрегатора отказывает ДО `pd.concat` (который потерял бы `attrs`).
* **S-10 / I-15** — менеджер с `price_source="aggregated"` создаётся только
  после живого фетча по ключам пользователя.
* **D-6** — одна бумага в разных валютах в двух источниках → отказ, а не
  молчаливое усреднение `_normalize_positions`.
* **Математика не меняется** — склейку дублей делает движок; агрегатор только
  конкатенирует (`AggregatorDoesNoMathTest`).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from finance.investment_logic import MAC3RiskEngine  # noqa: E402
from portfolio_aggregation import (  # noqa: E402
    AGGREGATED_SOURCE,
    AggregatedNotPermitted,
    AggregationRefused,
    FreedomSource,
    KEY_ORIGIN_ADMIN_SERVICE,
    ManualSource,
    PortfolioAggregator,
    SourceResult,
    aggregated_manager,
    apply_edit,
    canonical_text,
    entries_of,
    load_with_budget,
    remove_at,
    version_tag,
)

BROKER_ROWS = [
    {"Ticker": "AAPL", "Quantity": 5.0, "Purchase_Price": 140.0,
     "Broker_Current_Price": 190.0, "Asset_Type": "Акция",
     "Raw_Ticker": "AAPL.US", "Currency": "USD"},
    {"Ticker": "USD", "Quantity": 100.0, "Purchase_Price": 1.0,
     "Broker_Current_Price": 1.0, "Asset_Type": "Кэш",
     "Raw_Ticker": "USD", "Currency": "USD"},
]

MANUAL_TEXT = "AAPL 10 150\nTLT 20 95\nCASH:KZT 1 000 000\nCASH:USD -300\n"


def _engine() -> MAC3RiskEngine:
    return MAC3RiskEngine(price_source="manual")


def _connector_returning(rows):
    class _C:
        def __init__(self, *_a) -> None:
            pass

        def fetch_portfolio(self):
            return pd.DataFrame(rows)
    return _C


def _connector_raising(exc):
    class _C:
        def __init__(self, *_a) -> None:
            pass

        def fetch_portfolio(self):
            raise exc
    return _C


def _live_freedom(rows=BROKER_ROWS) -> SourceResult:
    return FreedomSource("key", "secret",
                         connector_factory=_connector_returning(rows)).load()


def _manual(text=MANUAL_TEXT) -> SourceResult:
    return ManualSource(text, _engine()).load()


def _real_connector_failing_with(exc):
    """НАСТОЯЩИЙ `FreedomConnector` с падающим транспортом — его fallback-мок."""
    import finance.broker_api as ba

    class _Boom:
        def __init__(self, *a, **k) -> None:
            pass

        def get_portfolio(self):
            raise exc
    return patch.object(ba, "TradernetClient", _Boom)


# ── источники ─────────────────────────────────────────────────────────────────

class FreedomSourceTest(unittest.TestCase):

    def test_live_fetch_is_ok_and_carries_proof(self) -> None:
        res = _live_freedom()
        self.assertTrue(res.ok)
        self.assertTrue(res.live)
        self.assertEqual(res.key_origin, "vault")
        self.assertEqual(res.positions, 2)

    def test_fallback_mock_becomes_failure_with_empty_frame(self) -> None:
        """S-1: мок брокера наружу не выходит, причина сохраняется (§−94)."""
        from freedom_portfolio.client import BrokerAPIError, CloudflareBlockError

        for exc, reason in ((BrokerAPIError("SSL EOF"), "api_error"),
                            (CloudflareBlockError("blocked"), "waf_block"),
                            (ValueError("malformed"), "parse_error")):
            with self.subTest(reason=reason), _real_connector_failing_with(exc):
                res = FreedomSource("real-key", "real-secret").load()
                self.assertFalse(res.ok)
                self.assertFalse(res.live)
                self.assertEqual(res.failure_reason, reason)
                self.assertTrue(res.frame.empty, "мок обязан быть отброшен")
                self.assertTrue(res.error_id)

    def test_auth_and_empty_are_distinct_reasons(self) -> None:
        from finance.broker_api import BrokerAuthError, BrokerEmptyPortfolioError

        auth = FreedomSource("k", connector_factory=_connector_raising(
            BrokerAuthError("bad"))).load()
        empty = FreedomSource("k", connector_factory=_connector_raising(
            BrokerEmptyPortfolioError("none"))).load()
        self.assertEqual((auth.ok, auth.failure_reason), (False, "auth"))
        self.assertEqual((empty.ok, empty.failure_reason), (False, "empty"))

    def test_empty_key_never_reaches_connector(self) -> None:
        """Пустой ключ коннектор подменил бы СЕРВИСНЫМ — то есть чужим портфелем."""
        called = []

        def _factory(*a):
            called.append(a)
            raise AssertionError("коннектор не должен создаваться")

        res = FreedomSource("", connector_factory=_factory).load()
        self.assertEqual(res.failure_reason, "auth")
        self.assertFalse(called)

    def test_demo_key_is_not_live(self) -> None:
        res = FreedomSource("demo").load()
        self.assertFalse(res.ok)
        self.assertEqual(res.failure_reason, "not_live")
        self.assertTrue(res.frame.empty)

    def test_unexpected_error_hides_text(self) -> None:
        """S-2: текст исключения может нести тело ответа брокера — наружу код."""
        with self.assertLogs("portfolio_aggregation.sources", level="ERROR") as logs:
            res = FreedomSource("k", connector_factory=_connector_raising(
                RuntimeError("SECRET-BODY-xyz"))).load()
        self.assertEqual(res.failure_reason, "error")
        self.assertTrue(res.error_id)
        self.assertNotIn("SECRET-BODY-xyz", "\n".join(logs.output))

    def test_repr_has_no_keys(self) -> None:
        """S-3: ключи не печатаются ни в repr, ни в логах."""
        src = FreedomSource("API-KEY-123", "SECRET-456")
        self.assertNotIn("API-KEY-123", repr(src))
        self.assertNotIn("SECRET-456", repr(src))

    def test_unknown_key_origin_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FreedomSource("k", key_origin="service")


class ManualSourceTest(unittest.TestCase):

    def test_parses_to_contract_frame(self) -> None:
        res = _manual()
        self.assertTrue(res.ok)
        self.assertEqual(res.frame.attrs.get("_ramp_source"), "manual")
        self.assertEqual(res.detail["parsed"], 4)

    def test_empty_text_is_failure(self) -> None:
        self.assertEqual(_manual("").failure_reason, "empty")
        self.assertEqual(_manual("??? ??").failure_reason, "empty")

    def test_51st_position_is_refused(self) -> None:
        """S-7: `MANUAL_MAX_POSITIONS` (дефолт 50)."""
        text = "\n".join(f"T{i:03d} 1 10 USD" for i in range(51))
        with patch.dict(os.environ, {"MANUAL_MAX_POSITIONS": ""}):
            res = ManualSource(text, _engine()).load()
        self.assertFalse(res.ok)
        self.assertEqual(res.failure_reason, "too_many")
        self.assertEqual(res.detail["limit"], 50)

    def test_repr_has_no_positions(self) -> None:
        self.assertNotIn("AAPL", repr(ManualSource(MANUAL_TEXT, _engine())))


class BudgetTest(unittest.IsolatedAsyncioTestCase):

    async def test_timeout_becomes_failure(self) -> None:
        class _Slow:
            name = "freedom"

            def load(self):
                time.sleep(0.5)
                return SourceResult(name="freedom", frame=pd.DataFrame(), ok=True)

        res = await load_with_budget(_Slow(), 0.05)
        self.assertFalse(res.ok)
        self.assertEqual(res.failure_reason, "timeout")

    async def test_fast_source_passes_through(self) -> None:
        res = await load_with_budget(
            FreedomSource("k", connector_factory=_connector_returning(BROKER_ROWS)), 5)
        self.assertTrue(res.ok)


# ── I-15 ─────────────────────────────────────────────────────────────────────

class AggregatedRequiresLiveBrokerFetchTest(unittest.TestCase):
    """S-10: `test_aggregated_requires_live_broker_fetch` (уровень гейта).

    Тот же инвариант на уровне бота — `test_hybrid_bot_flow`.
    """

    def _assert_manager_not_built(self, freedom: SourceResult) -> None:
        built = []
        with self.assertRaises(AggregatedNotPermitted):
            aggregated_manager(freedom, lambda **kw: built.append(kw))
        self.assertFalse(built, "менеджер с aggregated создан без доказательства")

    def test_aggregated_requires_live_broker_fetch(self) -> None:
        from finance.broker_api import BrokerAuthError
        from freedom_portfolio.client import BrokerAPIError

        with _real_connector_failing_with(BrokerAPIError("down")):
            fallback = FreedomSource("real-key", "real-secret").load()
        auth = FreedomSource("k", connector_factory=_connector_raising(
            BrokerAuthError("revoked"))).load()
        no_keys = FreedomSource("").load()
        for name, res in (("fallback", fallback), ("auth", auth), ("no_keys", no_keys)):
            with self.subTest(case=name):
                self._assert_manager_not_built(res)

    def test_forged_result_with_mock_marker_is_refused(self) -> None:
        df = pd.DataFrame(BROKER_ROWS)
        df.attrs["_ramp_is_fallback"] = True
        forged = SourceResult(name="freedom", frame=df, ok=True, live=True,
                              key_origin="vault")
        self._assert_manager_not_built(forged)

    def test_live_vault_fetch_builds_manager(self) -> None:
        built = []
        aggregated_manager(_live_freedom(), lambda **kw: built.append(kw))
        self.assertEqual(built, [{"price_source": AGGREGATED_SOURCE}])

    def test_admin_service_keys_are_accepted(self) -> None:
        res = FreedomSource("k", key_origin=KEY_ORIGIN_ADMIN_SERVICE,
                            connector_factory=_connector_returning(BROKER_ROWS)).load()
        built = []
        aggregated_manager(res, lambda **kw: built.append(kw))
        self.assertTrue(built)


# ── агрегатор ─────────────────────────────────────────────────────────────────

class AggregatorTest(unittest.TestCase):

    def test_fallback_mock_never_merged(self) -> None:
        """S-1: мок брокера не попадает в слияние ни при каком сбое."""
        from finance.demo_portfolio import build_demo_portfolio
        from freedom_portfolio.client import BrokerAPIError

        with _real_connector_failing_with(BrokerAPIError("down")):
            freedom = FreedomSource("real-key", "real-secret").load()
        with self.assertRaises(AggregatedNotPermitted):
            PortfolioAggregator().merge(freedom, _manual())
        demo_tickers = set(build_demo_portfolio()["Ticker"].astype(str))
        self.assertFalse(set(freedom.frame.get("Ticker", [])) & demo_tickers)

    def test_merged_frame_is_marked_aggregated_only(self) -> None:
        res = PortfolioAggregator().merge(_live_freedom(), _manual())
        self.assertEqual(res.frame.attrs, {"_ramp_source": "aggregated"})
        self.assertEqual(len(res.frame), 2 + 4)
        self.assertEqual(res.overlaps, ["AAPL"], "кэш USD — не пересечение бумаг")
        self.assertEqual(res.composition(),
                         {"freedom_positions": 2, "manual_positions": 4,
                          "overlaps": ["AAPL"]})

    def test_manual_rows_keep_no_broker_price(self) -> None:
        """D-2: у ручной бумаги цены брокера нет — движок возьмёт её из матрицы."""
        res = PortfolioAggregator().merge(_live_freedom(), _manual())
        tlt = res.frame[res.frame["Ticker"] == "TLT"].iloc[0]
        self.assertTrue(pd.isna(tlt["Broker_Current_Price"]))

    def test_currency_conflict_is_refused(self) -> None:
        """D-6: `_normalize_positions` молча взял бы первую валюту."""
        with self.assertRaises(AggregationRefused) as ctx:
            PortfolioAggregator().merge(_live_freedom(), _manual("AAPL 10 70000 KZT"))
        self.assertEqual(ctx.exception.reason, "currency_conflict")
        self.assertEqual(ctx.exception.tickers, ("AAPL",))

    def test_manual_must_be_present(self) -> None:
        with self.assertRaises(AggregationRefused) as ctx:
            PortfolioAggregator().merge(_live_freedom(), _manual(""))
        self.assertEqual(ctx.exception.reason, "manual_empty")

    def test_aggregated_limit(self) -> None:
        """S-7: `AGGREGATED_MAX_POSITIONS` (дефолт 150) — до склейки."""
        broker = [dict(BROKER_ROWS[0], Ticker=f"B{i:03d}", Raw_Ticker=f"B{i:03d}.US")
                  for i in range(120)]
        manual = "\n".join(f"M{i:03d} 1 10 USD" for i in range(31))
        with patch.dict(os.environ, {"AGGREGATED_MAX_POSITIONS": ""}):
            with self.assertRaises(AggregationRefused) as ctx:
                PortfolioAggregator().merge(_live_freedom(broker),
                                            ManualSource(manual, _engine(),
                                                         max_positions=100).load())
        self.assertEqual((ctx.exception.reason, ctx.exception.limit), ("too_many", 150))

    def test_taxonomy_columns_are_filled_not_nan(self) -> None:
        """`pdf_payload` читает `Asset_Class_Label or ...` — NaN истинен."""
        rows = [dict(BROKER_ROWS[0], Asset_Class="equity", Asset_Class_Label="Акции")]
        res = PortfolioAggregator().merge(_live_freedom(rows), _manual("TLT 5 90"))
        tlt = res.frame[res.frame["Ticker"] == "TLT"].iloc[0]
        self.assertIsInstance(tlt["Asset_Class_Label"], str)
        self.assertTrue(tlt["Asset_Class_Label"])
        self.assertNotEqual(tlt["Asset_Class_Label"].lower(), "nan")


class AggregatorDoesNoMathTest(unittest.TestCase):
    """Склейка — дело движка: VWAP и кэш по валюте те же, что у брокера (A-1)."""

    def test_engine_merges_duplicates_exactly_as_before(self) -> None:
        from finance.investment_logic import UniversalPortfolioManager

        merged = PortfolioAggregator().merge(_live_freedom(), _manual())
        # Агрегатор НЕ склеивает: AAPL и USD — по две строки.
        self.assertEqual(int((merged.frame["Ticker"] == "AAPL").sum()), 2)
        upm = UniversalPortfolioManager(price_source="aggregated")
        df, merged_rows, _dropped, source = upm._stage_normalize(merged.frame)
        self.assertEqual(source, "aggregated")
        self.assertEqual(dict(merged_rows), {"AAPL": 2, "USD": 2})
        self.assertAlmostEqual(df.loc["AAPL", "Quantity"], 15.0)
        self.assertAlmostEqual(df.loc["AAPL", "Purchase_Price"],
                               (5 * 140.0 + 10 * 150.0) / 15)
        self.assertAlmostEqual(df.loc["AAPL", "Broker_Current_Price"], 190.0)
        self.assertAlmostEqual(df.loc["USD", "Quantity"], 100.0 - 300.0,
                               msg="отрицательный ручной кэш (маржа) допустим — D-6")


# ── ценовой слой и чекеры (D-12) ─────────────────────────────────────────────

class AggregatedPriceSourceTest(unittest.TestCase):

    def test_known_source_and_tradernet_provider(self) -> None:
        from finance import price_providers as pp

        self.assertIn("aggregated", pp._KNOWN_SOURCES)
        prov = pp.provider_for_source("aggregated", client=object())
        self.assertIsInstance(prov, pp.TradernetProvider)

    def test_branch_is_explicit(self) -> None:
        """Явная ветка, а не падение в хвост `return TradernetProvider(...)`."""
        src = (_SRC / "finance" / "price_providers.py").read_text(encoding="utf-8")
        self.assertIn('if src == "aggregated":', src)

    def test_check_profile_is_legacy_explicitly(self) -> None:
        from finance import data_checks as dc

        self.assertIs(dc.profile_for_source("aggregated"), dc.CheckProfile.LEGACY)
        src = (_SRC / "finance" / "data_checks.py").read_text(encoding="utf-8")
        self.assertIn('if src == "aggregated":', src)

    def test_engine_builds_tradernet_client_for_aggregated(self) -> None:
        """`risk_engine._get_price_provider` не правился — проверяем поведение."""
        from finance import price_providers as pp

        eng = MAC3RiskEngine(price_source="aggregated")
        sentinel = object()
        with patch.object(MAC3RiskEngine, "_get_tradernet_client",
                          return_value=sentinel) as made:
            prov = eng._get_price_provider()
        made.assert_called_once()
        self.assertIsInstance(prov, pp.TradernetProvider)

    def test_manual_still_refuses_tradernet(self) -> None:
        """I-12 не ослаблен: `manual` по-прежнему без клиента Tradernet."""
        eng = MAC3RiskEngine(price_source="manual")
        with patch.object(MAC3RiskEngine, "_get_tradernet_client") as made:
            try:
                eng._get_price_provider()
            except Exception:                     # noqa: BLE001 — Stooq может быть не настроен
                pass
        made.assert_not_called()


# ── CoVe (§I.6) ──────────────────────────────────────────────────────────────

class LineageTest(unittest.TestCase):

    def test_aggregated_row(self) -> None:
        from finance.data_lineage import _aggregated_source_status

        row = _aggregated_source_status({
            "portfolio_source": "aggregated",
            "aggregated_composition": {"freedom_positions": 3, "manual_positions": 2,
                                       "overlaps": ["AAPL"]},
        })
        self.assertEqual(row["status"], "degrade")
        self.assertIn("Freedom Broker (3 поз.) + ручной ввод (2 поз.)", row["source"])
        self.assertIn("Tradernet для всех позиций", row["note"])
        self.assertIn("AAPL", row["note"])

    def test_row_absent_for_other_sources(self) -> None:
        from finance.data_lineage import _aggregated_source_status

        for src in ("freedom", "manual", "demo", None):
            self.assertIsNone(_aggregated_source_status({"portfolio_source": src}))

    def test_fallback_note_on_manual_row(self) -> None:
        from finance.data_lineage import _manual_source_status

        row = _manual_source_status({"portfolio_source": "manual",
                                     "broker_fallback_reason": "таймаут ответа"})
        self.assertIn("Freedom Broker был недоступен (таймаут ответа)", row["note"])
        plain = _manual_source_status({"portfolio_source": "manual"})
        self.assertNotIn("недоступен", plain["note"])


# ── правки ручного портфеля (PR-1 §2) ────────────────────────────────────────

class EditsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = _engine()
        self.text, n, bad = canonical_text(
            "AAPL 10 150\nкаспи 200 104,5 KZT\nCASH:USD 3000", self.engine)
        self.assertEqual((n, bad), (3, 0))

    def test_canonical_text_reparses_identically(self) -> None:
        again, n, bad = canonical_text(self.text, self.engine)
        self.assertEqual(again, self.text)
        self.assertIn("KSPI.KZ 200 104.5 KZT", self.text)

    def test_add_new_and_vwap(self) -> None:
        res = apply_edit(self.text, "+AAPL 10 170\n+TLT 5 90", self.engine)
        self.assertTrue(res.ok, res.error)
        aapl = next(e for e in entries_of(res.new_text, self.engine) if e.key == "AAPL")
        self.assertEqual(aapl.quantity, 20)
        self.assertAlmostEqual(aapl.price, 160.0)
        self.assertEqual(len(res.applied), 2)

    def test_add_in_other_currency_is_refused(self) -> None:
        res = apply_edit(self.text, "+AAPL 1 70000 KZT", self.engine)
        self.assertIn("одна валюта", res.error.lower())
        self.assertEqual(res.new_text, self.text)

    def test_trim_keeps_price(self) -> None:
        res = apply_edit(self.text, "-AAPL 4", self.engine)
        aapl = next(e for e in entries_of(res.new_text, self.engine) if e.key == "AAPL")
        self.assertEqual((aapl.quantity, aapl.price), (6, 150.0))

    def test_trim_to_zero_suggests_full_remove(self) -> None:
        res = apply_edit(self.text, "-AAPL 10", self.engine)
        self.assertIn("-AAPL", res.error)

    def test_remove_whole(self) -> None:
        res = apply_edit(self.text, "-AAPL", self.engine)
        self.assertNotIn("AAPL", res.new_text)

    def test_cash_cannot_cross_zero(self) -> None:
        self.assertIsNotNone(apply_edit(self.text, "-USD 5000", self.engine).error)
        gone = apply_edit(self.text, "-USD 3000", self.engine)
        self.assertNotIn("CASH:USD", gone.new_text)
        more = apply_edit(self.text, "+USD 500", self.engine)
        self.assertIn("CASH:USD 3500", more.new_text)

    def test_all_or_nothing(self) -> None:
        res = apply_edit(self.text, "+TLT 5 90\n-ZZZZ", self.engine)
        self.assertIsNotNone(res.error)
        self.assertEqual(res.new_text, self.text)

    def test_line_without_sign_is_refused(self) -> None:
        self.assertIsNotNone(apply_edit(self.text, "AAPL 1 1", self.engine).error)

    def test_51st_position_via_edit_is_refused(self) -> None:
        """S-7 на пути правки."""
        text = "\n".join(f"T{i:03d} 1 10 USD" for i in range(50))
        res = apply_edit(text, "+NEWT 1 10 USD", self.engine)
        self.assertIn("50", res.error)
        ok = apply_edit(text, "+T000 1 10 USD", self.engine)
        self.assertTrue(ok.ok, "докупка существующей бумаги лимит не трогает")

    def test_remove_at_checks_version_and_bounds(self) -> None:
        """S-5: индекс — в границах и с хэшем версии."""
        tag = version_tag(self.text)
        self.assertIsNotNone(remove_at(self.text, 999, tag, self.engine).error)
        self.assertIsNotNone(remove_at(self.text, -1, tag, self.engine).error)
        self.assertIsNotNone(remove_at(self.text, 0, "deadbeef", self.engine).error)
        ok = remove_at(self.text, 0, tag, self.engine)
        self.assertTrue(ok.ok)
        self.assertNotIn("AAPL", ok.new_text)


class LayerTest(unittest.TestCase):
    """Пакет — L1: не знает `tg_bot`, vault и БД (ключи и текст даёт L4)."""

    def test_package_imports_stay_low(self) -> None:
        import ast

        forbidden = {"tg_bot", "db_tokenomics", "aiogram", "entrypoint"}
        for path in (_SRC / "portfolio_aggregation").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                for mod in mods:
                    with self.subTest(file=path.name, mod=mod):
                        self.assertNotIn(mod.split(".")[0], forbidden)
                        self.assertNotEqual(mod, "finance.security")


if __name__ == "__main__":
    unittest.main()


# ── аудит §−123: найденные при перепроверке дефекты ──────────────────────────

class AuditFindingsTest(unittest.TestCase):
    """Каждый тест — дефект, найденный двойной проверкой гибрида."""

    def test_unpriced_manual_position_is_named(self) -> None:
        """Ручная бумага без ряда у провайдера и без цены брокера выпадала из
        отчёта МОЛЧА: ни в таблице, ни в `dropped_rows`, ни у чекеров."""
        sys.path.insert(0, str(_SRC.parent / "tests"))
        import golden_support as gs
        from portfolio_aggregation import unpriced_positions

        text = "AAPL 20 160.0\nTLT 120 95.0\nZZZQ 10 100 USD\nCASH:USD 1500\n"
        with patch.object(gs, "AGGREGATED_MANUAL_TEXT", text):
            results = gs.run_analyze_all("aggregated")
            from finance.investment_logic import UniversalPortfolioManager
            frame, _comp = gs.build_aggregated_frame(UniversalPortfolioManager().engine)
        perf = set(results["performance_table"]["Ticker"])
        self.assertNotIn("ZZZQ", perf, "движок не изменён: бумага без цены не оценивается")
        self.assertEqual(unpriced_positions(frame, results), ["ZZZQ"])

    def test_unpriced_has_no_false_positives(self) -> None:
        """Прокси, молодой листинг, отброшенная строка, дубли — не «выпавшие»."""
        sys.path.insert(0, str(_SRC.parent / "tests"))
        import golden_support as gs
        from portfolio_aggregation import unpriced_positions

        results = gs.run_analyze_all("base")
        self.assertEqual(unpriced_positions(pd.DataFrame(gs.PORTFOLIO_ROWS), results), [])

    def test_unpriced_note_in_cove(self) -> None:
        from finance.data_lineage import _aggregated_source_status, _manual_source_status

        for fn, src in ((_aggregated_source_status, "aggregated"),
                        (_manual_source_status, "manual")):
            row = fn({"portfolio_source": src, "unpriced_positions": ["ZZZQ"]})
            self.assertIn("не вошли в расчёт (нет рыночной цены): ZZZQ", row["note"])

    def test_one_counting_rule_for_the_limit(self) -> None:
        """50 бумаг + кэш: правки пропускают — значит и источник обязан."""
        from portfolio_aggregation import count_positions

        engine = _engine()
        text = "\n".join(f"T{i:03d} 1 10 USD" for i in range(50))
        text += "\nCASH:USD 100\nCASH:KZT 1000"
        with patch.dict(os.environ, {"MANUAL_MAX_POSITIONS": ""}):
            res = ManualSource(text, engine).load()
        self.assertTrue(res.ok, res.failure_reason)
        self.assertEqual(count_positions(entries_of(text, engine)), 50)
        self.assertIsNotNone(apply_edit(text, "+NEWT 1 10 USD", engine).error)
        self.assertTrue(apply_edit(text, "+CASH:EUR 5", engine).ok, "кэш лимит не трогает")

    def test_numbers_roundtrip_losslessly(self) -> None:
        """`:.10f` срезал знаки: 0.123456789012 → 0.123456789, 1e-11 → 0."""
        from portfolio_aggregation.edits import fmt_amount

        engine = _engine()
        for qty in (0.123456789012, 1e-11, 155.45454545454547, 2500000.0):
            text = f"BTC-USD {fmt_amount(qty)} 30000 USD"
            entries = entries_of(text, engine)
            self.assertEqual(len(entries), 1, text)
            self.assertEqual(entries[0].quantity, qty)
        vwap = apply_edit("AAPL.US 3 100 USD", "+AAPL 7 101", engine)
        aapl = entries_of(vwap.new_text, engine)[0]
        self.assertEqual(aapl.price, (3 * 100 + 7 * 101) / 10)

    def test_dash_variants_are_minus(self) -> None:
        engine = _engine()
        for dash in ("–", "—", "−"):
            res = apply_edit("AAPL.US 3 100 USD", f"{dash}AAPL", engine)
            self.assertTrue(res.ok, dash)
            self.assertEqual(res.new_text, "")

    def test_taxonomy_uses_canonical_not_user_input(self) -> None:
        rows = [dict(BROKER_ROWS[0], Asset_Class="EQUITY", Asset_Class_Label="Акции")]
        res = PortfolioAggregator().merge(_live_freedom(rows), _manual("TLT 5 90"))
        tlt = res.frame[res.frame["Ticker"] == "TLT"].iloc[0]
        self.assertEqual(tlt["Asset_Class"], "FIXED_INCOME")
