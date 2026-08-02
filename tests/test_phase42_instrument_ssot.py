"""A-5 · SSOT классификации инструментов.

В проекте жили ШЕСТЬ независимых копий «какой это инструмент», и они
разошлись. Замерено ДО правки:

    ticker   display   mandate   Asset_Type  sec_skip  credit_na
    VWOB     Прочее    Stocks_US Акция       False     False      ← прокси §−38!
    VWO      Акции США Stocks_US Акция       False     False      ← прокси §−38!
    SHV      Акции США Stocks_US ETF         False     False
    VCIT     Прочее    Stocks_US Акция       False     False
    EMB      Облигации Bonds     Акция       False     False
    GOVT     Прочее    Stocks_US Акция       False     True
    FFSPC…AIX Акции KZ Bonds     Структ.нота True      False      ← противоречие

Что здесь зафиксировано:
  1. Перечни тикеров лежат В ОДНОМ месте (`finance.asset_taxonomy`), и вторая
     копия в любом из шести модулей роняет тест.
  2. Согласованность там, где вопрос ОДИН: облигационный ETF обязан быть
     облигацией и для подписи, и для лимита мандата, и для AssetClass.
  3. РАЗЛИЧИЯ, которые обязаны СОХРАНИТЬСЯ: `TLT` для лимитов «Bonds»
     (экономическая экспозиция), а для колонки `Asset_Type` — «ETF» (форма
     инструмента); C-пиллар неприменим к суверенному долгу DM, но применим к
     `LQD`/`HYG`/`EMB` — у них кредитный риск реален.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance import asset_taxonomy as tx
from finance.asset_taxonomy import AssetClass, from_freedom_metadata
from finance.scoring import classify_asset_class
from finance.broker_api import _classify_instrument
from finance.sec_edgar import _should_skip
from finance.scoring_orchestrator import _is_credit_not_applicable
from agent.gatekeeper import _classify_to_asset_key


# ──────────────────────────────────────────────────────────────────────────
# 1. Единственность источника
# ──────────────────────────────────────────────────────────────────────────
class NoSecondCopyTest(unittest.TestCase):
    """Перечень тикеров живёт ровно в одном файле."""

    _CONSUMERS = (
        "src/finance/scoring.py",
        "src/agent/gatekeeper.py",
        "src/finance/broker_api.py",
        "src/finance/sec_edgar.py",
        "src/finance/scoring_orchestrator.py",
    )

    #: Тикеры-маркеры: если они появились в потребителе литералом — значит
    #: кто-то завёл ВТОРУЮ копию перечня вместо пополнения SSOT.
    _MARKERS = ("TLT", "AGG", "HYG", "SOXX", "PDBC", "MTUM")

    def test_consumers_do_not_relist_tickers(self):
        for rel in self._CONSUMERS:
            src = Path(rel).read_text(encoding="utf-8")
            # Комментарии не считаем — там тикеры упоминаются как примеры.
            code = "\n".join(ln for ln in src.splitlines()
                             if not ln.lstrip().startswith("#"))
            for m in self._MARKERS:
                self.assertNotIn(f'"{m}"', code,
                                 f"{rel}: тикер {m} перечислен литералом — "
                                 f"пополняй finance/asset_taxonomy.py")

    def test_every_consumer_imports_the_ssot(self):
        for rel in self._CONSUMERS:
            src = Path(rel).read_text(encoding="utf-8")
            self.assertIn("asset_taxonomy", src, f"{rel} не читает SSOT")

    def test_taxonomy_exports_the_sets(self):
        for name in ("BOND_ETFS", "COMMODITY_ETFS", "EQUITY_ETFS", "ETF_BASES",
                     "SOVEREIGN_BOND_ETFS", "CASH_CCYS", "CRYPTO_BASES",
                     "KZ_BLUE_CHIPS"):
            self.assertIn(name, tx.__all__, name)
            self.assertTrue(getattr(tx, name), f"{name} пуст")


# ──────────────────────────────────────────────────────────────────────────
# 2. Согласованность там, где вопрос один
# ──────────────────────────────────────────────────────────────────────────
class BondEtfsAgreeEverywhereTest(unittest.TestCase):
    """Каждый облигационный ETF — облигация для ВСЕХ, кто спрашивает про класс."""

    def test_display_label(self):
        for t in sorted(tx.BOND_ETFS):
            self.assertEqual(classify_asset_class(t), "Облигации", t)

    def test_mandate_limit_key(self):
        for t in sorted(tx.BOND_ETFS):
            self.assertEqual(_classify_to_asset_key(t), "Bonds", t)

    def test_asset_class_enum(self):
        for t in sorted(tx.BOND_ETFS):
            self.assertEqual(from_freedom_metadata(ticker=t),
                             AssetClass.FIXED_INCOME, t)

    def test_suffixed_form_agrees_with_bare(self):
        """Брокер присылает и «LQD», и «LQD.US» — ответ обязан совпадать."""
        for t in sorted(tx.BOND_ETFS | tx.EQUITY_ETFS | tx.COMMODITY_ETFS):
            self.assertEqual(classify_asset_class(t),
                             classify_asset_class(f"{t}.US"), t)
            self.assertEqual(_classify_to_asset_key(t),
                             _classify_to_asset_key(f"{t}.US"), t)


class ProxyTickersAreKnownTest(unittest.TestCase):
    """🔴 Прокси раунда §−38 приходят в модель каждым прогоном.

    До A-5 `VWOB`/`VWO` не знал НИ ОДИН из шести классификаторов: EM-облигация
    печаталась как «Прочее»/акция США и уходила в SEC за несуществующим 10-K.
    """

    def test_em_bond_proxy_is_a_bond(self):
        self.assertIn("VWOB", tx.BOND_ETFS)
        self.assertEqual(classify_asset_class("VWOB.US"), "Облигации")
        self.assertEqual(_classify_to_asset_key("VWOB.US"), "Bonds")

    def test_em_equity_proxy_is_an_equity_etf(self):
        self.assertIn("VWO", tx.BROAD_EQUITY_ETFS)
        self.assertEqual(_classify_to_asset_key("VWO.US"), "GlobalETFs")

    def test_neither_proxy_goes_to_sec(self):
        for t in ("VWOB", "VWOB.US", "VWO", "VWO.US"):
            self.assertTrue(_should_skip(t), t)

    def test_external_diversifiers_are_all_known(self):
        """`simulate.EXTERNAL_DIVERSIFIERS` покупаются планом — класс обязан быть.

        И более того: `asset_key`, объявленный в самой таблице диверсификаторов,
        обязан совпасть с тем, что вернёт мандатный классификатор — иначе
        Effect покупает бумагу «в облигации», а панель считает её акцией.
        """
        from finance.simulate import EXTERNAL_DIVERSIFIERS
        for profile, rows in EXTERNAL_DIVERSIFIERS.items():
            for row in rows:
                t = row["ticker"]
                self.assertIn(tx.ticker_base(t), tx.ETF_BASES,
                              f"{t} ({profile}) не в SSOT")
                self.assertEqual(_classify_to_asset_key(t), row["asset_key"],
                                 f"{t} ({profile}): таблица и классификатор "
                                 f"расходятся")

    def test_factor_tickers_are_all_known(self):
        """Факторные ETF движка тоже могут лежать в портфеле (см. §−38)."""
        from finance.investment_logic import MAC3RiskEngine
        for t in MAC3RiskEngine().factor_tickers.values():
            base = tx.ticker_base(t)
            if base.startswith("^"):
                continue                      # индексы, не инструменты
            self.assertTrue(base in tx.ETF_BASES or base in tx.BOND_ETFS,
                            f"факторный тикер {t} неизвестен SSOT")


class AixNoteIsConsistentTest(unittest.TestCase):
    """Живое противоречие: одна нота — «Облигации» и «Акции KZ» одновременно."""

    NOTE = "FFSPC6.1028.AIX"

    def test_display_now_agrees_with_the_mandate(self):
        self.assertEqual(_classify_to_asset_key(self.NOTE), "Bonds")
        self.assertEqual(classify_asset_class(self.NOTE), "Облигации")

    def test_structured_nature_is_not_lost(self):
        """Колонка `Asset_Type` по-прежнему называет её нотой, а не облигацией."""
        self.assertEqual(_classify_instrument(self.NOTE), "Структ.нота")
        self.assertEqual(from_freedom_metadata(ticker=self.NOTE),
                         AssetClass.STRUCTURED)

    def test_risk_proxy_agrees_it_is_debt(self):
        """Ответ согласован с §−38: прокси ноты — EM govt bond."""
        from finance.investment_logic import MAC3RiskEngine
        self.assertEqual(MAC3RiskEngine.INSTRUMENT_PROXY_MAP["AIX_SPC"],
                         "VWOB.US")


# ──────────────────────────────────────────────────────────────────────────
# 3. Различия, которые ОБЯЗАНЫ сохраниться
# ──────────────────────────────────────────────────────────────────────────
class DeliberateDivergenceTest(unittest.TestCase):
    """Шесть потребителей отвечают на РАЗНЫЕ вопросы — слить их нельзя."""

    def test_bond_etf_is_bonds_for_mandate_but_etf_for_asset_type(self):
        self.assertEqual(_classify_to_asset_key("TLT"), "Bonds")     # экспозиция
        self.assertEqual(_classify_instrument("TLT"), "ETF")         # форма

    def test_credit_pillar_off_for_sovereign_on_for_credit_risk(self):
        for t in ("TLT", "IEF", "SHY", "AGG", "BND", "BIL", "GOVT"):
            self.assertTrue(_is_credit_not_applicable(t, None), t)
        # …а здесь кредитный риск реален и пиллар обязан работать
        for t in ("LQD", "HYG", "EMB", "VWOB"):
            self.assertFalse(_is_credit_not_applicable(t, None), t)

    def test_em_debt_is_not_treated_as_risk_free(self):
        """Та же логика, что в §−38: EM-долг ≠ US Treasuries."""
        self.assertNotIn("EMB", tx.SOVEREIGN_BOND_ETFS)
        self.assertNotIn("VWOB", tx.SOVEREIGN_BOND_ETFS)

    def test_commodity_etf_keeps_its_own_label(self):
        self.assertEqual(classify_asset_class("GLD"), "Сырьё")
        self.assertEqual(_classify_to_asset_key("GLD"), "Commodities")
        self.assertEqual(_classify_instrument("GLD"), "ETF")


# ──────────────────────────────────────────────────────────────────────────
# 4. Предикаты и границы
# ──────────────────────────────────────────────────────────────────────────
class PredicateTest(unittest.TestCase):

    def test_ticker_base_and_suffix(self):
        self.assertEqual(tx.ticker_base("KMGZ.KZ"), "KMGZ")
        self.assertEqual(tx.ticker_suffix("KMGZ.KZ"), "KZ")
        self.assertEqual(tx.ticker_base("AAPL"), "AAPL")
        self.assertEqual(tx.ticker_suffix("AAPL"), "")
        self.assertEqual(tx.ticker_base(" aapl "), "AAPL")

    def test_empty_and_none_are_safe(self):
        for bad in ("", None, "   "):
            self.assertFalse(tx.is_bond_like(bad))
            self.assertFalse(tx.is_crypto(bad))
            self.assertFalse(tx.is_etf(bad))
            self.assertEqual(tx.ticker_base(bad), "")

    def test_cash_covers_every_reported_currency(self):
        for c in ("USD", "EUR", "RUB", "RUR", "KZT", "GBP", "CHF", "CNY",
                  "JPY", "CASH"):
            self.assertTrue(tx.is_cash(c), c)
            self.assertEqual(_classify_to_asset_key(c), "Cash", c)

    def test_no_substring_false_positive_on_bond(self):
        """«ABOND»-подобный тикер не должен ловиться подстрокой невпопад.

        Историческое поведение СОХРАНЕНО: маркер «BOND» ищется в базе тикера,
        поэтому явный `USBOND` — облигация, а обычная акция — нет.
        """
        self.assertFalse(tx.is_bond_like("AAPL"))
        self.assertFalse(tx.is_bond_like("MSFT"))
        self.assertTrue(tx.is_bond_like("USBOND"))

    def test_isin_prefixes_still_recognised(self):
        for t in ("KZ2P0Y10E827", "KZ1P00000123", "XS1234567890",
                  "US912828XX12"):
            self.assertTrue(tx.is_bond_like(t), t)

    def test_kz_listed(self):
        for t in ("KSPI.KZ", "KSPI", "HSBK.KZ", "SOMETHING.AIX", "TEVA.IL"):
            self.assertTrue(tx.is_kz_listed(t), t)
        self.assertFalse(tx.is_kz_listed("AAPL.US"))

    def test_etf_membership_is_the_union(self):
        self.assertEqual(tx.ETF_BASES,
                         tx.EQUITY_ETFS | tx.BOND_ETFS | tx.COMMODITY_ETFS)
        self.assertEqual(tx.EQUITY_ETFS,
                         tx.BROAD_EQUITY_ETFS | tx.FACTOR_ETFS | tx.SECTOR_ETFS)

    def test_sets_do_not_overlap_across_classes(self):
        """Один тикер не может быть одновременно облигацией и сырьём."""
        pairs = [("BOND_ETFS", "COMMODITY_ETFS"),
                 ("BOND_ETFS", "EQUITY_ETFS"),
                 ("COMMODITY_ETFS", "EQUITY_ETFS"),
                 ("CRYPTO_BASES", "ETF_BASES"),
                 ("CASH_CCYS", "ETF_BASES")]
        for a, b in pairs:
            overlap = getattr(tx, a) & getattr(tx, b)
            self.assertEqual(overlap, frozenset(), f"{a} ∩ {b} = {overlap}")


class SecSkipTest(unittest.TestCase):
    """Нет 10-K — нет запроса в SEC (лишняя сеть + пустой фундаментал)."""

    def test_every_etf_is_skipped(self):
        for t in sorted(tx.ETF_BASES):
            self.assertTrue(_should_skip(t), t)

    def test_crypto_cash_and_kz_are_skipped(self):
        for t in ("BTC-USD", "ETH", "USD", "KZT", "KSPI", "KSPI.KZ",
                  "FFSPC6.1028.AIX"):
            self.assertTrue(_should_skip(t), t)

    def test_real_us_equities_are_not_skipped(self):
        for t in ("AAPL", "MSFT", "NVDA", "JPM", "XOM", "AAPL.US"):
            self.assertFalse(_should_skip(t), t)


if __name__ == "__main__":
    unittest.main()
