"""Ручной ввод: ЧТО именно попадает в портфель (MP-05 · ЧК-05.0).

Раунд: `AUDIT §−72`, ревизия Фазы 04 перед Фазой 05
(`docs/roadmap/manual_portfolio/PHASE_05_FSM_FLOW.md`).

Три находки ревизии, каждая — молчаливая
----------------------------------------
**1. Прокси подменял САМУ БУМАГУ.** Парсер брал `Ticker` из
`resolve_tickers`, а тот отдаёт факторный заменитель неликвидной бумаги.
AIX-нота пользователя становилась `VWOB`: цена — рыночная цена ETF вместо
цены покупки, `Asset_Type` — «ETF» вместо облигации, а две РАЗНЫЕ ноты с одним
прокси склеивались движком в ОДНУ позицию (склейка идёт по `Ticker`).
Это возврат дефекта `§−38` через слой ввода и прямое нарушение инварианта
«прокси влияет ТОЛЬКО на риск» (`src/finance/CLAUDE.md`).

**2. Маржинальный заём отвергался.** `CASH:USD -5000` — не короткая продажа, а
отрицательный остаток по счёту; брокерский путь отдаёт такие строки штатно
(`PHASE_04 §5a.2`). Пока парсер их запрещал, ручной ввод не мог описать книгу,
которую freedom-путь считает без оговорок, — то есть I-1 ломался на первом же
портфеле с плечом.

**3. Разряды в строке кэша.** `CASH:KZT 500 000` отвергался сообщением про
цену, тогда как `CASH:KZT 2 500 000` разбирался: у первой арность 3 (легальная
для кэша), у второй — 4, и только вторая попадала в склейку разрядов. Тенговый
остаток пишут с разрядами всегда — на нём флоу и спотыкался бы.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from finance.investment_logic import MAC3RiskEngine  # noqa: E402
from finance.manual_portfolio import parse_portfolio_text  # noqa: E402

#: Неликвидная бумага, у которой ЕСТЬ прокси. Берётся не константой: тест
#: обязан ловить и запись, добавленную в `INSTRUMENT_PROXY_MAP` завтра.
PROXIED_TICKER = "FFSPC6.1028.AIX"


class CanonicalTickerTest(unittest.TestCase):
    """Нормализация имени и подмена бумаги — РАЗНЫЕ вопросы к движку."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MAC3RiskEngine()

    def test_proxy_is_visible_only_through_resolve(self) -> None:
        """Один вход, два разных ответа — и это не противоречие, а слой."""
        canon = self.engine.canonical_ticker(PROXIED_TICKER)
        resolved = self.engine.resolve_tickers([PROXIED_TICKER])
        self.assertEqual(canon, PROXIED_TICKER,
                         "canonical_ticker обязан вернуть БУМАГУ, а не её прокси")
        self.assertEqual(resolved, [self.engine.proxy_for(PROXIED_TICKER)])
        self.assertNotEqual(canon, resolved[0])

    def test_cash_has_no_ticker(self) -> None:
        """У валюты нет тикера — `None`, а не строка-подделка."""
        for ccy in self.engine.NON_RISK_ASSETS:
            with self.subTest(ccy=ccy):
                self.assertIsNone(self.engine.canonical_ticker(ccy))

    def test_normalisation_matches_resolve_where_no_proxy(self) -> None:
        """Там, где подмены нет, обе функции обязаны совпадать ДОСЛОВНО.

        Расхождение здесь означало бы два набора правил нормализации — ровно
        ловушка Т-4, ради которой шаги и вынесены в общий код.
        """
        for raw in ("AAPL", "AAPL.US", "каспи", "KSPI", "BTC", "BTC-USD",
                    "BRK.B", "TLT", "KMG", "BITCOIN"):
            with self.subTest(raw=raw):
                canon = self.engine.canonical_ticker(raw)
                self.assertIsNone(self.engine.proxy_for(raw.upper()),
                                  "тест построен на бумагах БЕЗ прокси")
                self.assertEqual([canon], self.engine.resolve_tickers([raw]))

    def test_bond_map_key_still_reaches_its_proxy(self) -> None:
        """Регресс разреза: `proxy_for` спрашивается на ИМЕНИ, не на формате.

        Ключи `BOND_CLASSIFICATION_MAP` — голые имена (`US_TREASURY`). Если
        спросить прокси уже после дописывания `.US`, точное совпадение
        пропадёт, и `US_TREASURY` МОЛЧА поедет в фактор-модель сам собой
        вместо `BIL.US` — без истории и без единого признака ошибки.
        """
        for name, proxy in (("US_TREASURY", "BIL.US"),
                            ("KZ_GOV_BOND", "VWOB.US"),
                            ("KASPI_BOND", "VWOB.US")):
            with self.subTest(name=name):
                self.assertEqual(self.engine.resolve_tickers([name]), [proxy])


class ProxyNeverBecomesThePositionTest(unittest.TestCase):
    """🔴 Главная находка ревизии: бумага пользователя обязана остаться собой."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MAC3RiskEngine()
        cls.report = parse_portfolio_text(f"{PROXIED_TICKER} 10 103.5\n", cls.engine)

    def test_position_keeps_the_real_security(self) -> None:
        pos = self.report.valid[0]
        proxy = self.engine.proxy_for(PROXIED_TICKER)
        self.assertNotIn(str(proxy).split(".")[0], pos.ticker,
                         "в портфель уехал ПРОКСИ: нота исчезла, а вместе с ней "
                         "её цена покупки и её тип инструмента")
        self.assertTrue(PROXIED_TICKER.startswith(pos.ticker),
                        f"{pos.ticker!r} не выводится из {PROXIED_TICKER!r}")

    def test_substitution_is_disclosed_separately_from_synonyms(self) -> None:
        """Подмена бумаги и синоним — разные факты, и списки разные (H-1)."""
        self.assertEqual(self.report.proxied,
                         [(PROXIED_TICKER, self.engine.proxy_for(PROXIED_TICKER))])
        self.assertFalse(self.report.auto_resolved,
                         "подмена прокси попала в список синонимов — читатель "
                         "решит, что бумагу просто переименовали")

    def test_two_different_notes_stay_two_positions(self) -> None:
        """Склейка идёт по `Ticker`: общий прокси слепил бы разные бумаги.

        Замер до правки: обе строки получали `Ticker='VWOB'`, движок склеивал
        их в одну позицию со средневзвешенной ценой ДВУХ РАЗНЫХ нот.
        """
        report = parse_portfolio_text(
            "FFSPC6.1028.AIX 10 103.5\nFFSPC7.0515.AIX 20 98.0\n", self.engine)
        self.assertFalse(report.failed)
        tickers = {p.ticker for p in report.valid}
        self.assertEqual(len(tickers), 2, f"две ноты схлопнулись в {tickers}")

    def test_synonym_is_not_reported_as_a_substitution(self) -> None:
        """Обратная сторона: `KSPI` → `KSPI.KZ` подменой НЕ является."""
        report = parse_portfolio_text("каспи 100 104,5 KZT\n", self.engine)
        self.assertFalse(report.proxied)
        self.assertEqual(report.auto_resolved, [("КАСПИ", "KSPI.KZ")])

    def test_pure_formatting_is_not_reported_at_all(self) -> None:
        """`AAPL.US` → `AAPL` — не находка, а форма записи.

        Пара «AAPL.US → AAPL.US» в списке «мы поняли это так» — чистый шум:
        узнать из неё нечего, а внимание к настоящим синонимам она размывает.
        """
        for raw in ("AAPL", "AAPL.US", "KSPI"):
            with self.subTest(raw=raw):
                report = parse_portfolio_text(f"{raw} 10 150\n", self.engine)
                self.assertFalse(report.auto_resolved)
                self.assertFalse(report.proxied)


class MarginLoanIsNotAShortSaleTest(unittest.TestCase):
    """Отрицательный КЭШ — заём; отрицательное КОЛИЧЕСТВО — короткая продажа."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MAC3RiskEngine()

    def test_negative_cash_is_accepted(self) -> None:
        report = parse_portfolio_text("AAPL 10 150\nCASH:USD -5000\n", self.engine)
        self.assertFalse(report.failed, f"заём отвергнут: {report.failed}")
        cash = [p for p in report.valid if p.is_cash]
        self.assertEqual(len(cash), 1)
        self.assertEqual(cash[0].quantity, -5000.0)
        self.assertEqual(cash[0].price, 1.0)

    def test_negative_cash_survives_thousand_separators(self) -> None:
        report = parse_portfolio_text("CASH:KZT -2 500 000\n", self.engine)
        self.assertFalse(report.failed)
        self.assertEqual(report.valid[0].quantity, -2_500_000.0)

    def test_short_sale_is_still_refused(self) -> None:
        """Отрицательный ВЕС молча ломает Euler и composite_risk_score."""
        report = parse_portfolio_text("AAPL -10 150\n", self.engine)
        self.assertFalse(report.valid)
        self.assertIn("корот", report.failed[0][1].lower())

    def test_zero_cash_is_refused(self) -> None:
        """Ноль — не заём и не остаток: строка ничего не описывает."""
        report = parse_portfolio_text("CASH:USD 0\n", self.engine)
        self.assertFalse(report.valid)
        self.assertTrue(report.failed)

    def test_leveraged_book_reaches_the_frame(self) -> None:
        """Сквозная проверка: заём доезжает до контрактного фрейма."""
        report = parse_portfolio_text("AAPL 100 150\nCASH:USD -5000\n", self.engine)
        df = report.to_dataframe()
        cash_row = df[df["Asset_Type"] == "Кэш"].iloc[0]
        self.assertLess(float(cash_row["Quantity"]), 0.0)
        self.assertEqual(float(cash_row["Broker_Current_Price"]), 1.0)


class CashThousandSeparatorsTest(unittest.TestCase):
    """Разряды в остатке — норма для тенге, а не повод отвергнуть строку."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MAC3RiskEngine()

    def test_all_magnitudes_parse_the_same_way(self) -> None:
        """Арность строки не должна решать, понял её парсер или нет."""
        for text, expected in (("CASH:KZT 500 000", 500_000.0),
                               ("CASH:KZT 2 500 000", 2_500_000.0),
                               ("CASH:USD 5 000", 5_000.0),
                               ("USD 5 000", 5_000.0),
                               ("CASH:KZT 500000", 500_000.0)):
            with self.subTest(text=text):
                report = parse_portfolio_text(text, self.engine)
                self.assertFalse(report.failed, f"{text}: {report.failed}")
                self.assertEqual(report.valid[0].quantity, expected)

    def test_redundant_unit_price_is_still_accepted(self) -> None:
        """Третье поле «1.0» — избыточно, но верно; отвергать его не за что."""
        report = parse_portfolio_text("CASH:USD 5000 1.0\n", self.engine)
        self.assertFalse(report.failed)
        self.assertEqual(report.valid[0].quantity, 5000.0)

    def test_wrong_unit_price_is_still_refused(self) -> None:
        """Цена кэша ≠ 1.0 — по-прежнему ошибка строки с объяснением."""
        report = parse_portfolio_text("CASH:USD 5000 1.5\n", self.engine)
        self.assertFalse(report.valid)
        self.assertIn("1.0", report.failed[0][1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
