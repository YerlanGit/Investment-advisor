"""`§−125`: ручной отчёт без котировок — ОДИН внятный отказ, а не два ложных.

Живой дефект (2026-10-02, первый прогон ручного портфеля в проде): бот ответил

    ❌ Шаг 2 не удался: движок риск-анализа упал. Код ошибки …
    ⚠️ Анализ невозможен … Проверьте подключение к брокеру …

Ни одной бумаги ручного портфеля не нашлось в базе котировок, движок выбросил
все строки (цены брокера у ручной строки нет) и упёрся в гард «стоимость = 0».
Оба сообщения лгали: движок не падал, брокера у ручного портфеля нет. Причину
провайдер знал по каждому тикеру — она уходила только в лог.

Что охраняется
--------------
* шаг 1 ручного отчёта без единой котировки останавливается ДО движка и
  называет тикеры с причиной провайдера; текст исключения базы наружу не идёт;
* брокерский путь этой проверкой не затронут (I-9);
* штатный отказ движка на шаге 2 — одно сообщение, без «движок упал»;
* экран ручного портфеля предупреждает о бумагах без котировок сразу.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from test_bot_navigation import assert_markdown_balanced  # noqa: E402
from test_broker_fallback import FallbackTestBase, _BotRecorder  # noqa: E402

_ENGINE_CRASH = "движок риск-анализа упал"


def _preview(*, loaded=3, portfolio=("KSPI.KZ", "XYZ.US"), portfolio_loaded=0,
             failed=None):
    portfolio = list(portfolio)
    return SimpleNamespace(
        data=pd.DataFrame({"SPY.US": [1.0, 1.1]}) if loaded else pd.DataFrame(),
        risky_tickers=list(portfolio), resolved_portfolio=list(portfolio),
        history_result=SimpleNamespace(failed=dict(failed or {}), retried=[]),
        loaded_count=loaded, internal_tickers={"SPY.US"},
        portfolio_loaded=portfolio_loaded, portfolio_total=len(portfolio),
        proxy_map={})


class _Base(FallbackTestBase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        for p in self._extra:
            p.stop()
        self._extra = []
        self.analyzed: list[str] = []

    async def _run(self, preview, *, source="manual", analyze=None) -> _BotRecorder:
        analyzed = self.analyzed

        class _Manager:
            def __init__(self, price_source="freedom") -> None:
                self.engine = SimpleNamespace(price_source=price_source)

            def prefetch_market_data(self, _candidates):
                return preview

            def analyze_all(self, df, profile_benchmark=None, risk_mandate=None):
                analyzed.append(self.engine.price_source)
                if analyze is not None:
                    return analyze()
                raise RuntimeError("стоп после шага 2")

        bot = _BotRecorder()
        df = pd.DataFrame({"Ticker": ["KSPI", "XYZ"], "Quantity": [1, 2]})
        with patch("finance.investment_logic.UniversalPortfolioManager", _Manager), \
             patch.object(self.tg, "UniversalPortfolioManager", _Manager):
            await self.tg._run_analysis_background(
                bot=bot, chat_id=self.USER_ID, user_id=self.USER_ID, tier="base",
                cost=1, df=df, bench_tick=None, source=source)
        return bot


class ManualWithoutQuotesTest(_Base):

    async def test_refused_before_engine_with_reasons(self) -> None:
        bot = await self._run(_preview(failed={
            "KSPI.KZ": "нет в базе котировок Stooq",
            "XYZ.US": "нет торгов 9 торговых дн. подряд (последний бар 2026-09-20)",
        }))
        texts = [t for t, _ in bot.sent]
        refusal = next(t for t in texts if "нет котировок" in t)
        self.assertIn("KSPI.KZ", refusal)
        self.assertIn("нет в базе котировок Stooq", refusal)
        self.assertIn("нет торгов 9 торговых дн.", refusal)
        assert_markdown_balanced(self, refusal)
        self.assertEqual(self.analyzed, [], "движок не должен запускаться")
        joined = "\n".join(texts)
        self.assertNotIn(_ENGINE_CRASH, joined)
        self.assertNotIn("Freedom API", joined)
        self.assertNotIn("подключение к брокеру", joined)
        self.assertIn("не списан", joined)
        datas = [b.callback_data for _t, kw in bot.sent if kw.get("reply_markup")
                 for row in kw["reply_markup"].inline_keyboard for b in row]
        self.assertIn("mp:show", datas)
        self.assertTrue(await self._slot_is_free())

    async def test_base_down_hides_exception_text(self) -> None:
        bot = await self._run(_preview(loaded=0, failed={
            "KSPI.KZ": "база котировок недоступна: /mnt/state/quotes.sqlite missing",
        }))
        joined = "\n".join(t for t, _ in bot.sent)
        self.assertIn("база котировок для ручного портфеля", joined)
        self.assertNotIn("/mnt", joined)
        self.assertNotIn("Freedom API", joined)
        self.assertEqual(self.analyzed, [])

    async def test_partial_quotes_go_to_engine(self) -> None:
        await self._run(_preview(portfolio_loaded=1))
        self.assertEqual(self.analyzed, ["manual"])

    async def test_broker_path_untouched(self) -> None:
        """I-9: у брокера есть своя цена строки — проверка его не касается."""
        await self._run(_preview(portfolio_loaded=0), source="freedom")
        self.assertEqual(self.analyzed, ["freedom"])


class EngineRefusalIsOneMessageTest(_Base):

    async def test_real_portfolio_required_is_not_a_crash(self) -> None:
        from finance.broker_api import RealPortfolioRequired

        def _refuse():
            raise RealPortfolioRequired("Стоимость портфеля = 0.")

        bot = await self._run(_preview(portfolio_loaded=2), analyze=_refuse)
        joined = "\n".join(t for t, _ in bot.sent)
        self.assertNotIn(_ENGINE_CRASH, joined)
        self.assertNotIn("Код ошибки", joined)
        self.assertIn("Анализ невозможен", joined)
        self.assertTrue(await self._slot_is_free())

    async def test_real_crash_still_reports_support_id(self) -> None:
        bot = await self._run(_preview(portfolio_loaded=2))
        joined = "\n".join(t for t, _ in bot.sent)
        self.assertIn(_ENGINE_CRASH, joined)
        self.assertIn("Код ошибки", joined)


class NoQuotesTextTest(unittest.TestCase):

    def test_long_list_is_capped_and_balanced(self) -> None:
        from tg_bot import _manual_no_quotes_text

        tickers = [f"T{i}_X*" for i in range(14)]
        text = _manual_no_quotes_text(tickers, {}, base_down=False)
        self.assertIn("и ещё 4", text)
        self.assertIn("нет котировок", text)
        assert_markdown_balanced(self, text)


class ScreenCoverageNoteTest(unittest.TestCase):
    """Экран портфеля называет бумаги без котировок ДО отчёта."""

    def test_note_names_missing_ticker(self) -> None:
        import tg_bot

        lookup = lambda tickers, _days: {t: (1 if t.startswith("AAPL") else 0)  # noqa: E731
                                         for t in tickers}
        with patch("finance.manual_portfolio._default_coverage_lookup",
                   return_value=lookup):
            note = tg_bot._mp_coverage_note_sync("AAPL 10 150 USD\nZZZQ 5 20 USD")
        self.assertIn("ZZZQ", note)
        self.assertNotIn("AAPL", note)

    def test_failure_is_silent(self) -> None:
        import tg_bot

        with patch("finance.manual_portfolio._default_coverage_lookup",
                   side_effect=OSError("нет базы")):
            self.assertEqual(tg_bot._mp_coverage_note_sync("AAPL 10 150 USD"), "")

    def test_note_is_rendered_on_screen(self) -> None:
        import tg_bot

        entries = tg_bot._mp_entries_sync("AAPL 10 150 USD")
        screen = tg_bot._format_mp_screen(entries, None, "ℹ️ Ценовой истории нет по: X.")
        self.assertIn("Ценовой истории нет по", screen)
        assert_markdown_balanced(self, screen)


if __name__ == "__main__":
    unittest.main()
