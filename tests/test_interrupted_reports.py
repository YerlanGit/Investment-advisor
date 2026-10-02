"""`§−126`: отчёт, убитый редеплоем, не оставляет «⏳ Шаг 4/4» навсегда.

Живой случай (2026-10-02): PR смержен в 15:39, Cloud Build (build → test →
push → deploy) выкатил ревизию в окне 15:52–15:59 — а в 15:52 владелец запустил
DEEP по ручному портфелю. Cloud Run шлёт старому инстансу SIGTERM и через 10 с
SIGKILL; бот закрывал сессию Telegram (жёсткое правило против 409) и снимал
аренды, но владельцу расчёта не писал ничего. Последним сообщением в чате
оставалось «⏳ Шаг 4/4».

Что охраняется
--------------
* расчёт регистрируется в `_INFLIGHT_REPORTS` на всё время и снимается и при
  доставке, и при любом отказе;
* `notify_interrupted_reports` пишет КАЖДОМУ владельцу незавершённого расчёта;
  сбой одного адресата не мешает другим; токен не списан — так и сказано;
* на остановке уведомление идёт ПОСЛЕ закрытия основной сессии и через СВОЮ;
* худший случай ожидания модели — не больше 10 минут (обещание «5–10 минут»).
"""

from __future__ import annotations

import asyncio
import inspect
import os
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
from test_manual_no_quotes import _Base, _preview  # noqa: E402


class InflightRegistryTest(_Base):

    async def test_registered_while_running_and_cleared_after(self) -> None:
        seen: list[dict] = []

        def _analyze():
            seen.append(dict(self.tg._INFLIGHT_REPORTS))
            raise RuntimeError("стоп внутри шага 2")

        await self._run(_preview(portfolio_loaded=2), analyze=_analyze)
        self.assertEqual(seen, [{self.USER_ID: (self.USER_ID, "base")}])
        self.assertNotIn(self.USER_ID, self.tg._INFLIGHT_REPORTS)

    async def test_cleared_on_early_refusal(self) -> None:
        await self._run(_preview(portfolio_loaded=0))     # отказ шага 1
        self.assertNotIn(self.USER_ID, self.tg._INFLIGHT_REPORTS)


class NotifyInterruptedTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        import tg_bot
        self.tg = tg_bot
        self._saved = dict(tg_bot._INFLIGHT_REPORTS)
        tg_bot._INFLIGHT_REPORTS.clear()

    async def asyncTearDown(self) -> None:
        self.tg._INFLIGHT_REPORTS.clear()
        self.tg._INFLIGHT_REPORTS.update(self._saved)

    async def test_every_owner_is_told_even_if_one_send_fails(self) -> None:
        self.tg._INFLIGHT_REPORTS.update({1: (101, "deep"), 2: (202, "base"),
                                          3: (303, "base")})
        sent: list[tuple] = []

        async def _send(chat_id, text, kb):
            if chat_id == 202:
                raise RuntimeError("chat blocked")
            sent.append((chat_id, text, kb))

        delivered = await self.tg.notify_interrupted_reports(_send)
        self.assertEqual(delivered, 2)
        self.assertEqual(sorted(c for c, _t, _k in sent), [101, 303])
        text, kb = sent[0][1], sent[0][2]
        self.assertIn("прерван", text)
        self.assertIn("не списан", text)
        assert_markdown_balanced(self, text)
        datas = [b.callback_data for row in kb.inline_keyboard for b in row]
        self.assertIn("home:report", datas)

    async def test_nothing_running_sends_nothing(self) -> None:
        async def _send(*_a):
            raise AssertionError("слать некому")

        self.assertEqual(await self.tg.notify_interrupted_reports(_send), 0)


class ShutdownWiringTest(unittest.TestCase):

    def test_notice_goes_after_session_close_via_own_session(self) -> None:
        import tg_bot

        src = inspect.getsource(tg_bot.main)
        after_close = src.split("bot.session.close()", 1)[1]
        self.assertIn("notify_interrupted_reports(", after_close)
        self.assertIn("AiohttpSession(", after_close)


class ModelWaitIsBoundedTest(unittest.TestCase):

    def test_worst_case_fits_the_promise(self) -> None:
        import ai_narrative
        seen: dict = {}

        class _Client:
            def __init__(self, **kw):
                seen.update(kw)
                self.messages = SimpleNamespace(
                    create=lambda **_k: (_ for _ in ()).throw(RuntimeError("stub")))

        results = {"portfolio_metrics": {}, "total_value": 1.0,
                   "performance_table": pd.DataFrame([{"Ticker": "AAPL"}])}
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}), \
             patch("anthropic.Anthropic", _Client):
            ai_narrative.generate_narrative(results, tier="deep")
        self.assertEqual(seen.get("max_retries"), ai_narrative.ANTHROPIC_MAX_RETRIES)
        worst = seen["timeout"] * (seen["max_retries"] + 1)
        self.assertLessEqual(worst, 600.0, "шаг 4 обещан в 5–10 минут")


if __name__ == "__main__":
    unittest.main()
