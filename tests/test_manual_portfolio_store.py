"""Гибрид PR-1: постоянный ручной портфель — хранение, правки, удаление.

Задание `pico_prompt_hybrid_architecture_fallback_v2` (D-3, D-4), таблица §I.5:
S-4 (флаг на каждом нажатии), S-5 (callback_data — недоверенный ввод),
S-6 (правка под слотом), S-7 (лимит позиций), S-8 (`_md_safe`),
S-9 (шифрование at rest + `/forget_portfolio`).

Почему это отдельный портфель, а не черновик: черновик удаляется после
доставленного отчёта, а fallback-отчёт (PR-2) нужен ровно тогда, когда
брокер недоступен — и портфель к этому моменту обязан существовать.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from cryptography.fernet import Fernet  # noqa: E402

from test_manual_fsm_flow import (  # noqa: E402
    PORTFOLIO_TEXT,
    ManualFlowTestBase,
    _FakeCallback,
    _FakeMessage,
)


class StoreTestBase(ManualFlowTestBase):

    async def asyncSetUp(self) -> None:
        self._orig_key = os.environ.get("FINTECH_MASTER_KEY")
        os.environ["FINTECH_MASTER_KEY"] = Fernet.generate_key().decode()
        await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        if self._orig_key is None:
            os.environ.pop("FINTECH_MASTER_KEY", None)
        else:
            os.environ["FINTECH_MASTER_KEY"] = self._orig_key

    async def _mp(self, data: str) -> _FakeCallback:
        cb = _FakeCallback(data, user_id=self.USER_ID)
        await self.tg.cb_manual_portfolio(cb, self.state)
        return cb

    async def _edit(self, text: str) -> _FakeMessage:
        msg = _FakeMessage(text, user_id=self.USER_ID)
        await self.state.set_state(self.tg.ManualPortfolio.Edit)
        await self.tg.msg_manual_edit(msg, self.state)
        return msg

    async def _stored(self) -> str:
        return str((await self.db.get_manual_portfolio(self.USER_ID) or {}).get("text") or "")


# ── хранилище ────────────────────────────────────────────────────────────────

class StorageTest(StoreTestBase):

    async def test_roundtrip_and_upsert(self) -> None:
        self.assertIsNone(await self.db.get_manual_portfolio(self.USER_ID))
        self.assertFalse(await self.db.has_manual_portfolio(self.USER_ID))
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 10 150 USD")
        await self.db.save_manual_portfolio(self.USER_ID, "TLT.US 5 90 USD")
        self.assertEqual(await self._stored(), "TLT.US 5 90 USD")
        self.assertTrue(await self.db.has_manual_portfolio(self.USER_ID))

    async def test_encrypted_at_rest(self) -> None:
        """S-9: в SQLite нет открытого тикера."""
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 10 150 USD")
        con = sqlite3.connect(self.db.DB_PATH)
        try:
            blob = con.execute("SELECT payload_enc FROM manual_portfolio").fetchone()[0]
        finally:
            con.close()
        self.assertNotIn(b"AAPL", bytes(blob))
        raw = Path(self.db.DB_PATH).read_bytes()
        self.assertNotIn(b"AAPL.US 10 150", raw)

    async def test_rotated_key_is_a_clean_error(self) -> None:
        from finance.security import MasterKeyRotatedError

        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 10 150 USD")
        os.environ["FINTECH_MASTER_KEY"] = Fernet.generate_key().decode()
        with self.assertRaises(MasterKeyRotatedError):
            await self.db.get_manual_portfolio(self.USER_ID)
        cb = await self._mp("mp:show")
        self.assertIn("недоступен", cb.message.all_text)

    async def test_size_limit(self) -> None:
        with self.assertRaises(self.db.ManualDraftTooLarge):
            await self.db.save_manual_portfolio(
                self.USER_ID, "A" * (self.db.MANUAL_DRAFT_MAX_BYTES + 1))

    async def test_report_delivery_keeps_the_portfolio(self) -> None:
        """Удаление черновика после отчёта НЕ трогает постоянный портфель."""
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 10 150 USD")
        await self.db.save_manual_draft(self.USER_ID, "AAPL 10 150")
        await self.db.delete_manual_draft(self.USER_ID)
        self.assertTrue(await self._stored())


# ── перенос на подтверждении ──────────────────────────────────────────────────

class ConfirmPersistsTest(StoreTestBase):

    async def test_confirm_moves_draft_to_portfolio(self) -> None:
        await self._send_text(PORTFOLIO_TEXT)
        cb = await self._tap("confirm")
        stored = await self._stored()
        self.assertIn("AAPL.US 10 150.5 USD", stored)
        self.assertIn("KSPI.KZ 200 45000 KZT", stored)
        self.assertIn("CASH:KZT 2500000", stored)
        self.assertIn("/portfolio", cb.message.all_text)
        self.assertIsNotNone(await self.db.get_manual_draft(self.USER_ID),
                             "черновик живёт до доставленного отчёта")

    async def test_confirm_survives_missing_master_key(self) -> None:
        """Хранилище — удобство: без ключа расчёт всё равно доступен."""
        os.environ.pop("FINTECH_MASTER_KEY", None)
        await self._send_text(PORTFOLIO_TEXT)
        cb = await self._tap("confirm")
        self.assertIn("Выберите тип анализа", cb.message.all_text)
        self.assertTrue(await self._slot_is_free())

    async def test_too_long_portfolio_is_not_stored(self) -> None:
        """S-7: больше `MANUAL_MAX_POSITIONS` — не сохраняется, расчёт не отнят."""
        text = "\n".join(f"T{i:03d} 1 10 USD" for i in range(51))
        await self._send_text(text)
        cb = await self._tap("confirm")
        self.assertEqual(await self._stored(), "")
        self.assertIn("пределе 50", cb.message.all_text)
        self.assertIn("Выберите тип анализа", cb.message.all_text)


# ── правки ───────────────────────────────────────────────────────────────────

class EditFlowTest(StoreTestBase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self.db.save_manual_portfolio(
            self.USER_ID, "AAPL.US 10 150 USD\nCASH:USD 3000")

    async def test_screen_lists_positions_and_buttons(self) -> None:
        msg = _FakeMessage("/portfolio", user_id=self.USER_ID)
        await self.tg.cmd_portfolio(msg, self.state)
        text, kwargs = msg.sent[-1]
        self.assertIn("AAPL", text)
        self.assertIn("CASH:USD", text)
        datas = [b.callback_data for row in kwargs["reply_markup"].inline_keyboard
                 for b in row]
        self.assertEqual(datas, ["mp:add", "mp:rmlist", "mp:del", "mp:back"])

    async def test_add_vwap_and_change_line(self) -> None:
        await self._mp("mp:add")
        self.assertEqual(await self.state.get_state(), self.tg.ManualPortfolio.Edit)
        msg = await self._edit("+AAPL 10 170")
        self.assertIn("AAPL.US 20 160 USD", await self._stored())
        self.assertIn("Что изменилось", msg.all_text)
        self.assertIn("средневзвешенная", msg.all_text)
        self.assertTrue(await self._slot_is_free())

    async def test_trim_and_remove(self) -> None:
        await self._edit("-AAPL 4")
        self.assertIn("AAPL.US 6 150 USD", await self._stored())
        msg = await self._edit("-AAPL 6")
        self.assertIn("-AAPL", msg.all_text)
        self.assertIn("AAPL.US 6 150 USD", await self._stored(), "отказ ничего не меняет")
        await self._edit("-AAPL")
        self.assertNotIn("AAPL", await self._stored())

    async def test_md_unsafe_ticker_is_sanitized(self) -> None:
        """S-8: тикер `A_B*C` в ответе не ломает Markdown."""
        msg = await self._edit("-A_B*C")
        self.assertTrue(msg.sent)
        for text, _kw in msg.sent:
            self.assertNotIn("A_B*C", text)
            self.assertNotIn("A_B", text)

    async def test_51st_position_is_refused(self) -> None:
        """S-7 на пути правки бота."""
        base = "\n".join(f"T{i:03d}.US 1 10 USD" for i in range(50))
        await self.db.save_manual_portfolio(self.USER_ID, base)
        msg = await self._edit("+NEWT 1 10 USD")
        self.assertIn("50", msg.all_text)
        self.assertNotIn("NEWT", await self._stored())

    async def test_concurrent_edits_are_serialized(self) -> None:
        """S-6: два одновременных `+AAPL` — одна правка, второй — «уже идёт»."""
        gate = asyncio.Event()
        real = self.tg._mp_apply_sync

        def _slow(text, ops):
            # Первый вызов держит слот, пока второй пытается войти.
            asyncio.run_coroutine_threadsafe(_noop(), loop).result()
            return real(text, ops)

        async def _noop():
            await gate.wait()

        loop = asyncio.get_running_loop()
        with patch.object(self.tg, "_mp_apply_sync", _slow):
            first = asyncio.ensure_future(self._edit("+AAPL 10 150"))
            await asyncio.sleep(0.05)
            second = await self._edit("+AAPL 10 150")
            gate.set()
            await first
        self.assertIn("уже идёт обработка", second.all_text)
        self.assertIn("AAPL.US 20 150 USD", await self._stored())
        self.assertTrue(await self._slot_is_free())

    async def test_logs_carry_no_positions(self) -> None:
        """S-3: в логах правки — счётчики, а не состав портфеля."""
        with self.assertLogs(level="INFO") as logs:          # корневой — все модули
            await self._edit("+MSFT 3 410")
        joined = "\n".join(logs.output)
        self.assertNotIn("MSFT", joined)
        self.assertNotIn("410", joined)


# ── кнопки удаления: версия, границы, подделка ────────────────────────────────

class RemoveButtonsTest(StoreTestBase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.text = "AAPL.US 10 150 USD\nTLT.US 5 90 USD"
        await self.db.save_manual_portfolio(self.USER_ID, self.text)

    def _tag(self, text: str) -> str:
        from portfolio_aggregation import version_tag
        return version_tag(text)

    async def test_rmlist_buttons_carry_version(self) -> None:
        cb = await self._mp("mp:rmlist")
        _t, kwargs = cb.message.sent[-1]
        datas = [b.callback_data for row in kwargs["reply_markup"].inline_keyboard
                 for b in row]
        tag = self._tag(self.text)
        self.assertIn(f"mp:rm:0:{tag}", datas)
        self.assertIn(f"mp:rm:1:{tag}", datas)
        self.assertTrue(all(len(d.encode()) <= 64 for d in datas))

    async def test_remove_by_button(self) -> None:
        await self._mp(f"mp:rm:0:{self._tag(self.text)}")
        self.assertEqual(await self._stored(), "TLT.US 5 90 USD")

    async def test_stale_button_does_not_remove_another_position(self) -> None:
        stale = self._tag(self.text)
        await self._edit("-AAPL")                      # портфель изменился
        cb = await self._mp(f"mp:rm:0:{stale}")
        self.assertIn("изменился", cb.message.all_text)
        self.assertIn("TLT.US", await self._stored())

    async def test_forged_callbacks(self) -> None:
        """S-5: `mp:rm:999`, мусорный тег, неизвестное действие."""
        tag = self._tag(self.text)
        for data in (f"mp:rm:999:{tag}", "mp:rm:0:zzzzzzzz", "mp:rm:0",
                     "mp:hack", "mp:rm:-1:" + tag):
            with self.subTest(data=data):
                await self._mp(data)
                self.assertEqual(await self._stored(), self.text)
                self.assertTrue(await self._slot_is_free())


# ── флаг и удаление ──────────────────────────────────────────────────────────

class FlagAndForgetTest(StoreTestBase):

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 10 150 USD")
        await self.db.save_manual_draft(self.USER_ID, "AAPL 10 150")

    async def test_old_buttons_with_flag_off(self) -> None:
        """S-4: кнопка из прошлой ревизии при выключенном флаге ничего не меняет."""
        from portfolio_aggregation import version_tag

        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        tag = version_tag("AAPL.US 10 150 USD")
        for data in ("mp:show", "mp:add", "mp:rmlist", f"mp:rm:0:{tag}"):
            with self.subTest(data=data):
                cb = await self._mp(data)
                self.assertIn("недоступен", cb.message.all_text)
        msg = await self._edit("+TLT 5 90")
        self.assertIn("недоступен", msg.all_text)
        self.assertEqual(await self._stored(), "AAPL.US 10 150 USD")

    async def test_forget_deletes_both_tables(self) -> None:
        """S-9: `/forget_portfolio` удаляет портфель И черновик — даже при
        выключенном флаге (право на удаление своих данных)."""
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        msg = _FakeMessage("/forget_portfolio", user_id=self.USER_ID)
        await self.tg.cmd_forget_portfolio(msg, self.state)
        _t, kwargs = msg.sent[-1]
        datas = [b.callback_data for row in kwargs["reply_markup"].inline_keyboard
                 for b in row]
        self.assertEqual(datas, ["mp:delyes", "mp:keep"])
        await self._mp("mp:delyes")
        self.assertIsNone(await self.db.get_manual_portfolio(self.USER_ID))
        self.assertIsNone(await self.db.get_manual_draft(self.USER_ID))
        self.assertTrue(await self._slot_is_free())

    async def test_commands_registered(self) -> None:
        dp = self.tg.build_dispatcher()
        names = {h.callback.__name__ for h in dp.message.handlers}
        self.assertIn("cmd_portfolio", names)
        self.assertIn("cmd_forget_portfolio", names)

    async def test_help_mentions_forget_only_with_flag(self) -> None:
        msg = _FakeMessage("/help", user_id=self.USER_ID)
        await self.tg.cmd_help(msg)
        self.assertIn("forget", msg.all_text)
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        msg = _FakeMessage("/help", user_id=self.USER_ID)
        await self.tg.cmd_help(msg)
        self.assertNotIn("forget", msg.all_text, "I-9: без флага /help прежний")


if __name__ == "__main__":
    unittest.main()
