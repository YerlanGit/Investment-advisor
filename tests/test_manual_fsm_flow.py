"""Ручной ввод: FSM, экран подтверждения, черновик (MP-05 · ЧК-05.1…05.5).

Раунд: `AUDIT §−72`, фаза Manual Portfolio 05
(`docs/roadmap/manual_portfolio/PHASE_05_FSM_FLOW.md`).

Что охраняется и почему именно это
----------------------------------
**F-4 — порядок веток источника.** `_resolve_portfolio_source` считает ключи в
vault доказательством привязки брокера и САМО-ЛЕЧИТ режим в `freedom`. Для
пользователя, который когда-то привязывал брокера, а сегодня вводит портфель
руками, это означало бы молчаливый отказ: он выбрал `manual`, а отчёт
построился бы по брокерскому счёту. Ветка `manual` обязана стоять ДО проверки
vault — и одновременно само-лечение обязано остаться живым для `template`/None
(регресс инцидента 2026-07-14).

**Экран подтверждения (§3.4).** Опечатка в количестве — лишний ноль — не видна
ни в одном числе отчёта, но полностью искажает веса, TRC и CVaR. Заметной её
делает ДОЛЯ позиции, поэтому доля на экране обязательна, а не желательна.

**Т-9 — утечка слота.** В `cb_confirm` слот освобождается девятью отдельными
вызовами. Ручной флоу добавляет свои ветки выхода, и забытый `release` запирает
пользователя на 30 минут (аренда), показывая ему «у вас уже выполняется
анализ» — ошибку, которую он сам исправить не может.

**E-6 — черновик.** `MemoryStorage` теряет состояние при КАЖДОМ передеплое
Cloud Run, а ввод портфеля на 20 позиций — это минуты работы.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# tg_bot читает токен при импорте — подставляем на этапе коллекции.
os.environ.setdefault("RAMP_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")

from aiogram.fsm.context import FSMContext          # noqa: E402
from aiogram.fsm.storage.base import StorageKey     # noqa: E402
from aiogram.fsm.storage.memory import MemoryStorage  # noqa: E402

from finance.investment_logic import MAC3RiskEngine  # noqa: E402

#: Курсы для офлайн-прогона. Без подмены `fx_quote_to_base` пошёл бы к
#: провайдеру FRED, то есть в сеть, — а тест обязан быть детерминированным.
_RATES = {"USD": (1.0, None), "KZT": (1.0 / 450.0, "2026-08-05")}

PORTFOLIO_TEXT = (
    "AAPL 10 150.50\n"
    "каспи 200 45000 KZT\n"
    "CASH:USD 5000\n"
    "CASH:KZT 2 500 000\n"
)


class _StubManager:
    """`UniversalPortfolioManager` ровно в той части, которую читает флоу."""

    def __init__(self, price_source: str = "manual") -> None:
        self.price_source = price_source
        self.engine = MAC3RiskEngine()
        self.engine.fx_quote_to_base = (                 # type: ignore[assignment]
            lambda ccy: _RATES.get(str(ccy).upper(), (None, None)))


class _FakeUser:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class _FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _FakeMessage:
    """Сообщение Telegram: помнит всё, что бот в него отправил и в нём правил."""

    def __init__(self, text: str = "", user_id: int = 1) -> None:
        self.text = text
        self.from_user = _FakeUser(user_id)
        self.chat = _FakeChat(user_id)
        self.bot = None
        self.sent: list[tuple[str, dict]] = []
        self.edited: list[tuple[str, dict]] = []
        self._user_id = user_id

    async def answer(self, text: str, **kwargs):
        self.sent.append((text, kwargs))
        child = _FakeMessage("", self._user_id)
        # Ответы на ответ должны попадать в ТУ ЖЕ ленту, иначе тест увидит
        # только первый экран цепочки.
        child.sent = self.sent
        child.edited = self.edited
        return child

    async def edit_text(self, text: str, **kwargs):
        self.edited.append((text, kwargs))
        return self

    async def delete(self):
        return True

    @property
    def all_text(self) -> str:
        return "\n".join(t for t, _ in self.sent + self.edited)


class _FakeBot:
    """`bot.send_message` — единственное, что нужно фоновой задаче."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_message(self, _chat_id, text: str, **_kwargs):
        self.sent.append(text)
        return _FakeMessage(text)


class _FakeCallback:
    def __init__(self, data: str, user_id: int = 1) -> None:
        self.data = data
        self.from_user = _FakeUser(user_id)
        self.message = _FakeMessage(user_id=user_id)
        self.answered = False

    async def answer(self, *_a, **_kw):
        self.answered = True


def _fresh_db() -> Path:
    return Path(tempfile.mkdtemp(prefix="ramp-manual-")) / "tokenomics.db"


class ManualFlowTestBase(unittest.IsolatedAsyncioTestCase):
    """Изолированная БД + включённый флаг + чистый слот на каждый тест."""

    USER_ID = 4242

    async def asyncSetUp(self) -> None:
        import db_tokenomics as db
        import tg_bot

        self.db = db
        self.tg = tg_bot
        self._orig_path = db.DB_PATH
        db.DB_PATH = _fresh_db()
        await db.init_db()

        # Валюта отчёта ПИНИТСЯ, а не наследуется из окружения: соседний
        # тест-файл вправе оставить после себя `REPORTING_CURRENCY=KZT`, и
        # тогда «KZT → USD» на экране не появится вовсе — падение было бы про
        # порядок файлов в прогоне, а не про ручной ввод.
        self._orig_env = {k: os.environ.get(k)
                          for k in ("MANUAL_PORTFOLIO_ENABLED", "REPORTING_CURRENCY")}
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "on"
        os.environ["REPORTING_CURRENCY"] = "USD"

        self._patcher = patch.object(tg_bot, "UniversalPortfolioManager", _StubManager)
        self._patcher.start()
        tg_bot._IN_FLIGHT_USERS.clear()

        self.storage = MemoryStorage()
        self.state = self._make_state(self.storage)

    async def asyncTearDown(self) -> None:
        self._patcher.stop()
        self.tg._IN_FLIGHT_USERS.clear()
        self.db.DB_PATH = self._orig_path
        for key, value in self._orig_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _make_state(self, storage: MemoryStorage) -> FSMContext:
        return FSMContext(storage=storage,
                          key=StorageKey(bot_id=1, chat_id=self.USER_ID,
                                         user_id=self.USER_ID))

    async def _send_text(self, text: str) -> _FakeMessage:
        message = _FakeMessage(text, user_id=self.USER_ID)
        await self.state.set_state(self.tg.ManualPortfolio.Input)
        await self.tg.msg_manual_input(message, self.state)
        return message

    async def _tap(self, action: str) -> _FakeCallback:
        callback = _FakeCallback(f"manual:{action}", user_id=self.USER_ID)
        await self.tg.cb_manual_action(callback, self.state)
        return callback

    async def _slot_is_free(self) -> bool:
        """Слот действительно свободен — проверяется ЗАХВАТОМ, а не флагом.

        Проверять только `_IN_FLIGHT_USERS` недостаточно: строка аренды живёт в
        SQLite и переживает процесс, поэтому утечку видно лишь попыткой взять
        слот заново.
        """
        if self.USER_ID in self.tg._IN_FLIGHT_USERS:
            return False
        won = await self.tg._try_acquire_user_slot(self.USER_ID)
        if won:
            await self.tg._release_user_slot(self.USER_ID)
        return won


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-05.1 · Источник портфеля (F-4)
# ═════════════════════════════════════════════════════════════════════════════

class ResolveManualSourceTest(unittest.IsolatedAsyncioTestCase):
    """Ветка `manual` стоит ДО vault — и не ломает само-лечение."""

    async def asyncSetUp(self) -> None:
        import tg_bot
        self.tg = tg_bot
        self._orig = (tg_bot._has_vault_keys_sync,
                      tg_bot.get_connection_mode_explicit,
                      tg_bot.save_connection_mode)
        self.saved: list[tuple[int, str]] = []

    async def asyncTearDown(self) -> None:
        (self.tg._has_vault_keys_sync,
         self.tg.get_connection_mode_explicit,
         self.tg.save_connection_mode) = self._orig

    def _patch(self, *, has_keys: bool, stored: str | None) -> None:
        self.tg._has_vault_keys_sync = lambda uid: has_keys

        async def fake_get(uid):
            return stored

        async def fake_save(uid, mode):
            self.saved.append((uid, mode))
            return True

        self.tg.get_connection_mode_explicit = fake_get
        self.tg.save_connection_mode = fake_save

    async def test_manual_wins_over_vault_keys(self) -> None:
        """🔴 F-4: иначе бывший клиент брокера не построит ручной отчёт."""
        self._patch(has_keys=True, stored="manual")
        source, stored = await self.tg._resolve_portfolio_source(7)
        self.assertEqual(source, "manual")
        self.assertEqual(stored, "manual")
        self.assertEqual(self.saved, [],
                         "режим само-вылечился в freedom — выбор пользователя "
                         "молча перезаписан фактом наличия ключей")

    async def test_manual_without_keys_is_manual(self) -> None:
        self._patch(has_keys=False, stored="manual")
        self.assertEqual((await self.tg._resolve_portfolio_source(7))[0], "manual")

    async def test_vault_selfheal_still_works_for_template(self) -> None:
        """Регресс инцидента 2026-07-14: ключи по-прежнему лечат `template`."""
        self._patch(has_keys=True, stored="template")
        source, _ = await self.tg._resolve_portfolio_source(7)
        self.assertEqual(source, "freedom")
        self.assertIn((7, "freedom"), self.saved)

    async def test_vault_selfheal_still_works_for_none(self) -> None:
        self._patch(has_keys=True, stored=None)
        source, _ = await self.tg._resolve_portfolio_source(7)
        self.assertEqual(source, "freedom")
        self.assertIn((7, "freedom"), self.saved)

    async def test_manual_report_costs_a_token(self) -> None:
        """I-4: ручной портфель — живой источник, бесплатно только демо."""
        self.assertEqual(self.tg._effective_cost("deep", "manual"),
                         self.tg.TIER_COST["deep"])
        self.assertEqual(self.tg._effective_cost("base", "demo"), 0)


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-05.2 · Флаг отката (I-9)
# ═════════════════════════════════════════════════════════════════════════════

class FeatureFlagTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        import tg_bot
        self.tg = tg_bot
        self._orig = os.environ.get("MANUAL_PORTFOLIO_ENABLED")

    async def asyncTearDown(self) -> None:
        if self._orig is None:
            os.environ.pop("MANUAL_PORTFOLIO_ENABLED", None)
        else:
            os.environ["MANUAL_PORTFOLIO_ENABLED"] = self._orig

    def _buttons(self) -> list[str]:
        return [b.callback_data
                for row in self.tg.kb_connect_choice().inline_keyboard for b in row]

    async def test_manual_button_hidden_when_flag_off(self) -> None:
        for value in ("off", "0", "", "false"):
            with self.subTest(value=value):
                os.environ["MANUAL_PORTFOLIO_ENABLED"] = value
                self.assertEqual(self._buttons(),
                                 ["connect:template", "connect:freedom"])

    async def test_flag_defaults_to_off(self) -> None:
        """Дефолт «выключено»: провайдера цен для manual нет до Фазы 9."""
        os.environ.pop("MANUAL_PORTFOLIO_ENABLED", None)
        self.assertFalse(self.tg.manual_portfolio_enabled())
        self.assertNotIn("connect:manual", self._buttons())

    async def test_manual_button_appears_when_flag_on(self) -> None:
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "on"
        self.assertIn("connect:manual", self._buttons())

    async def test_stale_button_is_refused_after_flag_off(self) -> None:
        """Сообщение с кнопкой живёт в чате вечно — гард обязан быть и в хендлере."""
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        storage = MemoryStorage()
        state = FSMContext(storage=storage,
                           key=StorageKey(bot_id=1, chat_id=5, user_id=5))
        callback = _FakeCallback("connect:manual", user_id=5)
        await self.tg.cb_connect_choice(callback, state)
        self.assertIn("недоступен", callback.message.all_text)
        self.assertIsNone(await state.get_state())


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-05.3 · Экран подтверждения
# ═════════════════════════════════════════════════════════════════════════════

class ConfirmationScreenTest(ManualFlowTestBase):

    async def test_confirm_screen_shows_weights(self) -> None:
        """Доля каждой позиции присутствует и суммируется в 100% ± 0.1пп."""
        from finance.manual_portfolio import build_confirmation, parse_portfolio_text

        engine = _StubManager().engine
        view = build_confirmation(parse_portfolio_text(PORTFOLIO_TEXT, engine), engine)
        shares = [r.share for r in view.positions + view.cash]
        self.assertTrue(all(s is not None for s in shares), "есть позиция без доли")
        self.assertAlmostEqual(sum(shares) * 100, 100.0, delta=0.1)

        message = await self._send_text(PORTFOLIO_TEXT)
        screen = message.all_text
        self.assertIn("Доля", screen)
        for row in view.positions:
            with self.subTest(ticker=row.ticker):
                self.assertIn(f"{row.share * 100:.1f}%", screen)

    async def test_confirm_screen_lists_excluded_rows(self) -> None:
        """Каждая исключённая строка — с НОМЕРОМ и человеческой причиной."""
        message = await self._send_text(
            "AAPL 10 150\nBADLINE\nAAPL -3 10\nCASH 5000\n")
        screen = message.all_text
        self.assertIn("Исключено строк: 3", screen)
        for line_no in (2, 3, 4):
            with self.subTest(line=line_no):
                self.assertIn(f"строка {line_no}", screen)
        self.assertIn("корот", screen.lower())

    async def test_confirm_screen_shows_fx_rate(self) -> None:
        """Курс и дата наблюдения видны — конверсия не должна быть невидимой."""
        message = await self._send_text(PORTFOLIO_TEXT)
        screen = message.all_text
        self.assertIn("KZT → USD", screen)
        self.assertIn("450", screen, "курс не показан")
        self.assertIn("2026-08-05", screen, "дата наблюдения курса не показана")

    async def test_currency_without_rate_is_named_not_silently_valued(self) -> None:
        """Т-5: молчаливого 1.0 нет ни на одном пути."""
        message = await self._send_text("AAPL 10 150\nCASH:EUR 1000\n")
        screen = message.all_text
        self.assertIn("Нет курса для EUR", screen)
        self.assertIn("не учтены", screen)

    async def test_proxy_substitution_is_shown_separately(self) -> None:
        """H-1: подмена для риск-модели ≠ переименование бумаги."""
        message = await self._send_text("FFSPC6.1028.AIX 10 103.5\nCASH:USD 100\n")
        screen = message.all_text
        self.assertIn("Для риск-модели заменены", screen)
        self.assertIn("VWOB.US", screen)
        self.assertIn("FFSPC6.1028.AIX", screen)

    async def test_typo_in_quantity_is_visible_as_a_weight(self) -> None:
        """Ради этого экран и существует: лишний ноль обязан бросаться в глаза."""
        from finance.manual_portfolio import build_confirmation, parse_portfolio_text

        engine = _StubManager().engine
        typo = build_confirmation(
            parse_portfolio_text("AAPL 100 150\nCASH:USD 5000\n", engine), engine)
        clean = build_confirmation(
            parse_portfolio_text("AAPL 10 150\nCASH:USD 5000\n", engine), engine)
        self.assertGreater(typo.positions[0].share, 0.7)
        self.assertLess(clean.positions[0].share, 0.3)

    async def test_screen_fits_the_telegram_limit_on_a_long_book(self) -> None:
        """Сообщение длиннее 4096 API отвергает ЦЕЛИКОМ — портфель не виден.

        Три формы длинной книги, потому что раздувают экран РАЗНЫЕ блоки, и
        ограничение таблицы позиций спасает только от первой:

        | Книга | Что растёт | Замер без защит |
        |---|---|---|
        | 200 обычных бумаг | таблица | в пределах лимита |
        | 200 синонимов | список «распознаны как» | 5 292 |
        | 200 неликвидных облигаций | список подмен прокси | 6 273 |

        Вторая форма лечится дедупликацией пар (у синонимов их всего несколько
        уникальных), третья — только ограничением списка и жёсткой обрезкой:
        двести РАЗНЫХ облигаций дают двести РАЗНЫХ пар, и дедуп бессилен.
        """
        synonyms = [k for k in MAC3RiskEngine.TICKER_MAP if not k.isascii()]
        books = {
            "обычные бумаги":
                [f"TICK{i} {i + 1} {100 + i}" for i in range(200)],
            "синонимы":
                [f"{synonyms[i % len(synonyms)]} {i + 1} 100 USD" for i in range(200)],
            "неликвидные облигации":
                [f"KZBOND{i} {i + 1} 100 USD" for i in range(200)],
        }
        for name, lines in books.items():
            with self.subTest(book=name):
                await self.state.set_state(self.tg.ManualPortfolio.Input)
                message = await self._send_text("\n".join(lines))
                screens = [t for t, _ in message.sent
                           if "проверьте перед расчётом" in t]
                self.assertTrue(screens, "экран подтверждения не показан")
                self.assertLessEqual(len(screens[0]), self.tg._TELEGRAM_TEXT_LIMIT,
                                     f"«{name}»: сообщение не пройдёт в Telegram")
                self.assertEqual(
                    screens[0].count("```") % 2, 0,
                    "непарный кодовый блок — Telegram отвергнет сообщение по "
                    "разметке, то есть обрезка сама себя обесценит")
                self.assertIn("и ещё", screens[0],
                              "усечение не названо — пользователь решит, что "
                              "остальные позиции потеряны")

    async def test_pairs_are_deduplicated_not_repeated_per_lot(self) -> None:
        """Двадцать лотов одной бумаги — это ОДНА пара, а не двадцать."""
        from finance.manual_portfolio import build_confirmation, parse_portfolio_text

        engine = _StubManager().engine
        text = "\n".join(f"каспи {i + 1} 100 KZT" for i in range(20))
        view = build_confirmation(parse_portfolio_text(text, engine), engine)
        self.assertEqual(view.auto_resolved, [("КАСПИ", "KSPI.KZ")])

    async def test_parse_failure_keeps_the_user_in_input(self) -> None:
        """Полный провал разбора — не тупик: остаёмся в вводе с перечнем ошибок."""
        message = await self._send_text("совсем не портфель\n")
        self.assertEqual(await self.state.get_state(),
                         self.tg.ManualPortfolio.Input.state)
        self.assertIn("строка 1", message.all_text)

    async def test_markdown_in_user_input_cannot_break_the_screen(self) -> None:
        """Незакрытая `*` в причине уронила бы ВСЁ сообщение целиком."""
        message = await self._send_text("AA*PL_ 10 150\nCASH:USD 100\n")
        screen = message.all_text
        self.assertNotIn("*PL", screen)
        self.assertEqual(screen.count("```") % 2, 0, "непарные кодовые блоки")


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-05.3 · Переходы FSM
# ═════════════════════════════════════════════════════════════════════════════

class FsmTransitionsTest(ManualFlowTestBase):

    async def test_input_to_confirm_and_back_to_edit(self) -> None:
        await self._send_text(PORTFOLIO_TEXT)
        self.assertEqual(await self.state.get_state(),
                         self.tg.ManualPortfolio.Confirm.state)
        callback = await self._tap("edit")
        self.assertEqual(await self.state.get_state(),
                         self.tg.ManualPortfolio.Input.state)
        self.assertIn("AAPL 10 150.50", callback.message.all_text,
                      "прошлый ввод не предложен для правки — набирать заново")

    async def test_correction_sent_from_the_confirm_screen_is_processed(self) -> None:
        """Присланный поверх экрана текст — исправление, а не мусор.

        Без фильтра на состояние `Confirm` такое сообщение не подходило ни под
        один хендлер (`msg_text_fallback` стоит под `StateFilter(None)`) и
        ПРОПАДАЛО молча: человек набрал портфель заново, а бот промолчал.
        """
        await self._send_text(PORTFOLIO_TEXT)
        self.assertEqual(await self.state.get_state(),
                         self.tg.ManualPortfolio.Confirm.state)

        corrected = "AAPL 11 150.50\nCASH:USD 5000\n"
        message = _FakeMessage(corrected, user_id=self.USER_ID)
        await self.tg.msg_manual_input(message, self.state)   # состояние — Confirm

        self.assertIn("проверьте перед расчётом", message.all_text)
        draft = await self.db.get_manual_draft(self.USER_ID)
        self.assertEqual(draft["text"], corrected,
                         "исправление не доехало до черновика")

    async def test_text_handler_is_registered_for_both_states(self) -> None:
        """Проверка САМОЙ РЕГИСТРАЦИИ, а не поведения функции.

        Вызов хендлера напрямую (как в тесте выше) проходит МИМО фильтра
        роутера и поэтому не может доказать, что сообщение вообще до него
        дойдёт. Здесь читаются фильтры зарегистрированного хендлера — единст-
        венное место, где решается, попадёт ли текст из состояния `Confirm`
        в обработку или пропадёт.
        """
        handlers = [h for h in self.tg.portfolio_router.message.handlers
                    if h.callback is self.tg.msg_manual_input]
        self.assertEqual(len(handlers), 1, "хендлер ручного ввода не зарегистрирован")
        states: set = set()
        for flt in handlers[0].filters:
            states |= set(getattr(flt.callback, "states", ()) or ())
        self.assertIn(self.tg.ManualPortfolio.Input, states)
        self.assertIn(self.tg.ManualPortfolio.Confirm, states,
                      "текст из состояния Confirm не дойдёт ни до одного "
                      "хендлера и пропадёт молча")

    async def test_confirm_leads_to_the_tier_menu(self) -> None:
        await self._send_text(PORTFOLIO_TEXT)
        callback = await self._tap("confirm")
        self.assertIsNone(await self.state.get_state())
        self.assertIn("Выберите тип анализа", callback.message.all_text)

    async def test_cancel_at_every_step(self) -> None:
        """Отмена и в Input, и в Confirm: состояние чисто, черновик удалён."""
        for step in ("input", "confirm"):
            with self.subTest(step=step):
                await self._send_text(PORTFOLIO_TEXT)
                if step == "input":
                    await self.state.set_state(self.tg.ManualPortfolio.Input)
                callback = await self._tap("cancel")
                self.assertIsNone(await self.state.get_state())
                self.assertIsNone(await self.db.get_manual_draft(self.USER_ID))
                self.assertIn("отменён", callback.message.all_text)
                self.assertTrue(await self._slot_is_free())

    async def test_oversized_input_is_refused_before_the_database(self) -> None:
        message = await self._send_text("A" * (self.db.MANUAL_DRAFT_MAX_BYTES + 1))
        self.assertIn("Слишком длинный ввод", message.all_text)
        self.assertIsNone(await self.db.get_manual_draft(self.USER_ID))


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-05.4 · Черновик
# ═════════════════════════════════════════════════════════════════════════════

class DraftTest(ManualFlowTestBase):

    async def test_draft_is_written_once_on_the_confirm_transition(self) -> None:
        await self._send_text(PORTFOLIO_TEXT)
        draft = await self.db.get_manual_draft(self.USER_ID)
        self.assertIsNotNone(draft)
        self.assertEqual(draft["text"], PORTFOLIO_TEXT)

    async def test_failed_parse_does_not_overwrite_a_good_draft(self) -> None:
        """Сбой разбора не имеет права стереть уже введённый портфель."""
        await self._send_text(PORTFOLIO_TEXT)
        await self.state.set_state(self.tg.ManualPortfolio.Input)
        await self._send_text("мусор\n")
        draft = await self.db.get_manual_draft(self.USER_ID)
        self.assertEqual(draft["text"], PORTFOLIO_TEXT)

    async def test_draft_survives_restart(self) -> None:
        """E-6: новый `MemoryStorage` = передеплой Cloud Run. Ввод обязан жить."""
        await self._send_text(PORTFOLIO_TEXT)
        # «Рестарт»: состояние FSM потеряно целиком, БД — нет.
        self.storage = MemoryStorage()
        self.state = self._make_state(self.storage)
        self.assertIsNone(await self.state.get_state())

        callback = await self._tap("resume")
        self.assertEqual(await self.state.get_state(),
                         self.tg.ManualPortfolio.Confirm.state)
        self.assertIn("проверьте перед расчётом", callback.message.all_text)
        self.assertIn("AAPL", callback.message.all_text)

    async def test_draft_is_offered_with_its_age_not_applied_silently(self) -> None:
        """§9: недельный черновик подставлять молча нельзя — портфель изменился."""
        await self.db.save_manual_draft(self.USER_ID, PORTFOLIO_TEXT)
        callback = _FakeCallback("connect:manual", user_id=self.USER_ID)
        await self.tg.cb_connect_choice(callback, self.state)
        screen = callback.message.all_text
        self.assertIn("Найден черновик", screen)
        self.assertIn("Продолжить", str(callback.message.edited))

    async def test_resume_failure_keeps_the_draft(self) -> None:
        """Сбой восстановления не имеет права стереть ввод пользователя."""
        await self.db.save_manual_draft(self.USER_ID, PORTFOLIO_TEXT)
        with patch.object(self.tg, "_manual_review_sync",
                          side_effect=RuntimeError("boom")):
            callback = await self._tap("resume")
        self.assertIn("Код ошибки для поддержки", callback.message.all_text)
        self.assertIsNotNone(await self.db.get_manual_draft(self.USER_ID))
        self.assertTrue(await self._slot_is_free())

    async def test_fresh_start_drops_the_draft(self) -> None:
        await self.db.save_manual_draft(self.USER_ID, PORTFOLIO_TEXT)
        await self._tap("fresh")
        self.assertIsNone(await self.db.get_manual_draft(self.USER_ID))
        self.assertEqual(await self.state.get_state(),
                         self.tg.ManualPortfolio.Input.state)

    async def test_draft_payload_size_limit(self) -> None:
        """Больше 64 КБ — отказ ДО записи, а не распухшая база на gcsfuse."""
        huge = "A" * (self.db.MANUAL_DRAFT_MAX_BYTES + 1)
        with self.assertRaises(self.db.ManualDraftTooLarge):
            await self.db.save_manual_draft(self.USER_ID, huge)
        self.assertIsNone(await self.db.get_manual_draft(self.USER_ID))

    async def test_draft_size_limit_counts_bytes_not_characters(self) -> None:
        """Кириллица в UTF-8 — два байта на символ; лимит на БАЙТАХ."""
        text = "я" * (self.db.MANUAL_DRAFT_MAX_BYTES // 2 + 1)
        self.assertLess(len(text), self.db.MANUAL_DRAFT_MAX_BYTES)
        with self.assertRaises(self.db.ManualDraftTooLarge):
            await self.db.save_manual_draft(self.USER_ID, text)

    async def test_draft_sql_is_parameterised(self) -> None:
        """Инъекция сохраняется как ДАННЫЕ; таблица остаётся на месте."""
        payload = "'; DROP TABLE manual_portfolio_draft; --"
        await self.db.save_manual_draft(self.USER_ID, payload)
        draft = await self.db.get_manual_draft(self.USER_ID)
        self.assertEqual(draft["text"], payload)
        # Таблица цела — иначе следующая запись упала бы.
        await self.db.save_manual_draft(self.USER_ID + 1, "AAPL 1 1")
        self.assertIsNotNone(await self.db.get_manual_draft(self.USER_ID + 1))

    async def test_draft_deleted_after_report(self) -> None:
        """Удаление — ПОСЛЕ доставки: упавший отчёт не отнимает ввод."""
        import inspect
        src = inspect.getsource(self.tg._run_analysis_background)
        after_send = src.split("_send_report(", 1)[1]
        self.assertIn("delete_manual_draft", after_send,
                      "черновик удаляется до доставки отчёта")
        self.assertNotIn("delete_manual_draft", src.split("_send_report(", 1)[0],
                         "черновик удаляется ДО доставки — упавший отчёт "
                         "заставит пользователя набирать портфель заново")

    async def test_draft_outlives_a_failed_report(self) -> None:
        """Прямая проверка того же: отказ расчёта черновик не трогает."""
        await self._send_text(PORTFOLIO_TEXT)
        callback = _FakeCallback("confirm:base:menu", user_id=self.USER_ID)

        async def fake_source(_uid):
            return "manual", "manual"

        with patch.object(self.tg, "_resolve_portfolio_source", fake_source):
            await self.tg.cb_confirm(callback, self.state)
        self.assertIsNotNone(await self.db.get_manual_draft(self.USER_ID))


# ═════════════════════════════════════════════════════════════════════════════
# ЧК-05.5 · Слот (Т-9)
# ═════════════════════════════════════════════════════════════════════════════

class SlotReleaseTest(ManualFlowTestBase):

    async def test_slot_released_in_every_branch(self) -> None:
        """Перебор веток ручного флоу: после каждой слот обязан быть свободен."""
        branches = {
            "успешный разбор":   lambda: self._send_text(PORTFOLIO_TEXT),
            "ошибка разбора":    lambda: self._send_text("мусор\n"),
            "пустой ввод":       lambda: self._send_text("\n\n"),
            "слишком длинный":   lambda: self._send_text(
                "A" * (self.db.MANUAL_DRAFT_MAX_BYTES + 1)),
            "отмена":            lambda: self._tap("cancel"),
            "правка":            lambda: self._tap("edit"),
            "подтверждение":     lambda: self._tap("confirm"),
        }
        for name, run in branches.items():
            with self.subTest(branch=name):
                await run()
                self.assertTrue(await self._slot_is_free(),
                                f"слот утёк в ветке «{name}»")

    async def test_slot_released_when_review_raises(self) -> None:
        """Неожиданное исключение внутри разбора — тоже ветка выхода."""
        with patch.object(self.tg, "_manual_review_sync",
                          side_effect=RuntimeError("boom")):
            message = await self._send_text(PORTFOLIO_TEXT)
        self.assertIn("Код ошибки для поддержки", message.all_text)
        self.assertTrue(await self._slot_is_free())

    async def test_busy_slot_refuses_without_leaking(self) -> None:
        """Занятый слот — отказ, а не отнятая аренда чужого расчёта."""
        self.assertTrue(await self.tg._try_acquire_user_slot(self.USER_ID))
        try:
            message = await self._send_text(PORTFOLIO_TEXT)
            self.assertIn("уже идёт обработка", message.all_text)
            self.assertIn(self.USER_ID, self.tg._IN_FLIGHT_USERS,
                          "чужая аренда снята обработчиком отказа")
        finally:
            await self.tg._release_user_slot(self.USER_ID)
        self.assertTrue(await self._slot_is_free())

    async def test_context_manager_releases_on_exception(self) -> None:
        with self.assertRaises(ValueError):
            async with self.tg.user_slot(self.USER_ID):
                raise ValueError("внутри")
        self.assertTrue(await self._slot_is_free())

    async def test_context_manager_refuses_second_entry(self) -> None:
        async with self.tg.user_slot(self.USER_ID):
            with self.assertRaises(self.tg.SlotBusy):
                async with self.tg.user_slot(self.USER_ID):
                    pass                                   # pragma: no cover
        self.assertTrue(await self._slot_is_free())


# ═════════════════════════════════════════════════════════════════════════════
# Граница фазы: ручной источник не имеет права стать демо
# ═════════════════════════════════════════════════════════════════════════════

class ManualSourceBoundaryTest(ManualFlowTestBase):

    @staticmethod
    def _buttons(message) -> list[str]:
        """Кнопки со ВСЕХ экранов — и правленых, и новых.

        Смотреть только `edited` нельзя: отказы после строки «⏳ Собираю…»
        приходят новым сообщением, как и брокерские, — иначе пользователь
        потеряет статус из виду.
        """
        markups = [kw.get("reply_markup")
                   for _t, kw in message.sent + message.edited]
        return [b.callback_data for m in markups if m
                for row in m.inline_keyboard for b in row]

    async def _confirm_as_manual(self, callback) -> None:
        """Нажатие «Рассчитать» пользователем, чей источник — ручной ввод.

        Баланс пополняется намеренно: проверка средств стоит РАНЬШЕ разбора
        портфеля (как и для брокерского пути — не работаем для того, кто не
        может заплатить), и на нулевом балансе тесты ниже проверяли бы
        сообщение о токенах, а не ручную ветку.
        """
        await self.db.init_user(self.USER_ID)
        await self.db.credit_tokens(self.USER_ID, 5, reason="test")

        async def fake_source(_uid):
            return "manual", "manual"

        with patch.object(self.tg, "_resolve_portfolio_source", fake_source):
            await self.tg.cb_confirm(callback, self.state)

    async def test_manual_never_falls_through_to_demo(self) -> None:
        """🔴 Молчаливое демо было бы ПЛАТНЫМ: `_effective_cost` тут не ноль.

        До MP-06 инвариант держался отказом («фаза не готова»). Расчёт
        включён, инвариант прежний: ручной источник НЕ ходит к брокеру и НЕ
        подставляет демо-ключи. Проверяется по факту вызова, а не по тексту.
        """
        callback = _FakeCallback("confirm:deep:menu", user_id=self.USER_ID)
        await self.tg.save_manual_draft(self.USER_ID, "AAPL 10 150")

        started: list[str] = []

        async def fake_background(**kwargs):
            started.append(kwargs.get("source"))

        with patch.object(self.tg, "_fetch_portfolio_sync") as fetch, \
             patch.object(self.tg, "_run_analysis_background", fake_background):
            await self._confirm_as_manual(callback)
            # Анализ уходит в `asyncio.create_task` — без уступки циклу задача
            # не успеет стартовать, и тест проверял бы планировщик, а не ветку.
            await asyncio.sleep(0)

        fetch.assert_not_called()
        self.assertEqual(started, ["manual"],
                         "расчёт обязан стартовать именно ручным источником:\n"
                         + callback.message.all_text)

    async def test_missing_draft_refuses_without_charging(self) -> None:
        """Черновика нет — честный отказ, а не пустой отчёт за полную цену."""
        callback = _FakeCallback("confirm:deep:menu", user_id=self.USER_ID)
        await self.tg.delete_manual_draft(self.USER_ID)

        with patch.object(self.tg, "_fetch_portfolio_sync") as fetch:
            await self._confirm_as_manual(callback)

        fetch.assert_not_called()
        screen = callback.message.all_text
        self.assertIn("не списан", screen)
        self.assertTrue(await self._slot_is_free())

    async def test_refusal_offers_a_way_out(self) -> None:
        """Иначе режим `manual` — ловушка: /start ведёт в меню, меню сюда.

        Ровно ту же петлю закрыли 2026-07-16 для `undetermined` (второй
        пользователь застрял): восстановление обязано быть в ОДИН тап.
        """
        callback = _FakeCallback("confirm:base:menu", user_id=self.USER_ID)
        await self.tg.delete_manual_draft(self.USER_ID)
        await self._confirm_as_manual(callback)
        buttons = self._buttons(callback.message)
        self.assertIn("connect:manual", buttons)
        self.assertIn("connect:template", buttons)

    async def test_disabled_flag_stops_the_calculation(self) -> None:
        """🔴 Флаг отката обязан работать и для УЖЕ выбравших ручной режим.

        Режим хранится в профиле: такой пользователь приходит к расчёту прямо
        из меню тиров, мимо `cb_manual_action`, где флаг проверяется. Без
        проверки здесь выключение фичи в проде не остановило бы ничего — то
        есть откат перестал бы быть откатом.
        """
        callback = _FakeCallback("confirm:deep:menu", user_id=self.USER_ID)
        await self.tg.save_manual_draft(self.USER_ID, "AAPL 10 150")

        with patch.dict("os.environ", {"MANUAL_PORTFOLIO_ENABLED": "0"}), \
             patch.object(self.tg, "_fetch_portfolio_sync") as fetch, \
             patch.object(self.tg, "_manual_frame_sync") as frame:
            await self._confirm_as_manual(callback)

        fetch.assert_not_called()
        frame.assert_not_called()
        self.assertIn("не списан", callback.message.all_text)
        self.assertIn("connect:template", self._buttons(callback.message))
        self.assertTrue(await self._slot_is_free())

    async def test_price_source_is_never_silently_freedom(self) -> None:
        """I-12: источник доезжает до движка КАК ЕСТЬ, а не сводится к freedom.

        Проверяется поведением, а не текстом функции: прежняя запись
        `price_source="demo" if source == "demo" else "freedom"` означала, что
        ЛЮБОЙ новый источник молча поедет на брокерском фиде — то есть данные
        Tradernet ушли бы не-клиенту Freedom без единого признака в коде.
        """
        seen: list[str] = []

        class _Recorder:
            def __init__(self, price_source: str = "freedom") -> None:
                seen.append(price_source)
                raise RuntimeError("остановка сразу после выбора провайдера")

        for source in ("manual", "demo", "freedom"):
            with self.subTest(source=source):
                seen.clear()
                with patch("finance.investment_logic.UniversalPortfolioManager",
                           _Recorder):
                    await self.tg._run_analysis_background(
                        bot=_FakeBot(), chat_id=self.USER_ID, user_id=self.USER_ID,
                        tier="base", cost=1, df=None, bench_tick=None,
                        source=source)
                self.assertEqual(seen, [source])

    async def test_provider_for_manual_is_not_the_broker(self) -> None:
        """Гард юридической границы жив и без бота.

        До MP-09 тест проверял ОТКАЗ («провайдер не реализован») — это пинило
        границу фазы, а не инвариант. Провайдер написан, отказа больше нет, и
        проверять надо то, что переживёт следующую фазу: чем бы `manual` ни
        обслуживался, это не брокерский фид (I-12).
        """
        from finance.price_providers import TradernetProvider, provider_for_source
        got = provider_for_source("manual", client=object())
        self.assertNotIsInstance(got, TradernetProvider)
        self.assertEqual(got.name, "stooq")


# ═════════════════════════════════════════════════════════════════════════════
# Гейт §10 — время. Что измеримо СЕГОДНЯ, а что нет
# ═════════════════════════════════════════════════════════════════════════════

class ManualReviewLatencyTest(ManualFlowTestBase):
    """Замер той части, которую эта фаза действительно добавила.

    Гейт §10 задания («время отчёта на 25 ручных позициях не хуже `freedom`
    более чем на 20%») сегодня НЕИЗМЕРИМ и подделывать его нельзя: живой
    ручной отчёт обязан ходить в Stooq (I-12), а `StooqProvider` появится в
    Фазе 9 — то есть измерять пока попросту нечего. Сравнение обеих веток на
    Tradernet-фикстуре дало бы красивое число ни о чём.

    Что измеримо и измеряется здесь: собственная задержка ручного шага —
    разбор, доли, pre-flight. Она и есть весь вклад Фазы 5 в общее время.
    """

    async def test_review_of_25_positions_is_not_a_bottleneck(self) -> None:
        import time

        text = "\n".join(f"TICK{i} {i + 1} {100 + i}" for i in range(25))
        started = time.perf_counter()
        report, view, _cov = self.tg._manual_review_sync(text)
        elapsed = time.perf_counter() - started
        self.assertEqual(len(report.valid), 25)
        self.assertEqual(len(view.positions), 25)
        # Порог намеренно грубый: он ловит АЛГОРИТМИЧЕСКУЮ регрессию (сетевой
        # поход на позицию, O(n²) в разборе), а не дрожание CI.
        self.assertLess(elapsed, 5.0, f"разбор 25 позиций занял {elapsed:.2f} с")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
