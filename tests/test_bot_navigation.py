"""Навигация бота и раскатка флагов (`§−124`).

Повод — скриншот владельца: в «меню анализа» и в «Мой мандат и настройки» нет
ни ручного портфеля, ни агрегированного отчёта. Причин было две:

1. **Флаги не доезжали до прода.** `gcloud run deploy --set-env-vars` ЗАМЕНЯЕТ
   весь набор переменных, а `MANUAL_PORTFOLIO_ENABLED` / `HYBRID_PORTFOLIO_ENABLED`
   в нём не было — прод всегда работал с выключенными фичами (тот же класс
   аварии, что у `PREMIUM_REPORT_ENABLED`).
2. **Навигации не было.** Вернувшийся пользователь попадал сразу в тиры: ни
   главного меню, ни пути к смене источника портфеля (демо-пользователь не мог
   подключить брокера), а `/mandate` вёл только к мандату.

Здесь стерегутся: ступень раскатки `admins`, пины флагов в `cloudbuild.yaml`,
главное меню и «Мой портфель», честная цена для демо, команды Telegram,
валидность разметки каждого экрана (битая разметка = ни одного экрана, `§−104`).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from test_hybrid_bot_flow import make_profile  # noqa: E402
from test_manual_fsm_flow import _FakeCallback, _FakeMessage  # noqa: E402
from test_manual_portfolio_store import StoreTestBase  # noqa: E402

ADMIN = 777
_TG_CMD = re.compile(r"^[a-z0-9_]{1,32}$")


def _buttons_of(kb) -> list[str]:
    return [b.callback_data for row in kb.inline_keyboard for b in row]


def _all_buttons(message) -> list[str]:
    out: list[str] = []
    for _t, kw in message.sent + message.edited:
        if kw.get("reply_markup") is not None:
            out += _buttons_of(kw["reply_markup"])
    return out


def _last(message) -> tuple[str, dict]:
    return (message.sent + message.edited)[-1]


def assert_markdown_balanced(testcase: unittest.TestCase, text: str) -> None:
    """Legacy Markdown Telegram: непарная `*`/`_`/`` ` `` роняет сообщение целиком."""
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    testcase.assertEqual(body.count("`") % 2, 0, f"непарный `: {text!r}")
    body = re.sub(r"`[^`]*`", "", body)
    body = body.replace("\\_", "")
    testcase.assertEqual(body.count("*") % 2, 0, f"непарная *: {text!r}")
    testcase.assertEqual(body.count("_") % 2, 0, f"непарный _: {text!r}")


class NavTestBase(StoreTestBase):
    """Пользователь с пройденной анкетой; ключи брокера — переключатель `self.keys`."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._nav_env = {k: os.environ.get(k)
                         for k in ("HYBRID_PORTFOLIO_ENABLED", "ADMIN_USER_IDS")}
        os.environ.pop("HYBRID_PORTFOLIO_ENABLED", None)
        os.environ["ADMIN_USER_IDS"] = str(ADMIN)
        await make_profile(self.db, self.USER_ID)
        await self.db.init_user(self.USER_ID)              # 10 приветственных токенов
        self.keys = False
        self._nav_patches = [
            patch.object(self.tg, "_has_vault_keys_sync", lambda _uid: self.keys),
        ]
        for p in self._nav_patches:
            p.start()

    async def asyncTearDown(self) -> None:
        for p in self._nav_patches:
            p.stop()
        for k, v in self._nav_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        await super().asyncTearDown()

    async def _connect_freedom(self) -> None:
        self.keys = True
        await self.db.save_connection_mode(self.USER_ID, "freedom")

    async def _tap_home(self, action: str) -> _FakeCallback:
        cb = _FakeCallback(f"home:{action}", user_id=self.USER_ID)
        await self.tg.cb_home(cb, self.state)
        return cb

    async def _tap_pf(self, data: str) -> _FakeCallback:
        cb = _FakeCallback(data, user_id=self.USER_ID)
        await self.tg.cb_portfolio_card(cb, self.state)
        return cb


# ── раскатка флагов ──────────────────────────────────────────────────────────

class RolloutFlagTest(unittest.TestCase):

    def setUp(self) -> None:
        import tg_bot
        self.tg = tg_bot
        self._env = {k: os.environ.get(k) for k in
                     ("MANUAL_PORTFOLIO_ENABLED", "HYBRID_PORTFOLIO_ENABLED", "ADMIN_USER_IDS")}
        os.environ["ADMIN_USER_IDS"] = str(ADMIN)

    def tearDown(self) -> None:
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_modes_are_parsed_fail_closed(self) -> None:
        from portfolio_aggregation import rollout_mode
        for raw, mode in (("on", "on"), ("1", "on"), ("TRUE", "on"), ("admins", "admins"),
                          ("Admin", "admins"), ("off", "off"), ("", "off"), ("maybe", "off")):
            with self.subTest(raw=raw):
                os.environ["MANUAL_PORTFOLIO_ENABLED"] = raw
                self.assertEqual(rollout_mode("MANUAL_PORTFOLIO_ENABLED"), mode)

    def test_admins_mode_is_per_user(self) -> None:
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "admins"
        os.environ["HYBRID_PORTFOLIO_ENABLED"] = "admins"
        self.assertTrue(self.tg.manual_portfolio_enabled(ADMIN))
        self.assertTrue(self.tg.hybrid_portfolio_enabled(ADMIN))
        self.assertFalse(self.tg.manual_portfolio_enabled(12345))
        self.assertFalse(self.tg.hybrid_portfolio_enabled(12345))
        self.assertFalse(self.tg.manual_portfolio_enabled(None),
                         "без пользователя видно только «on»")

    def test_hybrid_still_requires_manual(self) -> None:
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        os.environ["HYBRID_PORTFOLIO_ENABLED"] = "on"
        self.assertFalse(self.tg.hybrid_portfolio_enabled(ADMIN))

    def test_connect_keyboard_follows_the_user(self) -> None:
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "admins"
        self.assertIn("connect:manual", _buttons_of(self.tg.kb_connect_choice(ADMIN)))
        self.assertNotIn("connect:manual", _buttons_of(self.tg.kb_connect_choice(12345)))
        self.assertEqual(_buttons_of(self.tg.kb_connect_choice(ADMIN))[0], "connect:freedom",
                         "брокер — первым: это основной путь")

    def test_commands(self) -> None:
        """Команды — разделы меню; `/forget_portfolio` — только при «on» для всех."""
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "admins"
        cmds = self.tg.bot_commands()
        names = [c.command for c in cmds]
        self.assertEqual(names, ["start", "report", "portfolio", "mandate", "balance",
                                 "help", "support"])
        for c in cmds:
            with self.subTest(command=c.command):
                self.assertRegex(c.command, _TG_CMD)
                self.assertTrue(0 < len(c.description) <= 256)
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "on"
        self.assertEqual(self.tg.bot_commands()[-1].command, "forget_portfolio")

    def test_every_command_has_a_handler(self) -> None:
        import inspect
        src = inspect.getsource(self.tg.build_dispatcher)
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "on"
        for c in self.tg.bot_commands():
            with self.subTest(command=c.command):
                if c.command == "start":
                    self.assertIn("CommandStart()", src)
                else:
                    self.assertIn(f'F.text == "/{c.command}"', src)

    def test_navigation_handlers_registered(self) -> None:
        import inspect
        src = inspect.getsource(self.tg.build_dispatcher)
        for name, prefix in (("cb_home", "home:"), ("cb_portfolio_card", "pf:"),
                             ("cmd_report", "/report")):
            with self.subTest(handler=name):
                self.assertIn(name, src)
                self.assertIn(prefix, src)


class CloudbuildPinsFlagsTest(unittest.TestCase):
    """`--set-env-vars` заменяет ВЕСЬ набор: флаг вне него живёт до деплоя."""

    def setUp(self) -> None:
        path = Path(__file__).resolve().parent.parent / "cloudbuild.yaml"
        if not path.exists():
            self.skipTest("cloudbuild.yaml не входит в образ деплоя")
        import yaml
        self.doc = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_flags_are_on_the_deploy_env_line(self) -> None:
        deploy = next(s for s in self.doc["steps"] if s.get("id") == "deploy")
        env = next(a for a in deploy["args"] if a.startswith("--set-env-vars"))
        for flag in ("MANUAL_PORTFOLIO_ENABLED", "HYBRID_PORTFOLIO_ENABLED"):
            with self.subTest(flag=flag):
                self.assertIn(f"{flag}=${{_{flag}}}", env)
                value = self.doc["substitutions"][f"_{flag}"]
                self.assertIn(value, ("off", "admins", "on"))


# ── главное меню ─────────────────────────────────────────────────────────────

class HomeScreenTest(NavTestBase):

    async def test_returning_user_lands_in_home(self) -> None:
        await self._connect_freedom()
        msg = _FakeMessage("/start", user_id=self.USER_ID)
        await self.tg.cmd_start(msg, self.state)
        text, kw = _last(msg)
        self.assertIn("Главное меню", text)
        self.assertIn("Freedom Broker", text)
        self.assertIn("10 токенов", text)
        self.assertEqual(_buttons_of(kw["reply_markup"]),
                         ["home:report", "home:portfolio", "home:mandate",
                          "home:balance", "home:help"])
        assert_markdown_balanced(self, text)

    async def test_undetermined_source_still_asks_first(self) -> None:
        msg = _FakeMessage("/start", user_id=self.USER_ID)
        await self.tg.cmd_start(msg, self.state)
        self.assertIn("Сначала подключите источник", msg.all_text)
        self.assertIn("connect:freedom", _all_buttons(msg))

    async def test_sections_edit_in_place(self) -> None:
        await self._connect_freedom()
        for action, needle in (("balance", "Баланс: 10 токенов"),
                               ("topup", "Пополнение"),
                               ("help", "Как пользоваться"),
                               ("portfolio", "Мой портфель"),
                               ("menu", "Главное меню")):
            with self.subTest(action=action):
                cb = await self._tap_home(action)
                self.assertEqual(cb.message.sent, [], "меню правится на месте")
                text, kw = cb.message.edited[-1]
                self.assertIn(needle, text)
                assert_markdown_balanced(self, text)
                self.assertTrue(all(len(d.encode()) <= 64
                                    for d in _buttons_of(kw["reply_markup"])))

    async def test_open_sends_a_new_message(self) -> None:
        """`home:open` — под отчётом: строку списания затирать нельзя."""
        await self._connect_freedom()
        cb = await self._tap_home("open")
        self.assertEqual(cb.message.edited, [])
        self.assertIn("Главное меню", cb.message.all_text)

    async def test_single_portfolio_goes_straight_to_tiers(self) -> None:
        """Один настоящий портфель — вопрос «по какому?» не задаётся."""
        await self._connect_freedom()
        cb = await self._tap_home("report")
        self.assertIn("Выберите тип анализа* · Freedom Broker", cb.message.all_text)
        datas = _all_buttons(cb.message)
        self.assertEqual(datas, ["rpt:freedom:base", "rpt:freedom:scenario",
                                 "rpt:freedom:deep", "home:portfolio", "home:menu"])

    async def test_two_portfolios_ask_which_one(self) -> None:
        """Брокер + ручной без гибрида: выбор портфеля, «Freedom + ручной» не упомянут."""
        await self._connect_freedom()
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 1 100 USD")
        cb = await self._tap_home("report")
        self.assertIn("шаг 1 из 2", cb.message.all_text)
        self.assertNotIn("Freedom + ручной", cb.message.all_text, "I-9 гибрида")
        self.assertEqual(_all_buttons(cb.message),
                         ["src:freedom", "src:manual", "src:demo",
                          "home:portfolio", "home:menu"])

    async def test_report_with_hybrid_asks_portfolio_first(self) -> None:
        await self._connect_freedom()
        os.environ["HYBRID_PORTFOLIO_ENABLED"] = "on"
        cb = await self._tap_home("report")
        self.assertIn("шаг 1 из 2", cb.message.all_text)
        self.assertIn("src:freedom", _all_buttons(cb.message))

    async def test_mandate_from_home(self) -> None:
        cb = await self._tap_home("mandate")
        self.assertIn("инвестиционный мандат", cb.message.all_text)
        datas = _all_buttons(cb.message)
        self.assertIn("home:portfolio", datas, "мандат ведёт и к портфелю")
        self.assertIn("home:menu", datas)

    async def test_forged_and_profileless(self) -> None:
        cb = await self._tap_home("hack")
        self.assertEqual(cb.message.all_text, "")
        other = _FakeCallback("home:menu", user_id=999001)
        await self.tg.cb_home(other, self.state)
        self.assertIn("/start", other.message.all_text)

    async def test_help_names_features_by_flag(self) -> None:
        cb = await self._tap_home("help")
        text = cb.message.all_text
        for needle in ("/portfolio", "/report", "/mandate", "/balance", "forget"):
            self.assertIn(needle, text)
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        cb = await self._tap_home("help")
        self.assertNotIn("forget", cb.message.all_text, "I-9: без флага — без ручного")

    async def test_report_command(self) -> None:
        await self._connect_freedom()
        msg = _FakeMessage("/report", user_id=self.USER_ID)
        await self.tg.cmd_report(msg, self.state)
        self.assertIn("Выберите тип анализа", msg.all_text)


# ── «Мой портфель» ───────────────────────────────────────────────────────────

class PortfolioHubTest(NavTestBase):

    async def test_hub_lists_every_source(self) -> None:
        await self._connect_freedom()
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 1 100 USD\nCASH:USD 5")
        msg = _FakeMessage("/portfolio", user_id=self.USER_ID)
        await self.tg.cmd_portfolio(msg, self.state)
        text, kw = _last(msg)
        self.assertIn("Freedom Broker — подключён", text)
        self.assertIn("Ручной портфель — 1 позиция", text)
        self.assertEqual(_buttons_of(kw["reply_markup"]),
                         ["pf:freedom", "mp:show", "pf:demo", "home:menu"])
        assert_markdown_balanced(self, text)

    async def test_hub_without_manual_flag(self) -> None:
        """I-9: без флага — ни строки, ни кнопки ручного портфеля."""
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        msg = _FakeMessage("/portfolio", user_id=self.USER_ID)
        await self.tg.cmd_portfolio(msg, self.state)
        self.assertNotIn("Ручной", msg.all_text)
        self.assertNotIn("mp:show", _all_buttons(msg))

    async def test_demo_user_can_now_connect_the_broker(self) -> None:
        """Раньше пути не было вовсе: экран подключения — только в онбординге."""
        await self.db.save_connection_mode(self.USER_ID, "template")
        cb = await self._tap_pf("pf:freedom")
        self.assertIn("шаг 1 из 3", cb.message.all_text)
        self.assertEqual(await self.state.get_state(),
                         self.tg.PortfolioConnection.Login)
        self.assertIn("cancel", _all_buttons(cb.message))

    async def test_freedom_card_with_keys(self) -> None:
        await self._connect_freedom()
        cb = await self._tap_pf("pf:freedom")
        self.assertIn("подключён", cb.message.all_text)
        self.assertEqual(_all_buttons(cb.message),
                         ["src:freedom", "connect:freedom", "home:portfolio"])

    async def _tap(self, data: str, handler) -> _FakeCallback:
        cb = _FakeCallback(data, user_id=self.USER_ID)
        await handler(cb, self.state)
        await asyncio.sleep(0)
        return cb

    async def test_demo_report_stays_demo_for_a_broker_client(self) -> None:
        """🔴 Ключи в vault «перелечивают» режим `template` в `freedom`
        (`_resolve_portfolio_source`, решение 2026-07-16). Кнопка демо-отчёта,
        построенная на смене режима по умолчанию, молча строила бы ПЛАТНЫЙ
        брокерский отчёт. Портфель едет в callback_data явно — демо остаётся демо."""
        await self._connect_freedom()
        started: list[dict] = []

        async def _bg(**kw):
            started.append(kw)

        cb = await self._tap_pf("pf:demo")
        self.assertIn("src:demo", _all_buttons(cb.message))
        cb = await self._tap("src:demo", self.tg.cb_report_source)
        self.assertIn("демо-портфель", cb.message.all_text)
        self.assertIn("rpt:demo:deep", _all_buttons(cb.message))
        cb = await self._tap("rpt:demo:deep", self.tg.cb_report_tier)
        self.assertIn("Бесплатно", cb.message.all_text)
        with patch.object(self.tg, "_run_analysis_background", _bg):
            await self._tap("rptgo:demo:deep", self.tg.cb_report_tier)
        self.assertEqual(len(started), 1)
        self.assertEqual((started[0]["source"], started[0]["cost"]), ("demo", 0))
        self.assertEqual(await self.db.get_connection_mode_explicit(self.USER_ID),
                         "freedom", "режим по умолчанию не тронут")

    async def test_manual_report_entry(self) -> None:
        cb = await self._tap("src:manual", self.tg.cb_report_source)
        self.assertEqual(await self.state.get_state(), self.tg.ManualPortfolio.Input,
                         "пустой ручной портфель — сразу к вводу")
        await self.db.save_manual_portfolio(self.USER_ID, "AAPL.US 1 100 USD")
        cb = await self._tap("src:manual", self.tg.cb_report_source)
        self.assertIn("ручной портфель", cb.message.all_text)
        self.assertIn("rpt:manual:base", _all_buttons(cb.message))
        os.environ["MANUAL_PORTFOLIO_ENABLED"] = "off"
        cb = await self._tap("src:manual", self.tg.cb_report_source)
        self.assertIn("недоступен", cb.message.all_text, "S-4: флаг на каждом нажатии")

    async def test_freedom_report_without_keys_points_to_connect(self) -> None:
        cb = await self._tap("rpt:freedom:base", self.tg.cb_report_tier)
        self.assertIn("недоступен: подключите Freedom Broker", cb.message.all_text)
        self.assertIn("pf:freedom", _all_buttons(cb.message))

    async def test_forged_pf_callbacks(self) -> None:
        for data in ("pf:use:demo", "pf:use:stooq", "pf:hack", "pf:"):
            with self.subTest(data=data):
                cb = await self._tap_pf(data)
                self.assertEqual(cb.message.all_text, "")


# ── цена перед запуском ──────────────────────────────────────────────────────

class PriceScreenTest(NavTestBase):

    async def _price(self, tier: str) -> _FakeCallback:
        cb = _FakeCallback(f"analysis:{tier}", user_id=self.USER_ID)
        await self.tg.cb_analysis_choice(cb, self.state)
        return cb

    async def test_demo_is_free_not_one_token(self) -> None:
        """Прежний экран обещал демо-пользователю списать 1 токен."""
        await self.db.save_connection_mode(self.USER_ID, "template")
        cb = await self._price("base")
        text = cb.message.all_text
        self.assertIn("Бесплатно", text)
        self.assertNotIn("Стоимость", text)
        self.assertIn("confirm:base:menu", _all_buttons(cb.message))

    async def test_paid_source_shows_real_cost(self) -> None:
        await self._connect_freedom()
        cb = await self._price("deep")
        self.assertIn("Стоимость: *2 токена*", cb.message.all_text)
        assert_markdown_balanced(self, cb.message.all_text)

    async def test_insufficient_balance_offers_topup(self) -> None:
        await self._connect_freedom()
        await self.db.deduct_tokens(self.USER_ID, 9, reason="test")
        cb = await self._price("deep")
        self.assertIn("Не хватает токенов", cb.message.all_text)
        datas = _all_buttons(cb.message)
        self.assertIn("home:topup", datas)
        self.assertNotIn("confirm:deep:menu", datas)

    async def test_forged_tier(self) -> None:
        cb = await self._price("vip")
        self.assertEqual(cb.message.all_text, "")


# ── мелкие экраны ────────────────────────────────────────────────────────────

class SmallScreensTest(NavTestBase):

    async def test_cancel_and_fallback_lead_home(self) -> None:
        cb = _FakeCallback("cancel", user_id=self.USER_ID)
        await self.tg.cb_cancel(cb, self.state)
        self.assertIn("не списаны", cb.message.all_text)
        self.assertIn("home:menu", _all_buttons(cb.message))
        msg = _FakeMessage("привет", user_id=self.USER_ID)
        await self.tg.msg_text_fallback(msg, self.state)
        self.assertIn("home:menu", _all_buttons(msg))

    async def test_mandate_close_goes_home(self) -> None:
        cb = _FakeCallback("mandate:close", user_id=self.USER_ID)
        await self.tg.cb_mandate_action(cb, self.state)
        self.assertIn("Главное меню", cb.message.all_text)

    async def test_scenario_cta_keeps_the_report_message(self) -> None:
        datas = _buttons_of(self.tg._kb_scenario_cta())
        self.assertEqual(datas, ["scenario:cached", "home:open"])
        self.assertEqual(_buttons_of(self.tg.kb_nav(new_message=True)), ["home:open"])

    async def test_plural_tokens(self) -> None:
        for n, word in ((0, "токенов"), (1, "токен"), (2, "токена"), (5, "токенов"),
                        (11, "токенов"), (21, "токен"), (22, "токена"), (111, "токенов")):
            self.assertEqual(self.tg._tokens(n), f"{n} {word}")

    async def test_screen_does_not_duplicate_on_same_content(self) -> None:
        class _NotModified(_FakeMessage):
            async def edit_text(self, *_a, **_k):
                raise RuntimeError("Bad Request: message is not modified")

        cb = _FakeCallback("home:menu", user_id=self.USER_ID)
        cb.message = _NotModified(user_id=self.USER_ID)
        await self.tg._screen(cb, "x", None)
        self.assertEqual(cb.message.sent, [], "повторное нажатие — без дубля")

        class _Gone(_FakeMessage):
            async def edit_text(self, *_a, **_k):
                raise RuntimeError("Bad Request: message to edit not found")

        cb.message = _Gone(user_id=self.USER_ID)
        await self.tg._screen(cb, "x", None)
        self.assertEqual(len(cb.message.sent), 1, "не вышло править — новое сообщение")


if __name__ == "__main__":
    unittest.main()
