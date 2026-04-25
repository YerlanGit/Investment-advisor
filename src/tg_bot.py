"""
RAMP Telegram Bot — aiogram 3.x async entry point.

Deep-link format: t.me/RampBot?start=<source_slug>
Analysis tiers:
  - base  : 1 token  → MAC3 CVaR + allocation table
  - deep  : 2 tokens → base + scenario analysis + fundamental signals

Onboarding FSM (new users only):
  Q1 → Q2 → Q3 → Q4 → Universe → MandateReview → PortfolioConnection → Analysis

PortfolioConnection FSM:
  connect:template → save mode → Analysis menu
  connect:freedom  → Login → ApiKey → save encrypted → Analysis menu
"""

import asyncio
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramConflictError
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from db_tokenomics import (
    approve_mandate,
    deduct_tokens,
    get_balance,
    get_connection_mode,
    get_profile,
    init_db,
    init_user,
    save_connection_mode,
    save_profile,
)
from finance.broker_api import FreedomConnector
from finance.investment_logic import UniversalPortfolioManager
from finance.security import SecureVault
from agent.gatekeeper import run_gatekeeper
from pdf_generator import MOCK_DATA, generate_portfolio_pdf
from profile_manager import ASSET_DISPLAY, ASSET_KEYS, PROFILE_BENCH_TICKER, RiskProfileManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger("ramp_bot")

BOT_TOKEN: str = os.environ["RAMP_BOT_TOKEN"]
VAULT_DB: str  = str(Path(__file__).parent.parent / "data" / "users_vault.db")

# ── FSM ───────────────────────────────────────────────────────────────────────

class Onboarding(StatesGroup):
    Q1            = State()
    Q2            = State()
    Q3            = State()
    Q4            = State()
    Universe      = State()
    MandateReview = State()


class PortfolioConnection(StatesGroup):
    Login     = State()
    ApiKey    = State()
    SecretKey = State()


class AnalysisFlow(StatesGroup):
    awaiting_approval = State()


# ── Onboarding question definitions ──────────────────────────────────────────
# Each option tuple: (button_label, score_points, callback_data)

QUESTIONS: list[dict] = [
    {
        "state": Onboarding.Q1,
        "text":  "🕐 *Вопрос 1 из 4*\nКаков ваш инвестиционный горизонт?",
        "options": [
            ("Менее 1 года",                         1, "ob:q1:1"),
            ("От 1 до 3 лет",                        2, "ob:q1:2"),
            ("Более 3 лет",                          3, "ob:q1:3"),
        ],
    },
    {
        "state": Onboarding.Q2,
        "text":  "🎯 *Вопрос 2 из 4*\nВаша главная инвестиционная цель?",
        "options": [
            ("Сохранение капитала",                  1, "ob:q2:1"),
            ("Умеренный рост",                       2, "ob:q2:2"),
            ("Максимальный рост",                    3, "ob:q2:3"),
        ],
    },
    {
        "state": Onboarding.Q3,
        "text":  "📉 *Вопрос 3 из 4*\nЕсли ваш портфель упадёт на 20%, вы:",
        "options": [
            ("Продам всё, чтобы остановить потери",  1, "ob:q3:1"),
            ("Подожду восстановления",               2, "ob:q3:2"),
            ("Докуплю на просадке",                  3, "ob:q3:3"),
        ],
    },
    {
        "state": Onboarding.Q4,
        "text":  "📚 *Вопрос 4 из 4*\nВаш опыт в инвестировании?",
        "options": [
            ("Нет опыта",                            1, "ob:q4:1"),
            ("До 3 лет",                             2, "ob:q4:2"),
            ("Более 3 лет / профессионал",           3, "ob:q4:3"),
        ],
    },
]

# ── Constants ─────────────────────────────────────────────────────────────────

TIER_COST  = {"base": 1, "deep": 2}
TIER_LABEL = {"base": "Базовый отчёт", "deep": "Глубокий сценарный анализ"}

SOURCE_GREETING: dict[str, str] = {
    "news_apple": "новостей Apple",
    "news_kaspi":  "новостей Kaspi",
}


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _source_label(slug: str) -> str:
    return SOURCE_GREETING.get(slug, f"канала «{slug}»")


def _classify_asset(ticker: str) -> str:
    t = ticker.upper()
    if any(x in t for x in ("BTC", "ETH", "SOL", "BNB")):
        return "Крипто"
    if any(x in t for x in ("GOLD", "GLD", "DBC", "OIL", "PDBC")):
        return "Сырьё"
    if any(x in t for x in ("CASH", "USD", "EUR", "KZT", "RUB")):
        return "Ден. средства"
    if any(x in t for x in ("KAP", "KSPI", "HSBK", "KZTK")):
        return "Акции KZ"
    if any(x in t for x in ("TLT", "AGG", "BND", "TNX", "BOND")):
        return "Облигации"
    return "Акции"


def _get_keys_sync(user_id: int):
    """Blocking helper — must be run in a thread executor."""
    vault = SecureVault(db_name=VAULT_DB)
    return vault.get_user_keys(str(user_id))


def _save_keys_sync(user_id: int, login: str, api_key: str, secret_key: str) -> None:
    """Blocking helper — must be run in a thread executor."""
    vault = SecureVault(db_name=VAULT_DB)
    vault.save_user_keys(str(user_id), login, api_key, secret_key)


def _fetch_and_analyze_sync(api_key: str, secret_key: str = "",
                            bench_ticker: str | None = None) -> dict:
    """
    Blocking: fetches portfolio from broker, runs MAC3 engine.
    Must be run in a thread executor.
    """
    df = FreedomConnector(api_key, secret_key).fetch_portfolio()
    return UniversalPortfolioManager().analyze_all(df, profile_benchmark=bench_ticker)


# ── PDF payload mapping ───────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _build_pdf_payload(results: dict, tier: str) -> dict:
    """
    Map UniversalPortfolioManager.analyze_all() output to the schema
    expected by report.html / MOCK_DATA:
      cvar, sharpe, max_drawdown, risk_pct, assets, scenarios (deep only).
    """
    metrics    = results.get("portfolio_metrics") or {}
    perf_df    = results.get("performance_table")
    total_val  = _safe_float(results.get("total_value"), 1.0) or 1.0

    cvar_raw   = _safe_float(metrics.get("CVaR_95_Daily"),        0.0)
    sharpe_raw = _safe_float(metrics.get("Sharpe_Ratio"),         float("nan"))
    var_raw    = _safe_float(metrics.get("VaR_95_Daily"),         0.0)
    vol_raw    = _safe_float(metrics.get("Total_Volatility_Ann"), 0.0)

    cvar_str   = f"{cvar_raw * 100:.1f}%"
    sharpe_str = f"{sharpe_raw:.2f}" if not math.isnan(sharpe_raw) else "—"
    max_dd_str = f"{var_raw * 100:.1f}%"
    risk_pct   = min(100, max(0, int(vol_raw / 0.40 * 100)))

    assets: list[dict] = []
    if perf_df is not None and not perf_df.empty:
        for _, row in perf_df.iterrows():
            ticker    = str(row.get("Ticker", "—"))
            cur_val   = _safe_float(row.get("Current_Value"), 0.0)
            weight_pct = cur_val / total_val * 100
            euler     = _safe_float(row.get("Euler_Risk_Contribution_Pct"), 0.0)
            assets.append({
                "ticker":      ticker,
                "weight":      f"{weight_pct:.1f}%",
                "asset_class": _classify_asset(ticker),
                "euler_risk":  f"{euler:.1f}%",
            })

    payload: dict = {
        "cvar":         cvar_str,
        "sharpe":       sharpe_str,
        "max_drawdown": max_dd_str,
        "risk_pct":     risk_pct,
        "assets":       assets or MOCK_DATA["assets"],  # fallback if engine returned nothing
    }

    if tier == "deep":
        bm_data   = results.get("benchmark_comparison") or {}
        scenarios = []
        for bm_name, bm in bm_data.items():
            excess  = _safe_float(bm.get("Excess_Return"), 0.0)
            pnl_str = f"+{excess*100:.1f}%" if excess >= 0 else f"{excess*100:.1f}%"
            ir      = _safe_float(bm.get("Information_Ratio"), 0.0)
            beat    = "✅ Обыгрывает" if bm.get("Beating_Benchmark") else "❌ Отстаёт"
            scenarios.append({
                "name":        bm_name,
                "probability": f"IR: {ir:.2f}" if ir else "—",
                "pnl":         pnl_str,
                "driver":      f"{beat} бенчмарк",
            })
        payload["scenarios"] = scenarios

    return payload


# ── MAC3 pipeline ─────────────────────────────────────────────────────────────

async def _build_analysis_payload(user_id: int, tier: str) -> dict:
    """Async orchestrator: vault → broker → MAC3 engine → PDF payload."""
    loop      = asyncio.get_running_loop()
    conn_mode = await get_connection_mode(user_id)

    profile    = await get_profile(user_id)
    bench_tick = PROFILE_BENCH_TICKER.get(profile["profile_name"]) if profile else None

    if conn_mode == "freedom":
        keys = await loop.run_in_executor(None, _get_keys_sync, user_id)
        if keys is None:
            # Fall back to service-level env var credentials when vault has no keys.
            api_key    = os.getenv("FREEDOM_API_KEY", "demo")
            secret_key = os.getenv("FREEDOM_API_SECRET", "")
        else:
            _, api_key, secret_key = keys
    else:
        api_key    = "demo"
        secret_key = ""

    try:
        results = await loop.run_in_executor(
            None, _fetch_and_analyze_sync, api_key, secret_key, bench_tick
        )
    except RuntimeError as exc:
        logger.error("Freedom Broker API ошибка для %s: %s", user_id, exc)
        raise RuntimeError(
            f"Не удалось получить данные от Freedom Broker: {exc}\n\n"
            "Проверьте, что ваш API-ключ действителен, и повторите попытку."
        ) from exc

    if profile:
        gate_limits = {
            "max_portfolio_volatility": profile["target_volatility"] * 1.2
        }
        gate = run_gatekeeper(results, user_limits=gate_limits)
        if not gate["passed"]:
            logger.warning("Gatekeeper нарушения: %s", gate["critical"])

    return _build_pdf_payload(results, tier)


# ── Keyboard builders ─────────────────────────────────────────────────────────

def kb_question(options: list[tuple[str, int, str]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=cb)]
        for label, _, cb in options
    ])


def kb_universe(selected: set[str]) -> InlineKeyboardMarkup:
    rows = []
    for key in ASSET_KEYS:
        label = ("✅ " if key in selected else "") + ASSET_DISPLAY[key]
        rows.append([InlineKeyboardButton(text=label, callback_data=f"ob:uni:{key}")])
    rows.append([
        InlineKeyboardButton(text="Подтвердить выбор ➡️", callback_data="ob:uni:confirm")
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_mandate_review() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Утвердить мандат", callback_data="ob:mandate:approve"),
        InlineKeyboardButton(text="✏️ Изменить",        callback_data="ob:mandate:edit"),
    ]])


def kb_connect_choice() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📋 Демо-режим (Шаблон)", callback_data="connect:template")],
        [InlineKeyboardButton(text="🔗 Freedom Broker API",  callback_data="connect:freedom")],
    ])


def kb_analysis_choice() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📊 Базовый (1 токен)",   callback_data="analysis:base"),
        InlineKeyboardButton(text="🔬 Глубокий (2 токена)", callback_data="analysis:deep"),
    ]])


def kb_confirm(tier: str, context_slug: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="✅ Одобрить",
            callback_data=f"confirm:{tier}:{context_slug}",
        ),
        InlineKeyboardButton(text="❌ Отмена", callback_data="cancel"),
    ]])


# ── Shared UI helpers ─────────────────────────────────────────────────────────

async def _edit_or_answer(
    target: Message | CallbackQuery,
    state: FSMContext,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    """Edit the active onboarding message when possible; otherwise send new."""
    data      = await state.get_data()
    ob_msg_id = data.get("ob_message_id")

    if isinstance(target, CallbackQuery) and ob_msg_id:
        try:
            await target.message.edit_text(
                text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )
            return
        except Exception:
            pass

    if isinstance(target, CallbackQuery):
        sent = await target.message.answer(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
        )
    else:
        sent = await target.answer(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
        )
    await state.update_data(ob_message_id=sent.message_id)


async def send_question(
    target: Message | CallbackQuery,
    state: FSMContext,
    q_idx: int,
) -> None:
    """Advance to Q q_idx (0-based) or to the Universe step when q_idx == 4."""
    if q_idx < 4:
        q = QUESTIONS[q_idx]
        await state.set_state(q["state"])
        await _edit_or_answer(target, state, q["text"], kb_question(q["options"]))
    else:
        data     = await state.get_data()
        selected = set(data.get("universe", []))
        await state.set_state(Onboarding.Universe)
        if "universe" not in data:
            await state.update_data(universe=[])
        await _edit_or_answer(
            target, state,
            "🌍 *Выбор классов активов*\n\n"
            "Отметьте классы активов, которые вы хотите включить в портфель:",
            kb_universe(selected),
        )


async def _show_analysis_menu(message: Message, slug: str) -> None:
    """Send the final analysis choice, with a deep-link context if slug is set."""
    if slug:
        await message.answer(
            f"👋 Кстати, вы пришли из нашего канала _{_source_label(slug)}_.\n\n"
            "Хотите узнать, как эта новость влияет на ваш портфель?\n\n"
            "💰 *Базовый отчёт:* 1 токен\n"
            "🔬 *Глубокий анализ:* 2 токена",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_analysis_choice(),
        )
    else:
        await message.answer(
            "🚀 *Выберите тип анализа вашего портфеля:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_analysis_choice(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARDING ROUTER
# ══════════════════════════════════════════════════════════════════════════════

onboarding_router = Router(name="onboarding")


# ── /start ────────────────────────────────────────────────────────────────────

async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    user_id = message.from_user.id
    slug    = message.text.split(maxsplit=1)[1].strip() if " " in (message.text or "") else ""
    profile = await get_profile(user_id)

    if profile is None:
        # NEW user — launch onboarding; token grant is deferred until mandate approval.
        await state.update_data(slug=slug)
        await state.set_state(Onboarding.Q1)
        q    = QUESTIONS[0]
        sent = await message.answer(
            "👋 *Добро пожаловать в RAMP — Risk & Asset Management Platform!*\n\n"
            "Прежде чем начать, пройдите короткое анкетирование (4 вопроса), "
            "чтобы мы могли составить ваш персональный инвестиционный мандат.\n\n"
            + q["text"],
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_question(q["options"]),
        )
        await state.update_data(ob_message_id=sent.message_id)

    else:
        # RETURNING user — original flow unchanged.
        await init_user(user_id)  # returns False — no double-grant
        if slug:
            await state.update_data(context_slug=slug)
            await message.answer(
                f"👋 Привет! Я вижу, вы пришли из нашего канала _{_source_label(slug)}_.\n\n"
                "Я могу проанализировать, как эта новость повлияет на ваш портфель.\n\n"
                "💰 *Стоимость базового отчета:* 1 токен\n"
                "🔬 *Глубокий сценарный анализ:* 2 токена\n\n"
                "Начать анализ?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_analysis_choice(),
            )
        else:
            balance = await get_balance(user_id)
            await message.answer(
                f"📊 *RAMP — Risk & Asset Management Platform*\n\n"
                f"Ваш баланс: *{balance} токен(а)*\n\n"
                "Выберите тип анализа вашего портфеля:",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_analysis_choice(),
            )


# ── Q1–Q4 answer handler ──────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.regexp(r"^ob:q[1-4]:\d$"),
    StateFilter(Onboarding.Q1, Onboarding.Q2, Onboarding.Q3, Onboarding.Q4),
)
async def cb_question_answer(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, q_part, pts_str = callback.data.split(":")
    q_num = int(q_part[1])
    pts   = int(pts_str)
    await state.update_data(**{f"q{q_num}": pts})
    await send_question(callback, state, q_num)


# ── Universe toggle ───────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data.startswith("ob:uni:"),
    ~F.data.endswith("confirm"),
    StateFilter(Onboarding.Universe),
)
async def cb_universe_toggle(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    asset_key = callback.data[len("ob:uni:"):]
    data      = await state.get_data()
    universe: list[str] = list(data.get("universe", []))
    if asset_key in universe:
        universe.remove(asset_key)
    else:
        universe.append(asset_key)
    await state.update_data(universe=universe)
    await callback.message.edit_reply_markup(reply_markup=kb_universe(set(universe)))


# ── Universe confirm ──────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:uni:confirm",
    StateFilter(Onboarding.Universe),
)
async def cb_universe_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data     = await state.get_data()
    universe = data.get("universe", [])

    if not universe:
        await callback.message.answer("⚠️ Выберите хотя бы один класс активов.")
        return

    score   = sum(data.get(f"q{i}", 0) for i in range(1, 5))
    profile = RiskProfileManager.score_to_profile(score)
    limits  = RiskProfileManager.apply_universe(profile, universe)
    summary = RiskProfileManager.build_mandate_summary(profile, limits)

    await state.update_data(profile_data={
        "name":       profile["name"],
        "target_vol": profile["target_vol"],
        "target_te":  profile["target_te"],
        "score":      score,
        "limits":     limits,
    })
    await state.set_state(Onboarding.MandateReview)
    await _edit_or_answer(callback, state, summary, kb_mandate_review())


# ── Mandate approve ───────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:mandate:approve",
    StateFilter(Onboarding.MandateReview),
)
async def cb_mandate_approve(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    user_id  = callback.from_user.id
    data     = await state.get_data()
    prof     = data["profile_data"]
    slug     = data.get("slug", "")
    universe = data.get("universe", [])

    await save_profile(
        telegram_id       = user_id,
        score             = prof["score"],
        profile_name      = prof["name"],
        target_volatility = prof["target_vol"],
        target_te         = prof["target_te"],
        selected_assets   = universe,
        limits_dict       = prof["limits"],
    )
    await approve_mandate(user_id)
    await init_user(user_id)   # grants tokens — user is new here

    # Preserve slug for the connection sub-flow, then reset everything else.
    await state.clear()
    if slug:
        await state.update_data(slug=slug)

    balance = await get_balance(user_id)
    await callback.message.answer(
        f"🎉 *Мандат утверждён! Добро пожаловать в RAMP.*\n\n"
        f"Ваш профиль: *{prof['name']}*\n"
        f"На ваш счёт зачислено *{balance} токен(а)*.\n\n"
        "Последний шаг: подключите источник данных о вашем портфеле.",
        parse_mode=ParseMode.MARKDOWN,
    )
    await callback.message.answer(
        "📡 *Как вы хотите подключить ваш портфель?*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_connect_choice(),
    )


# ── Mandate edit ──────────────────────────────────────────────────────────────

@onboarding_router.callback_query(
    F.data == "ob:mandate:edit",
    StateFilter(Onboarding.MandateReview),
)
async def cb_mandate_edit(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data     = await state.get_data()
    universe = set(data.get("universe", []))
    await state.set_state(Onboarding.Universe)
    await _edit_or_answer(
        callback, state,
        "🌍 *Выбор классов активов*\n\nОтметьте классы активов для вашего портфеля:",
        kb_universe(universe),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONNECTION ROUTER
# ══════════════════════════════════════════════════════════════════════════════

portfolio_router = Router(name="portfolio_connection")


@portfolio_router.callback_query(F.data.startswith("connect:"))
async def cb_connect_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, mode  = callback.data.split(":", 1)
    user_id  = callback.from_user.id
    fsm_data = await state.get_data()
    slug     = fsm_data.get("slug", "")

    if mode == "template":
        await save_connection_mode(user_id, "template")
        await callback.message.edit_text(
            "✅ *Демо-режим активирован.*\n\n"
            "Для анализа будет использоваться шаблонный институциональный портфель.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        await _show_analysis_menu(callback.message, slug)

    elif mode == "freedom":
        await state.update_data(slug=slug)
        await state.set_state(PortfolioConnection.Login)
        await callback.message.edit_text(
            "⚠️ *Важно:* RAMP использует API исключительно для режима ЧТЕНИЯ (Read-Only) "
            "сырых данных для глубокого квантового анализа. "
            "Мы не имеем права совершать сделки. "
            "В целях безопасности, после ввода ключей, пожалуйста, "
            "удалите свои сообщения из истории чата.\n\n"
            "🔐 Введите ваш *Логин* в Freedom Broker:",
            parse_mode=ParseMode.MARKDOWN,
        )


@portfolio_router.message(StateFilter(PortfolioConnection.Login))
async def msg_login(message: Message, state: FSMContext) -> None:
    await state.update_data(connect_login=message.text.strip())
    await state.set_state(PortfolioConnection.ApiKey)
    await message.answer(
        "🔑 Введите ваш *API Key*:",
        parse_mode=ParseMode.MARKDOWN,
    )


@portfolio_router.message(StateFilter(PortfolioConnection.ApiKey))
async def msg_api_key(message: Message, state: FSMContext) -> None:
    await state.update_data(connect_api_key=message.text.strip())
    await state.set_state(PortfolioConnection.SecretKey)
    await message.answer(
        "🔑 Введите ваш *Secret Key* (приватный ключ):",
        parse_mode=ParseMode.MARKDOWN,
    )


@portfolio_router.message(StateFilter(PortfolioConnection.SecretKey))
async def msg_secret_key(message: Message, state: FSMContext) -> None:
    secret_key = message.text.strip()
    data       = await state.get_data()
    login      = data.get("connect_login", "")
    api_key    = data.get("connect_api_key", "")
    slug       = data.get("slug", "")
    user_id    = message.from_user.id

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _save_keys_sync, user_id, login, api_key, secret_key)
    await save_connection_mode(user_id, "freedom")
    await state.clear()

    await message.answer(
        "🔐 *Подключение успешно. Ваши данные под квантовой защитой.*\n\n"
        "✅ API-ключ и Secret Key надёжно зашифрованы. Пожалуйста, удалите "
        "предыдущие сообщения с ключами из истории чата.",
        parse_mode=ParseMode.MARKDOWN,
    )
    await _show_analysis_menu(message, slug)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FLOW
# ══════════════════════════════════════════════════════════════════════════════

async def _send_pdf(
    bot: Bot,
    chat_id: int,
    user_id: int,
    tier: str,
    payload: dict | None = None,
) -> None:
    report_type = TIER_LABEL[tier]
    pdf_path    = await generate_portfolio_pdf(payload, user_id=user_id, report_type=report_type)

    with open(pdf_path, "rb") as fh:
        doc = BufferedInputFile(fh.read(), filename=f"RAMP_Report_{user_id}.pdf")

    await bot.send_document(
        chat_id,
        document=doc,
        caption=(
            f"📄 *{report_type}* готов.\n\n"
            "Данный отчёт сформирован на основе институциональных моделей рисков "
            "(MAC3, Euler Decomposition, CVaR). Помните: штурвал всегда у вас."
        ),
        parse_mode=ParseMode.MARKDOWN,
    )


async def cb_analysis_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    _, tier  = callback.data.split(":", 1)
    cost     = TIER_COST[tier]
    balance  = await get_balance(callback.from_user.id)
    ctx      = await state.get_data()
    context_slug = ctx.get("context_slug", "menu")

    await state.update_data(tier=tier)
    await callback.message.edit_text(
        f"⚠️ *Внимание:*\n\n"
        f"Данный анализ потребует сложных нейросетевых и квантовых вычислений.\n\n"
        f"С вашего баланса будет списано *{cost} токен(а)*.\n"
        f"Текущий баланс: *{balance} токен(а)*.\n\n"
        "Одобрить?",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_confirm(tier, context_slug),
    )
    await state.set_state(AnalysisFlow.awaiting_approval)


async def cb_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()  # acknowledge immediately to prevent "query is too old"
    _, tier, context_slug = callback.data.split(":", 2)
    user_id = callback.from_user.id
    cost    = TIER_COST[tier]

    success = await deduct_tokens(user_id, cost, reason=f"{tier}_analysis")
    if not success:
        balance = await get_balance(user_id)
        await callback.message.edit_text(
            f"❌ *Недостаточно токенов.*\n\n"
            f"Требуется: *{cost}*, доступно: *{balance}*.\n\n"
            "Пополните баланс командой /topup.",
            parse_mode=ParseMode.MARKDOWN,
        )
        await state.clear()
        return

    balance_after = await get_balance(user_id)
    await callback.message.edit_text(
        f"⚙️ Генерирую *{TIER_LABEL[tier]}*…\n"
        "⏳ Загружаю рыночные данные и запускаю MAC3 движок...\n\n"
        f"Остаток баланса: *{balance_after} токен(а)*.",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        payload = await _build_analysis_payload(user_id, tier)
        await _send_pdf(callback.message.bot, callback.message.chat.id, user_id, tier, payload)
    except Exception as exc:
        logger.exception("Ошибка генерации PDF для %s: %s", user_id, exc)
        error_detail = str(exc)[:200]  # Truncate very long errors
        await callback.message.answer(
            f"⚠️ *Ошибка при анализе портфеля:*\n\n"
            f"`{error_detail}`\n\n"
            "Проверьте настройки API-ключа или попробуйте позже.\n"
            "Токены не потеряны — обратитесь в поддержку /support.",
            parse_mode=ParseMode.MARKDOWN,
        )

    await state.clear()


async def cb_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.clear()
    await callback.message.edit_text("❌ Анализ отменён. Токены не списаны.")


# ── Utility commands ──────────────────────────────────────────────────────────

async def cmd_balance(message: Message) -> None:
    balance = await get_balance(message.from_user.id)
    await message.answer(
        f"💳 *Ваш баланс:* {balance} токен(а)\n\n"
        "Пополнить: /topup\n"
        "Тарифы: 10 токенов = 5 000 KZT (1 токен = 500 KZT)",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_topup(message: Message) -> None:
    await message.answer(
        "💰 *Пополнение баланса*\n\n"
        "Тариф: *10 токенов за 5 000 KZT* (1 токен = 500 KZT)\n\n"
        "Для оплаты обратитесь к администратору или используйте платёжный шлюз (скоро).",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_support(message: Message) -> None:
    await message.answer(
        "🛟 *Поддержка RAMP*\n\n"
        "По всем вопросам пишите: @ramp_support_bot\n"
        "Часы работы: пн–пт, 09:00–18:00 (UTC+5).",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── Dispatcher assembly ───────────────────────────────────────────────────────

def build_dispatcher() -> Dispatcher:
    dp = Dispatcher(storage=MemoryStorage())

    # Routers first — StateFilter guards prevent cross-fire with AnalysisFlow.
    dp.include_router(onboarding_router)
    dp.include_router(portfolio_router)

    # Message commands
    dp.message.register(cmd_start,   CommandStart())
    dp.message.register(cmd_balance, F.text == "/balance")
    dp.message.register(cmd_topup,   F.text == "/topup")
    dp.message.register(cmd_support, F.text == "/support")

    # Analysis flow callbacks
    dp.callback_query.register(cb_analysis_choice, F.data.startswith("analysis:"))
    dp.callback_query.register(cb_confirm,          F.data.startswith("confirm:"))
    dp.callback_query.register(cb_cancel,           F.data == "cancel")

    return dp


async def main() -> None:
    await init_db()
    bot = Bot(token=BOT_TOKEN)
    dp  = build_dispatcher()
    logger.info("RAMP Bot запущен.")
    while True:
        try:
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
            break
        except TelegramConflictError:
            delay = random.uniform(3, 6)
            logger.warning(
                "TelegramConflictError: другой экземпляр занял polling. "
                "Повтор через %.1f сек...", delay
            )
            await asyncio.sleep(delay)


if __name__ == "__main__":
    asyncio.run(main())
