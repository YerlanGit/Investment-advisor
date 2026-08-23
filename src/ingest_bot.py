"""
OMBRI Ingest Bot — второй Telegram-бот, только для загрузки данных (IB-3…IB-7).

Что это и чем он НЕ является
────────────────────────────
Оператор пересылает сюда дневной срез Stooq — бот применяет его к базе
котировок и публикует базу. Отчётов он не строит, токеномику не трогает и в
бакет с балансами и ключами брокера не ходит: у него нет ни кода, ни прав
(`PLAN §7.1`).

**Слой L4 Delivery.** Здесь только чтение сообщения и печать сводки; весь цикл
живёт в `services/quote_ingest`, разбор файлов — в `finance/stooq_ingest`. Тот
же разрез, которым `analyze_all` отделён от `tg_bot`, и по той же причине:
логика, отделённая от доставки, проверяется без бота и без токена.

🔴 Этот модуль **не импортирует `tg_bot` и не импортируется им**. Общее у двух
ботов — `env_config`, `finance/*` и `services/*`, то есть L0/L1. Иначе они
срастутся, и правка онбординга сломает загрузку котировок.

Почему бот отдельный — пять причин в `PLAN §2`; коротко: главный бот открывает
базу `mode=ro`, и это инвариант, который писатель в том же процессе разрушил бы
по построению.
"""

from __future__ import annotations

import asyncio
import html
import importlib
import logging
import os
import signal
import tempfile
from pathlib import Path
from typing import Optional

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramConflictError
from aiogram.filters import CommandStart
from aiogram.types import (CallbackQuery, InlineKeyboardButton,
                           InlineKeyboardMarkup, Message, TelegramObject)

import ingest_access as access
from env_config import env_int
from services import quote_ingest as qi
from services.quote_publisher import PublisherUnavailable, publisher_from_env

logger = logging.getLogger("ombri.ingest")

#: Токен ЭТОГО бота.  Проект переезжает с имени RAMP на OMBRI, поэтому имя
#: переменной объявлено, а не вписано в четыре места: сообщение об ошибке,
#: страж общего токена и два теста читают его отсюда.
TOKEN_ENV = "OMBRI_INGEST_BOT_TOKEN"
#: Прежнее имя той же переменной.  Читается ВТОРЫМ и с предупреждением: пока
#: секрет в Secret Manager не переименован, бот обязан подниматься, а не
#: молчать про «пустой токен», — но и делать вид, что всё в порядке, он не
#: вправе, иначе переезд не закончится никогда.
LEGACY_TOKEN_ENV = "RAMP_INGEST_BOT_TOKEN"

#: Токен ГЛАВНОГО бота — под обоими именами.  Здесь он нужен ровно для одного:
#: убедиться, что это НЕ ТОТ ЖЕ токен (`_refuse_shared_token`).  Главный бот
#: этим модулем не настраивается и не переименовывается: он работает, и
#: трогать его имя переменной из пристройки было бы ровно тем срастанием
#: двух ботов, которого весь этот файл и избегает.
MAIN_TOKEN_ENVS = ("OMBRI_BOT_TOKEN", "RAMP_BOT_TOKEN")


def read_bot_token() -> str:
    """Токен бота: новое имя, при его отсутствии — прежнее, с предупреждением."""
    value = str(os.getenv(TOKEN_ENV, "") or "").strip()
    if value:
        return value
    value = str(os.getenv(LEGACY_TOKEN_ENV, "") or "").strip()
    if value:
        logger.warning(
            "%s пуст — взял устаревший %s. Переименуйте секрет в Secret Manager "
            "(подстановка `_INGEST_BOT_TOKEN_SECRET` в cloudbuild.yaml).",
            TOKEN_ENV, LEGACY_TOKEN_ENV)
    return value


BOT_TOKEN = read_bot_token()

#: Потолок ожидания одной операции. Цикл — скачать базу, применить, залить;
#: замер применения 0.2 с, остальное упирается в сеть. Пять минут с запасом,
#: и они нужны как ГРАНИЦА: без неё зависшая загрузка держала бы замок вечно,
#: и оператор не смог бы прислать следующий файл, не поняв почему.
APPLY_TIMEOUT_S = env_int("INGEST_APPLY_TIMEOUT_S", 300, lo=30, hi=3600)

#: Потолок сообщения Telegram — 4096 символов. Сводка короче, но `/universe` и
#: `/prune` печатают списки, а обрезанное сообщение всё же лучше НЕ доставленного:
#: ошибка длины возвращается 400-м и убивает ответ целиком.
_TG_LIMIT = 3900

#: Сколько раз пережидать 409 при старте. Cloud Run на деплое держит старую
#: ревизию несколько секунд, и обе держат `getUpdates` — без ожидания новая
#: падает, и сервис остаётся лежать. Приём и числа — из `tg_bot`.
_CONFLICT_RETRIES = env_int("INGEST_CONFLICT_RETRIES", 5, lo=1, hi=20)
_CONFLICT_DELAY_S = env_int("INGEST_CONFLICT_DELAY_S", 5, lo=1, hi=60)

#: Что рабочие потоки тянут ПЕРВЫМ обращением. Всё это импортируется ЛЕНИВО,
#: внутри функций, которые бот исполняет через `asyncio.to_thread`:
#:   `_engine_universe`    → `finance.data_checks`   (numpy + pandas)
#:   `_market_stale_limit` → `finance.stooq_provider` (pandas)
#:   `_working_set`        → `finance.stooq_symbols`
#:   `GcsQuotePublisher`   → `google.cloud.storage`
#:
#: 🔴 `§−99`, тот же класс, что уронил главный бот в `§−98`. Команды под
#: `_LOCK` не стоят: `/status`, `/universe`, `/check` и плановая проверка
#: `run_scheduled_check` идут МИМО замка. Значит два рабочих потока могут
#: войти в numpy одновременно — оператор шлёт файл (поток применяет и зовёт
#: `_c1`) и в ту же секунду жмёт `/status` (второй поток зовёт
#: `_market_stale_limit`). CPython отдаёт одному из них полуинициализированный
#: модуль, и импорт падает `cannot import name 'NDArray' …`.
#:
#: Здесь это не убивает контейнер — исключения ловятся, — и потому ХУЖЕ: `_c1`
#: ловит ровно `ImportError` и печатает «C-1 НЕ ПРОВЕРЕН», а
#: `_market_stale_limit` возвращает `None`, и срок до блокировки исчезает из
#: сводки. То есть гонка тихо подменяет ответ оператору ровно в тот момент,
#: когда он этот ответ и читает.
#:
#: Список СВОЙ, а не общий с `entrypoint._BOOT_INGEST_HEAVY_IMPORTS`: у
#: загрузчика другие потоки и другая работа. Общий здесь — приём, а не факт;
#: одну редакцию правды (`finance/*`) оба списка и читают.
_WORKER_HEAVY_IMPORTS: tuple[str, ...] = (
    "numpy",                       # общий корень гонки — грузим самым первым
    "pandas",
    "finance.data_checks",         # → quote_ingest._engine_universe
    "finance.stooq_provider",      # → quote_ingest._market_stale_limit
    "finance.stooq_symbols",       # → quote_ingest._working_set, si.resolve_symbol
    "google.cloud.storage",        # → quote_publisher.GcsQuotePublisher._bucket
)


def preimport_worker_deps() -> None:
    """Загрузить тяжёлое В ГЛАВНОМ ПОТОКЕ, до первого `asyncio.to_thread`.

    Лечение то же, что в `§−98`: не ретрай, а ПОРЯДОК. К моменту, когда
    рабочий поток впервые попросит `finance.data_checks`, numpy и pandas уже
    целиком в `sys.modules`, и гонке не на чем возникнуть.

    Зовётся из `main` ПОСЛЕ того, как health-сервер занял порт (`_serve`
    создаётся первой задачей `gather` и биндит сокет до первого await здесь),
    поэтому стартовая проба Cloud Run от этих секунд не зависит.

    Ошибка импорта не фатальна: в офлайн-режиме `google.cloud.storage` может
    отсутствовать вовсе, а без pandas бот обязан не падать, а честно сказать
    «C-1 не проверен» — это уже поведение `quote_ingest`.
    """
    for what in _WORKER_HEAVY_IMPORTS:
        try:
            importlib.import_module(what)
        except Exception as exc:                        # noqa: BLE001
            logger.info("предзагрузка %s не удалась (%s) — "
                        "команда, которой он нужен, скажет это сама", what, exc)


#: Одна операция за раз. Единственная конкуренция, которая здесь реальна, —
#: два файла, присланных одним человеком подряд: цикл read-modify-write не
#: терпит наложения, а CAS отбил бы второй файл конфликтом, и оператор увидел
#: бы тревогу там, где никакого нарушения не было.
_LOCK = asyncio.Lock()

#: Бот, созданный в `main`. Нужен планировщику (IB-7), который приходит
#: снаружи по HTTP и своего экземпляра не имеет.
_BOT: Optional[Bot] = None

_HELP = (
    "🛠 <b>OMBRI Ingest</b> — обновление базы котировок\n\n"
    "<b>Как обновлять базу</b>\n"
    "1. скачайте дневной срез со stooq.com/db/ (<code>YYYYMMDD_d.txt</code>, ≈700 КБ)\n"
    "2. перешлите файл сюда\n"
    "3. прочитайте сводку\n\n"
    "<b>Добор бумаги.</b> Пришлите файл истории из архива "
    "(<code>aapl.us.txt</code>) — бумага заведётся в базе.\n\n"
    "<b>Команды</b>\n"
    "/status — поколение базы, свежесть рынков, допуск C-1\n"
    "/missing — какие дневные файлы переслать\n"
    "/universe — покрытие факторов C-1 с глубиной истории\n"
    "/check ТИКЕР — есть ли бумага в базе (тикером движка)\n"
    "/prune — вычистить бумаги с обрывочной историей\n"
    "/help — эта справка\n\n"
    "🔴 Бутстрап через чат невозможен: архив истории весит 0.84 ГБ, "
    "Telegram отдаёт боту не больше 20 МБ. Сборка базы с нуля остаётся "
    "операцией на вашей машине."
)


class AdminOnlyMiddleware(BaseMiddleware):
    """Fail-CLOSED: пустой список админов = никого.

    🔴 Зеркальная противоположность `tg_bot.WhitelistMiddleware`, и это
    сознательно. Там пустая настройка пропускает всех, потому что запертая
    бета хуже открытой; здесь пустая настройка не должна открывать запись в
    базу цен. Разбор — `ingest_access`.
    """

    async def __call__(self, handler, event: TelegramObject, data: dict):
        user = getattr(event, "from_user", None)
        if user is None:
            inner = getattr(event, "message", None)
            user = getattr(inner, "from_user", None)
        user_id = getattr(user, "id", None)
        if access.is_admin(user_id):
            return await handler(event, data)
        access.note_stranger(user_id, f"событие {type(event).__name__}")
        try:
            answer = getattr(event, "answer", None)
            if callable(answer):
                await answer(access.DENIAL_TEXT)
        except Exception as exc:                        # noqa: BLE001
            logger.debug("отказ не доставлен: %s", exc)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Доставка текста
# ═════════════════════════════════════════════════════════════════════════════

def render_block(text: str, *, limit: int = _TG_LIMIT) -> str:
    """Моноширинный блок для Telegram — HTML, а не Markdown.

    🔴 Legacy-Markdown разбирает содержимое даже внутри тройных кавычек и
    отвечает 400-м на несбалансированную разметку. Сводка при этом не
    доставляется ВООБЩЕ — а доставка сводки и есть смысл этого бота: правило 9
    существует ровно ради того, чтобы человек УЗНАЛ про недокачанный файл.
    `html.escape` снимает вопрос целиком: в `<pre>` уезжает текст, а не разметка.
    """
    body = str(text or "")
    if len(body) > limit:
        body = body[:limit] + "\n… сообщение обрезано, полный вывод в логах"
    return f"<pre>{html.escape(body)}</pre>"


async def _send_block(message: Message, text: str) -> None:
    await message.answer(render_block(text), parse_mode=ParseMode.HTML)


# ═════════════════════════════════════════════════════════════════════════════
# Хендлеры
# ═════════════════════════════════════════════════════════════════════════════

async def cmd_start(message: Message) -> None:
    await message.answer(_HELP, parse_mode=ParseMode.HTML)


async def cmd_help(message: Message) -> None:
    await message.answer(_HELP, parse_mode=ParseMode.HTML)


async def _guarded(message: Message, work, *args) -> None:
    """Общая обвязка команд: блокирующая работа в поток, отказы — текстом.

    Ни одна команда не имеет права уронить polling: упавший хендлер оставит
    оператора без ответа и без объяснения, а второго канала связи у него нет.
    """
    try:
        result = await asyncio.to_thread(work, *args)
    except PublisherUnavailable as exc:
        await _send_block(message, f"🔴 хранилище не настроено\n   {exc}")
        return
    except Exception as exc:                            # noqa: BLE001
        logger.exception("%s упал", getattr(work, "__name__", work))
        await _send_block(message, f"🔴 не смог выполнить: {exc}")
        return
    await _send_block(message, result)


async def cmd_status(message: Message) -> None:
    await _guarded(message, lambda: qi.format_status(qi.status()))


async def cmd_missing(message: Message) -> None:
    await _guarded(message, lambda: qi.format_missing(qi.status()))


async def cmd_universe(message: Message) -> None:
    await _guarded(message, lambda: qi.format_universe(qi.universe_report()))


async def cmd_check(message: Message) -> None:
    """`/check BRK.B` — найдёт ли отчёт эту бумагу и под какой формой символа."""
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Формат: <code>/check AAPL.US</code>",
                             parse_mode=ParseMode.HTML)
        return
    ticker = parts[1]
    await _guarded(message, lambda: qi.format_probe(qi.check_ticker(ticker)))


async def cmd_prune(message: Message) -> None:
    """Чистка показывает `--dry-run` ПЕРВЫМ и ждёт подтверждения кнопкой.

    🔴 Единственная необратимая команда бота. В чате сообщение отправляется
    одним касанием, и отменить его нечем — поэтому между намерением и
    удалением стоит ещё одно осознанное действие.
    """
    async def _work() -> tuple[str, bool]:
        outcome = await asyncio.to_thread(qi.prune, dry_run=True,
                                          actor=_actor(message))
        return qi.format_prune(outcome), bool(outcome.ok and outcome.removed)

    try:
        text, offer = await _work()
    except Exception as exc:                            # noqa: BLE001
        logger.exception("prune --dry-run упал")
        await _send_block(message, f"🔴 не смог посчитать чистку: {exc}")
        return
    markup = None
    if offer:
        markup = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="🗑 Выполнить чистку",
                                 callback_data="prune:go"),
            InlineKeyboardButton(text="Отмена", callback_data="prune:no"),
        ]])
    await message.answer(render_block(text), parse_mode=ParseMode.HTML,
                         reply_markup=markup)


async def cb_prune(callback: CallbackQuery) -> None:
    await callback.answer()
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:                                   # noqa: BLE001
        pass
    if callback.data == "prune:no":
        await callback.message.answer("Отменено — база не тронута.")
        return
    async with _LOCK:
        try:
            outcome = await asyncio.wait_for(
                asyncio.to_thread(qi.prune, dry_run=False,
                                  actor=_actor(callback)),
                timeout=APPLY_TIMEOUT_S)
            text = qi.format_prune(outcome)
        except asyncio.TimeoutError:
            text = f"🔴 чистка не уложилась в {APPLY_TIMEOUT_S} с."
        except Exception as exc:                        # noqa: BLE001
            logger.exception("prune упал")
            text = f"🔴 не смог вычистить: {exc}"
    await callback.message.answer(render_block(text), parse_mode=ParseMode.HTML)


def _actor(event) -> str:
    return str(getattr(getattr(event, "from_user", None), "id", "?"))


async def on_document(message: Message) -> None:
    """Приём файла: лимиты → скачивание → применение → сводка.

    🔴 Порядок обязателен. Лимиты проверяются ПО МЕТАДАННЫМ, до `getFile`:
    Telegram отдаёт размер в описании документа, и отказать по нему дешевле,
    чем скачать архив в `/tmp`, который на Cloud Run является оперативной
    памятью.
    """
    document = message.document
    if document is None:                                # pragma: no cover
        return
    decision = qi.classify_upload(document.file_name, document.file_size)
    if not decision.accepted:
        await _send_block(message, f"🔴 файл не принят\n   {decision.reason}")
        return

    if _LOCK.locked():
        await message.answer("⏳ занят предыдущим файлом — этот встанет следом.")

    async with _LOCK:
        note = await message.answer(
            "Применяю "
            + ("дневную дельту…" if decision.kind == "daily"
               else "файл истории…"))
        try:
            summary = await asyncio.wait_for(
                _apply_document(message, document, decision,
                                actor=_actor(message)),
                timeout=APPLY_TIMEOUT_S)
        except asyncio.TimeoutError:
            summary = (f"🔴 операция не уложилась в {APPLY_TIMEOUT_S} с. "
                       "База могла остаться нетронутой — проверьте /status "
                       "и пришлите файл ещё раз, повтор безвреден.")
        except Exception as exc:                        # noqa: BLE001
            logger.exception("применение %s упало", document.file_name)
            summary = f"🔴 не смог применить {document.file_name}: {exc}"
        try:
            await note.delete()
        except Exception:                               # noqa: BLE001
            pass
        await _send_block(message, summary)


async def _apply_document(message: Message, document, decision, *,
                          actor: str) -> str:
    """Скачать во временный каталог, применить, вернуть готовую сводку.

    Временный каталог удаляется в любом исходе: `/tmp` на Cloud Run — это
    оперативная память, и забытый файл там расходует ту же память, что нужна
    базе.
    """
    with tempfile.TemporaryDirectory(prefix="ombri-upload-") as tmp:
        target = Path(tmp) / Path(document.file_name).name
        await message.bot.download(document, destination=target)
        apply = (qi.apply_daily if decision.kind == "daily"
                 else qi.apply_history)
        outcome = await asyncio.to_thread(apply, target, actor=actor)
    if not outcome.ok:
        logger.warning("INGEST: %s не применён — %s",
                       outcome.file_name, outcome.reason)
    return qi.format_summary(outcome)


async def msg_fallback(message: Message) -> None:
    """Бот принимает ФАЙЛЫ. Текст — только команды, остальное мягко отбивается."""
    await message.answer(
        "Пришлите файл дневного среза <code>YYYYMMDD_d.txt</code> — или /help.",
        parse_mode=ParseMode.HTML)


# ═════════════════════════════════════════════════════════════════════════════
# IB-7 — плановая проверка снаружи
# ═════════════════════════════════════════════════════════════════════════════

async def run_scheduled_check() -> str:
    """Проверить базу и написать оператору, ЕСЛИ есть о чём.

    🔴 Молчание — штатный исход, а не сбой. Напоминание, приходящее каждый день
    независимо от состояния, перестают читать за неделю — и тогда оно не
    сработает в тот единственный раз, когда было нужно.

    Зовётся планировщиком по HTTP (`ingest_entrypoint`), поэтому опирается на
    `_BOT`, созданный в `main`: своего экземпляра у планировщика нет.
    """
    if _BOT is None:
        return "бот ещё не запущен"
    try:
        state = await asyncio.to_thread(qi.status)
    except Exception as exc:                            # noqa: BLE001
        logger.exception("плановая проверка не смогла прочитать базу")
        text = f"🔴 плановая проверка не смогла прочитать базу: {exc}"
    else:
        text = qi.build_reminder(state)
    if text is None:
        logger.info("плановая проверка: всё в порядке, молчу")
        return "тихо"
    delivered = 0
    for user_id in sorted(access.admin_ids()):
        try:
            await _BOT.send_message(user_id, text)
            delivered += 1
        except Exception as exc:                        # noqa: BLE001
            logger.warning("напоминание не доставлено %s: %s", user_id, exc)
    return f"отправлено {delivered}"


# ═════════════════════════════════════════════════════════════════════════════
# Сборка
# ═════════════════════════════════════════════════════════════════════════════

def build_dispatcher() -> Dispatcher:
    """Диспетчер отдельной функцией — чтобы тест собирал его без сети и токена."""
    dp = Dispatcher()
    # Admin-гейт ПЕРВЫМ и на ОБЕИХ шинах: кнопка подтверждения чистки приходит
    # колбэком, и шина без гейта пропустила бы чужое нажатие к необратимой
    # операции.
    dp.message.middleware(AdminOnlyMiddleware())
    dp.callback_query.middleware(AdminOnlyMiddleware())

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, F.text == "/help")
    dp.message.register(cmd_status, F.text == "/status")
    dp.message.register(cmd_missing, F.text == "/missing")
    dp.message.register(cmd_universe, F.text == "/universe")
    dp.message.register(cmd_prune, F.text == "/prune")
    dp.message.register(cmd_check, F.text.startswith("/check"))
    dp.message.register(on_document, F.document)
    dp.callback_query.register(cb_prune, F.data.startswith("prune:"))
    # Последним — иначе перехватит команды.
    dp.message.register(msg_fallback, F.text, ~F.text.startswith("/"))
    return dp


def _refuse_shared_token() -> None:
    """Отказаться стартовать, если токен тот же, что у главного бота.

    🔴 Цена ошибки НЕСИММЕТРИЧНА, и это главное. Два процесса с одним токеном
    оба зовут `getUpdates`, Telegram отвечает 409 обоим, и каждый переживает
    ровно `_CONFLICT_RETRIES` попыток. Дальше главный бот бросает исключение
    из `asyncio.run`, контейнер выходит с кодом 1, Cloud Run печатает
    «The instance was not started» — то есть КОПИЯ СЕКРЕТА В ЧУЖОЙ СЕРВИС
    роняет прод, и симптом при этом ничем не отличается от `§−98`.

    Проверка работает там, где ошибку и совершают: человек, заводя второй
    сервис, копирует существующий секрет. Если оба имени видны одному
    процессу — сравниваем и отказываемся. Когда `RAMP_BOT_TOKEN` в окружении
    нет (штатный деплой загрузчика), сравнивать нечего, и молчание здесь
    честное: гарантию даёт не эта функция, а разные секреты.
    """
    for name in MAIN_TOKEN_ENVS:
        main_token = str(os.getenv(name, "") or "").strip()
        if main_token and main_token == BOT_TOKEN:
            raise RuntimeError(
                f"токен загрузчика совпадает с {name} — токеном ГЛАВНОГО бота. "
                "Два бота на одном токене дерутся за getUpdates: Telegram "
                "отвечает 409 обоим, и после нескольких попыток ГЛАВНЫЙ бот "
                "падает — прод встанет из-за загрузчика. Заведите второго бота "
                "у @BotFather и положите ЕГО токен в "
                f"{TOKEN_ENV}.")


async def main() -> None:
    """Запуск бота. Обе проверки конфигурации — ДО первого апдейта.

    Отказ стартовать при пустом списке админов повторяет приём `M-9`
    (`db_tokenomics.assert_persistent_state`): молчаливая деградация в
    «работает, но бесполезен» скрыла бы ошибку настройки, а падение видно в
    Cloud Run сразу.
    """
    global _BOT

    if not BOT_TOKEN:
        raise RuntimeError(
            f"{TOKEN_ENV} пуст — второму боту нужен СВОЙ токен, а не токен "
            f"главного бота. (Прежнее имя {LEGACY_TOKEN_ENV} тоже принимается, "
            "но лучше переименовать секрет.)")
    if not access.configured():
        raise RuntimeError(
            f"{access.ENV_NAME} пуст — говорить с ботом некому. Пустой список "
            "означает «никого», и это не аварийный режим, а требование: "
            "сервис пишет в базу котировок.")
    _refuse_shared_token()

    publisher = publisher_from_env()
    logger.info("OMBRI Ingest Bot: хранилище — %s, админов %d",
                publisher.describe(), len(access.admin_ids()))

    # 🔴 `§−99`. Тяжёлое грузит ГЛАВНЫЙ поток и ДО первого `to_thread`.
    # Разбор — в докстроке `preimport_worker_deps`.
    preimport_worker_deps()

    bot = Bot(token=BOT_TOKEN)
    _BOT = bot
    dp = build_dispatcher()

    # Собственный id в логе — самый дешёвый способ увидеть «два сервиса, один
    # бот»: id совпадёт с тем, что печатает `tg_bot` («Starting bot id=…»).
    try:
        me = await bot.get_me()
        logger.info("Ingest bot id=%s (@%s)", me.id, me.username)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("get_me не ответил (%s) — id бота неизвестен", exc)

    try:
        from aiogram.types import BotCommand                # noqa: PLC0415
        await bot.set_my_commands([
            BotCommand(command="status", description="Состояние базы котировок"),
            BotCommand(command="missing", description="Какие файлы переслать"),
            BotCommand(command="universe", description="Покрытие факторов C-1"),
            BotCommand(command="prune", description="Вычистить обрывочные бумаги"),
            BotCommand(command="help", description="Как обновлять базу"),
        ])
    except Exception as exc:                            # noqa: BLE001
        logger.warning("set_my_commands failed: %s", exc)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(_sig, stop.set)
        except (NotImplementedError, RuntimeError):      # pragma: no cover
            pass

    async def _shutdown() -> None:
        await stop.wait()
        logger.info("Сигнал остановки — освобождаю getUpdates.")
        try:
            await dp.stop_polling()
        except Exception:                               # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(bot.session.close(), timeout=2.5)
        except Exception:                               # noqa: BLE001
            pass

    watcher = asyncio.create_task(_shutdown())
    try:
        for attempt in range(_CONFLICT_RETRIES):
            try:
                # Вебхук и polling вместе дают 409 — снимаем его перед стартом.
                await bot.delete_webhook(drop_pending_updates=True)
                await dp.start_polling(bot, handle_signals=False,
                                       drop_pending_updates=True)
                break
            except TelegramConflictError:
                # Старая ревизия Cloud Run ещё держит `getUpdates`. Без
                # ожидания новая падает, и сервис остаётся лежать после
                # каждого деплоя.
                if stop.is_set() or attempt == _CONFLICT_RETRIES - 1:
                    raise
                logger.warning("409 Conflict — жду %d с (попытка %d/%d)",
                               _CONFLICT_DELAY_S, attempt + 2, _CONFLICT_RETRIES)
                await asyncio.sleep(_CONFLICT_DELAY_S)
    finally:
        stop.set()
        await asyncio.gather(watcher, return_exceptions=True)
        _BOT = None


__all__ = ["AdminOnlyMiddleware", "LEGACY_TOKEN_ENV", "MAIN_TOKEN_ENVS",
           "TOKEN_ENV", "build_dispatcher", "main", "preimport_worker_deps",
           "read_bot_token", "render_block", "run_scheduled_check"]
