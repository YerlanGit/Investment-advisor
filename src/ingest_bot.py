"""
RAMP Ingest Bot — второй Telegram-бот, только для загрузки данных (IB-3, IB-4).

Что это и чем он НЕ является
────────────────────────────
Оператор пересылает сюда дневной срез Stooq — бот применяет его к базе
котировок и публикует базу.  Отчётов он не строит, токеномику не трогает и в
бакет с балансами и ключами брокера не ходит: у него нет ни кода, ни прав
(`PLAN §7.1`).

**Слой L4 Delivery.**  Здесь только чтение сообщения и печать сводки; весь
цикл живёт в `services/quote_ingest`, разбор файлов — в `finance/stooq_ingest`.
Тот же разрез, которым `analyze_all` отделён от `tg_bot`, и по той же причине:
логика, отделённая от доставки, проверяется без бота и без токена.

🔴 Этот модуль **не импортирует `tg_bot` и не импортируется им**.  Общее у двух
ботов — `env_config`, `finance/*` и `services/*`, то есть L0/L1.  Иначе они
срастутся, и правка онбординга сломает загрузку котировок.

Почему бот отдельный — пять причин в `PLAN §2`; коротко: главный бот открывает
базу `mode=ro`, и это инвариант, который писатель в том же процессе разрушил бы
по построению.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import tempfile
from pathlib import Path

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, TelegramObject

import ingest_access as access
from env_config import env_int
from services import quote_ingest as qi
from services.quote_publisher import PublisherUnavailable, publisher_from_env

logger = logging.getLogger("ramp.ingest")

BOT_TOKEN = os.getenv("RAMP_INGEST_BOT_TOKEN", "").strip()

#: Потолок ожидания одной операции.  Цикл — скачать базу, применить, залить;
#: замер применения 0.2 с, остальное упирается в сеть.  Пять минут с запасом,
#: и они нужны как ГРАНИЦА: без неё зависшая загрузка держала бы замок вечно,
#: и оператор не смог бы прислать следующий файл, не поняв почему.
APPLY_TIMEOUT_S = env_int("INGEST_APPLY_TIMEOUT_S", 300, lo=30, hi=3600)

#: Одна операция за раз.  Единственная конкуренция, которая здесь реальна, —
#: два файла, присланных одним человеком подряд: цикл read-modify-write не
#: терпит наложения, а CAS отбил бы второй файл конфликтом, и оператор увидел
#: бы тревогу там, где никакого нарушения не было.
_LOCK = asyncio.Lock()

_HELP = (
    "🛠 *RAMP Ingest* — обновление базы котировок\n\n"
    "*Как обновлять базу*\n"
    "1. скачайте дневной срез со stooq.com/db/ (`YYYYMMDD_d.txt`, ≈700 КБ)\n"
    "2. перешлите файл сюда\n"
    "3. прочитайте сводку\n\n"
    "*Добор бумаги.* Пришлите файл истории из архива (`aapl.us.txt`) — "
    "бумага заведётся в базе.\n\n"
    "*Команды*\n"
    "/status — поколение базы, свежесть рынков, допуск C-1, пропавшие дни\n"
    "/help — эта справка\n\n"
    "🔴 Бутстрап через чат невозможен: архив истории весит 0.84 ГБ, "
    "Telegram отдаёт боту не больше 20 МБ. Сборка базы с нуля остаётся "
    "операцией на вашей машине."
)


class AdminOnlyMiddleware(BaseMiddleware):
    """Fail-CLOSED: пустой список админов = никого.

    🔴 Зеркальная противоположность `tg_bot.WhitelistMiddleware`, и это
    сознательно.  Там пустая настройка пропускает всех, потому что запертая
    бета хуже открытой; здесь пустая настройка не должна открывать запись в
    базу цен.  Разбор — `ingest_access`.
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
# Хендлеры
# ═════════════════════════════════════════════════════════════════════════════

async def cmd_start(message: Message) -> None:
    await message.answer(_HELP, parse_mode=ParseMode.MARKDOWN)


async def cmd_help(message: Message) -> None:
    await message.answer(_HELP, parse_mode=ParseMode.MARKDOWN)


async def cmd_status(message: Message) -> None:
    """Состояние базы.  Блокирующее чтение уходит в поток."""
    try:
        state = await asyncio.to_thread(qi.status)
    except PublisherUnavailable as exc:
        await message.answer(f"🔴 хранилище не настроено\n   {exc}")
        return
    except Exception as exc:                            # noqa: BLE001
        logger.exception("status упал")
        await message.answer(f"🔴 не смог прочитать базу: {exc}")
        return
    await message.answer(f"```\n{qi.format_status(state)}\n```",
                         parse_mode=ParseMode.MARKDOWN)


async def on_document(message: Message) -> None:
    """Приём файла: лимиты → скачивание → применение → сводка.

    🔴 Порядок обязателен.  Лимиты проверяются ПО МЕТАДАННЫМ, до `getFile`:
    Telegram отдаёт размер в описании документа, и отказать по нему дешевле,
    чем скачать архив в `/tmp`, который на Cloud Run является оперативной
    памятью.
    """
    document = message.document
    if document is None:                                # pragma: no cover
        return
    decision = qi.classify_upload(document.file_name, document.file_size)
    if not decision.accepted:
        await message.answer(f"🔴 файл не принят\n   {decision.reason}")
        return

    if _LOCK.locked():
        await message.answer("⏳ занят предыдущим файлом — этот встанет следом.")

    async with _LOCK:
        note = await message.answer(
            "Применяю "
            + ("дневную дельту…" if decision.kind == "daily"
               else "файл истории…"))
        actor = str(getattr(message.from_user, "id", "?"))
        try:
            summary = await asyncio.wait_for(
                _apply_document(message, document, decision, actor=actor),
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
        await message.answer(f"```\n{summary}\n```",
                             parse_mode=ParseMode.MARKDOWN)


async def _apply_document(message: Message, document, decision, *,
                          actor: str) -> str:
    """Скачать во временный каталог, применить, вернуть готовую сводку.

    Временный каталог удаляется в любом исходе: `/tmp` на Cloud Run — это
    оперативная память, и забытый файл там расходует ту же память, что нужна
    базе.
    """
    with tempfile.TemporaryDirectory(prefix="ramp-upload-") as tmp:
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
        "Пришлите файл дневного среза `YYYYMMDD_d.txt` — или /help.",
        parse_mode=ParseMode.MARKDOWN)


# ═════════════════════════════════════════════════════════════════════════════
# Сборка
# ═════════════════════════════════════════════════════════════════════════════

def build_dispatcher() -> Dispatcher:
    """Диспетчер отдельной функцией — чтобы тест собирал его без сети и токена."""
    dp = Dispatcher()
    # Admin-гейт ПЕРВЫМ: чужой апдейт не должен дойти ни до одного хендлера.
    dp.message.middleware(AdminOnlyMiddleware())

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_help, F.text == "/help")
    dp.message.register(cmd_status, F.text == "/status")
    dp.message.register(on_document, F.document)
    # Последним — иначе перехватит команды.
    dp.message.register(msg_fallback, F.text, ~F.text.startswith("/"))
    return dp


async def main() -> None:
    """Запуск бота.  Обе проверки конфигурации — ДО первого апдейта.

    Отказ стартовать при пустом списке админов повторяет приём `M-9`
    (`db_tokenomics.assert_persistent_state`): молчаливая деградация в
    «работает, но бесполезен» скрыла бы ошибку настройки, а падение видно в
    Cloud Run сразу.
    """
    if not BOT_TOKEN:
        raise RuntimeError(
            "RAMP_INGEST_BOT_TOKEN пуст — второму боту нужен СВОЙ токен, "
            "а не токен главного бота.")
    if not access.configured():
        raise RuntimeError(
            f"{access.ENV_NAME} пуст — говорить с ботом некому. Пустой список "
            "означает «никого», и это не аварийный режим, а требование: "
            "сервис пишет в базу котировок.")

    publisher = publisher_from_env()
    logger.info("RAMP Ingest Bot: хранилище — %s, админов %d",
                publisher.describe(), len(access.admin_ids()))

    bot = Bot(token=BOT_TOKEN)
    dp = build_dispatcher()

    try:
        from aiogram.types import BotCommand                # noqa: PLC0415
        await bot.set_my_commands([
            BotCommand(command="status", description="Состояние базы котировок"),
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
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, handle_signals=False,
                               drop_pending_updates=True)
    finally:
        stop.set()
        await asyncio.gather(watcher, return_exceptions=True)


__all__ = ["AdminOnlyMiddleware", "build_dispatcher", "main"]
