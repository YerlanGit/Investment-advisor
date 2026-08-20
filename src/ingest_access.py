"""
Кто вправе говорить с ботом-загрузчиком (IB-3).

Почему отдельным модулем, а не внутри `ingest_bot`
──────────────────────────────────────────────────
`ingest_bot` импортирует aiogram на уровне модуля — иначе не объявить хендлеры.
Значит там, где aiogram не установлен, он не импортируется, и вместе с ним
становится непроверяемой **политика доступа** — самая чувствительная часть
сервиса, который пишет в базу котировок.  Здесь только stdlib, поэтому
fail-closed проверяется в любом окружении, включая офлайн-разработку.

🔴 Копировать `tg_bot.WhitelistMiddleware` НЕЛЬЗЯ, и это не вкусовщина.
Тот при пустом списке пропускает всех — «misconfiguration should fail OPEN for
the beta».  Для публичного продукта это осознанный выбор: пустая переменная не
должна запирать пользователей.  Здесь ровно наоборот — пустая переменная не
должна открывать запись в базу цен.  Один процесс не может нести оба дефолта,
и это пятая причина, по которой бот отдельный (`PLAN §2`).

Список СВОЙ (`INGEST_ADMIN_IDS`), а не общий `ADMIN_USER_IDS`: право начислять
себе токены и право менять цены — разные права, и человек, у которого есть
первое, не обязан иметь второе.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

ENV_NAME = "INGEST_ADMIN_IDS"

#: Ответ чужому.  Тот же приём, что у `tg_bot.cmd_grant`: команда остаётся
#: невидимой, а не отвечает «доступ запрещён» — второе подтверждает, что бот
#: делает что-то, ради чего стоит подбирать доступ.
DENIAL_TEXT = "Неизвестная команда."


def parse_admin_ids(raw: Optional[str]) -> frozenset[int]:
    """`"1, 2; 3"` → `{1, 2, 3}`.  Мусор молча отбрасывается, пустое → пусто.

    Разделители те же, что у `tg_bot._admin_users` (запятая и точка с запятой):
    человек, настраивавший главного бота, не обязан помнить второй синтаксис.
    """
    out: set[int] = set()
    for token in str(raw or "").replace(";", ",").split(","):
        token = token.strip()
        if token.lstrip("-").isdigit():
            out.add(int(token))
    return frozenset(out)


def admin_ids() -> frozenset[int]:
    return parse_admin_ids(os.getenv(ENV_NAME))


def is_admin(user_id: Optional[int]) -> bool:
    """🔴 Пустой список = НИКОГО.  Не «все», не «владелец по умолчанию»."""
    if user_id is None:
        return False
    return int(user_id) in admin_ids()


def note_stranger(user_id: Optional[int], what: str) -> None:
    """Чужое сообщение — событие безопасности, а не опечатка.

    Легитимный отправитель ровно один, поэтому любое другое обращение стоит
    того, чтобы остаться в логе с идентификатором: при одном операторе это
    единственный способ заметить, что токен бота куда-то утёк.
    """
    logger.warning("INGEST: обращение не от админа — user_id=%s, %s",
                   user_id, what)


def configured() -> bool:
    """Настроен ли доступ вообще.  Ложь → бот обязан сказать это на старте."""
    return bool(admin_ids())


__all__ = ["DENIAL_TEXT", "ENV_NAME", "admin_ids", "configured", "is_admin",
           "note_stranger", "parse_admin_ids"]
