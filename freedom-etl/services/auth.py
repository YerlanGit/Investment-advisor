"""Подпись запросов Tradernet (HMAC-SHA256, схема APIv3).

Что именно подписывается
────────────────────────
    payload   = json.dumps(params, separators=(",", ":"))   # БЕЗ пробелов
    timestamp = str(int(time.time()))                        # UNIX, секунды
    sig       = HMAC_SHA256(secret_key, payload + timestamp).hexdigest()

Заголовки::

    Content-Type:      application/json
    X-NtApi-PublicKey: <публичный ключ>
    X-NtApi-Timestamp: <тот же timestamp, что вошёл в подпись>
    X-NtApi-Sig:       <sig>

Команда едет в ПУТИ (`/api/getHloc`), а не в теле, — в отличие от легаси-схемы,
где `cmd` был полем подписываемого словаря.

Откуда взята схема
──────────────────
Сверено с исходником официального SDK `tradernet-sdk` 2.2.0 (PyPI):
`tradernet/core.py::authorized_request` (version=3) и
`tradernet/common/string_utils.py::stringify` / `::sign`. Значения совпадают
дословно, включая `separators=(",", ":")` и `str(int(time()))`.

Три подводных камня, каждый из которых даёт «Invalid signature» (code=4)
────────────────────────────────────────────────────────────────────────
1. **Пробелы в JSON.** `json.dumps` по умолчанию ставит `", "` и `": "`.
   Сервер считает HMAC от компактной формы; лишний пробел ломает подпись, а
   диагностика выглядит как «неверные ключи».
2. **Двойная сериализация.** Подписывать надо ТУ ЖЕ строку, которая уйдёт в
   тело. Поэтому :meth:`TradernetAuth.sign` возвращает и заголовки, и body —
   пересобрать JSON второй раз просто нечем, и рассинхрон невозможен.
3. **Расходящийся timestamp.** Тот, что в подписи, обязан совпасть с тем, что
   в заголовке. Здесь он вычисляется один раз и используется дважды.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

__all__ = ["TradernetAuth", "SignedRequest"]


@dataclass(frozen=True, slots=True)
class SignedRequest:
    """Готовый к отправке запрос: тело и заголовки согласованы по построению."""

    body: str
    headers: dict[str, str]

    @property
    def timestamp(self) -> str:
        return self.headers["X-NtApi-Timestamp"]

    @property
    def signature(self) -> str:
        return self.headers["X-NtApi-Sig"]


class TradernetAuth:
    """Формирует подписанные заголовки Tradernet APIv3.

    Параметры
    ---------
    public_key, secret_key
        Пара ключей ЕДИНСТВЕННОЙ учётной записи сервиса (ограничение ТЗ:
        одно подсоединение). Класс не хранит состояния между вызовами и
        безопасен для конкурентного использования из нескольких задач.
    clock
        Источник времени. Вынесен параметром, чтобы тест мог зафиксировать
        timestamp и сверить подпись с эталонным вектором: без этого
        единственный способ проверить HMAC — повторить его же формулу в тесте,
        то есть проверить код самим собой.
    """

    __slots__ = ("_public_key", "_secret_key", "_clock")

    def __init__(self, public_key: str, secret_key: str, *,
                 clock: Callable[[], float] = time.time) -> None:
        self._public_key = (public_key or "").strip()
        self._secret_key = (secret_key or "").strip()
        self._clock = clock

    # ── Публичный интерфейс ──────────────────────────────────────────────────

    @staticmethod
    def serialize(params: Mapping[str, Any]) -> str:
        """Компактный JSON — ровно та строка, от которой считается HMAC.

        `sort_keys` НЕ включён намеренно: схема APIv3 подписывает байты тела
        как есть, порядок ключей значения не имеет, а принудительная
        сортировка создала бы ложное впечатление, что имеет.
        """
        return json.dumps(params, separators=(",", ":"), ensure_ascii=False)

    def sign(self, params: Mapping[str, Any]) -> SignedRequest:
        """Подписать `params` и вернуть согласованную пару (тело, заголовки)."""
        body = self.serialize(params)
        timestamp = str(int(self._clock()))
        signature = self._hmac(body + timestamp)

        return SignedRequest(
            body=body,
            headers={
                "Content-Type":      "application/json",
                "Accept":            "application/json",
                "X-NtApi-PublicKey": self._public_key,
                "X-NtApi-Timestamp": timestamp,
                "X-NtApi-Sig":       signature,
            },
        )

    # ── Внутреннее ───────────────────────────────────────────────────────────

    def _hmac(self, message: str) -> str:
        return hmac.new(
            self._secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def __repr__(self) -> str:  # pragma: no cover — только для отладочного лога
        # Ни ключ, ни его префикс не печатаются: по префиксу публичного ключа
        # учётная запись опознаётся, а лог уезжает в общий сборщик.
        return (f"TradernetAuth(key_present={bool(self._public_key)}, "
                f"key_len={len(self._public_key)}, secret_len={len(self._secret_key)})")
