"""
Точка входа сервиса-загрузчика: health-сервер, планировщик и бот (IB-3, IB-7).

Отдельный файл, а не `python -m ingest_bot`, по той же причине, что у главного
бота: Cloud Run требует, чтобы контейнер слушал `$PORT`, иначе стартовая проба
не проходит, а бот работает long-polling и входящих запросов не принимает.

🔴 **Прод этот файл НЕ запускает.** `Dockerfile` заканчивается
`CMD ["python", "/app/src/entrypoint.py"]` — там главный бот, и он не изменён.
Загрузчик поднимается ОТДЕЛЬНЫМ сервисом из того же образа переопределением
команды (`PLAN §9.1`).

Два маршрута, и они разные по смыслу
────────────────────────────────────
* `GET /` — стартовая проба Cloud Run. Всегда 200, без условий: проба не
  должна зависеть ни от базы, ни от Telegram, иначе недоступность котировок
  будет выглядеть как неживой контейнер и ревизия не поднимется;
* `POST /tasks/check` — плановая проверка (IB-7). Закрыта ДВАЖДЫ: снаружи —
  OIDC-аутентификацией Cloud Scheduler (сервис деплоится
  `--no-allow-unauthenticated`), внутри — общим секретом. Без секрета маршрут
  ОТКЛЮЧЁН: незакрытая ручка, шлющая сообщения владельцу, — это канал для
  чужого шума в единственном канале связи оператора.
"""

import asyncio
import hmac
import logging
import os
import sys

from env_config import env_int

logger = logging.getLogger("ombri.ingest.entrypoint")

TASK_PATH = "/tasks/check"
TASK_TOKEN_HEADER = "x-ingest-task-token"


def _reply(code: int, body: str) -> bytes:
    payload = body.encode("utf-8")
    reason = {200: "OK", 401: "Unauthorized", 405: "Method Not Allowed",
              503: "Service Unavailable"}.get(code, "OK")
    return (f"HTTP/1.0 {code} {reason}\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n").encode("utf-8") + payload


def _parse(raw: bytes) -> tuple[str, str, dict[str, str]]:
    """`(метод, путь, заголовки)`. Разбор терпимый: мусор → пустые значения."""
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:                                   # noqa: BLE001
        return "", "", {}
    head = text.split("\r\n\r\n", 1)[0].splitlines()
    if not head:
        return "", "", {}
    parts = head[0].split()
    method = parts[0].upper() if parts else ""
    path = parts[1] if len(parts) > 1 else ""
    headers: dict[str, str] = {}
    for line in head[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
    return method, path, headers


def _task_token() -> str:
    return str(os.getenv("INGEST_TASK_TOKEN", "") or "").strip()


async def _serve(on_task) -> None:
    """HTTP-сервер: проба и плановая проверка."""
    port = env_int("PORT", 8080, lo=1, hi=65535)

    async def _handle(reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        try:
            raw = b""
            try:
                raw = await asyncio.wait_for(reader.read(8192), timeout=5)
            except asyncio.TimeoutError:
                pass
            method, path, headers = _parse(raw)
            if path.split("?")[0] == TASK_PATH:
                writer.write(await _handle_task(method, headers, on_task))
            else:
                writer.write(_reply(200, "OK"))
            await writer.drain()
        except Exception as exc:                        # noqa: BLE001
            logger.warning("HTTP-запрос не обработан: %s", exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:                           # noqa: BLE001
                pass

    server = await asyncio.start_server(_handle, "0.0.0.0", port)
    logger.info("Health server listening on :%s (задача — POST %s)",
                port, TASK_PATH)
    async with server:
        await server.serve_forever()


async def _handle_task(method: str, headers: dict, on_task) -> bytes:
    token = _task_token()
    if not token:
        logger.warning("плановая проверка запрошена, но INGEST_TASK_TOKEN пуст "
                       "— маршрут отключён")
        return _reply(503, "INGEST_TASK_TOKEN не задан — маршрут отключён")
    if method != "POST":
        return _reply(405, "только POST")
    # `compare_digest`, а не `!=`: сравнение секрета не должно зависеть от
    # длины совпавшего префикса. Маршрут закрыт ещё и OIDC снаружи, но это
    # ровно тот случай, где правильный идиом стоит одну строку.
    if not hmac.compare_digest(headers.get(TASK_TOKEN_HEADER, ""), token):
        logger.warning("плановая проверка с неверным секретом — отказ")
        return _reply(401, "неверный секрет")
    try:
        result = await on_task()
    except Exception as exc:                            # noqa: BLE001
        logger.exception("плановая проверка упала")
        return _reply(200, f"ошибка: {exc}")
    return _reply(200, str(result))


async def _main() -> None:
    from ingest_bot import main as bot_main, run_scheduled_check  # noqa: PLC0415

    await asyncio.gather(_serve(run_scheduled_check), bot_main())


if __name__ == "__main__":                              # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s")
    try:
        asyncio.run(_main())
    except RuntimeError as exc:
        # Ошибка конфигурации — не traceback на пол-экрана, а одна строка,
        # по которой видно, что чинить.
        logger.error("Загрузчик не запущен: %s", exc)
        sys.exit(2)
