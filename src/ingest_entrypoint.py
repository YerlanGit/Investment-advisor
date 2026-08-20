"""
Точка входа сервиса-загрузчика: health-сервер + бот (IB-3).

Отдельный файл, а не `python -m ingest_bot`, по той же причине, что у главного
бота: Cloud Run требует, чтобы контейнер слушал `$PORT`, иначе стартовая проба
не проходит, а бот работает long-polling и входящих запросов не принимает.

🔴 **Прод этот файл НЕ запускает.**  `Dockerfile` заканчивается
`CMD ["python", "/app/src/entrypoint.py"]` — там главный бот, и он не изменён.
Загрузчик поднимается ОТДЕЛЬНЫМ сервисом из того же образа переопределением
команды (`PLAN §9.1`), и пока такого сервиса нет, этот модуль просто лежит.
"""

import asyncio
import logging
import sys

from env_config import env_int

logger = logging.getLogger("ramp.ingest.entrypoint")


async def _health_server() -> None:
    """Минимальный HTTP/1.0: любой GET → 200 OK.  Копия приёма `entrypoint.py`."""
    port = env_int("PORT", 8080, lo=1, hi=65535)

    async def _handle(reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.wait_for(reader.read(4096), timeout=5)
        except asyncio.TimeoutError:
            pass
        writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n"
                     b"Content-Length: 2\r\n\r\nOK")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(_handle, "0.0.0.0", port)
    logger.info("Health server listening on :%s", port)
    async with server:
        await server.serve_forever()


async def _main() -> None:
    from ingest_bot import main as bot_main            # noqa: PLC0415

    await asyncio.gather(_health_server(), bot_main())


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
