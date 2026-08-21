"""
Публикация базы котировок: скачать → применить → загрузить обратно (IB-1).

Слой
────
**L1 Data.**  Модуль похож на `services/report_storage.py` — тот тоже кладёт
файл в GCS, — но лежит на другом слое, и это не придирка: `report_storage`
ДОСТАВЛЯЕТ отчёт человеку и потому L4, а этот кладёт данные, которые потом
читает движок.  То же место в графе, что у `services/fx_feed.py`.  Ошибиться
здесь значило бы завести L4-модуль, которому можно импортировать всё, и
обессмыслить гейт слоёв.

Почему файл целиком, а не запись на месте
─────────────────────────────────────────
SQLite поверх объектного хранилища не имеет работающих файловых блокировок.
Проект это уже знает и платит: `db_tokenomics` держит `journal_mode=DELETE`,
потому что WAL на gcsfuse портит базу outright.  Для базы в 1 МБ это
приемлемо; для 58.6 МБ и миллиона UPSERT-ов в день — нет.

Compare-and-swap, и 412 — это ОТКАЗ, а не повод повторить
──────────────────────────────────────────────────────────
`upload(..., if_generation_match=gen)` публикует, только если объект не менялся
с момента скачивания.  Оператор ОДИН (`PLAN §1.2`), поэтому конфликт означает
не «не повезло», а «базу трогали мимо бота» — и это единственный признак
такого события.  Тихий повтор его бы стёр, поэтому конфликт возвращается
вызывающему КАК РЕЗУЛЬТАТ (`UploadResult.conflict`), а не как исключение и не
как повод для ретрая.

Два бэкенда
───────────
* `LocalQuotePublisher` — каталог на диске.  **Дефолт**: он позволяет
  прогонять весь цикл без облака, без ключей и без сети;
* `GcsQuotePublisher` — бакет.  🔴 Против живого бакета НЕ проверялся: в этой
  редакции проекта включён офлайн-режим.  Тесты гоняют его на подменённом
  клиенте, то есть доказывают форму вызовов, а не работу GCS.

Оба реализуют один протокол, и `quote_ingest` не знает, с каким работает.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

from env_config import env_int

logger = logging.getLogger(__name__)

#: Имена объектов.  Совпадают с тем, что ждёт главный бот в `STOOQ_DB_PATH`,
#: поэтому локальный корень можно навести прямо на каталог базы и получить
#: сквозной офлайн-цикл без единой правки конфигурации.
DB_OBJECT_NAME = "prices.sqlite"
CURSOR_OBJECT_NAME = "cursor.json"
INBOX_DIR_NAME = "inbox"

#: Сколько дат помнит курсор.  Не «на всякий случай»: `/missing` печатает по
#: нему список файлов к пересылке, а список длиной в год нечитаем.  Старые даты
#: уходят сами — это дешевле, чем команда для ручной чистки.
CURSOR_HISTORY_LIMIT = env_int("INGEST_CURSOR_HISTORY", 120, lo=10, hi=2000)


class PublisherUnavailable(RuntimeError):
    """Хранилища нет или в нём нет базы.

    Отдельный тип — по той же причине, что `StooqStore.StoreUnavailable`:
    «не настроено» и «сломано» это разные состояния с разной реакцией, и
    вызывающий обязан их различать, а не ловить `OSError` вслепую.
    """


@dataclass(frozen=True)
class Snapshot:
    """Скачанная копия базы и поколение, с которым её взяли."""

    path: Path
    generation: int
    size: int


@dataclass(frozen=True)
class UploadResult:
    """Итог публикации.  `conflict` — это 412, и он НЕ ошибка транспорта."""

    published: bool
    generation: Optional[int] = None
    conflict: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class Cursor:
    """Журнал бота ВНЕ базы — единственный детектор отката.

    Откатившаяся база внутри себя непротиворечива: календарь рынка выводится
    из тех же строк, что и бары, поэтому каждая бумага честно рапортует
    свежесть 0 (`stooq_store.market_staleness_days` заведён из-за этого же).
    Курсор, лежащий В базе, откатился бы вместе с ней — значит он обязан
    лежать снаружи.

    🔴 Дата попадает сюда, только если бары РЕАЛЬНО записаны.  Файл, из
    которого не легло ни одного бара (все тикеры вне рабочего набора), не
    оставляет следа в `daily_bars` — записав его в курсор, бот сам себе
    сфабриковал бы «пропавшую» дату и звал бы оператора чинить пустоту.
    """

    applied_dates: tuple[int, ...] = ()
    generation: Optional[int] = None
    at: Optional[str] = None
    last_run: dict = field(default_factory=dict)

    def with_applied(self, trade_date: Optional[int], *,
                     generation: Optional[int],
                     last_run: Optional[dict] = None) -> "Cursor":
        dates = set(self.applied_dates)
        if trade_date is not None:
            dates.add(int(trade_date))
        ordered = tuple(sorted(dates))[-CURSOR_HISTORY_LIMIT:]
        return Cursor(applied_dates=ordered, generation=generation,
                      at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      last_run=dict(last_run or {}))

    def to_json(self) -> str:
        return json.dumps({
            "applied_dates": list(self.applied_dates),
            "generation": self.generation,
            "at": self.at,
            "last_run": self.last_run,
        }, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(raw: str) -> "Cursor":
        """Разбор курсора.  Мусор читается как «курсора нет», а не как ноль дат.

        Разница существенна: «я ничего не применял» и «я не смог прочитать свой
        журнал» ведут к разным выводам о том, был ли откат.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("курсор не является объектом JSON")
        dates = data.get("applied_dates") or []
        if not isinstance(dates, Sequence) or isinstance(dates, (str, bytes)):
            raise ValueError("applied_dates не является списком")
        return Cursor(
            applied_dates=tuple(sorted(int(d) for d in dates)),
            generation=(int(data["generation"])
                        if data.get("generation") is not None else None),
            at=data.get("at"),
            last_run=dict(data.get("last_run") or {}),
        )


@runtime_checkable
class QuotePublisher(Protocol):
    """Контракт хранилища базы котировок."""

    name: str

    def describe(self) -> str: ...
    def download(self, dest: Path) -> Snapshot: ...
    def upload(self, src: Path, *, if_generation_match: int) -> UploadResult: ...
    def read_cursor(self) -> Optional[Cursor]: ...
    def write_cursor(self, cursor: Cursor) -> None: ...
    def archive_source(self, src: Path, name: str) -> None: ...


# ═════════════════════════════════════════════════════════════════════════════
# Локальный бэкенд — офлайн-режим
# ═════════════════════════════════════════════════════════════════════════════

class LocalQuotePublisher:
    """Каталог на диске вместо бакета.

    Поколение — `st_mtime_ns` файла базы, и после каждой публикации оно
    принудительно сдвигается вперёд (`_stamp`).  Это не украшение: на
    файловой системе с грубым разрешением времени две публикации подряд могли
    бы получить ОДНО поколение, и CAS перестал бы отличать «не менялось» от
    «менялось дважды».  Монотонность здесь та же, что у GCS, и получена она
    явно, а не предположением о точности часов.
    """

    name = "local"

    def __init__(self, root) -> None:
        self._root = Path(root)

    # ── пути ─────────────────────────────────────────────────────────────
    @property
    def root(self) -> Path:
        return self._root

    @property
    def db_path(self) -> Path:
        return self._root / DB_OBJECT_NAME

    @property
    def cursor_path(self) -> Path:
        return self._root / CURSOR_OBJECT_NAME

    def describe(self) -> str:
        return f"локальный каталог {self._root}"

    # ── поколение ────────────────────────────────────────────────────────
    def _generation(self) -> Optional[int]:
        try:
            return int(self.db_path.stat().st_mtime_ns)
        except OSError:
            return None

    def _stamp(self, previous: Optional[int]) -> int:
        """Пометить файл новым, строго большим поколением."""
        moment = time.time_ns()
        if previous is not None and moment <= previous:
            moment = previous + 1
        os.utime(self.db_path, ns=(moment, moment))
        return moment

    # ── протокол ─────────────────────────────────────────────────────────
    def download(self, dest: Path) -> Snapshot:
        if not self.db_path.exists():
            raise PublisherUnavailable(
                f"базы котировок нет по пути {self.db_path}. "
                "Её собирает БУТСТРАП из архива истории на машине оператора "
                "(docs/roadmap/manual_portfolio/OPERATOR_STOOQ.md §5) — "
                "бот базу с нуля не создаёт.")
        generation = self._generation()
        if generation is None:                          # pragma: no cover
            raise PublisherUnavailable("база исчезла между проверкой и чтением")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.db_path, dest)
        return Snapshot(path=dest, generation=generation,
                        size=dest.stat().st_size)

    def upload(self, src: Path, *, if_generation_match: int) -> UploadResult:
        current = self._generation()
        if current is None:
            return UploadResult(
                published=False, conflict=True,
                reason="база исчезла из хранилища, пока я применял файл")
        if int(current) != int(if_generation_match):
            return UploadResult(
                published=False, conflict=True,
                reason=(f"поколение изменилось: было {if_generation_match}, "
                        f"стало {current}"))
        scratch = self.db_path.with_name(self.db_path.name + ".part")
        shutil.copyfile(Path(src), scratch)
        os.replace(scratch, self.db_path)
        return UploadResult(published=True, generation=self._stamp(current))

    def read_cursor(self) -> Optional[Cursor]:
        try:
            raw = self.cursor_path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return Cursor.from_json(raw)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("курсор %s нечитаем (%s) — считаю, что его нет",
                           self.cursor_path, exc)
            return None

    def write_cursor(self, cursor: Cursor) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        scratch = self.cursor_path.with_name(self.cursor_path.name + ".part")
        scratch.write_text(cursor.to_json(), encoding="utf-8")
        os.replace(scratch, self.cursor_path)

    def archive_source(self, src: Path, name: str) -> None:
        target_dir = self._root / INBOX_DIR_NAME
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(src), target_dir / name)
        except OSError as exc:                          # pragma: no cover
            logger.warning("аудит-след %s не сохранён: %s", name, exc)


# ═════════════════════════════════════════════════════════════════════════════
# GCS-бэкенд
# ═════════════════════════════════════════════════════════════════════════════

def _is_precondition_failed(exc: BaseException) -> bool:
    """412 от GCS, определённый БЕЗ импорта `google.api_core`.

    Импортировать исключение ради `except` значило бы сделать библиотеку
    жёсткой зависимостью модуля, который в офлайн-режиме её не касается.
    Проверяются оба признака, потому что версии клиента различаются формой.
    """
    return (getattr(exc, "code", None) == 412
            or type(exc).__name__ == "PreconditionFailed")


class GcsQuotePublisher:
    """Бакет Cloud Storage.

    🔴 **Отдельный бакет, а не `ramp-bot-state`.**  Там лежат `tokenomics.db`
    (балансы) и `users_vault.db` (зашифрованные ключи брокера).  Ограничивается
    не человек — у оператора доступ и так есть, — а ВСЕГДА РАБОТАЮЩИЙ СЕРВИС:
    у него свой принципал, свой токен и свой код, который никто не
    перечитывает перед каждым запуском (`PLAN §7.1`).
    """

    name = "gcs"

    def __init__(self, bucket: str, prefix: str = "stooq/", *,
                 client=None) -> None:
        self._bucket_name = str(bucket)
        self._prefix = str(prefix or "")
        if self._prefix and not self._prefix.endswith("/"):
            self._prefix += "/"
        self._client = client

    def describe(self) -> str:
        return f"gs://{self._bucket_name}/{self._prefix}"

    def _bucket(self):
        if self._client is None:
            try:
                from google.cloud import storage      # noqa: PLC0415
            except ImportError as exc:
                raise PublisherUnavailable(
                    "google-cloud-storage не установлен — GCS-бэкенд "
                    "недоступен. Офлайн-режим: QUOTES_BACKEND=local") from exc
            self._client = storage.Client()
        return self._client.bucket(self._bucket_name)

    def _blob(self, name: str):
        return self._bucket().blob(self._prefix + name)

    def download(self, dest: Path) -> Snapshot:
        """Скачать объект и запомнить его поколение.

        🔴 **Между `get_blob` и скачиванием есть окно.** Если объект в него
        изменится, мы получим новые байты со старым поколением. Это НЕ приводит
        к порче: публикация пойдёт с записанным поколением, GCS ответит 412, и
        бот громко откажет. То есть окно даёт ложную тревогу, а не тихую
        ошибку, — и при одном операторе оно практически недостижимо. Названо
        здесь, чтобы не выглядеть недосмотром.
        """
        blob = self._bucket().get_blob(self._prefix + DB_OBJECT_NAME)
        if blob is None:
            raise PublisherUnavailable(
                f"в {self.describe()} нет {DB_OBJECT_NAME}. Базу собирает "
                "бутстрап на машине оператора — бот её с нуля не создаёт.")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        return Snapshot(path=dest, generation=int(blob.generation),
                        size=dest.stat().st_size)

    def upload(self, src: Path, *, if_generation_match: int) -> UploadResult:
        blob = self._blob(DB_OBJECT_NAME)
        try:
            blob.upload_from_filename(
                str(src), if_generation_match=int(if_generation_match))
        except Exception as exc:                        # noqa: BLE001
            if _is_precondition_failed(exc):
                return UploadResult(
                    published=False, conflict=True,
                    reason=("объект изменился с поколения "
                            f"{if_generation_match} (412 Precondition Failed)"))
            logger.error("публикация базы не удалась: %s", exc)
            return UploadResult(published=False, reason=str(exc))
        return UploadResult(published=True,
                            generation=int(getattr(blob, "generation", 0) or 0))

    def read_cursor(self) -> Optional[Cursor]:
        try:
            blob = self._bucket().get_blob(self._prefix + CURSOR_OBJECT_NAME)
            if blob is None:
                return None
            return Cursor.from_json(blob.download_as_text())
        except Exception as exc:                        # noqa: BLE001
            logger.warning("курсор не прочитан (%s) — считаю, что его нет", exc)
            return None

    def write_cursor(self, cursor: Cursor) -> None:
        try:
            self._blob(CURSOR_OBJECT_NAME).upload_from_string(
                cursor.to_json(), content_type="application/json")
        except Exception as exc:                        # noqa: BLE001
            logger.error("курсор НЕ сохранён (%s) — следующий запуск может "
                         "принять применённые дни за пропавшие", exc)

    def archive_source(self, src: Path, name: str) -> None:
        try:
            self._blob(f"{INBOX_DIR_NAME}/{name}").upload_from_filename(str(src))
        except Exception as exc:                        # noqa: BLE001
            logger.warning("аудит-след %s не сохранён: %s", name, exc)


# ═════════════════════════════════════════════════════════════════════════════
# Выбор бэкенда
# ═════════════════════════════════════════════════════════════════════════════

def default_local_root() -> Path:
    """Корень офлайн-хранилища.  Репо-локальный, как у `stooq_ingest`."""
    fallback = Path(__file__).resolve().parents[2] / "data" / "quotes"
    return Path(os.getenv("QUOTES_LOCAL_ROOT", str(fallback)))


def publisher_from_env() -> QuotePublisher:
    """Хранилище по переменным окружения.  **Дефолт — локальный каталог.**

    Выбор ЯВНЫЙ (`QUOTES_BACKEND`), а не выводится из наличия `QUOTES_BUCKET`:
    выведенный режим означал бы, что опечатка в имени бакета молча роняет
    сервис в офлайн и он «работает», ничего никуда не публикуя.
    """
    backend = str(os.getenv("QUOTES_BACKEND", "local") or "local").strip().lower()
    if backend == "gcs":
        bucket = str(os.getenv("QUOTES_BUCKET", "") or "").strip()
        if not bucket:
            raise PublisherUnavailable(
                "QUOTES_BACKEND=gcs, но QUOTES_BUCKET пуст — публиковать некуда")
        return GcsQuotePublisher(bucket, os.getenv("QUOTES_PREFIX", "stooq/"))
    if backend != "local":
        raise PublisherUnavailable(
            f"неизвестный QUOTES_BACKEND={backend!r}. Допустимы: local, gcs")
    return LocalQuotePublisher(default_local_root())


__all__ = [
    "CURSOR_OBJECT_NAME",
    "Cursor",
    "DB_OBJECT_NAME",
    "GcsQuotePublisher",
    "LocalQuotePublisher",
    "PublisherUnavailable",
    "QuotePublisher",
    "Snapshot",
    "UploadResult",
    "default_local_root",
    "publisher_from_env",
]
