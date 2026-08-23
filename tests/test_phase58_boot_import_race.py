"""Старт контейнера: тяжёлые импорты — только в главном потоке (`§−98`).

Что охраняется и почему
-----------------------
`entrypoint.py` поднимал демон-поток `rag-boot-ingest` в блоке `__main__`, то
есть ДО того, как главный поток входил в `from tg_bot import main`. Дальше два
потока одновременно тянули один и тот же numpy: фоновый — через
`agent.rag_engine` → chromadb → onnxruntime, главный — через `tg_bot` →
pandas/sklearn. CPython разрывает взаимную блокировку модульных локов, отдавая
одному из потоков ПОЛУИНИЦИАЛИЗИРОВАННЫЙ модуль, и импорт падает::

    cannot import name 'NDArray' from partially initialized module
    'numpy._typing' (most likely due to a circular import)

Кто проиграет гонку — решает планировщик, поэтому дефект ПЛАВАЮЩИЙ, и именно
поэтому он прожил долго: чаще проигрывал фоновый поток, а его исключение
глотает `try/except` внутри `_boot_ingest_from_inbox` — в логе оставалась
безобидная строка «RAG boot-ingest skipped (...)». Когда же проигрывал ГЛАВНЫЙ
поток, падал `from tg_bot import main`, исключение выходило из
`asyncio.run(_main())`, контейнер завершался с кодом 1, и Cloud Run печатал
«Default STARTUP TCP probe failed … The instance was not started» — бот НЕ
ЗАПУСКАЛСЯ. Замер по логам `ramp-bot` 17.08.2026: 17:51:26 и 18:53:49 — в обоих
случаях трейсбек главного потока и предупреждение фонового стоят в ОДНОЙ
секунде и называют один и тот же numpy.

Лечение — порядок, а не ретрай: главный поток дотягивает все тяжёлые импорты
(`tg_bot`, затем `_preimport_boot_ingest_deps`), и только потом стартует
демон-поток. На `--cpu=1` это ничего не замедляет: параллельно эти импорты и
так не шли.

Почему это ТЕСТ, а не комментарий: гонку нельзя поймать прогоном — она
воспроизводится раз в N стартов (40 локальных запусков дали 0 падений при
живом падении в проде). Проверять здесь можно только СТРУКТУРУ, зато её
проверять обязательно: любая будущая правка, вернувшая старт потока раньше
импортов или добавившая в тело потока новый тяжёлый импорт без предзагрузки,
вернёт и дефект — молча.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Sequence

_ENTRYPOINT = Path(__file__).resolve().parent.parent / "src" / "entrypoint.py"

#: Имя фонового потока, ради которого весь этот гейт.
_THREAD_TARGET = "_boot_ingest_from_inbox"
#: Функция, обязанная загрузить зависимости потока в главном потоке.
_PREIMPORT = "_preimport_boot_ingest_deps"


def _tree() -> ast.Module:
    return ast.parse(_ENTRYPOINT.read_text(encoding="utf-8"), filename=str(_ENTRYPOINT))


def _func(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} не найдена в entrypoint.py")


def _imported_modules(node: ast.AST) -> set[str]:
    """ПОЛНЫЕ имена модулей, которые импортирует поддерево `node`.

    Имя берётся целиком, а не до первой точки. Разница не косметическая:
    `finance.data_checks` в реестре НЕ загружает `finance.stooq_provider`,
    и сравнение по верхнему пакету («оба — `finance`») объявило бы второй
    предзагруженным, хотя он приедет на поток как ни в чём не бывало.
    """
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            out.update(a.name for a in sub.names)
        elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0:
            out.add(sub.module)
    return out


def _covered(module: str, registry: Sequence[str]) -> bool:
    """Загрузит ли реестр модуль `module` до первого обращения потока?

    Покрывает СВОЙ модуль и своих предков: `import chromadb.utils.embedding_
    functions` исполняет и `chromadb`. Обратное неверно — родитель не тянет
    потомка, — и именно это различие делает проверку небесполезной.
    """
    return any(r == module or r.startswith(module + ".") for r in registry)


def _uncovered(modules: set[str], registry: Sequence[str],
               stdlib: set[str]) -> list[str]:
    return sorted(m for m in modules
                  if m.split(".")[0] not in stdlib and not _covered(m, registry))


def _lazy_imports_inside_functions(path: Path) -> set[str]:
    """Пакеты, которые модуль `path` импортирует ВНУТРИ функций (лениво).

    Ленивый импорт исполняется у ВЫЗЫВАЮЩЕГО — то есть на фоновом потоке.
    Именно так `chromadb` попадал на поток, хотя `agent.rag_engine`
    предзагружен: сам модуль лёгкий, тяжесть спрятана в `__init__`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    top_level = {id(n) for n in tree.body}
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import) and id(sub) not in top_level:
                out.update(a.name for a in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0 \
                    and id(sub) not in top_level:
                out.add(sub.module)
    return out


def _registry(path: Path, name: str) -> list[str]:
    """Строки из кортежа-реестра `name` в модуле `path` — из AST, без импорта."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        target_names = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
        elif isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name in target_names:
            return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    raise AssertionError(f"{name} не найден в {path.name}")


def _preload_registry() -> list[str]:
    return _registry(_ENTRYPOINT, "_BOOT_INGEST_HEAVY_IMPORTS")


def _starts_boot_thread(node: ast.AST) -> bool:
    """Есть ли в поддереве `threading.Thread(target=_boot_ingest_from_inbox…)`?"""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        is_thread = (isinstance(func, ast.Attribute) and func.attr == "Thread") or (
            isinstance(func, ast.Name) and func.id == "Thread")
        if not is_thread:
            continue
        for kw in sub.keywords:
            if kw.arg == "target" and isinstance(kw.value, ast.Name) \
                    and kw.value.id == _THREAD_TARGET:
                return True
    return False


class BootImportOrderTest(unittest.TestCase):
    """Порядок «сначала импорты главного потока, потом поток» — исполняем."""

    def test_thread_not_started_in_main_guard(self):
        """В блоке `if __name__ == "__main__"` поток НЕ поднимается.

        Именно оттуда он стартовал одновременно с импортом `tg_bot`.
        """
        tree = _tree()
        guards = [n for n in tree.body if isinstance(n, ast.If)]
        self.assertTrue(guards, "блок `if __name__ == '__main__'` пропал")
        for guard in guards:
            self.assertFalse(
                _starts_boot_thread(guard),
                "демон-поток RAG стартует в `__main__` — снова ДО `from tg_bot "
                "import main`; это и есть гонка импорта numpy (§−98)",
            )

    def test_thread_starts_inside_main_after_heavy_imports(self):
        """В `_main` поток стартует ПОСЛЕ импорта `tg_bot` и предзагрузки."""
        main = _func(_tree(), "_main")
        self.assertTrue(_starts_boot_thread(main),
                        "поток RAG больше не поднимается вовсе — фолбэк ингеста потерян")

        line_tg_import = None
        line_preimport = None
        line_thread = None
        for sub in ast.walk(main):
            if isinstance(sub, ast.ImportFrom) and sub.module == "tg_bot":
                line_tg_import = sub.lineno
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                    and sub.func.id == _PREIMPORT:
                line_preimport = sub.lineno
        if _starts_boot_thread(main):
            for sub in ast.walk(main):
                if isinstance(sub, ast.Call) and _starts_boot_thread(sub):
                    line_thread = sub.lineno if line_thread is None else min(line_thread, sub.lineno)

        self.assertIsNotNone(line_tg_import, "`from tg_bot import …` исчез из `_main`")
        self.assertIsNotNone(line_preimport, f"`{_PREIMPORT}()` не вызывается в `_main`")
        self.assertIsNotNone(line_thread, "старт потока не найден в `_main`")
        self.assertLess(line_tg_import, line_preimport,
                        "предзагрузка идёт РАНЬШЕ импорта `tg_bot` — порядок сломан")
        self.assertLess(line_preimport, line_thread,
                        "поток стартует РАНЬШЕ предзагрузки — гонка §−98 вернулась")

    def test_every_thread_import_is_preloaded(self):
        """Всё, что импортирует ТЕЛО потока, грузит главный поток.

        Это и есть настоящий инвариант: не «вызови функцию», а «в фоновом
        потоке не осталось ни одного импорта, которого главный поток не сделал
        раньше». Новый тяжёлый импорт в теле ингеста уронит этот тест, а не
        прод.
        """
        tree = _tree()
        in_thread   = _imported_modules(_func(tree, _THREAD_TARGET))
        # `_upload_chroma_db_to_store` вызывается из того же потока.
        in_thread  |= _imported_modules(_func(tree, "_upload_chroma_db_to_store"))
        # stdlib-модули гонки не создают: они уже загружены интерпретатором
        # к моменту старта потока (и не тянут numpy).
        stdlib = {"os", "sys", "tempfile", "threading", "asyncio", "logging", "shutil",
                  "time", "json", "pathlib", "typing", "__future__", "importlib"}
        missing = _uncovered(in_thread, _preload_registry(), stdlib)
        self.assertEqual(
            missing, [],
            f"фоновый поток импортирует {missing} первым — главный поток обязан "
            f"загрузить это раньше (добавь в `_BOOT_INGEST_HEAVY_IMPORTS`), иначе "
            f"возвращается гонка импорта numpy (§−98)",
        )

    def test_lazy_imports_of_rag_engine_are_preloaded(self):
        """ЛЕНИВЫЕ импорты `agent.rag_engine` тоже грузит главный поток.

        🔴 Первая редакция починки была НЕПОЛНОЙ ровно здесь: предзагружался
        `agent.rag_engine`, но сам он лёгкий — `chromadb` он импортирует внутри
        `FinancialRAG.__init__`, а `pymupdf4llm` внутри `ingest_pdf`. Оба вызова
        делает фоновый поток, то есть numpy приезжал бы на поток как и раньше,
        и «починка» лечила бы только видимую половину.

        Проверка смотрит В САМ `rag_engine`: любой новый ленивый импорт там
        обязан появиться в реестре предзагрузки.
        """
        rag = _ENTRYPOINT.parent / "agent" / "rag_engine.py"
        if not rag.exists():                                  # pragma: no cover
            self.skipTest("agent/rag_engine.py отсутствует")
        stdlib = {"os", "sys", "re", "math", "time", "json", "logging", "pathlib",
                  "datetime", "typing", "tempfile", "collections", "hashlib",
                  "itertools", "functools", "shutil", "uuid", "importlib"}
        lazy = _lazy_imports_inside_functions(rag)
        missing = _uncovered(lazy, _preload_registry(), stdlib)
        self.assertEqual(
            missing, [],
            f"`agent.rag_engine` лениво импортирует {missing} — это исполнится НА "
            f"ФОНОВОМ ПОТОКЕ. Добавь в `_BOOT_INGEST_HEAVY_IMPORTS`, иначе гонка "
            f"импорта numpy возвращается (§−98)",
        )

    def test_numpy_is_preloaded_first(self):
        """numpy — общий корень гонки, он грузится раньше всех остальных."""
        registry = _preload_registry()
        self.assertTrue(registry, "реестр предзагрузки пуст")
        self.assertEqual(
            registry[0], "numpy",
            "numpy грузится не первым: именно его модульные локи и рвали два "
            "потока (§−98)",
        )

    def test_preimport_never_raises(self):
        """Предзагрузка обёрнута в try/except: старт бота не зависит от RAG."""
        pre = _func(_tree(), _PREIMPORT)
        self.assertTrue(
            any(isinstance(n, ast.Try) for n in ast.walk(pre)),
            f"`{_PREIMPORT}` обязана глотать ошибку импорта: RAG — фолбэк, "
            "а не условие старта бота",
        )


class IngestBotWorkerImportsTest(unittest.TestCase):
    """`§−99`: тот же инвариант — для второго бота.

    У загрузчика нет демон-потока, зато есть `asyncio.to_thread`: каждая
    команда уходит в РАБОЧИЙ поток пула. Под `_LOCK` стоят только приём файла
    и чистка — `/status`, `/universe`, `/check` и плановая проверка идут мимо,
    поэтому два потока могут войти в numpy одновременно. Разница с `§−98` не в
    механизме, а в последствии: здесь исключение ловится, и вместо падения
    оператор получает подменённый ответ («C-1 НЕ ПРОВЕРЕН», исчезнувший срок
    до блокировки) — то есть дефект ТИШЕ и потому опаснее.
    """

    def setUp(self) -> None:
        self.bot = _ENTRYPOINT.parent / "ingest_bot.py"
        if not self.bot.exists():
            self.skipTest("src/ingest_bot.py отсутствует")

    def _chain_modules(self) -> set[str]:
        """Ленивые импорты всей цепочки, которую бот зовёт из потоков."""
        src = _ENTRYPOINT.parent
        out: set[str] = set()
        for rel in ("services/quote_ingest.py", "services/quote_publisher.py"):
            path = src / rel
            if path.exists():
                out |= _lazy_imports_inside_functions(path)
        return out

    def test_registry_is_preloaded_before_any_thread_work(self):
        """`preimport_worker_deps` вызывается в `main` до старта polling."""
        tree = ast.parse(self.bot.read_text(encoding="utf-8"), filename=str(self.bot))
        main = _func(tree, "main")
        call_line = poll_line = None
        for sub in ast.walk(main):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                    and sub.func.id == "preimport_worker_deps":
                call_line = sub.lineno
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                    and sub.func.attr == "start_polling":
                poll_line = sub.lineno
        self.assertIsNotNone(call_line,
                             "`preimport_worker_deps()` не вызывается в `main` — "
                             "рабочие потоки снова импортируют numpy наперегонки (§−99)")
        self.assertIsNotNone(poll_line, "`start_polling` исчез из `main`")
        self.assertLess(call_line, poll_line,
                        "предзагрузка идёт ПОСЛЕ старта polling — команда может "
                        "приехать раньше неё")

    def test_every_lazy_import_of_the_chain_is_preloaded(self):
        """Каждый ленивый импорт `quote_*` есть в реестре загрузчика."""
        stdlib = {"os", "sys", "tempfile", "logging", "pathlib", "typing",
                  "sqlite3", "json", "dataclasses", "datetime", "importlib",
                  "asyncio", "html", "signal", "shutil", "time", "__future__"}
        missing = _uncovered(self._chain_modules(),
                             _registry(self.bot, "_WORKER_HEAVY_IMPORTS"), stdlib)
        self.assertEqual(
            missing, [],
            f"цепочка загрузчика лениво импортирует {missing} — это исполнится в "
            f"РАБОЧЕМ ПОТОКЕ. Добавь в `_WORKER_HEAVY_IMPORTS` (§−99)")

    def test_numpy_is_preloaded_first(self):
        registry = _registry(self.bot, "_WORKER_HEAVY_IMPORTS")
        self.assertTrue(registry, "реестр загрузчика пуст")
        self.assertEqual(registry[0], "numpy",
                         "numpy грузится не первым — он общий корень гонки")

    def test_preimport_never_raises(self):
        tree = ast.parse(self.bot.read_text(encoding="utf-8"), filename=str(self.bot))
        fn = _func(tree, "preimport_worker_deps")
        self.assertTrue(
            any(isinstance(n, ast.Try) for n in ast.walk(fn)),
            "предзагрузка обязана глотать ошибку импорта: в офлайне "
            "`google.cloud.storage` может отсутствовать вовсе")


class SharedTokenIsRefusedTest(unittest.TestCase):
    """`§−99`: один токен на два бота роняет ГЛАВНЫЙ бот — старт запрещён."""

    def setUp(self) -> None:
        if not (_ENTRYPOINT.parent / "ingest_bot.py").exists():
            self.skipTest("src/ingest_bot.py отсутствует")

    def test_guard_is_called_before_the_bot_object_is_built(self):
        bot = _ENTRYPOINT.parent / "ingest_bot.py"
        tree = ast.parse(bot.read_text(encoding="utf-8"), filename=str(bot))
        main = _func(tree, "main")
        guard_line = build_line = None
        for sub in ast.walk(main):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id == "_refuse_shared_token":
                    guard_line = sub.lineno
                elif sub.func.id == "Bot":
                    build_line = sub.lineno
        self.assertIsNotNone(guard_line,
                             "`_refuse_shared_token()` не вызывается: копия секрета "
                             "главного бота уронит прод (§−99)")
        self.assertIsNotNone(build_line, "`Bot(...)` исчез из `main`")
        self.assertLess(guard_line, build_line,
                        "страж срабатывает ПОСЛЕ создания Bot — поздно")

    _KEYS = ("OMBRI_BOT_TOKEN", "RAMP_BOT_TOKEN", "OMBRI_INGEST_BOT_TOKEN",
             "RAMP_INGEST_BOT_TOKEN", "INGEST_ADMIN_IDS")

    def _reloaded(self, env: dict):
        """Перезагрузить `ingest_bot` с заданным окружением.

        Токен читается на уровне МОДУЛЯ, поэтому проверять его можно только
        через перезагрузку — иначе тест мерил бы окружение своего процесса,
        а не то, что увидит бот на старте.
        """
        import importlib as _il                           # noqa: PLC0415
        import os                                         # noqa: PLC0415
        for key in self._KEYS:
            os.environ.pop(key, None)
        os.environ.update(env)
        try:
            module = _il.import_module("ingest_bot")
        except ModuleNotFoundError:                       # pragma: no cover
            self.skipTest("aiogram не установлен")
        return _il.reload(module)

    def setUp(self) -> None:
        super().setUp()
        import os                                         # noqa: PLC0415
        self._prev = {k: os.environ.get(k) for k in self._KEYS}

        def _restore() -> None:
            for key, value in self._prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(_restore)

    def test_identical_token_refuses(self):
        ingest_bot = self._reloaded({"OMBRI_BOT_TOKEN": "111:SAME",
                                     "OMBRI_INGEST_BOT_TOKEN": "111:SAME",
                                     "INGEST_ADMIN_IDS": "42"})
        with self.assertRaises(RuntimeError) as ctx:
            ingest_bot._refuse_shared_token()
        self.assertIn("OMBRI_BOT_TOKEN", str(ctx.exception))

    def test_different_tokens_are_silent(self):
        ingest_bot = self._reloaded({"OMBRI_BOT_TOKEN": "222:OTHER",
                                     "OMBRI_INGEST_BOT_TOKEN": "111:MINE",
                                     "INGEST_ADMIN_IDS": "42"})
        ingest_bot._refuse_shared_token()                 # молчание — штатно

    def test_legacy_name_of_the_main_bot_is_still_compared(self):
        """Главный бот ещё под именем `RAMP_BOT_TOKEN` — совпадение ловится.

        🔴 Переезд РАЗНЕСЁН во времени: загрузчик переименован, главный бот
        работает под прежним именем и не тронут. Если бы страж смотрел только
        на новое имя, то ровно в это окно — то есть СЕЙЧАС — он ничего бы не
        поймал, а копия секрета уронила бы прод.
        """
        ingest_bot = self._reloaded({"RAMP_BOT_TOKEN": "111:SAME",
                                     "OMBRI_INGEST_BOT_TOKEN": "111:SAME",
                                     "INGEST_ADMIN_IDS": "42"})
        with self.assertRaises(RuntimeError) as ctx:
            ingest_bot._refuse_shared_token()
        self.assertIn("RAMP_BOT_TOKEN", str(ctx.exception))


if __name__ == "__main__":                                        # pragma: no cover
    unittest.main()
