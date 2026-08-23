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
    """Верхнеуровневые пакеты, которые импортирует поддерево `node`.

    `import agent.rag_engine` и `from agent.rag_engine import X` дают одно и
    то же имя: важен ПАКЕТ, чьи модульные локи и участвуют в гонке.
    """
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            out.update(a.name.split(".")[0] for a in sub.names)
        elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0:
            out.add(sub.module.split(".")[0])
    return out


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
                out.update(a.name.split(".")[0] for a in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0 \
                    and id(sub) not in top_level:
                out.add(sub.module.split(".")[0])
    return out


def _preload_registry() -> list[str]:
    """Строки из `_BOOT_INGEST_HEAVY_IMPORTS` — читаем из AST, без импорта."""
    for node in _tree().body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "_BOOT_INGEST_HEAVY_IMPORTS":
            value = node.value
            break
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_BOOT_INGEST_HEAVY_IMPORTS"
                for t in node.targets):
            value = node.value
            break
    else:
        raise AssertionError("_BOOT_INGEST_HEAVY_IMPORTS не найден в entrypoint.py")
    return [e.value for e in value.elts if isinstance(e, ast.Constant)]


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
        preloaded   = {m.split(".")[0] for m in _preload_registry()}
        # stdlib-модули гонки не создают: они уже загружены интерпретатором
        # к моменту старта потока (и не тянут numpy).
        stdlib = {"os", "sys", "tempfile", "threading", "asyncio", "logging", "shutil",
                  "time", "json", "pathlib", "typing", "__future__", "importlib"}
        missing = sorted(in_thread - preloaded - stdlib)
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
        preloaded = {m.split(".")[0] for m in _preload_registry()}
        stdlib = {"os", "sys", "re", "math", "time", "json", "logging", "pathlib",
                  "datetime", "typing", "tempfile", "collections", "hashlib",
                  "itertools", "functools", "shutil", "uuid", "importlib"}
        lazy = _lazy_imports_inside_functions(rag)
        missing = sorted(lazy - preloaded - stdlib)
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


if __name__ == "__main__":                                        # pragma: no cover
    unittest.main()
