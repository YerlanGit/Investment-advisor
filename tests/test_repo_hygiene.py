"""Гигиена репозитория: инструкция не разрастается, карта тестов не врёт.

Раунд: архитектурный трек `docs/ARCHITECTURE_FOR_AGENTS.md §4`, фазы Арх-1 и Арх-4.

Почему это ТЕСТ, а не соглашение
--------------------------------
Корневой `CLAUDE.md` уже один раз превратился из инструкции в журнал: за три дня
он вырос на 45% (21 010 → 30 448 символов ≈ 7 600 токенов в КАЖДОЙ сессии), 95%
объёма ушло в мега-строки, а самая длинная строка достигла 18 252 символов —
прочитать её глазами невозможно, и именно поэтому в ней полгода жила неверная
команда запуска тестов (без `PYTHONPATH=src`). Соглашение «пиши коротко» этого
не остановило, потому что нарушать его было нечем.

В этом репозитории уже доказано, что работает обратное: инвариант, закреплённый
исполняемым гейтом. AST-сканер против `int(os.getenv)` (`AUDIT §−50`) поймал
регресс автора того же правила на следующий день; `NoSecondCopyTest` (`§−46`)
не даёт завести вторую копию перечня тикеров. Здесь — тот же приём.

🔴 ГАРДЫ ОБРАЗА. `Dockerfile` копирует только `requirements*`, `src/`,
`SYSTEM_PROMPT.md` и `tests/`. Ни `CLAUDE.md`, ни `docs/`, ни `scripts/`, ни
`design/` в образе НЕТ, а deploy-гейт Cloud Build гоняет этот же набор тестов
внутри образа. Безусловное чтение такого файла даёт зелёный GitHub CI и
падение деплоя с `FileNotFoundError` — так уже было. Поэтому каждый тест ниже
начинается с проверки существования и `skipTest`.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CLAUDE_MD = _ROOT / "CLAUDE.md"
_SCRIPTS = _ROOT / "scripts"
_TEST_MAP = _ROOT / "tests" / "TEST_MAP.md"
_AUDIT_MD = _ROOT / "docs" / "audit" / "AUDIT.md"

# Пороги — не «красиво», а «читаемо глазами в точке правки».
# Замер после Арх-1 (2026-08-04): 122 строки, максимум 115 символов.
# Запас оставлен намеренно: правила будут добавляться, журнал — нет.
#
# 🔴 `§−99`, потолок поднят 200 → 210 ОДИН раз и по делу. Запас съели не
# пересказы раундов, ради которых гейт и стоит, а две новые ПОДСИСТЕМЫ: гонка
# импорта на потоках (`§−98`, оба бота) и сам бот-загрузчик с его правилами про
# токены и писателя базы. Перед подъёмом файл сжат там, где текст дублировался:
# два правила про `analyze_all` сведены в одно, блок зеркала деплой-образа — с
# семи строк до пяти. Замер после сжатия: 203 строки.
# Гейт от этого не слабеет: дамп истории раунда — это сотни строк, а не десять.
MAX_LINES = 210
MAX_LINE_CHARS = 300


class RootInstructionStaysReadableTest(unittest.TestCase):
    """`CLAUDE.md` — инструкция. Журнал живёт в `docs/audit/AUDIT.md`."""

    def setUp(self) -> None:
        if not _CLAUDE_MD.exists():
            self.skipTest("CLAUDE.md отсутствует (Docker deploy-gate)")
        self.lines = _CLAUDE_MD.read_text(encoding="utf-8").splitlines()

    def test_line_count_within_budget(self) -> None:
        self.assertLessEqual(
            len(self.lines), MAX_LINES,
            f"CLAUDE.md разросся до {len(self.lines)} строк (потолок {MAX_LINES}). "
            "Скорее всего в него дописали историю раунда — её место в docs/audit/AUDIT.md.",
        )

    def test_no_mega_lines(self) -> None:
        """Мега-строка = правило, которое физически нельзя прочитать."""
        offenders = [
            (i + 1, len(ln)) for i, ln in enumerate(self.lines) if len(ln) > MAX_LINE_CHARS
        ]
        self.assertFalse(
            offenders,
            "строки длиннее "
            f"{MAX_LINE_CHARS} символов: {offenders}. Одна мысль = одна строка; "
            "внутри такой строки правило невозможно вычитать глазами (так и жила "
            "неверная команда запуска тестов).",
        )

    def test_verification_command_is_correct(self) -> None:
        """Команда прогона обязана нести `PYTHONPATH=src`.

        Без префикса `import finance…` не находится: в репозитории нет ни
        `conftest.py`, ни `pyproject.toml`. Инструкция, в которой команда не
        работает, хуже отсутствующей.
        """
        text = "\n".join(self.lines)
        self.assertIn("PYTHONPATH=src python -m pytest tests/ -q", text)

    def test_points_at_the_journal(self) -> None:
        """Правило «история → AUDIT.md» должно быть НАПИСАНО, иначе файл отрастёт заново."""
        text = "\n".join(self.lines)
        self.assertIn("docs/audit/AUDIT.md", text)


#: Замер §−91: навигатор после расслоения — 337 строк, максимум 142 символа.
#: Потолки с запасом: доска и правила будут расти, ЖУРНАЛ — нет (он в rounds/).
AUDIT_MAX_LINES = 600
AUDIT_MAX_LINE_CHARS = 400


class AuditJournalStaysNavigableTest(unittest.TestCase):
    """`AUDIT.md` — навигатор; полные записи раундов живут в `rounds/`.

    Гейт написан по ИЗМЕРЕННОЙ деградации (`§−91`), а не на всякий случай:
    журнал дорос до 382 КБ, шапка до 76 КБ, одна строка до 27.6 КБ, а 14
    раундов вообще не получили секции — при том, что на 10 из них ссылается
    боевой код. Дефект был не косметическим: устаревшая доска, утонувшая в
    той самой мега-строке, увела за собой план работ по Manual Portfolio.

    Правило без гейта отрастает обратно — это уже проверено на `CLAUDE.md`.
    """

    def setUp(self) -> None:
        if not _AUDIT_MD.exists():
            self.skipTest("docs/audit/AUDIT.md отсутствует (Docker deploy-gate)")
        self.text = _AUDIT_MD.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()

    def test_navigator_stays_small_enough_to_read_whole(self) -> None:
        self.assertLessEqual(
            len(self.lines), AUDIT_MAX_LINES,
            f"AUDIT.md разросся до {len(self.lines)} строк (потолок "
            f"{AUDIT_MAX_LINES}). Полная запись раунда идёт в "
            "docs/audit/rounds/, сюда — одна строка индекса.")

    def test_no_mega_lines(self) -> None:
        """Строку в 27 КБ невозможно ни ревьюить, ни дифать, ни прочитать."""
        offenders = [(i + 1, len(ln)) for i, ln in enumerate(self.lines)
                     if len(ln) > AUDIT_MAX_LINE_CHARS]
        self.assertFalse(
            offenders,
            f"строки длиннее {AUDIT_MAX_LINE_CHARS} символов: {offenders}. "
            "Так и вырос блок «Версия»: препенд за препендом в одну строку.")

    def test_version_blob_does_not_come_back(self) -> None:
        """🔴 Именно этот блок и был болезнью: препенд-лог, дублирующий раунды.

        Он писал тот же факт дважды, и читаемым первым оказывался худший —
        неструктурированный — экземпляр.
        """
        # Дата в шапке меняется каждый раунд, поэтому сверять её литералом
        # нельзя — гейт сломался бы на первом же обновлении (и сломался).
        # Проверяем ФОРМУ: строка «Версия» допустима РОВНО ОДНА и обязана
        # оставаться однострочной сводкой, а не пересказом раунда.
        version_lines = [ln for ln in self.lines if "**Версия:**" in ln]
        self.assertLessEqual(
            len(version_lines), 1,
            f"строк «Версия» в AUDIT.md: {len(version_lines)}. Так и рос "
            "препенд-лог: раунд за раундом в шапку, пока она не стала 76 КБ. "
            "Запись раунда живёт в rounds/, здесь — одна строка индекса.")
        for ln in version_lines:
            self.assertLessEqual(
                len(ln), 160,
                "строка «Версия» разрослась в пересказ раунда — оставь в ней "
                "только дату, номер последнего раунда и их количество.")

    def test_every_round_has_an_index_line(self) -> None:
        """Индекс — единственный способ найти раунд, не читая 350 КБ."""
        import re
        rounds_dir = _ROOT / "docs" / "audit" / "rounds"
        if not rounds_dir.exists():
            self.skipTest("docs/audit/rounds/ отсутствует")
        have = set()
        for f in rounds_dir.glob("ROUNDS_*.md"):
            have |= set(re.findall(r"^## −([\d.]+)\.",
                                   f.read_text(encoding="utf-8"), re.M))
        indexed = set(re.findall(r"\|\s*`§−([\d.]+)`\s*\|", self.text))
        missing = sorted(have - indexed, key=float, reverse=True)
        self.assertFalse(
            missing,
            f"раунды есть в rounds/, но их нет в индексе AUDIT.md §4: {missing}. "
            "Ненайденный раунд — это висячая ссылка из кода (так §−76…§−89 "
            "прожили 14 раундов без секции).")

    def test_index_points_only_at_existing_rounds(self) -> None:
        """Обратная сторона: индекс не должен обещать несуществующее."""
        import re
        rounds_dir = _ROOT / "docs" / "audit" / "rounds"
        if not rounds_dir.exists():
            self.skipTest("docs/audit/rounds/ отсутствует")
        have = set()
        for f in rounds_dir.glob("ROUNDS_*.md"):
            have |= set(re.findall(r"^## −([\d.]+)\.",
                                   f.read_text(encoding="utf-8"), re.M))
        indexed = set(re.findall(r"\|\s*`§−([\d.]+)`\s*\|", self.text))
        self.assertFalse(sorted(indexed - have, key=float),
                         "индекс ссылается на раунды, которых нет в rounds/")


class RoundReferencesResolveTest(unittest.TestCase):
    """Каждая ссылка `§−N` из кода и доков обязана указывать на живой раунд.

    До `§−91` десять номеров, на которые ссылались `stooq_ingest.py`,
    `stooq_store.py`, `stooq_provider.py`, `data_lineage.py`, `tg_bot.py` и
    тесты, не имели секции вовсе — агент, идущий по ссылке, не находил ничего.
    """

    def test_no_dangling_section_references(self) -> None:
        import re
        rounds_dir = _ROOT / "docs" / "audit" / "rounds"
        if not rounds_dir.exists():
            self.skipTest("docs/audit/rounds/ отсутствует")
        have = set()
        for f in rounds_dir.glob("ROUNDS_*.md"):
            have |= set(re.findall(r"^## −([\d.]+)\.",
                                   f.read_text(encoding="utf-8"), re.M))
        refs: set[str] = set()
        for path in _ROOT.rglob("*"):
            if path.is_dir() or "node_modules" in path.parts or ".git" in path.parts:
                continue
            if path.suffix not in (".py", ".md", ".jsx"):
                continue
            if "docs/audit/rounds/" in path.as_posix():
                continue
            try:
                refs |= set(re.findall(r"§−(\d+(?:\.\d+)?)",
                                       path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError):
                continue
        missing = sorted(refs - have, key=float, reverse=True)
        self.assertFalse(
            missing,
            f"ссылки на несуществующие раунды: {missing}. Либо раунд не "
            "записан в rounds/, либо номер в ссылке опечатан.")


class TestMapIsGeneratedAndFreshTest(unittest.TestCase):
    """`tests/TEST_MAP.md` собирается скриптом и не разъезжается с кодом."""

    def test_map_exists_and_is_marked_generated(self) -> None:
        if not _TEST_MAP.exists():
            self.skipTest("tests/TEST_MAP.md отсутствует")
        head = _TEST_MAP.read_text(encoding="utf-8")[:600]
        self.assertIn("СГЕНЕРИРОВАН", head,
                      "карта обязана честно говорить, что её не правят руками")

    def test_map_is_up_to_date(self) -> None:
        """Свежесть проверяется тем же генератором (`--check`).

        Скрипт живёт в `scripts/`, которого в образе деплоя нет, — отсюда skip.
        Паттерн тот же, что у `CompiledAssetsTest` для Premium-бандлов.
        """
        if not (_SCRIPTS / "gen_test_map.py").exists():
            self.skipTest("scripts/ отсутствует (Docker deploy-gate)")
        if not _TEST_MAP.exists():
            self.fail("tests/TEST_MAP.md не сгенерирован: python scripts/gen_test_map.py")
        proc = subprocess.run(
            [sys.executable, str(_SCRIPTS / "gen_test_map.py"), "--check"],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(
            proc.returncode, 0,
            "карта тестов устарела — перегенерируй `python scripts/gen_test_map.py`.\n"
            f"{proc.stdout}{proc.stderr}",
        )


class TestsThatReadRepoFilesMustSkipInTheImageTest(unittest.TestCase):
    """Тест, читающий файл ВНЕ образа, обязан сам себя пропустить (`§−99`).

    🔴 Цена ошибки — весь деплой, а не один тест. Гейт `cloudbuild` гоняет
    `unittest discover` ВНУТРИ образа, а образ несёт только то, что копирует
    `Dockerfile`: `src/`, `tests/`, `SYSTEM_PROMPT.md` и requirements. Тест,
    открывающий `cloudbuild.yaml` или `scripts/`, падает там `FileNotFoundError`
    — сборка краснеет, и НЕ ДЕПЛОИТСЯ НИКТО, включая главный бот.

    Так и случилось с `DeployStepTest` бота-загрузчика: пять падений в образе
    при зелёном GitHub CI, потому что CI видит полный чекаут. Правило это в
    `CLAUDE.md` есть, но названо было только про `design/` — а помнить его надо
    про ЛЮБОЙ путь из корня репо. Поэтому здесь оно исполняемое.

    Проверка намеренно ГРУБАЯ: достаточно, чтобы в теле класса или функции,
    где путь строится, БЫЛ ВЫЗОВ `skipTest` (или `raise SkipTest`). Доказать
    срабатывание по всем ветвям статически нельзя, а «есть ли вообще выход»
    ловит ровно ту ошибку, которую люди и совершают, — забыли, а не написали
    неверно.

    🔴 Ищется именно ВЫЗОВ в AST, а не подстрока. Первая редакция этого гейта
    считала слово `skipTest` в тексте узла — и пропустила собственную мутацию:
    слово стояло в КОММЕНТАРИИ рядом. Тот же дефект, что `§−97` E-9, где
    текстовые гейты читали комментарии; повторён здесь через неделю после того,
    как был записан, — поэтому проверка теперь структурная.

    Сам этот тест в образе пропускается: он читает `Dockerfile`, которого там
    нет. Ловить он и должен на полном чекауте — до того, как образ собран.
    """

    #: `Path(...).parents[1] / "имя"` и `parent.parent / "имя"` — обе формы,
    #: которыми тесты этого репозитория добираются до корня.
    _PATH_RE = re.compile(r'(?:parents\[1\]|parent\.parent)\s*/\s*"([^"]+)"')

    @staticmethod
    def _has_skip(node: ast.AST) -> bool:
        """Есть ли в поддереве ВЫЗОВ `skipTest` или `raise SkipTest`."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                if isinstance(func, ast.Attribute) and func.attr == "skipTest":
                    return True
                if isinstance(func, ast.Name) and func.id in ("skipTest",
                                                              "SkipTest"):
                    return True
            if isinstance(sub, ast.Raise):
                exc = sub.exc
                name = getattr(exc, "func", exc)
                if isinstance(name, ast.Attribute) and name.attr == "SkipTest":
                    return True
                if isinstance(name, ast.Name) and name.id == "SkipTest":
                    return True
        return False

    def _image_manifest(self) -> set[str]:
        """Что `Dockerfile` реально кладёт в образ — из него самого, не списком."""
        dockerfile = _ROOT / "Dockerfile"
        if not dockerfile.is_file():
            self.skipTest("Dockerfile отсутствует (сам деплой-гейт)")
        out: set[str] = set()
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.upper().startswith("COPY "):
                continue
            parts = stripped.split()[1:]
            for token in parts[:-1]:                     # последний — назначение
                if token.startswith("--"):
                    continue
                out.add(token.rstrip("/").split("/")[0])
        return out

    def test_every_repo_path_read_by_a_test_is_guarded(self) -> None:
        manifest = self._image_manifest()
        self.assertIn("src", manifest,
                      "разбор Dockerfile сломался: в манифесте нет даже src/")
        offenders: list[str] = []
        for path in sorted((_ROOT / "tests").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                    continue
                body = ast.get_source_segment(source, node) or ""
                names = set(self._PATH_RE.findall(body))
                outside = {n for n in names if n.rstrip("/").split("/")[0]
                           not in manifest}
                if outside and not self._has_skip(node):
                    offenders.append(f"{path.name}::{node.name} → {sorted(outside)}")
        self.assertEqual(
            offenders, [],
            "эти тесты читают файлы, которых в деплой-образе НЕТ, и не умеют "
            "пропуститься — в образе они упадут FileNotFoundError и завалят "
            f"весь деплой (§−99):\n  " + "\n  ".join(offenders))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
