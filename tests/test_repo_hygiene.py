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
MAX_LINES = 200
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
        self.assertNotIn(
            "**Версия:** 2026", self.text.replace("**Версия:** 2026-08-12 ·", ""),
            "в AUDIT.md вернулся блок «Версия» — пересказ раундов в шапке. "
            "Одна мысль живёт в одном месте: запись раунда — в rounds/.")

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
