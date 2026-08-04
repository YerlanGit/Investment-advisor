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
