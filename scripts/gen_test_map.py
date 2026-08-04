#!/usr/bin/env python3
"""Генератор `tests/TEST_MAP.md` — «какие тесты защищают этот модуль» (Арх-4).

Зачем
-----
91% тест-файлов названы по НОМЕРУ ФАЗЫ (`test_phase28_*.py`), то есть по тому,
КОГДА они написаны. Вопрос агента перед правкой — ЧТО защищено у модуля, на
который он собирается покуситься, — по такому имени не решается: покрытие
`finance/regime.py` размазано по семи файлам, и ни один не назван «regime».
Цена: либо гонять весь набор (~3 минуты) на каждую гипотезу, либо править
вслепую.

Массовое переименование ОТКЛОНЕНО осознанно (`AUDIT §−50`): оно ломает ссылки
в `AUDIT.md`, доках и истории PR. Поэтому навигация строится рядом — картой,
которая генерируется из кода и потому не устаревает молча.

Как работает
------------
Читает AST каждого `tests/test_*.py`, собирает импорты модулей из `src/`
(включая ОТЛОЖЕННЫЕ импорты внутри функций — в этом репозитории так
импортируют `tg_bot`/`ai_narrative`, чтобы не тянуть тяжёлые зависимости на
этапе коллекции) и считает тест-функции. Импорт — доказательство того, что
файл реально трогает модуль; упоминание в строке — нет, поэтому строки не в счёт.

Запуск
------
    python scripts/gen_test_map.py            # перезаписать tests/TEST_MAP.md
    python scripts/gen_test_map.py --check    # не писать, вернуть 1 при расхождении

`--check` использует тест `tests/test_repo_hygiene.py`, чтобы карта не
разъезжалась с реальностью.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
OUT = TESTS / "TEST_MAP.md"

_HEADER = """# TEST_MAP.md — какие тесты защищают какой модуль

<!-- nav | area:tests | code:(все модули src/) | read-before:правка любого модуля — сначала посмотри, что уже защищено -->

> **Файл СГЕНЕРИРОВАН** — не правь руками:
> `python scripts/gen_test_map.py` (проверка свежести — `--check`).
>
> Зачем он есть: 91% тест-файлов названы по номеру фазы, то есть по тому, КОГДА
> написаны. Эта карта отвечает на другой вопрос — ЧТО защищено у модуля,
> который ты собрался править. Связь установлена по ИМПОРТАМ (включая
> отложенные, внутри функций), а не по совпадению строк.
>
> Переименование `test_phase*.py` отклонено осознанно (`AUDIT §−50`): оно рвёт
> ссылки в `AUDIT.md` и доках. Для НОВЫХ файлов конвенция другая —
> `test_<область>_<тема>.py`, номер раунда в докстринге.
"""


def _src_modules() -> set[str]:
    """Имена модулей, импортируемые с `PYTHONPATH=src`."""
    mods: set[str] = set()
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
            if not parts:
                continue
        else:
            parts[-1] = parts[-1][: -len(".py")]
        mods.add(".".join(parts))
    return mods


def _imports_of(tree: ast.AST) -> set[str]:
    """Все импортируемые имена, включая отложенные (внутри функций)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # относительный импорт внутри tests/ — не про src
                continue
            if node.module:
                found.add(node.module)
    return found


def _test_count(tree: ast.AST) -> int:
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
    )


def build_map() -> str:
    src_mods = _src_modules()
    # модуль → {тест-файл: число тестов}
    by_module: dict[str, dict[str, int]] = defaultdict(dict)
    file_tests: dict[str, int] = {}
    untied: list[str] = []

    for path in sorted(TESTS.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        name = path.name
        n_tests = _test_count(tree)
        file_tests[name] = n_tests
        imported = _imports_of(tree)
        touched = set()
        for imp in imported:
            # `from finance.regime import X` → finance.regime;
            # `import tg_bot` → tg_bot; префиксный матч ловит подпакеты.
            if imp in src_mods:
                touched.add(imp)
            else:
                head = imp.split(".")[0]
                if head in src_mods:
                    touched.add(head)
        if not touched:
            untied.append(name)
        for mod in touched:
            by_module[mod][name] = n_tests

    lines: list[str] = [_HEADER, ""]
    total_files = len(file_tests)
    total_tests = sum(file_tests.values())
    lines.append(
        f"**Замер:** {total_files} тест-файлов · {total_tests} тест-функций · "
        f"{len(by_module)} модулей `src/` под покрытием.\n"
    )

    lines.append("## Модуль → тесты\n")
    lines.append(
        "> **Как читать колонку «тест-функций†».** Это сумма тест-функций В ФАЙЛАХ, которые\n"
        "> импортируют модуль, — **верхняя оценка, а не счётчик тестов именно этого модуля**:\n"
        "> один файл обычно проверяет несколько модулей сразу. Колонка отвечает на вопрос\n"
        "> «куда смотреть», а не «насколько покрыто». Для второго нужен coverage, а не импорты.\n"
    )
    lines.append("| Модуль `src/` | тест-функций† | Файлы (тест-функций в файле) |")
    lines.append("|---|---|---|")
    for mod in sorted(by_module):
        files = by_module[mod]
        total = sum(files.values())
        cells = " · ".join(
            f"`{f}` ({n})" for f, n in sorted(files.items(), key=lambda x: (-x[1], x[0]))
        )
        lines.append(f"| `{mod}` | {total} | {cells} |")

    uncovered = sorted(m for m in src_mods if m not in by_module and not m.endswith("__init__"))
    if uncovered:
        lines.append("\n## Модули без прямого тест-импорта\n")
        lines.append(
            "> Не обязательно «без покрытия»: модуль может проверяться через вызывающий код. "
            "Но правка здесь — без страховки в виде адресного теста.\n"
        )
        for mod in uncovered:
            lines.append(f"- `{mod}`")

    if untied:
        lines.append("\n## Тест-файлы, не привязанные к модулям `src/`\n")
        for f in sorted(untied):
            lines.append(f"- `{f}`")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="не писать файл; выйти с кодом 1, если карта устарела")
    args = ap.parse_args()

    content = build_map()
    if args.check:
        if not OUT.exists():
            print(f"НЕТ ФАЙЛА: {OUT} — запусти `python scripts/gen_test_map.py`")
            return 1
        if OUT.read_text(encoding="utf-8") != content:
            print(f"УСТАРЕЛА: {OUT} — перегенерируй `python scripts/gen_test_map.py`")
            return 1
        print(f"свежая: {OUT}")
        return 0

    OUT.write_text(content, encoding="utf-8")
    print(f"записано: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
