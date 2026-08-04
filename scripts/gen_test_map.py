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
Читает AST каждого `tests/test_*.py` и собирает импорты модулей из `src/`,
включая ОТЛОЖЕННЫЕ импорты внутри функций — в этом репозитории так импортируют
тяжёлые `tg_bot`/`ai_narrative`, чтобы не тянуть зависимости на этапе коллекции.
Импорт — доказательство того, что файл реально трогает модуль; упоминание в
строке — нет, поэтому строки не в счёт.

Чисел тестов карта НЕ печатает намеренно: сумма тест-функций по файлам, которые
импортируют модуль, — верхняя оценка, а не покрытие (один файл проверяет
несколько модулей). Заодно это избавляет от ложных срабатываний `--check`:
карта устаревает только при изменении СТРУКТУРЫ, а не от каждого нового теста.

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
>
> **Почему здесь НЕТ числа тестов на модуль.** Такое число было бы суммой
> тест-функций в файлах, которые модуль импортируют, то есть верхней оценкой,
> а не покрытием: один файл обычно проверяет несколько модулей. Печатать его
> рядом со словом «тесты» — ровно тот класс лукавых чисел, который в проекте
> ловят как дефект (`AUDIT §−52`, D-5). Карта отвечает на «куда смотреть»;
> на «насколько покрыто» отвечает coverage, а не импорты.
>
> Побочная выгода: без счётчиков карта не устаревает от каждого нового теста —
> `--check` краснеет только когда изменилась СТРУКТУРА (новый файл, новый
> импорт), то есть когда карта действительно врёт.
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


def build_map() -> str:
    src_mods = _src_modules()
    # модуль → множество тест-файлов, которые его импортируют
    by_module: dict[str, set[str]] = defaultdict(set)
    seen_files: list[str] = []
    untied: list[str] = []

    for path in sorted(TESTS.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        name = path.name
        seen_files.append(name)
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
            by_module[mod].add(name)

    lines: list[str] = [_HEADER, ""]
    total_files = len(seen_files)
    lines.append(
        f"**Замер:** {total_files} тест-файлов · "
        f"{len(by_module)} модулей `src/` имеют хотя бы один адресный тест-импорт.\n"
    )

    lines.append("## Модуль → тесты\n")
    lines.append("| Модуль `src/` | Файлов | Тест-файлы |")
    lines.append("|---|---|---|")
    for mod in sorted(by_module):
        files = sorted(by_module[mod])
        cells = " · ".join(f"`{f}`" for f in files)
        lines.append(f"| `{mod}` | {len(files)} | {cells} |")

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
