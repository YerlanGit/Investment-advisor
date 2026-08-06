"""`analyze_all` — оркестратор, и он обязан им ОСТАТЬСЯ (Арх-3.11).

Раунд: архитектурный трек `docs/ARCHITECTURE_FOR_AGENTS.md §4`, фаза Арх-3.11
(закрепление разреза). Имя — по конвенции Арх-4 (`test_<область>_<тема>.py`).

Зачем гейт, а не «мы же договорились»
------------------------------------
Разрез Арх-3 привёл `analyze_all` с 1 240 строк к 86. Но ровно так же когда-то
выглядел и `CLAUDE.md`, который потом дорос до мега-строк — по одной разумной
правке за раунд, каждая из которых по отдельности была оправдана (`§2.3`).
Функция отрастает тем же способом: «сюда быстрее дописать, чем заводить
стадию». Без исполняемого потолка возврат к простыне — вопрос времени, а не
намерения.

Что охраняется
--------------
1. **Размер.** Тело ≤ `MAX_BODY_LINES` строк и ≤ `MAX_TOP_STATEMENTS`
   операторов верхнего уровня.
2. **Порядок стадий.** Он не косметика: стадии общаются через состояние
   движка (`_last_*`), читаемое из `getattr`, — перестановка изменит числа
   МОЛЧА, а обычный скан атрибутов такую связь не находит (`§4 Арх-3 §0a`).
   Замерено (`§−66`): всё, что идёт после риск-модели, зависит от неё; поэтому
   порядок пинится целиком, а не «первые три».
3. **Полнота проводки.** Каждая объявленная стадия вызывается РОВНО один раз:
   заведённая, но не подключённая стадия — мёртвый код, а вызванная дважды —
   удвоенные побочные эффекты.
4. **Один выход.** Ранний `return` из оркестратора означал бы, что часть
   контракта собирается в обход `_build_results`.

Почему источник ищется через КЛАСС, а не по пути
------------------------------------------------
`inspect.getsourcefile(UniversalPortfolioManager)` следует за кодом, куда бы
его ни переложили. Это не педантизм: Арх-3.10 превращает
`finance/investment_logic.py` в фасад, и тест, прибитый к этому пути, начал бы
читать почти пустой файл. Для `assertIn`-проверок это громкое падение, а для
`assertNotIn` — ТИХОЕ позеленение: защита исчезает, а отчёт зелёный
(тот же класс, что `§−64`).
"""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from finance.investment_logic import UniversalPortfolioManager

#: Потолок тела оркестратора. План Арх-3.11 называл «≈150 строк»; фактический
#: размер после 3.8 — 87, так что запас двукратный. Порог намеренно НЕ
#: подтягивается вплотную к факту: тест обязан ловить возврат к простыне, а не
#: мешать добавить стадию.
MAX_BODY_LINES = 150

#: Операторов верхнего уровня. После 3.8 их 11 (докстринг, гейт мандата,
#: семь стадий, сборка контекста, return).
MAX_TOP_STATEMENTS = 20

#: ПОРЯДОК стадий конвейера. Менять этот список можно только вместе с
#: осознанным решением переставить стадии — а его на треке Арх-3 нет.
EXPECTED_STAGE_ORDER = [
    "_stage_normalize",
    "_stage_price_and_fx",
    "_stage_data_quality",
    "_stage_risk_model",
    "_stage_benchmarks",
    "_stage_scoring",
    "_stage_overlay",
    "_build_results",
]


def _class_source_tree() -> tuple[ast.Module, str]:
    """AST файла, где РЕАЛЬНО определён класс движка."""
    path = Path(inspect.getsourcefile(UniversalPortfolioManager))
    text = path.read_text(encoding="utf-8")
    return ast.parse(text), str(path)


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "UniversalPortfolioManager":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == name:
                    return member
    raise AssertionError(f"метод {name} не найден")


def _self_calls_in_order(fn: ast.FunctionDef) -> list[str]:
    """Имена вызванных `self.<...>` стадий В ПОРЯДКЕ следования по тексту.

    Порядок берётся по номеру строки ВЫЗОВА, а не по позиции в AST-обходе:
    `ast.walk` идёт в ширину и на вложенных вызовах даёт не тот порядок.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and (node.func.attr.startswith("_stage_")
                     or node.func.attr == "_build_results")):
            hits.append((node.lineno, node.func.attr))
    return [name for _, name in sorted(hits)]


class OrchestratorShapeTest(unittest.TestCase):
    """Форма `analyze_all` после Арх-3: диспетчер, а не вычислитель."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree, cls.path = _class_source_tree()
        cls.fn = _method(cls.tree, "analyze_all")

    def test_body_stays_small(self) -> None:
        body = [s for s in self.fn.body]
        first = body[0]
        # докстринг в счёт тела не идёт — он документация, а не логика
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            body = body[1:]
        lines = body[-1].end_lineno - body[0].lineno + 1
        self.assertLessEqual(
            lines, MAX_BODY_LINES,
            f"`analyze_all` разрослась до {lines} строк (потолок {MAX_BODY_LINES}) "
            f"в {self.path}. Новая логика — это НОВАЯ СТАДИЯ, а не ещё один блок "
            "в оркестраторе: именно так функция и доросла до 1 240 строк.",
        )
        self.assertLessEqual(
            len(self.fn.body), MAX_TOP_STATEMENTS,
            f"операторов верхнего уровня {len(self.fn.body)} "
            f"(потолок {MAX_TOP_STATEMENTS}) — оркестратор обрастает логикой",
        )

    def test_stages_are_called_in_the_pinned_order(self) -> None:
        """Порядок стадий задан не стилем, а состоянием движка (`_last_*`)."""
        self.assertEqual(
            _self_calls_in_order(self.fn), EXPECTED_STAGE_ORDER,
            "порядок вызова стадий изменился. Стадии общаются через состояние "
            "движка, читаемое из `getattr`, поэтому перестановка меняет числа "
            "МОЛЧА — golden-фикстура поймает не всё. Если перестановка "
            "НАМЕРЕННАЯ, её нужно обосновать замером, а не правкой этого списка.",
        )

    def test_every_declared_stage_is_wired_exactly_once(self) -> None:
        """Объявленная, но не вызванная стадия — мёртвый код; вызванная дважды —
        удвоенные побочные эффекты (стадии пишут в `self.engine`)."""
        declared = {
            m.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef) and node.name == "UniversalPortfolioManager"
            for m in node.body
            if isinstance(m, ast.FunctionDef) and m.name.startswith("_stage_")
        }
        called = _self_calls_in_order(self.fn)
        self.assertEqual(
            declared, {c for c in called if c.startswith("_stage_")},
            "множество ОБЪЯВЛЕННЫХ стадий разошлось с множеством ВЫЗВАННЫХ",
        )
        for name in sorted(declared):
            self.assertEqual(
                called.count(name), 1,
                f"стадия {name} вызвана {called.count(name)} раз(а), ожидался ровно один",
            )

    def test_orchestrator_has_a_single_exit(self) -> None:
        """Ранний выход означал бы сборку контракта в обход `_build_results`."""
        returns = [
            n for n in ast.walk(self.fn)
            if isinstance(n, ast.Return)
            # `return` вложенной функции принадлежит ей, а не оркестратору
            and not any(isinstance(p, (ast.FunctionDef, ast.Lambda))
                        for p in ast.walk(self.fn)
                        if p is not self.fn and n in ast.walk(p))
        ]
        self.assertEqual(
            len(returns), 1,
            f"в `analyze_all` {len(returns)} точек выхода — ожидалась одна",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
