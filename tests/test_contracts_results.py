"""Контракт `analyze_all` → потребители: форма описана в ОДНОМ месте (Арх-2).

Раунд: архитектурный трек `docs/ARCHITECTURE_FOR_AGENTS.md §4`, фаза Арх-2.
Имя файла — по НОВОЙ конвенции (`test_<область>_<тема>.py`, а не по номеру
фазы): номер раунда живёт здесь, в докстринге, а имя отвечает на «что
проверяется». Прежние `test_phase*.py` намеренно НЕ переименовывались —
это ломает ссылки в `AUDIT.md` и доках (решение `AUDIT §−50`).

Что охраняется
--------------
1. Множество ключей, которые движок РЕАЛЬНО кладёт в возвращаемый словарь,
   совпадает с `finance.contracts.AnalyzeResults`. Сверка идёт по AST самого
   `investment_logic.py`, поэтому забыть обновить контракт нельзя: тест
   назовёт и лишние, и недостающие ключи. С Арх-3.1 сборка живёт в
   `_build_results`, и тест дополнительно требует, чтобы `analyze_all` её
   БОЛЬШЕ НЕ делал — иначе «единственная точка сборки» снова станет
   пожеланием.
2. `portfolio_metrics` — то же самое для вложенного блока, который собирается
   в другой функции (`calculate_structural_risk`).
3. `analyze_all` аннотирован типом контракта — иначе связь «функция ↔ форма»
   существует только в голове автора.
4. Слой L0: `contracts.py` не тянет ничего из проекта (иначе он перестаёт
   быть общим основанием и рискует циклическим импортом).

Почему AST, а не вызов функции: `analyze_all` ходит в сеть (цены, SEC, FRED)
и считает Ridge/бутстрап — в юнит-тесте это неуместно, а форма контракта
статична и читается из исходника надёжнее, чем из прогона.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from finance.contracts import (
    PORTFOLIO_METRICS_KEYS,
    RESULTS_KEYS,
    AnalyzeResults,
)

_SRC = Path(__file__).resolve().parent.parent / "src"


def _source_of(cls) -> Path:
    """Файл, где РЕАЛЬНО определён класс.

    Арх-3.10 разложил ядро по пакету `finance/engine/`, и путь, прибитый к
    `investment_logic.py`, стал указывать на фасад. Поиск через класс следует
    за кодом сам — при следующем переезде тест не придётся чинить.
    """
    import inspect
    return Path(inspect.getsourcefile(cls))


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"функция {name} не найдена")


def _own_returns(fn: ast.FunctionDef) -> list[ast.Return]:
    """`return`-узлы САМОЙ функции, без вложенных хелперов.

    В `analyze_all` объявлены вложенные функции (`_dq_requested_days`,
    `_res_key`), и наивный `ast.walk` притащил бы их `return` тоже — тогда
    тест сверял бы контракт не с тем узлом.
    """
    out: list[ast.Return] = []

    def visit(node: ast.AST, top: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # тело вложенной функции — не наш контракт
            if isinstance(child, ast.Return):
                out.append(child)
            visit(child, False)

    visit(fn, True)
    return out


def _dict_keys(node: ast.Dict) -> set[str]:
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


class AnalyzeResultsContractTest(unittest.TestCase):
    """Ключи движка == ключи `AnalyzeResults`."""

    @classmethod
    def setUpClass(cls) -> None:
        from finance.investment_logic import MAC3RiskEngine, UniversalPortfolioManager

        cls.tree = ast.parse(
            _source_of(UniversalPortfolioManager).read_text(encoding="utf-8"))
        # `calculate_structural_risk` живёт у ДВИЖКА, а он с Арх-3.10 в другом
        # модуле пакета — второй разбор нужен именно поэтому.
        cls.engine_tree = ast.parse(
            _source_of(MAC3RiskEngine).read_text(encoding="utf-8"))
        cls.fn = _function_node(cls.tree, "analyze_all")
        # Арх-3.1: словарь собирается в выделенном методе-стадии, а не в теле
        # `analyze_all`. Инвариант «сборка в ОДНОМ месте» сохранён — изменился
        # адрес этого места, и тест следует за ним.
        cls.builder = _function_node(cls.tree, "_build_results")

    def test_single_contract_return(self) -> None:
        """Контрактный словарь собирается РОВНО в одном месте — `_build_results`.

        Это и есть основание для `total=True` в контракте: раз выход один,
        все ключи присутствуют всегда, а отсутствие данных выражается
        ЗНАЧЕНИЕМ. Появится второй выход — контракт станет условным, и его
        придётся пересмотреть, а не молча разойтись с реальностью.
        """
        dict_returns = [r for r in _own_returns(self.builder) if isinstance(r.value, ast.Dict)]
        self.assertEqual(
            len(dict_returns), 1,
            "ожидался ЕДИНСТВЕННЫЙ контрактный return-словарь в `_build_results`; "
            f"найдено {len(dict_returns)} — пересмотри total=True в contracts.py",
        )

    def test_analyze_all_no_longer_assembles_the_dict(self) -> None:
        """`analyze_all` — оркестратор: сам словарь он больше НЕ строит.

        Инвариант введён Арх-3.1 и охраняет главную цель фазы: пока сборка
        может происходить в двух местах, «единственная точка правды» —
        пожелание, а не свойство кода.
        """
        own = [r for r in _own_returns(self.fn) if isinstance(r.value, ast.Dict)]
        self.assertFalse(
            own,
            "в `analyze_all` снова появился словарь-литерал в return — "
            "сборка обязана жить только в `_build_results`",
        )

    def test_engine_keys_match_contract(self) -> None:
        dict_returns = [r for r in _own_returns(self.builder) if isinstance(r.value, ast.Dict)]
        engine_keys = _dict_keys(dict_returns[0].value)

        missing = engine_keys - RESULTS_KEYS
        extra = RESULTS_KEYS - engine_keys
        self.assertFalse(
            missing,
            f"движок кладёт ключи, которых нет в contracts.AnalyzeResults: {sorted(missing)}. "
            "Добавь их СНАЧАЛА в contracts.py — иначе потребитель не узнает форму.",
        )
        self.assertFalse(
            extra,
            f"contracts.AnalyzeResults описывает ключи, которых движок не кладёт: {sorted(extra)}. "
            "Удалил ключ из движка — удали и из контракта.",
        )

    def test_contract_size_is_pinned(self) -> None:
        """Замер зафиксирован числом: рост контракта должен быть ЗАМЕЧЕН.

        36 — не магия, а измерение: 35 на 2026-08-04 + `portfolio_source` (MP-04 ЧК-04.5). Тест не запрещает менять
        контракт, он требует делать это осознанно (и обновить
        `ARCHITECTURE_FOR_AGENTS.md §3.2`, где это число — аргумент).
        """
        self.assertEqual(len(RESULTS_KEYS), 36)

    def test_portfolio_metrics_block_matches(self) -> None:
        """Вложенный блок собирается в другой функции — сверяем отдельно."""
        risk_fn = _function_node(self.engine_tree, "calculate_structural_risk")
        literals = [
            n.value for n in ast.walk(risk_fn)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
            and any(isinstance(t, ast.Name) and t.id == "portfolio_metrics" for t in n.targets)
        ]
        self.assertEqual(len(literals), 1, "ожидалась одна сборка portfolio_metrics")
        self.assertEqual(_dict_keys(literals[0]), set(PORTFOLIO_METRICS_KEYS))

    def test_analyze_all_is_annotated(self) -> None:
        """Связь «функция ↔ форма» закреплена в коде, а не подразумевается."""
        for fn in (self.fn, self.builder):
            self.assertIsNotNone(fn.returns, f"у {fn.name} нет аннотации возврата")
            self.assertIn("AnalyzeResults", ast.unparse(fn.returns))


class ContractsLayerTest(unittest.TestCase):
    """`contracts.py` — слой L0: ничего из проекта не импортирует."""

    def test_no_project_imports(self) -> None:
        path = _SRC / "finance" / "contracts.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        project_roots = {"finance", "services", "agent", "freedom_portfolio",
                         "pdf_payload", "premium_payload", "tg_bot", "ai_narrative"}
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [a.name for a in node.names if a.name.split(".")[0] in project_roots]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in project_roots:
                    offenders.append(node.module)
        self.assertFalse(
            offenders,
            f"contracts.py — слой L0 и обязан оставаться свободным от проекта; найдено: {offenders}",
        )

    def test_keys_frozen_set_matches_annotations(self) -> None:
        self.assertEqual(set(RESULTS_KEYS), set(AnalyzeResults.__annotations__))


class ConsumerListCountsOnlyLIVEModulesTest(unittest.TestCase):
    """🔴 Контракт не имеет права числить МЁРТВЫЙ модуль среди потребителей.

    Докстрока `AnalyzeResults` сама предсказала эту гниль: «список пришлось
    править ОТДЕЛЬНЫМ раундом, потому что он устарел в момент того же коммита —
    иллюстрация правила "замер без исполняемого гейта разъезжается"». Он
    разъехался снова: аудит 26.08 нашёл `agent/advisor_bot` в списке из девяти
    при НУЛЕ импортёров во всём репозитории.

    Цена ошибки — не косметика. Список читают перед правкой `analyze_all`,
    чтобы понять радиус изменения. Мёртвый модуль в нём завышает связность
    контракта и заставляет закладываться на потребителя, которого нет.

    Гейт узкий намеренно: он не пересчитывает ОБРАЩЕНИЯ к ключам (это тот
    самый лукавый счётчик, от которого отказался `TEST_MAP`), а проверяет одно
    проверяемое утверждение — каждый названный потребитель ЖИВ.
    """

    #: Точки входа: их не импортируют, их запускают.
    _ENTRYPOINTS = {"tg_bot", "entrypoint", "ingest_bot", "ingest_entrypoint"}

    def _listed(self) -> list[str]:
        import re                                        # noqa: PLC0415

        doc = AnalyzeResults.__doc__ or ""
        head = doc.split("`total=True`")[0]
        return [m.replace("/", ".") for m in
                re.findall(r"`([\w/]+)`\s*\(\d+\)", head)]

    def test_the_docstring_actually_lists_consumers(self) -> None:
        """Гейт бесполезен, если список не распознан: пустой ⇒ нечего проверять."""
        listed = self._listed()
        self.assertGreaterEqual(len(listed), 5,
                                f"список потребителей не разобран: {listed}")
        self.assertIn("pdf_payload", listed)

    def test_every_listed_consumer_is_reachable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        src = root / "src"
        if not src.is_dir():
            self.skipTest("src/ отсутствует")
        blob = "\n".join(f.read_text(encoding="utf-8")
                         for f in src.rglob("*.py"))
        dead = []
        for mod in self._listed():
            short = mod.split(".")[-1]
            if short in self._ENTRYPOINTS:
                continue
            if not re.search(rf"^\s*(?:import|from)\s+[\w.]*\b{re.escape(short)}\b",
                             blob, re.MULTILINE):
                dead.append(mod)
        self.assertEqual(
            dead, [],
            "контракт числит потребителями модули, которых никто не импортирует "
            f"— радиус изменения `analyze_all` завышен: {dead}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
