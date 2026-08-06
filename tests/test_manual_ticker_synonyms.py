"""Синонимы тикеров: ОДИН словарь и одинаковый результат (MP-04 · ЧК-04.1).

Раунд: `AUDIT §−70`, фаза Manual Portfolio 04 (B-1 плана
`docs/roadmap/manual_portfolio/PHASE_04_MANUAL_SOURCE.md §B-1`).

Что охраняется и почему именно так
----------------------------------
**1. Синоним и канон резолвятся ОДИНАКОВО.**

План прямо запрещает заводить второй словарь в парсере ручного ввода (ловушка
Т-4): «пользователь, написавший „каспи“, и брокер, приславший `KSPI`, попадут в
разные ветки». Но одного словаря МАЛО — расхождение возможно и внутри него, из-за
порядка шагов `resolve_tickers`.

Замерено на живом коде до правки: маппинг применялся ПОСЛЕ подстановки прокси,
поэтому синоним, ведущий на неликвидную бумагу, минова́л проверку и возвращал сам
тикер, тогда как канон возвращал прокси. Для `FFSPC6.1028.AIX` это означало
бумагу без ценовой истории ВНУТРИ факторной модели — то есть тихо испорченную
ковариацию, а не громкую ошибку.

Разрыв был ЛАТЕНТНЫМ: ни одна запись `TICKER_MAP` на проксируемую бумагу не
ведёт. Активировал бы его ровно ручной ввод — там AIX-ноты приходят по имени.
Поэтому тест проверяет ВЕСЬ словарь целиком, а не выборочные пары: он обязан
сработать на записи, которую добавят завтра.

**2. Словарь — единственный.** Появление второго словаря синонимов в слое ввода
означает, что B-1 обойдён; тест это называет.

Почему не проверяем «правильность» самих символов
-------------------------------------------------
Тест не знает, что `KSPI.KZ` — это действительно Kaspi: это факт о внешнем мире,
а не о коде. Он проверяет СВОЙСТВО отображения (согласованность), которое в коде
живёт. Сверка символов с живым листингом KASE — отдельный чекпойнт ЧК-08.1, и
`KEGC.KZ` там помечен как требующий подтверждения.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from finance.investment_logic import MAC3RiskEngine

_SRC = Path(__file__).resolve().parent.parent / "src"


class TickerSynonymResolutionTest(unittest.TestCase):
    """Каждая запись `TICKER_MAP` резолвится так же, как её канон."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = MAC3RiskEngine()

    def test_map_is_not_empty(self) -> None:
        """Пустой словарь сделал бы остальные тесты вакуумными."""
        self.assertGreaterEqual(
            len(self.engine.TICKER_MAP), 8,
            "TICKER_MAP подозрительно мал — проверь, не потерялся ли он",
        )

    def test_synonym_resolves_exactly_like_its_canonical(self) -> None:
        """Главный инвариант B-1: написание не влияет на результат."""
        mismatches: list[str] = []
        for synonym, canonical in self.engine.TICKER_MAP.items():
            got_syn = self.engine.resolve_tickers([synonym])
            got_can = self.engine.resolve_tickers([canonical])
            if got_syn != got_can:
                mismatches.append(
                    f"{synonym!r} → {got_syn} , но канон {canonical!r} → {got_can}")
        self.assertFalse(
            mismatches,
            "синоним и канон резолвятся ПО-РАЗНОМУ — одна бумага попадёт в две "
            "ветки конвейера:\n  " + "\n  ".join(mismatches) +
            "\n\nСамая вероятная причина: запись ведёт на бумагу, которую "
            "`proxy_for` подменяет. Тогда чинить надо не словарь, а порядок "
            "шагов в `resolve_tickers` — маппинг обязан идти ДО прокси.",
        )

    def test_case_and_whitespace_do_not_matter(self) -> None:
        """Ручной ввод приходит как попало: «каспи», « KASPI », «Каспи»."""
        for raw in ("kaspi", " KASPI ", "Kaspi", "каспи", " Каспи "):
            with self.subTest(raw=raw):
                self.assertEqual(
                    self.engine.resolve_tickers([raw]), ["KSPI.KZ"],
                    f"{raw!r} не распознан — парсер ручного ввода получит "
                    "неизвестный тикер там, где пользователь написал понятное имя",
                )

    def test_cyrillic_and_latin_agree(self) -> None:
        """Кириллическое и латинское написание — одна бумага."""
        for cyr, lat in (("КАСПИ", "KASPI"), ("ХАЛЫК", "HALYK"),
                         ("КМГ", "KMG"), ("КАЗАТОМПРОМ", "KAZATOMPROM")):
            with self.subTest(pair=(cyr, lat)):
                self.assertEqual(
                    self.engine.resolve_tickers([cyr]),
                    self.engine.resolve_tickers([lat]),
                )

    def test_no_entry_points_at_a_proxied_instrument(self) -> None:
        """Прямая проверка условия, которое ломало согласованность.

        Держится ОТДЕЛЬНО от теста выше намеренно: тот проверяет следствие
        (результаты совпали), этот — причину. Когда однажды сломается порядок
        шагов, причина назовёт виновника сразу.
        """
        offenders = {
            syn: (canon, self.engine.proxy_for(canon))
            for syn, canon in self.engine.TICKER_MAP.items()
            if self.engine.proxy_for(canon) is not None
        }
        self.assertFalse(
            offenders,
            "запись словаря ведёт на ПРОКСИРУЕМУЮ бумагу: "
            f"{offenders}. Это допустимо только если `resolve_tickers` "
            "применяет маппинг ДО подстановки прокси — проверь порядок шагов.",
        )


class SingleSynonymDictionaryTest(unittest.TestCase):
    """B-1: словарь синонимов в проекте ровно один."""

    def test_no_second_synonym_map_in_src(self) -> None:
        """Второй словарь = два источника правды, которые разъедутся.

        Ищем по AST присваивания с именем, похожим на карту тикеров, вне того
        модуля, где живёт единственный разрешённый словарь.
        """
        allowed = (_SRC / "finance" / "engine" / "risk_engine.py").resolve()
        offenders: list[str] = []
        for path in _SRC.rglob("*.py"):
            if path.resolve() == allowed:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:                       # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    name = getattr(t, "id", None) or getattr(t, "attr", None)
                    if name and "TICKER_MAP" in name.upper():
                        offenders.append(f"{path.relative_to(_SRC)}:{node.lineno} {name}")
        self.assertFalse(
            offenders,
            "найден второй словарь синонимов тикера — B-1 требует ОДИН "
            "(`MAC3RiskEngine.TICKER_MAP`), а распознавание вести через "
            f"`resolve_tickers()`:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
