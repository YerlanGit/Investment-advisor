"""Golden-фикстура `analyze_all`: снимок ВСЕГО результата, сверяемый побайтово.

Раунд: архитектурный трек `docs/ARCHITECTURE_FOR_AGENTS.md §4`, фаза **Арх-3.0**
(блокирующий пререквизит разреза `analyze_all`).

Зачем именно снимок, а не набор точечных проверок
-------------------------------------------------
Арх-3 режет функцию на 1 240 строк, у которой шесть каналов НЕВИДИМОГО
состояния: движок пишет `_last_psd_repair`, `_last_regression_nobs`,
`_last_ortho_betas`, `_last_sparse_dropped`, `_last_port_log_returns`,
`_last_fx_records`, а стадии читают их через `getattr` — обычный скан
атрибутов такую связь не находит. Ошибка области видимости при разрезе НЕ
даёт исключения: она тихо меняет число в отчёте. Существующие 1 255 тестов
проверяют инварианты и отдельные величины; они НЕ заметят, если, например,
Sharpe сдвинется на 3% из-за переставленной стадии.

Снимок ловит ровно это: любое расхождение хоть в одном из 35 ключей.

Как обновлять
-------------
Снимок обновляется ТОЛЬКО осознанно, когда изменение поведения намеренное:

    GOLDEN_UPDATE=1 PYTHONPATH=src python -m pytest tests/test_contracts_golden.py

🔴 В фазе Арх-3 обновлять НЕЛЬЗЯ: каждая её подзадача обязана быть
поведенчески пустой, поэтому расхождение = ошибка разреза, а не новый эталон.
«Обновил снимок, чтобы позеленело» — способ спрятать регресс.

Ограничения фикстуры v1 честно перечислены в докстринге `golden_support`.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import golden_support as gs  # noqa: E402  (после правки sys.path — иначе не найдётся)


class GoldenResultsTest(unittest.TestCase):
    """`analyze_all` на детерминированном входе даёт зафиксированный результат."""

    def test_snapshot_matches(self) -> None:
        produced = gs.fixture_json()

        if os.getenv("GOLDEN_UPDATE") == "1":
            gs.FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            gs.FIXTURE_PATH.write_text(produced, encoding="utf-8")
            self.skipTest(f"снимок перезаписан: {gs.FIXTURE_PATH}")

        self.assertTrue(
            gs.FIXTURE_PATH.exists(),
            f"нет эталона {gs.FIXTURE_PATH} — создай его: "
            "GOLDEN_UPDATE=1 PYTHONPATH=src python -m pytest tests/test_contracts_golden.py",
        )
        expected = gs.FIXTURE_PATH.read_text(encoding="utf-8")
        if produced == expected:
            return

        # Расхождение: показать ИМЕННО те ключи, которые разошлись, — иначе
        # разбор 42 KB диффа превращается в археологию.
        import json
        got, want = json.loads(produced), json.loads(expected)
        changed = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        self.fail(
            "результат analyze_all разошёлся с эталоном.\n"
            f"расходятся ключи ({len(changed)}): {changed}\n"
            f"sha256 получено={gs.fixture_sha256(produced)[:16]} "
            f"ожидалось={gs.fixture_sha256(expected)[:16]}\n"
            "🔴 В фазе Арх-3 это ОШИБКА РАЗРЕЗА, а не повод обновить снимок."
        )

    def test_snapshot_covers_full_contract(self) -> None:
        """Снимок обязан покрывать ВЕСЬ контракт, а не удобное подмножество."""
        import json

        from finance.contracts import RESULTS_KEYS

        snapshot = json.loads(gs.FIXTURE_PATH.read_text(encoding="utf-8"))
        missing = sorted(set(RESULTS_KEYS) - set(snapshot))
        self.assertFalse(missing, f"эталон не содержит ключей контракта: {missing}")

    def test_run_is_deterministic(self) -> None:
        """Два прогона подряд дают идентичный текст.

        Без этого снимок бесполезен: «расхождение» нельзя будет отличить от
        собственного шума прогона. Проверяется каждый раз, а не однократно при
        создании, потому что недетерминизм может ПОЯВИТЬСЯ (новый вызов
        времени, случайный сид, порядок словаря из внешней библиотеки).
        """
        self.assertEqual(gs.fixture_json(), gs.fixture_json())


class GoldenIsolationTest(unittest.TestCase):
    """Прогон фикстуры не должен зависеть от сети и не должен пачкать окружение."""

    def test_env_pins_are_restored(self) -> None:
        marker = "HISTORY_LOOKBACK_DAYS"
        before = os.environ.get(marker)
        gs.run_analyze_all()
        self.assertEqual(os.environ.get(marker), before,
                         "прогон обязан вернуть окружение в исходное состояние")

    def test_fixture_still_covers_its_branches(self) -> None:
        """Фикстура не выродилась: ветви, ради которых собран состав, живы.

        Ключ с пустым значением бесполезен как страховка — его ПОТЕРЮ при
        разрезе снимок не заметит. Поэтому состав книги подобран так, чтобы
        «слепых» ключей было минимум, и этот тест стережёт именно покрытие,
        а не конкретные числа (числа стережёт снимок).

        История вопроса: первая редакция мока CDS возвращала `None` вместо
        `dict` (контракт `make_lookup` — `Callable[[str], dict]`), и скоринг
        молча пропускал три бумаги из четырёх; вторая — импортировала
        «настоящую» функцию инсайдеров уже ИЗ ПРОПАТЧЕННОГО модуля и уходила
        в рекурсию, оставляя `smart_money` пустым. Оба раза снимок бы
        зафиксировал обеднённое состояние как эталон.
        """
        r = gs.run_analyze_all()
        # 8 строк книги: минус дубль, минус отброшенная → 6 оценённых бумаг.
        self.assertEqual(len(r["asset_scores"]), 6)
        self.assertEqual([m["ticker"] for m in r["merged_positions"]], ["AAPL.US"])
        self.assertEqual([d["ticker"] for d in r["dropped_rows"]], ["BADQ.US"])
        self.assertIn("FFSPC6.1028.AIX", r["proxy_substitutions"])
        self.assertIn("FFSPC6.1028.AIX", r["priced_at_cost"])
        self.assertEqual(r["model_uncovered"]["names"], ["YOUNG.US"])
        self.assertTrue(r["macro_drivers"], "макро-пак обязан быть непустым")
        self.assertTrue(r["smart_money"], "инсайдерский блок обязан быть непустым")
        self.assertTrue(r["black_litterman"], "BL обязан посчитаться")
        self.assertTrue(r["stress_scenarios"], "стресс-сценарии обязаны посчитаться")

    def test_blind_keys_do_not_grow(self) -> None:
        """Слепых (пустых) ключей — не больше известного минимума.

        Это метрика ЦЕННОСТИ снимка: чем больше пустых ключей, тем меньше он
        ловит. Замер на 2026-08-04: ровно один — `fx_converted_rows`
        (ограничение v1, задокументировано в `golden_support`).
        """
        import json

        snapshot = json.loads(gs.FIXTURE_PATH.read_text(encoding="utf-8"))

        def _empty(v: object) -> bool:
            return v in (None, {}, [], "") or (
                isinstance(v, dict) and len(v) <= 2 and not any(v.values()))

        blind = sorted(k for k, v in snapshot.items() if _empty(v))
        self.assertEqual(
            blind, ["fx_converted_rows"],
            "изменился набор «слепых» ключей эталона: "
            f"{blind}. Ключ, ставший пустым, перестал страховать разрез.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
