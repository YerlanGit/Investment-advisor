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
        for scenario in gs.SCENARIOS:
            with self.subTest(scenario=scenario):
                produced = gs.fixture_json(scenario=scenario)
                path = gs.fixture_path(scenario)

                if os.getenv("GOLDEN_UPDATE") == "1":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(produced, encoding="utf-8")
                    continue

                self.assertTrue(
                    path.exists(),
                    f"нет эталона {path} — создай его: GOLDEN_UPDATE=1 "
                    "PYTHONPATH=src python -m pytest tests/test_contracts_golden.py",
                )
                expected = path.read_text(encoding="utf-8")
                if produced == expected:
                    continue

                # Расхождение: показать ИМЕННО те ключи, которые разошлись, —
                # иначе разбор 50 KB диффа превращается в археологию.
                import json
                got, want = json.loads(produced), json.loads(expected)
                changed = sorted(k for k in set(got) | set(want)
                                 if got.get(k) != want.get(k))
                self.fail(
                    f"сценарий «{scenario}» разошёлся с эталоном.\n"
                    f"расходятся ключи ({len(changed)}): {changed}\n"
                    f"sha256 получено={gs.fixture_sha256(produced)[:16]} "
                    f"ожидалось={gs.fixture_sha256(expected)[:16]}\n"
                    "🔴 В фазе Арх-3 это ОШИБКА РАЗРЕЗА, а не повод обновить снимок."
                )
        if os.getenv("GOLDEN_UPDATE") == "1":
            self.skipTest("снимки перезаписаны")

    def test_snapshot_covers_full_contract(self) -> None:
        """Каждый снимок покрывает ВЕСЬ контракт, а не удобное подмножество."""
        import json

        from finance.contracts import RESULTS_KEYS

        for scenario in gs.SCENARIOS:
            with self.subTest(scenario=scenario):
                snapshot = json.loads(gs.fixture_path(scenario).read_text(encoding="utf-8"))
                missing = sorted(set(RESULTS_KEYS) - set(snapshot))
                self.assertFalse(missing, f"эталон не содержит ключей контракта: {missing}")

    def test_run_is_deterministic(self) -> None:
        """Два прогона подряд дают идентичный текст — для КАЖДОГО сценария.

        Без этого снимок бесполезен: «расхождение» нельзя будет отличить от
        собственного шума прогона. Проверяется каждый раз, а не однократно при
        создании, потому что недетерминизм может ПОЯВИТЬСЯ (новый вызов
        времени, случайный сид, порядок словаря из внешней библиотеки).
        """
        for scenario in gs.SCENARIOS:
            with self.subTest(scenario=scenario):
                self.assertEqual(gs.fixture_json(scenario=scenario),
                                 gs.fixture_json(scenario=scenario))


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

    def test_leveraged_scenario_covers_its_branches(self) -> None:
        """Вторая книга обязана реально включать плечо, ETP и валюту.

        Иначе она просто удваивает время прогона, ничего не добавляя.
        """
        r = gs.run_analyze_all("leveraged_fx")
        self.assertTrue(r["leverage_metrics"]["is_leveraged"],
                        "книга задумана с маржинальным долгом")
        self.assertLess(r["leverage_metrics"]["cash_weight"], -0.001)
        self.assertGreater(r["leverage_metrics"]["leverage_ratio"], 1.0)
        fx = r["fx_converted_rows"]
        self.assertTrue(fx, "строка в тенге обязана попасть в реестр конверсий")
        self.assertEqual({row["currency"] for row in fx}, {"KZT"})
        self.assertIn("CONL.US", r["asset_scores"], "плечевой ETP должен быть оценён")

    def test_no_key_is_blind_in_every_scenario(self) -> None:
        """Ни один ключ не должен быть пустым ВО ВСЕХ сценариях сразу.

        Это метрика ЦЕННОСТИ эталонов: ключ, пустой везде, не страхует разрез —
        его потерю снимок не заметит. На одном сценарии таких было 8, после
        добавления второй книги — **ни одного** (замер 2026-08-04).

        Пустой ключ в ОДНОМ сценарии — норма: книга без плеча не обязана
        заполнять `fx_converted_rows`, а книга с плечом — `merged_positions`.
        """
        import json

        def _empty(v: object) -> bool:
            return v in (None, {}, [], "") or (
                isinstance(v, dict) and len(v) <= 2 and not any(v.values()))

        blind_sets = []
        for scenario in gs.SCENARIOS:
            snap = json.loads(gs.fixture_path(scenario).read_text(encoding="utf-8"))
            blind_sets.append({k for k, v in snap.items() if _empty(v)})

        blind_everywhere = sorted(set.intersection(*blind_sets))
        self.assertFalse(
            blind_everywhere,
            f"ключи, пустые во ВСЕХ сценариях: {blind_everywhere}. "
            "Такой ключ не страхует разрез — расширь состав книги.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
