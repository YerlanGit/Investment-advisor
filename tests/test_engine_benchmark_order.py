"""Порядок словаря бенчмарков: он НЕСУЩИЙ, и до §−64 его не стерёг никто.

Раунд: `AUDIT §−64` (ревизия фаз Арх-3.1…3.6 перед разрезом риск-модели).

Что здесь пинится и почему именно здесь
---------------------------------------
`analyze_all` кладёт выбранный пользователем бенчмарк в словарь ПЕРВЫМ ключом
(«Профильный бенчмарк»), и первый ключ доезжает до отчёта: `pdf_payload`
читает `next(iter(period_returns_table))` и по нему решает, чьим именем
подписать карточку «Рост против рынка». Собьётся порядок — все числа
останутся верными, а подпись станет чужой. Это тихий дефект: он не роняет
ничего и не меняет ни одной цифры.

Три причины, по которым до этого раунда порядок был не защищён:

1. **Golden-фикстура его не видит.** `golden_support.normalize` сортирует
   ключи словарей — намеренно, ради стабильности снимка. Значит перестановка
   бенчмарков оставляет эталон БИТ-В-БИТ прежним. Замерено: снимок с
   переставленным `period_returns_table` совпадает с исходным.
2. **Тесты потребителя проверяют не то.** `test_phase18_sprint5` и
   `test_phase31_benchmark_factor_propagation` подают в `pdf_payload`
   СИНТЕТИЧЕСКИЙ словарь, где профильный бенчмарк уже первый. Они проверяют,
   что потребитель правильно читает порядок, — но не что движок его создаёт.
3. **Комментарий в коде объяснял порядок НЕВЕРНО.** Он утверждал, что
   профильный бенчмарк первый «для корректного расчёта Tracking Error». С
   F-15 это не так: TE/IR считаются по КАЖДОМУ бенчмарку на своём pair
   inner-join, и порядок на них не влияет вовсе. Проверено мутацией: подмена
   первого бенчмарка на другой тикер оставляет эталон зелёным. Ложное
   обоснование опаснее отсутствующего — оно уводит от настоящей причины и
   создаёт впечатление, что порядок стережёт снимок.

Поэтому опора здесь — этот тест, а не эталон.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import golden_support as gs  # noqa: E402  (после правки sys.path)

#: Ключ-слот, под которым движок кладёт выбранный пользователем бенчмарк.
#: Именно его `pdf_payload` переименовывает в отображаемое имя.
PROFILE_SLOT = "Профильный бенчмарк"


class BenchmarkOrderTest(unittest.TestCase):
    """Движок ОБЯЗАН класть профильный бенчмарк первым ключом."""

    @classmethod
    def setUpClass(cls) -> None:
        # Один прогон на класс: `run_analyze_all` полностью офлайн и
        # детерминирован (то же окружение, что у golden-фикстуры), но стоит
        # несколько секунд — гонять его на каждый тест незачем.
        cls.results = gs.run_analyze_all("base")

    def test_profile_benchmark_is_first_in_comparison(self) -> None:
        bm = self.results["benchmark_comparison"]
        self.assertTrue(bm, "benchmark_comparison пуст — сравнивать нечего")
        self.assertEqual(
            next(iter(bm)), PROFILE_SLOT,
            "профильный бенчмарк обязан быть ПЕРВЫМ ключом: первый ключ "
            "доезжает до подписи карточки отчёта. Порядок сбит → числа верны, "
            "а подпись чужая. Эталон это НЕ поймает (normalize сортирует ключи).",
        )

    def test_profile_benchmark_is_first_in_period_table(self) -> None:
        prt = self.results["period_returns_table"]
        self.assertTrue(prt, "period_returns_table пуст")
        self.assertEqual(
            next(iter(prt)), PROFILE_SLOT,
            "именно этот словарь читает `pdf_payload` через next(iter(...)), "
            "чтобы назвать карточку «Рост против рынка»",
        )

    def test_tracking_error_is_per_benchmark_not_from_the_first(self) -> None:
        """TE у каждого бенчмарка СВОЙ — порядок на числа не влияет.

        Тест фиксирует ПРИЧИНУ, по которой порядок важен, чтобы ложное
        обоснование («первый нужен для Tracking Error») не вернулось: если бы
        TE считался от первого бенчмарка, у всех строк он был бы одинаковым.
        """
        bm = self.results["benchmark_comparison"]
        te = {name: row.get("Tracking_Error") for name, row in bm.items()}
        measured = {n: v for n, v in te.items() if v is not None}
        self.assertGreaterEqual(
            len(measured), 3, f"ожидались TE как минимум у трёх бенчмарков: {te}")
        self.assertGreater(
            len(set(measured.values())), 1,
            "все Tracking Error совпали — значит считаются НЕ по каждому "
            f"бенчмарку, и обоснование порядка пришлось бы пересматривать: {measured}",
        )

    def test_profile_slot_tracks_the_requested_ticker(self) -> None:
        """Слот профильного бенчмарка соответствует запрошенному тикеру.

        `run_analyze_all` запрашивает `SPY.US`, поэтому у слота и у «S&P 500»
        (тот же ETF) TE обязан совпасть — это и доказывает, что слот считается
        по СВОЕМУ тикеру, а не копирует соседа по позиции.
        """
        bm = self.results["benchmark_comparison"]
        self.assertIn(PROFILE_SLOT, bm)
        self.assertIn("S&P 500", bm)
        self.assertEqual(
            bm[PROFILE_SLOT]["Tracking_Error"], bm["S&P 500"]["Tracking_Error"],
            "профильный бенчмарк фикстуры — SPY.US, тот же ETF, что «S&P 500»; "
            "расхождение TE означало бы, что слот считается не по своему тикеру",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
