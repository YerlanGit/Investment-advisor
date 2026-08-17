"""Раунд §−92: паритет `manual` ↔ `freedom` НА СЛОЕ ОТЧЁТА + привязка карт к SSOT.

Владелец готовится к первому живому прогону ручного ввода и попросил
убедиться, что тот не конфликтует с уже работающими отчётами Freedom API.
Проверка показала, что существующий гейт паритета останавливается на полпути.

  C-1 🔴 **Паритет доказан на `results{}` и НЕ доказан на отчёте.**
      `test_manual_freedom_parity` сверяет 36 ключей контракта движка — то
      есть ЧИСЛА. Но требование владельца звучит про ОТЧЁТ, а отчёт рождается
      двумя слоями дальше: `pdf_payload.build_payload` →
      `premium_payload.build_design_data`. Именно там жили ВСЕ тринадцать
      дефектов раундов `§−90` и `§−91` (мёртвый `kpis`, литеральный статус,
      зелёная ложь валютного слоя). Совпадение чисел не гарантирует
      совпадения отчётов — это разные контракты.

  C-2 🔴 **Карта направления макро-драйверов ссылалась на несуществующий ряд.**
      `MACRO_DRIVER_DIRECTION` содержала ключ `hy_oas`, тогда как каталог
      `services.macro_data.MACRO_SERIES_CATALOG` зовёт ряд `hy_credit_spread`.
      Кредитный спред — один из шести драйверов — молча оставался нейтральным
      вместо окраски по направлению. Живой DEEP 15.08 это показывает:
      «US HY OAS ▼ −0.37 bp» имеет `trendTone: "flat"`, хотя сужение спреда
      это риск-он.
      🔴 Дефект пережил раунд `§−90` потому, что его тест сверял таблицу
      **саму с собой** (`assertEqual(M["hy_oas"], -1)`) и ни разу — с
      каталогом. Самосогласованный тест не ловит выдуманное имя.

  C-3 🔴 **Правка `§−90` A-6 сделала хуже, чем было.** Доля IT нормировалась
      на `Σ|w|` ПЕРЕДАННЫХ тикеров, а передаются только пережившие факторную
      регрессию. Бумага с короткой историей выпадала из ЗНАМЕНАТЕЛЯ, и доля
      росла. Живой DEEP 15.08: `SPCX` весом 6.17 % (сектор «Other», то есть
      даже не IT) вне модели → панель эффекта показала **59.6 %** против 56 %
      в предупреждении секторов. Расхождение 3.6 пп — ВТРОЕ больше того
      1.1 пп, ради которого правка делалась.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import premium_payload as pp
from finance.regime import MACRO_DRIVER_DIRECTION
from finance.simulate import _it_share


# ──────────────────────────────────────────────────────────────────────────
# C-2 · карта направлений связана с КАТАЛОГОМ, а не сама с собой
# ──────────────────────────────────────────────────────────────────────────

class MacroDirectionBoundToCatalogTest(unittest.TestCase):
    """🔴 Тест, который сверяет карту саму с собой, не ловит выдуманное имя.

    Ровно так `hy_oas` прожил раунд: `assertEqual(M["hy_oas"], -1)` проходил,
    потому что ключ был и в карте, и в проверке — а в каталоге его не было
    никогда. Здесь карта связывается с ЕДИНСТВЕННЫМ источником имён.
    """

    @staticmethod
    def _catalog_keys() -> set[str]:
        from services.macro_data import MACRO_SERIES_CATALOG
        return {spec.key for spec in MACRO_SERIES_CATALOG}

    def test_every_direction_key_exists_in_the_catalog(self) -> None:
        unknown = sorted(set(MACRO_DRIVER_DIRECTION) - self._catalog_keys())
        self.assertEqual(
            unknown, [],
            f"в карте направлений есть ряды, которых нет в каталоге FRED: "
            f"{unknown}. Такой ключ никогда не совпадёт с данными, и драйвер "
            f"молча останется нейтральным (так жил `hy_oas`).")

    def test_every_catalog_series_has_a_direction(self) -> None:
        """Обратная сторона: новый ряд без знака окрасится нейтрально и
        сделает это МОЛЧА — то есть выглядеть будет как «направления нет»."""
        missing = sorted(self._catalog_keys() - set(MACRO_DRIVER_DIRECTION))
        self.assertEqual(
            missing, [],
            f"ряды каталога без экономического направления: {missing}")

    def test_credit_spread_is_the_real_key(self) -> None:
        """Именно этот ряд был потерян — закрепляем поимённо."""
        self.assertIn("hy_credit_spread", MACRO_DRIVER_DIRECTION)
        self.assertEqual(MACRO_DRIVER_DIRECTION["hy_credit_spread"], -1)
        self.assertNotIn("hy_oas", MACRO_DRIVER_DIRECTION)


# ──────────────────────────────────────────────────────────────────────────
# C-3 · знаменатель доли IT — ВСЯ книга
# ──────────────────────────────────────────────────────────────────────────

class ItShareDenominatorIsTheWholeBookTest(unittest.TestCase):
    """Воспроизведение живого случая DEEP 15.08."""

    STRUCT = ["MSFT", "NVDA", "TLT", "GLD"]
    W = [0.1936, 0.1459, 0.11, 0.05]
    SECTORS = {"MSFT": "Technology", "NVDA": "Semiconductors",
               "TLT": "Bonds", "GLD": "Gold", "SPCX": "Other"}
    #: `SPCX` вне факторной модели (короткая история), кэш и маржинальный заём.
    OUTSIDE = {"SPCX": 0.0617, "USD": -0.0185, "KZT": 0.00005}

    def test_name_outside_the_model_stays_in_the_denominator(self) -> None:
        """`SPCX` вне модели — но в капитале. `KZT`/`USD` — кэш, вне капитала."""
        tech = 0.1936 + 0.1459
        book = sum(w for w in self.W if w > 0) + 0.0617   # без KZT: это кэш
        got = _it_share(self.STRUCT, self.W, self.SECTORS, self.OUTSIDE)
        self.assertAlmostEqual(got, tech / book, places=6)

    def test_positive_cash_is_excluded_like_the_sector_panel(self) -> None:
        """🔴 Третья итерация над одним числом — и она про кэш.

        Панель секторов пропускает денежные позиции (`NON_RISK_ASSETS`,
        правило `R-1`). Пока `_it_share` их учитывала, книга с ПОЛОЖИТЕЛЬНЫМ
        остатком расходилась с панелью: на фикстуре с кэшем 14.3 % панель
        показывала 36 %, а панель эффекта — 30.7 %.
        """
        struct = ["AAPL", "MSFT", "TLT", "USD"]
        w = [0.1710, 0.1355, 0.5502, 0.1433]
        sec = {"AAPL": "Technology", "MSFT": "Technology", "TLT": "Bonds"}
        got = _it_share(struct, w, sec)
        risk_book = 0.1710 + 0.1355 + 0.5502          # кэша здесь нет
        self.assertAlmostEqual(got, (0.1710 + 0.1355) / risk_book, places=6)
        self.assertGreater(got, 0.35, "кэш снова попал в знаменатель")

    def test_dropping_the_outside_book_inflates_the_share(self) -> None:
        """Это и был дефект: без книги знаменатель меньше — доля больше."""
        with_book = _it_share(self.STRUCT, self.W, self.SECTORS, self.OUTSIDE)
        without = _it_share(self.STRUCT, self.W, self.SECTORS)
        self.assertGreater(without, with_book)
        self.assertGreater(without - with_book, 0.05,   # живой разрыв ≈7 пп
                           "разрыв должен быть материальным, иначе тест "
                           "перестанет ловить регресс")

    def test_short_position_and_cash_never_enter_the_capital(self) -> None:
        """Маржинальный заём — финансирование, а не инвестированный капитал."""
        only_debt = _it_share(self.STRUCT, self.W, self.SECTORS, {"USD": -0.5})
        plain = _it_share(self.STRUCT, self.W, self.SECTORS)
        self.assertAlmostEqual(only_debt, plain, places=9)

    def test_it_name_outside_the_model_counts_in_both_parts(self) -> None:
        """Если вне модели оказалась IT-бумага, она обязана попасть и в
        числитель — иначе знаменатель вырос, а числитель нет, и доля упадёт."""
        got = _it_share(self.STRUCT, self.W, {**self.SECTORS, "AMD": "Technology"},
                        {"AMD": 0.10})
        tech = 0.1936 + 0.1459 + 0.10
        book = sum(w for w in self.W if w > 0) + 0.10
        self.assertAlmostEqual(got, tech / book, places=6)


# ──────────────────────────────────────────────────────────────────────────
# C-1 · паритет ДОХОДИТ ДО ОТЧЁТА, а не останавливается на числах
# ──────────────────────────────────────────────────────────────────────────

#: Ключи design-data, которым РАЗРЕШЕНО различаться между источниками,
#: и причина у каждого. Пусто — значит отчёт обязан совпадать целиком.
#:
#: `quality` и `cove` НЕ здесь сознательно: провенанс обязан называть
#: источник цен, но делает это ТЕКСТОМ ВНУТРИ строки, а состав строк и их
#: количество совпадать обязаны — иначе ручной отчёт потерял или приобрёл
#: целую проверку.
ALLOWED_DESIGN_DIFFS: dict[str, str] = {}


class ReportLayerParityTest(unittest.TestCase):
    """🔴 Требование владельца звучит про ОТЧЁТ, а не про числа движка.

    Существующий `test_manual_freedom_parity` доказывает совпадение
    `results{}` — 36 ключей. Но между `results` и тем, что видит читатель,
    лежат ДВА контракта: payload (`pdf_payload`) и design-data
    (`premium_payload`). Все тринадцать дефектов раундов `§−90`/`§−91` жили
    именно там и ни один из них не был бы виден на уровне `results`.

    Фикстура переиспользуется из паритетного теста ЦЕЛИКОМ: два прогона
    `analyze_all` на одних ценах, отличается только подпись провайдера. Так
    изолируется логика отчёта, а не разница источников.
    """

    @classmethod
    def setUpClass(cls) -> None:                            # noqa: N802
        from test_manual_freedom_parity import ManualFreedomParityTest as P

        cls._probe = P("test_price_matrix_itself_is_identical")
        cls._probe.setUp()
        # ВАЖНО: берём СЫРЫЕ results (не `gs.normalize`) — слою отчёта нужны
        # живые DataFrame'ы, а нормализация превращает их в словари.
        import contextlib
        from unittest import mock
        from finance.stooq_provider import StooqProvider

        class _SignedAsBroker(StooqProvider):
            name = "tradernet"

            def __init__(self, client=None) -> None:
                super().__init__()

        cls.manual_results = cls._probe._run("manual")
        cls.freedom_results = cls._probe._run(
            "freedom",
            extra=mock.patch("finance.price_providers.TradernetProvider",
                             _SignedAsBroker))

    @classmethod
    def tearDownClass(cls) -> None:                         # noqa: N802
        cls._probe.doCleanups()

    def _design(self, results: dict, tier: str) -> dict:
        from pdf_payload import build_payload
        return pp.build_design_data(build_payload(results, tier=tier), tier)

    def test_design_contract_matches_between_sources(self) -> None:
        for tier in ("base", "deep"):
            with self.subTest(tier=tier):
                m = self._design(self.manual_results, tier)
                f = self._design(self.freedom_results, tier)
                self.assertEqual(
                    set(m), set(f),
                    f"состав ключей отчёта {tier} разошёлся между источниками: "
                    f"только в ручном {sorted(set(m) - set(f))}, "
                    f"только в брокерском {sorted(set(f) - set(m))}")

    def test_quality_strip_has_the_same_number_of_checks(self) -> None:
        """Полоса качества обязана нести ОДИНАКОВОЕ число проверок.

        Различие в ТЕКСТЕ строки (источник цен) законно и обязательно (F-6);
        различие в КОЛИЧЕСТВЕ означает, что ручной отчёт потерял или
        приобрёл целую проверку.
        """
        for tier in ("base", "deep"):
            with self.subTest(tier=tier):
                m = self._design(self.manual_results, tier).get("quality") or []
                f = self._design(self.freedom_results, tier).get("quality") or []
                self.assertEqual(len(m), len(f))

    def test_risk_numbers_are_identical(self) -> None:
        """Числа риска не зависят от того, кто принёс цены."""
        for tier in ("base", "deep"):
            with self.subTest(tier=tier):
                m = self._design(self.manual_results, tier)
                f = self._design(self.freedom_results, tier)
                self.assertEqual(m["kpis"], f["kpis"])
                self.assertEqual(m["sectors"], f["sectors"])
                self.assertEqual(m["riskDecomp"], f["riskDecomp"])

    def test_it_share_agrees_with_the_sector_panel(self) -> None:
        """🔴 Прямая связка двух потребителей ОДНОЙ величины.

        Тесты выше проверяют формулу доли IT, а `test_contracts_golden` —
        её значение. Ни один из них не отвечает на вопрос, ради которого
        правка делалась: печатает ли отчёт ОДНО число в двух местах.
        Здесь берётся живой прогон и сверяется панель «Ожидаемый эффект»
        с предупреждением секторов — теми самыми 59.6 % против 56 %.
        """
        import re

        deep = self._design(self.manual_results, "deep")
        effect = {row["name"]: row for row in deep.get("effect") or []}
        it_row = effect.get("Доля IT")
        if not it_row or it_row.get("before") in (None, "", "–"):
            self.skipTest("на этой фикстуре доля IT не рассчитана")
        it_pct = float(str(it_row["before"]).replace("%", "").replace(",", "."))

        # Панель секторов считает от инвестированного капитала — тот же базис.
        tech = sum(s["pct"] for s in deep.get("sectors") or []
                   if re.search(r"tech|semicond|технолог|полупровод",
                                str(s.get("name", "")), re.I))
        if tech <= 0:
            self.skipTest("в фикстуре нет технологического сектора")
        self.assertLess(
            abs(it_pct - tech), 1.5,
            f"«Доля IT» в панели эффекта = {it_pct}%, а панель секторов даёт "
            f"{tech}% — одна величина напечатана двумя числами (§−90 A-6, "
            f"§−92 C-3). Разница обязана быть только округлением.")

    def test_source_is_still_named_in_the_provenance(self) -> None:
        """Обратная сторона паритета: отчёт ОБЯЗАН называть источник цен.

        Полное совпадение здесь было бы регрессом I-12/F-6 — читатель
        перестал бы видеть, откуда цены.
        """
        deep_m = self._design(self.manual_results, "deep")
        deep_f = self._design(self.freedom_results, "deep")
        cove_m = " ".join(str(c) for c in (deep_m.get("cove") or []))
        cove_f = " ".join(str(c) for c in (deep_f.get("cove") or []))
        self.assertNotEqual(cove_m, cove_f,
                            "провенанс обязан отличаться подписью источника")


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
