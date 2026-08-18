"""Раунд §−97: R-2 — гард на литералы дизайн-макета.

Класс дефекта повторился ПЯТЬ раз, и каждый раз его находил человек глазами:

  * статус KPI литералом в маппере — Sharpe 0.56 годами носил ярлык «good»
    (`§−90` A-1);
  * счётчик позиций константой при вычисляемом составе (`§−90` A-8);
  * счётчик стресс-сценариев константой при вычисляемом наборе (`§−91` B-4);
  * литеральная величина в промпте (`§−48`);
  * mock-дата из макета в проде (`R-2`, `§−40`).

Общее у всех пяти: ЧИСЛО, которое обязано печататься по данным, оказалось
записано словом. Глазами такое не ловится — число выглядит правдоподобно
именно потому, что когда-то было правдой.

🔴 **Гейт читает КОД, а не комментарии.** Babel сохраняет комментарии, и
разбор дефекта, написанный рядом с починкой, дважды за этот раунд валил
охраняющий её гейт. Комментарий до глаз читателя не доходит
(`tests/bundle_text.py`).

Константы МЕТОДА (определение CVaR, «сумма долей превышает 100%») числами по
данным не являются и живут в явном списке ниже — с обоснованием у каждой.
Список короткий намеренно: он растёт только осознанно.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from bundle_text import code_only                          # noqa: E402

_ASSETS = ROOT / "src" / "premium_assets"
_TEMPLATES = ROOT / "src" / "templates"

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_JS_STRING = re.compile(r"""(['"])((?:\\.|(?!\1).)*)\1""")

#: Счётчик, напечатанный словом: число вплотную к исчисляемой сущности.
_COUNTER = re.compile(
    r"\b\d+\s+(позици|сценари|иде[йяю]|отчёт|бумаг|банк|чанк|фактор|срез|"
    r"наблюдени|столп|проверк|актив)", re.IGNORECASE)
#: Величина с единицей внутри ПРОЗЫ (рядом кириллица) — обычно это результат
#: расчёта, случайно застывший в макете.
_NUM_UNIT = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|₸|п\.п\.)")
#: Конкретная дата — в шаблоне её быть не может, дата всегда из данных.
_DATE = re.compile(r"\b(?:20\d\d-\d\d-\d\d|\d{1,2}\.\d{2}\.20\d\d)\b")

#: Константы МЕТОДА: число описывает саму методику, а не результат прогона.
#: Ключ — точный фрагмент, значение — почему это не данные. Запись без
#: обоснования не принимается: иначе список станет свалкой исключений.
_METHOD_CONSTANTS: dict[str, str] = {
    "худшие 5% дней":
        "доверительный уровень CVaR 95% — параметр самой метрики, а не результат прогона",
    "сумма превышает 100%":
        "объяснение того, ПОЧЕМУ доли на маржинальном счёте не дают в сумме сто",
    "0% = долг не найден":
        "легенда шкалы: что означает нулевое значение показателя долговой нагрузки",
    "по 4 столпам":
        "4-Pillar — фиксированная структура модели (F+V+T+C), а не число из прогона",
}


def _prose_strings(js: str):
    """Строковые литералы КОДА, содержащие человеческий текст."""
    for m in _JS_STRING.finditer(js):
        s = m.group(2)
        if _CYRILLIC.search(s):
            yield s


def _offenders(text_source) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in text_source:
        if any(c in s for c in _METHOD_CONSTANTS):
            continue
        for name, rx in (("счётчик", _COUNTER),
                         ("величина с единицей", _NUM_UNIT),
                         ("дата", _DATE)):
            if rx.search(s):
                out.append((name, s[:110]))
    return out


class BundlesCarryNoMockupNumbersTest(unittest.TestCase):
    """Поставляемый артефакт — то, что реально едет в образ."""

    def _js(self, tier: str) -> str:
        return code_only((_ASSETS / f"{tier}-components.js").read_text(encoding="utf-8"))

    def test_no_literal_numbers_in_prose(self) -> None:
        for tier in ("base", "deep"):
            found = _offenders(_prose_strings(self._js(tier)))
            with self.subTest(tier=tier):
                self.assertEqual(
                    found, [],
                    f"в бандле {tier} число напечатано СЛОВОМ: {found}. "
                    "Счётчик и величина обязаны приходить из данных; если "
                    "число описывает саму МЕТОДИКУ — внеси его в "
                    "`_METHOD_CONSTANTS` с обоснованием.")

    def test_the_guard_catches_a_planted_literal(self) -> None:
        """🔴 Инструмент проверки сам обязан быть проверен (`§−68`).

        Пять раз подряд этот класс находил человек; гейт, который ничего не
        ловит, неотличим от его отсутствия.
        """
        planted = ('В портфеле 9 позиций',
                   'Прогон от 12.08.2026',
                   'Перевес технологий 8 %')
        for sample in planted:
            with self.subTest(sample=sample):
                self.assertTrue(_offenders([sample]),
                                f"гейт не увидел литерал: {sample!r}")

    def test_method_constants_stay_recognised(self) -> None:
        """Обратная сторона: обоснованная константа НЕ должна валить гейт."""
        for fragment in _METHOD_CONSTANTS:
            with self.subTest(fragment=fragment):
                self.assertEqual(
                    _offenders([f"Пояснение: {fragment} — так устроен показатель"]),
                    [])

    def test_every_allowance_carries_a_reason(self) -> None:
        for fragment, reason in _METHOD_CONSTANTS.items():
            with self.subTest(fragment=fragment):
                self.assertGreaterEqual(
                    len(reason), 40,
                    "исключение без внятного обоснования превращает список "
                    "в свалку — напиши, почему это константа МЕТОДА")

    def test_the_guard_actually_reads_something(self) -> None:
        """Пустой вход дал бы зелёный гейт ни за что."""
        for tier in ("base", "deep"):
            with self.subTest(tier=tier):
                self.assertGreater(len(list(_prose_strings(self._js(tier)))), 50)


class JinjaFallbackCarriesNoMockupNumbersTest(unittest.TestCase):
    """Фолбэк — тот же отчёт (`§−91` B-1), и правило для него то же."""

    def test_templates_have_no_literal_counters(self) -> None:
        for name in ("report_basic_v3.html", "report_deep_v3.html"):
            html = (_TEMPLATES / name).read_text(encoding="utf-8")
            # Комментарии Jinja и HTML до читателя тоже не доходят.
            body = re.sub(r"\{#.*?#\}", " ", html, flags=re.S)
            body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
            # Разрешения применяются ОДНИМ правилом на оба потребителя гейта:
            # иначе шаблон и бандл разъедутся в том, что считают константой
            # метода — а это ровно тот механизм, которым расходятся копии.
            found = sorted({
                m.group(0) for m in _COUNTER.finditer(body)
                if not any(c in body[max(0, m.start() - 60):m.end() + 60]
                           for c in _METHOD_CONSTANTS)})
            with self.subTest(tpl=name):
                self.assertEqual(
                    found, [],
                    f"{name}: счётчик напечатан константой {found}")


class MapperNeverInventsAToneTest(unittest.TestCase):
    """Статус карточки — ВЕРДИКТ движка; литерал в маппере запрещён."""

    def test_unknown_status_never_becomes_ok(self) -> None:
        import premium_payload as pp
        payload = {"cvar": "-3.4", "sharpe": "0.56", "max_drawdown": "-40.3",
                   "volatility": "20.5", "kpi_status": {}}
        for tier in ("base", "deep"):
            for card in pp.build_design_data(payload, tier)["kpis"]:
                with self.subTest(tier=tier, key=card["key"]):
                    self.assertNotEqual(card["status"], "ok")

    def test_tone_palette_lives_in_one_place(self) -> None:
        """Цвет следует за вердиктом через ОДНУ таблицу, а не по месту."""
        src = (ROOT / "src" / "premium_payload.py").read_text(encoding="utf-8")
        self.assertEqual(
            src.count("_KPI_TONE_COLOR = {"), 1,
            "палитра тона размножилась — копии разойдутся, как разошлись "
            "четыре копии имени эмитента (§−95)")


if __name__ == "__main__":
    unittest.main()
