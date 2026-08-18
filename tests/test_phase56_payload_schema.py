"""Раунд §−97: R-6 — схема контракта payload.

`build_payload` отдаёт больше сотни ключей, и до сих пор их состав не был
записан нигде: единственным описанием контракта был сам код, который его
собирает. Прямое следствие — дефект `§−90` A-2, где ключ `kpis` жил в
контракте BASE и не рендерился ВООБЩЕ: никто не мог сказать, что вообще
входит в payload и кто это читает.

Три контрактные границы конвейера теперь пинятся все три:
    results{} (`finance/contracts.py`) → payload (`pdf_payload.PAYLOAD_CONTRACT`)
    → design-data (`test_phase19_block_audit`).

Пинится РОД значения, а не тип: на пустом прогоне половина ключей равна `None`,
и проверка «тип» здесь врала бы. Род отвечает на вопрос потребителя — придёт
текст, число, список или словарь.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdf_payload import DEEP_ONLY_KEYS, PAYLOAD_CONTRACT, build_payload  # noqa: E402

#: Что означает каждый род и какие типы Python ему соответствуют.
_KIND_TYPES: dict[str, tuple] = {
    "text": (str,),
    "num":  (int, float),
    "flag": (bool,),
    "list": (list,),
    "map":  (dict,),
}
#: Роды, которым разрешено быть `None` — «величина не посчитана» / «блок не
#: собран». Текст и список такого права не имеют: у них есть пустое значение.
_NULLABLE = {"num", "map"}


class PayloadContractIsDeclaredTest(unittest.TestCase):

    def test_deep_emits_exactly_the_declared_keys(self) -> None:
        got = set(build_payload({}, tier="deep"))
        declared = set(PAYLOAD_CONTRACT)
        self.assertEqual(
            sorted(got - declared), [],
            "маппер отдаёт ключи, которых нет в схеме — объяви их в "
            "`PAYLOAD_CONTRACT` вместе с потребителем")
        self.assertEqual(
            sorted(declared - got), [],
            "схема обещает ключи, которых маппер больше не отдаёт — "
            "потребитель получит пусто там, где ждёт данные")

    def test_base_emits_the_declared_subset(self) -> None:
        got = set(build_payload({}, tier="base"))
        expect = set(PAYLOAD_CONTRACT) - set(DEEP_ONLY_KEYS)
        self.assertEqual(sorted(got ^ expect), [],
                         "состав BASE разошёлся со схемой")

    def test_deep_only_keys_really_are_deep_only(self) -> None:
        """Список «только для DEEP» обязан быть ПРАВДОЙ, а не намерением."""
        base = set(build_payload({}, tier="base"))
        deep = set(build_payload({}, tier="deep"))
        self.assertEqual(set(DEEP_ONLY_KEYS), deep - base)

    def test_every_value_matches_its_declared_kind(self) -> None:
        for tier in ("base", "deep"):
            payload = build_payload({}, tier=tier)
            for key, value in payload.items():
                kind = PAYLOAD_CONTRACT[key]
                with self.subTest(tier=tier, key=key, kind=kind):
                    if value is None:
                        self.assertIn(
                            kind, _NULLABLE,
                            f"{key}: род «{kind}» не допускает None — у текста "
                            "и списка есть пустое значение, и потребитель "
                            "вправе не проверять на None")
                        continue
                    self.assertIsInstance(value, _KIND_TYPES[kind])

    def test_kinds_are_from_the_known_vocabulary(self) -> None:
        unknown = sorted({k for k in PAYLOAD_CONTRACT.values()} - set(_KIND_TYPES))
        self.assertEqual(unknown, [], f"неизвестный род значения: {unknown}")

    def test_schema_is_not_trivially_small(self) -> None:
        """Гейт бесполезен, если схема вдруг опустеет."""
        self.assertGreaterEqual(len(PAYLOAD_CONTRACT), 90)


class PayloadContractCoversWhatTheReportReadsTest(unittest.TestCase):
    """Схема не должна разъезжаться с тем, что читает слой отчёта."""

    def test_kpi_notes_and_series_are_part_of_the_contract(self) -> None:
        """Ключи карточки KPI — предмет договора, а не случайные поля.

        Именно на них строится полоса показателей обоих тиров (`§−97`), и
        именно они молча терялись в маппере BASE.
        """
        for key in ("ai_cvar_note", "ai_sharpe_note", "ai_mdd_note",
                    "ai_vol_note", "kpi_status"):
            with self.subTest(key=key):
                self.assertIn(key, PAYLOAD_CONTRACT)

    def test_no_key_is_declared_twice_with_different_kinds(self) -> None:
        """Словарь этого не допускает — проверяем ФАЙЛ на дубли строк."""
        src = (ROOT / "src" / "pdf_payload.py").read_text(encoding="utf-8")
        block = src.split("PAYLOAD_CONTRACT", 1)[1].split("}", 1)[0]
        names = [ln.split('"')[1] for ln in block.splitlines()
                 if ln.strip().startswith('"')]
        dupes = sorted({n for n in names if names.count(n) > 1})
        self.assertEqual(dupes, [], f"ключ объявлен дважды: {dupes}")


if __name__ == "__main__":
    unittest.main()
