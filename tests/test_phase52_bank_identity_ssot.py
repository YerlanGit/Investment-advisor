"""§−95 — идентичность эмитента описана ОДИН раз, и это проверяется.

Что здесь охраняется
--------------------
**Копий было четыре, и они разошлись.** «Как зовут банк» жило в
`rag_engine._BANK_PATTERNS`, в `tg_bot._RAG_BANK_REMNANT_RE`, в трёх структурах
`ai_narrative` и литералами в шаблонах/бандлах. Добавление эмитента требовало
правки четырёх файлов, и НИ ОДИН тест их не связывал — расхождения копились
молча и были найдены только замером:

* `Citi` в прозе нарратива не засчитывался: `_BANK_NAME_RE` знал лишь
  `Citigroup|Citibank`, при том что ингест клал `bank="Citi"`, а `[Citi]` в
  теге засчитывался. Модель ссылалась на Citi — CoVe печатал «0 ссылок».
* `Merrill` не снимался с шапки выдержки, хотя нарратив его знал.
* **Пять эмитентов из одиннадцати** становились `Unknown` на собственной
  конвенции имён проекта (`wells_fargo_2026.pdf`, `ubs_outlook.pdf`): шаблоны
  писались через `\\s*`, а разделитель в именах файлов — `_`, который для `\\b`
  является СЛОВНЫМ символом.

Поэтому тесты ниже проверяют не «список правильный», а **согласованность
потребителей**: перечень выведен из реестра, поэтому новый эмитент доезжает до
всех четырёх мест сам. Тот же приём, что `ЧК-08.1` (список замера — из кода).

**Решения потребителей при этом РАЗНЫЕ и сливать их нельзя** (принцип
`asset_taxonomy`): ингест смотрит имя файла и обложку, нарратив — прозу,
чистильщик — шапку письма. Короткие формы (`MS`, `GS`) в прозе по-прежнему
запрещены (`§−14` C-8) — это тоже под тестом.

**Выдуманных фактов в поставляемом отчёте быть не может.** В BASE-шаблоне, в
блоке «ИСТОЧНИКИ», были напечатаны три НЕСУЩЕСТВУЮЩИХ названия отчётов
(«Q2 2026 Market Outlook» и др.) — ровно там, где читатель проверяет
происхождение данных.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("RAMP_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")


class RegistryIsTheSingleSourceTest(unittest.TestCase):
    """Реестр — один, и он не пуст."""

    def test_registry_covers_the_five_banks_in_use(self) -> None:
        from agent.rag_engine import BANK_ORDER
        for bank in ("JPMorgan", "Goldman Sachs", "Citi", "Jefferies", "Barclays"):
            self.assertIn(bank, BANK_ORDER)

    def test_no_consumer_keeps_its_own_issuer_list(self) -> None:
        """Литеральный перечень имён у потребителя = пятая копия. Потребители
        обязаны СТРОИТЬ свои regex из реестра."""
        for mod, must_build_from in (("ai_narrative.py", "bank_alias_regex"),
                                     ("tg_bot.py",       "bank_alias_regex")):
            src = (_SRC / mod).read_text(encoding="utf-8")
            # Комментарии описывают ПРОШЛЫЙ дефект и обязаны называть его
            # своим именем — сканируем только исполняемые строки.
            code = "\n".join(l for l in src.splitlines()
                             if not l.lstrip().startswith("#"))
            with self.subTest(module=mod):
                self.assertIn(must_build_from, code)
                # Признак старой копии: полное имя банка внутри строки-regex.
                self.assertNotIn(r"Wells\s+Fargo", code)
                self.assertNotIn("Citigroup|Citibank", code)

    def test_old_pattern_table_is_gone(self) -> None:
        src = (_SRC / "agent" / "rag_engine.py").read_text(encoding="utf-8")
        self.assertNotIn("_BANK_PATTERNS", src)

    def test_cloud_function_copy_is_identical(self) -> None:
        """Реестр живёт в rag_engine именно потому, что `cloud_function/`
        деплоится своим `--source` и `finance/` там нет. Копия обязана совпадать,
        иначе функция ингестит по СТАРОМУ перечню."""
        twin = _ROOT / "cloud_function" / "rag_engine.py"
        if not twin.exists():
            self.skipTest("cloud_function/ отсутствует (деплой-образ)")
        self.assertEqual((_SRC / "agent" / "rag_engine.py").read_text(encoding="utf-8"),
                         twin.read_text(encoding="utf-8"))


class FilenameConventionTest(unittest.TestCase):
    """Конвенция имён проекта — `<банк>_<что>_<период>.pdf` (так названы примеры
    в докстринге `rag_engine`). На ней обязаны опознаваться ВСЕ эмитенты."""

    def _slug(self, bank: str) -> str:
        return bank.lower().replace(" ", "_")

    def test_every_issuer_survives_underscore_separator(self) -> None:
        from agent.rag_engine import BANK_ORDER, FinancialRAG
        for bank in BANK_ORDER:
            fname = f"{self._slug(bank)}_outlook_2026.pdf"
            with self.subTest(bank=bank, file=fname):
                self.assertEqual(FinancialRAG._extract_bank(fname, ""), bank)

    def test_every_issuer_survives_space_and_dash(self) -> None:
        from agent.rag_engine import BANK_ORDER, FinancialRAG
        for bank in BANK_ORDER:
            for sep in (" ", "-", ""):
                fname = f"{bank.lower().replace(' ', sep)} outlook 2026.pdf"
                with self.subTest(bank=bank, sep=repr(sep)):
                    self.assertEqual(FinancialRAG._extract_bank(fname, ""), bank)

    def test_short_forms_work_in_filenames(self) -> None:
        from agent.rag_engine import BANK_SHORT, FinancialRAG
        for bank, shorts in BANK_SHORT.items():
            for short in shorts:
                with self.subTest(bank=bank, short=short):
                    self.assertEqual(
                        FinancialRAG._extract_bank(f"{short}_note_2026.pdf", ""), bank)

    def test_unlisted_issuer_is_still_unknown(self) -> None:
        """Реестр не должен матчить всё подряд — «Unknown» остаётся честным
        ответом и поводом завести эмитента осознанно."""
        from agent.rag_engine import FinancialRAG
        for fname in ("acme_capital_2026.pdf", "renaissance_note.pdf", "note.pdf"):
            self.assertEqual(FinancialRAG._extract_bank(fname, ""), "Unknown", fname)


class ConsumersAgreeTest(unittest.TestCase):
    """Связь потребителей — то, чего не было ни в одном тесте."""

    def test_ingested_bank_is_counted_by_the_narrative(self) -> None:
        """Ингест положил `bank=X` → прозаическое упоминание X обязано
        засчитаться. Именно здесь молча терялся Citi."""
        from agent.rag_engine import BANK_ALIASES, BANK_ORDER, FinancialRAG
        from ai_narrative import _count_rag_citations
        for bank in BANK_ORDER:
            fname = f"{bank.lower().replace(' ', '_')}_2026.pdf"
            ingested = FinancialRAG._extract_bank(fname, "")
            # Каждое ПОЛНОЕ написание должно давать тот же канон в нарративе.
            for alias in BANK_ALIASES[bank]:
                with self.subTest(bank=bank, alias=alias):
                    cited = _count_rag_citations([f"{alias} ожидает роста в 2027."])
                    self.assertIn(ingested, cited["banks"])

    def test_letterhead_of_every_issuer_is_stripped(self) -> None:
        """Шапка письма в начале выдержки снимается для ЛЮБОГО эмитента —
        раньше `Merrill` проезжал, потому что копия в tg_bot о нём не знала."""
        from agent.rag_engine import BANK_ALIASES, BANK_ORDER
        from tg_bot import _clean_rag_excerpt
        for bank in BANK_ORDER:
            for alias in BANK_ALIASES[bank]:
                text = f"{alias.title()} Equities remain attractive into year-end"
                with self.subTest(bank=bank, alias=alias):
                    self.assertTrue(_clean_rag_excerpt(text).startswith("Equities"))

    def test_bank_tails_are_stripped_too(self) -> None:
        """Разрыв шапки оставляет одинокий хвост двусловного имени."""
        from tg_bot import _clean_rag_excerpt
        for tail in ("Sachs", "Stanley", "Chase"):
            self.assertTrue(
                _clean_rag_excerpt(f"{tail} Equities remain attractive").startswith("Equities"),
                tail)

    def test_canonical_bank_round_trips_every_alias(self) -> None:
        from agent.rag_engine import BANK_ALIASES, BANK_ORDER, BANK_SHORT, canonical_bank
        for bank in BANK_ORDER:
            for alias in BANK_ALIASES[bank] + BANK_SHORT.get(bank, ()):
                with self.subTest(bank=bank, alias=alias):
                    self.assertEqual(canonical_bank(alias), bank)
            self.assertEqual(canonical_bank(bank), bank)

    def test_unknown_name_is_returned_as_is(self) -> None:
        """Незнакомого эмитента нельзя терять: он уже напечатан в отчёте."""
        from agent.rag_engine import canonical_bank
        self.assertEqual(canonical_bank("Renaissance Capital"), "Renaissance Capital")
        self.assertEqual(canonical_bank(""), "")


class ShortFormsStayOutOfProseTest(unittest.TestCase):
    """`§−14` C-8 — регресс-гард на самое дорогое послабление реестра."""

    def test_bare_short_forms_are_not_bank_mentions(self) -> None:
        from ai_narrative import _count_rag_citations
        for phrase in ("latency 5 MS observed", "GS выросли на 3%",
                       "результат BAC не важен"):
            self.assertEqual(_count_rag_citations([phrase])["banks"], [], phrase)

    def test_short_forms_count_inside_a_tag(self) -> None:
        from ai_narrative import _count_rag_citations
        for tag, want in (("[MS]", "Morgan Stanley"), ("[JPM]", "JPMorgan"),
                          ("[GS]", "Goldman Sachs"), ("[JEF]", "Jefferies"),
                          ("[Wells Fargo]", "Wells Fargo")):
            with self.subTest(tag=tag):
                self.assertIn(want, _count_rag_citations([f"Идея {tag}."])["banks"])

    def test_cover_page_short_form_does_not_name_a_bank(self) -> None:
        """Короткая форма годится в ИМЕНИ ФАЙЛА, но не на обложке."""
        from agent.rag_engine import FinancialRAG
        self.assertEqual(FinancialRAG._extract_bank("note.pdf", "latency 5 MS"), "Unknown")


class NoFabricatedSourcesTest(unittest.TestCase):
    """Поставляемый отчёт не называет банки и отчёты, которых нет в базе."""

    def test_basic_template_has_no_invented_report_titles(self) -> None:
        tpl = (_SRC / "templates" / "report_basic_v3.html").read_text(encoding="utf-8")
        for invented in ("Q2 2026 Market Outlook",
                         "Technology Sector Outlook 2026",
                         "Q2 2026 Strategy &amp; Allocation"):
            self.assertNotIn(invented, tpl, f"выдуманное название отчёта: {invented}")

    def test_templates_read_banks_from_data(self) -> None:
        for name in ("report_basic_v3.html", "report_deep_v3.html"):
            tpl = (_SRC / "templates" / name).read_text(encoding="utf-8")
            with self.subTest(template=name):
                self.assertIn("data.rag_banks", tpl)

    def test_shipped_bundles_have_no_hardcoded_issuers(self) -> None:
        """Пиним ПОСТАВЛЯЕМЫЙ артефакт, а не исходник `.jsx`."""
        for name in ("deep-components.js", "base-components.js"):
            js = (_SRC / "premium_assets" / name).read_text(encoding="utf-8")
            code = "\n".join(l for l in js.splitlines() if not l.lstrip().startswith("//"))
            with self.subTest(bundle=name):
                self.assertNotIn("GS / MS / JPM", code)
                self.assertIn("ragSource", code)

    def test_payload_carries_the_real_banks(self) -> None:
        from pdf_payload import build_payload  # noqa: F401  (import smoke)
        src = (_SRC / "pdf_payload.py").read_text(encoding="utf-8")
        self.assertIn('"rag_banks"', src)
        self.assertIn("rag_kb_banks", src)

    def test_design_data_meta_carries_the_real_banks(self) -> None:
        from premium_payload import build_design_data
        d = build_design_data({"assets": [], "rag_banks": ["Citi", "Jefferies"]}, "deep")
        self.assertEqual(d["meta"]["ragBanks"], ["Citi", "Jefferies"])

    def test_design_data_meta_defaults_to_empty(self) -> None:
        """Нет базы → пустой список, а бандл печатает просто «ChromaDB»."""
        from premium_payload import build_design_data
        d = build_design_data({"assets": []}, "base")
        self.assertEqual(d["meta"]["ragBanks"], [])

    def test_sample_fallbacks_do_not_reseed_the_literal(self) -> None:
        """`*-data.sample.json` — ФОЛБЭК рендера (`§−91` B-3), а не картинка:
        литерал в нём вернулся бы на экран."""
        p = _SRC / "premium_assets" / "deep-data.sample.json"
        if not p.exists():
            self.skipTest("sample отсутствует")
        self.assertNotIn("GS / MS / JPM", json.dumps(
            json.loads(p.read_text(encoding="utf-8")), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
