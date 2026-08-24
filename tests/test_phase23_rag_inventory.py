"""§−93 — новые отчёты инвестбанков: распознавание эмитента, честная подпись
источника и инвентарь базы.

Что здесь охраняется
--------------------
**Эмитент распознаётся ПО ПЕРЕЧНЮ, и перечень был неполон.** Jefferies в
`rag_engine._BANK_PATTERNS` отсутствовал, поэтому его отчёты ингестились с
``bank="Unknown"``, а «Unknown» отбрасывается И в заголовке выдачи
(`get_market_sentiment`), И в отборе «одна цитата на банк»
(`tg_bot._fetch_rag_context`) — цитата доезжала до отчёта БЕЗ автора, и
`ai_narrative` не засчитывал ссылку на банк.

**Подпись источника была ЛИТЕРАЛОМ.** `data_lineage` печатал «ChromaDB · GS /
MS / JPM PDF reports» независимо от содержимого базы: сколько бы отчётов Citi /
Jefferies / Barclays ни доехало, строка CoVe утверждала одно и то же. Подпись
теперь ВЫВОДИТСЯ из той же инвентаризации, что даёт отчёты/чанки, — так они не
могут разойтись.

**Инвентарь — из кода, а не из головы.** Числа «база: N отчётов · M чанков»
проверяются глазами по строке отчёта; скрипт `scripts/rag_inventory.py` считает
их и сверяет INBOX со store. Здесь пиньтся его чистые функции: ключ сверки
обязан совпадать с ключом `entrypoint._boot_ingest_from_inbox`, иначе скрипт
покажет «всё ингестировано» там, где бот думает иначе.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_SCRIPT = _ROOT / "scripts" / "rag_inventory.py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# tg_bot читает токен на ИМПОРТЕ — подставляем заглушку до первого импорта.
os.environ.setdefault("OMBRI_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")


def _load_inventory_module():
    """Скрипт живёт в `scripts/`, которого НЕТ в деплой-образе — отсутствие
    каталога обязано быть `skipTest`, иначе падает деплой-гейт."""
    if not _SCRIPT.exists():
        raise unittest.SkipTest("scripts/rag_inventory.py отсутствует (деплой-образ)")
    spec = importlib.util.spec_from_file_location("rag_inventory", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BankPatternTest(unittest.TestCase):
    """Эмитент из имени файла / обложки — по перечню `_BANK_PATTERNS`."""

    def test_new_issuers_are_recognised(self) -> None:
        from agent.rag_engine import FinancialRAG
        cases = [
            ("jefferies_us_equity_strategy_Q3_2026.pdf", "Jefferies"),
            ("jef_outlook_2026.pdf",                     "Jefferies"),
            ("citi_global_outlook_2026.pdf",             "Citi"),
            ("barclays_equity_gilt_2026.pdf",            "Barclays"),
            ("jpm_guide_to_markets_Q3_2026.pdf",         "JPMorgan"),
            ("goldman_sachs_outlook_Q3_2026.pdf",        "Goldman Sachs"),
        ]
        for fname, want in cases:
            self.assertEqual(FinancialRAG._extract_bank(fname, ""), want, fname)

    def test_jefferies_detected_from_cover_page(self) -> None:
        """Имя файла бывает безликим («report_03.pdf») — тогда решает обложка."""
        from agent.rag_engine import FinancialRAG
        self.assertEqual(
            FinancialRAG._extract_bank(
                "report_03.pdf", "Jefferies LLC | Equity Research | July 2026"),
            "Jefferies")

    def test_unknown_still_returned_for_unlisted_issuer(self) -> None:
        """«Unknown» — не баг, а честный ответ: добавлять эмитента нужно
        осознанно, в `_BANK_PATTERNS`, а не расширять regex до всего подряд."""
        from agent.rag_engine import FinancialRAG
        self.assertEqual(FinancialRAG._extract_bank("acme_capital_2026.pdf", ""),
                         "Unknown")

    def test_cloud_function_copy_is_identical(self) -> None:
        """`cloud_function/rag_engine.py` держится ИДЕНТИЧНЫМ оригиналу —
        иначе Cloud Function ингестит по СТАРОМУ перечню банков и метаданные
        зависят от того, каким путём приехал PDF."""
        src = _SRC / "agent" / "rag_engine.py"
        twin = _ROOT / "cloud_function" / "rag_engine.py"
        if not twin.exists():
            self.skipTest("cloud_function/ отсутствует (деплой-образ)")
        self.assertEqual(src.read_text(encoding="utf-8"),
                         twin.read_text(encoding="utf-8"))


class NarrativeBankCitationTest(unittest.TestCase):
    """Аудит цитирования обязан узнавать те же имена, что и ингест."""

    def test_jefferies_counts_as_bank_citation(self) -> None:
        from ai_narrative import _count_rag_citations
        out = _count_rag_citations(["Jefferies ожидает замедления в 2027 году."])
        self.assertIn("Jefferies", out["banks"])

    def test_bracket_tag_is_canonicalised(self) -> None:
        from ai_narrative import _count_rag_citations
        out = _count_rag_citations(["Циклические в фаворе [JEF]."])
        self.assertIn("Jefferies", out["banks"])


class KbBanksTest(unittest.TestCase):
    """Список эмитентов для подписи — из инвентаря, покрытием вниз."""

    def test_sorted_by_chunk_coverage(self) -> None:
        from tg_bot import _kb_banks
        docs = [
            {"bank": "Citi",          "chunks": 50},
            {"bank": "Goldman Sachs", "chunks": 300},
            {"bank": "Jefferies",     "chunks": 120},
        ]
        self.assertEqual(_kb_banks(docs), ["Goldman Sachs", "Jefferies", "Citi"])

    def test_chunks_are_summed_across_reports_of_one_bank(self) -> None:
        from tg_bot import _kb_banks
        docs = [{"bank": "Citi", "chunks": 100},
                {"bank": "Citi", "chunks": 100},
                {"bank": "Barclays", "chunks": 150}]
        self.assertEqual(_kb_banks(docs), ["Citi", "Barclays"])

    def test_unknown_and_blank_are_dropped(self) -> None:
        """«Unknown» — ОТСУТСТВИЕ эмитента; печатать его как источник нельзя."""
        from tg_bot import _kb_banks
        docs = [{"bank": "Unknown", "chunks": 900}, {"bank": "", "chunks": 5},
                {"bank": "Citi", "chunks": 10}]
        self.assertEqual(_kb_banks(docs), ["Citi"])

    def test_ties_are_stable(self) -> None:
        from tg_bot import _kb_banks
        docs = [{"bank": "UBS", "chunks": 10}, {"bank": "Citi", "chunks": 10}]
        self.assertEqual(_kb_banks(docs), ["Citi", "UBS"])   # по имени


class RagSourceLabelTest(unittest.TestCase):
    """Строка CoVe «Bank RAG (выдержки)» обязана называть РЕАЛЬНЫЕ банки."""

    def test_label_names_the_banks_in_the_store(self) -> None:
        from finance.data_lineage import _rag_source_label
        self.assertEqual(
            _rag_source_label(["Goldman Sachs", "JPMorgan", "Citi"]),
            "ChromaDB · Goldman Sachs / JPMorgan / Citi PDF reports")

    def test_long_roster_is_truncated_with_counter(self) -> None:
        from finance.data_lineage import _rag_source_label
        banks = ["Goldman Sachs", "JPMorgan", "Citi", "Barclays", "Jefferies",
                 "UBS", "HSBC"]
        label = _rag_source_label(banks)
        self.assertTrue(label.endswith("+2 PDF reports"), label)
        self.assertIn("Jefferies", label)
        self.assertNotIn("HSBC", label)

    def test_empty_roster_falls_back_to_historic_literal(self) -> None:
        """Сводки, снятые до §−93, поля `rag_kb_banks` не несут — строка
        обязана остаться прежней, а не превратиться в «ChromaDB ·  PDF»."""
        from finance.data_lineage import _rag_source_label
        for empty in (None, [], ["", "  "]):
            self.assertEqual(_rag_source_label(empty),
                             "ChromaDB · GS / MS / JPM PDF reports")

    def test_row_uses_the_dynamic_label(self) -> None:
        from finance.data_lineage import _rag_status
        row = _rag_status({"rag_status": "used", "rag_kb_docs": 38,
                           "rag_kb_chunks": 4102, "rag_context": "x\n\ny",
                           "rag_kb_banks": ["Jefferies", "Citi"]})
        self.assertEqual(row["source"], "ChromaDB · Jefferies / Citi PDF reports")
        self.assertIn("38 отчётов · 4102 чанков", row["note"])


class InventoryHelpersTest(unittest.TestCase):
    """Чистые функции скрипта инвентаризации."""

    def test_norm_name_matches_boot_ingest_key(self) -> None:
        """Ключ сверки обязан СОВПАДАТЬ с ключом `_boot_ingest_from_inbox`
        (basename → strip → lower).  Разойдутся — скрипт соврёт про то, что
        бот собирается доингестить."""
        inv = _load_inventory_module()
        for raw, want in [("reports/JPM_Q3_2026.PDF", "jpm_q3_2026.pdf"),
                          ("  gs_outlook.pdf  ", "gs_outlook.pdf"),
                          (None, "")]:
            self.assertEqual(inv.norm_name(raw), want)

    def test_bank_tally_counts_reports_and_chunks(self) -> None:
        inv = _load_inventory_module()
        docs = [{"bank": "Citi", "chunks": 10}, {"bank": "Citi", "chunks": 20},
                {"bank": "Jefferies", "chunks": 100}]
        self.assertEqual(inv.bank_tally(docs),
                         [("Jefferies", 1, 100), ("Citi", 2, 30)])

    def test_bank_tally_keeps_unknown_last(self) -> None:
        """В ДИАГНОСТИКЕ «Unknown» не прячется — это и есть находка."""
        inv = _load_inventory_module()
        docs = [{"bank": "Unknown", "chunks": 900}, {"bank": "Citi", "chunks": 1}]
        self.assertEqual([b for b, _, _ in inv.bank_tally(docs)],
                         ["Citi", "Unknown"])

    def test_diff_inbox_splits_present_missing_orphans(self) -> None:
        inv = _load_inventory_module()
        diff = inv.diff_inbox(
            ["JPM_Q3_2026.pdf", "jefferies_2026.pdf", "citi_2026.pdf"],
            ["jpm_q3_2026.pdf", "old_gs_2024.pdf"])
        self.assertEqual(diff["present"], ["JPM_Q3_2026.pdf"])
        self.assertEqual(diff["missing"], ["jefferies_2026.pdf", "citi_2026.pdf"])
        self.assertEqual(diff["orphans"], ["old_gs_2024.pdf"])

    def test_format_delta_reports_growth(self) -> None:
        inv = _load_inventory_module()
        self.assertEqual(inv.format_delta("чанков", 4102, 2984),
                         "чанков: 2984 → 4102 (+1118)")
        self.assertEqual(inv.format_delta("отчётов", 28, 28), "отчётов: 28 → 28 (±0)")
        self.assertEqual(inv.format_delta("чанков", 2900, 2984),
                         "чанков: 2984 → 2900 (-84)")
        self.assertEqual(inv.format_delta("чанков", 2984, None), "чанков: 2984")


if __name__ == "__main__":
    unittest.main()
