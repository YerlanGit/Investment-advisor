"""Раунд §−97: качество чанков RAG и честность панели провенанса.

Владелец попросил проверить, «как он режет цитаты и как учитывает их в отчёте».
Замер на разметке банковского PDF и разбор живого DEEP 17.08 дали четыре
дефекта — три в нарезке, один в счётчике.

  🔴 **Рез шёл по символу.** Окно бралось как `sec[start:start+1200]`, поэтому
     три чанка из пяти начинались с середины предложения и столько же ими
     обрывались. Слой отчёта это чинил постфактум (`_clean_rag_excerpt` ищет
     начало предложения и отрезает огрызок) — то есть лечил следствие ценой
     потери текста.

  🔴 **Заголовки глубже третьего уровня не были точками разреза.** `#{1,3}` не
     видит `#### Sector view`; подраздел вливался в предыдущую секцию, и в
     подписи выдержки читатель видел ЧУЖОЕ имя секции.

  🔴 **Секции короче 150 символов выбрасывались целиком.** «### Rates /
     Duration risk is back.» не попадал в базу вообще — а у банковского отчёта
     короткая секция это обычно самый резкий тезис.

  🔴 **«Прочитано 97 отрывков» при шести запрошенных.** Счётчик считал непустые
     АБЗАЦЫ (`split("\\n\\n")`), а тела чанков — markdown из PDF с десятками
     пустых строк. Завышение в панели, где читатель проверяет отчёт.

Плюс пятый, на стыке промпта и аудита: нарратив написал «против совета GS», а
панель отчиталась об одном банке (Barclays). Короткой форме модель научил сам
промпт, где источники показаны как `[GS]/[JPM]`.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.rag_engine import FinancialRAG, count_snippets   # noqa: E402

_SENT = ("Equity valuations remain elevated relative to history, and we expect "
         "multiple compression to continue into the second half. ")


def _report_md() -> str:
    """Разметка, повторяющая структуру реального отчёта банка."""
    return (
        "# Global Outlook 2026\nPublished by the research desk.\n"
        "## Equities\n" + _SENT * 22 + "\n"
        "#### Sector view\n" + _SENT * 6 + "\n"
        "### Rates\nDuration risk is back.\n"
        "## Credit\n" + _SENT * 4 + "\n"
        "## Appendix\n\n"
        "## Page footer\n12\n"
    )


class ChunkBoundariesAreSentencesTest(unittest.TestCase):
    """Цитата не должна начинаться и обрываться на середине фразы."""

    def setUp(self) -> None:
        self.chunks = FinancialRAG._chunk_markdown(_report_md())

    def test_no_chunk_starts_mid_sentence(self) -> None:
        bad = [c[:60] for _, c in self.chunks if c[:1].islower()]
        self.assertEqual(
            bad, [],
            "чанк открывается серединой предложения — именно такие обрывки "
            "потом чинил `_clean_rag_excerpt`, теряя текст")

    def test_no_chunk_ends_mid_sentence(self) -> None:
        bad = [c[-60:] for _, c in self.chunks
               if not c.rstrip().endswith((".", "!", "?", ":", "…"))]
        self.assertEqual(bad, [], "чанк обрывается на середине предложения")

    def test_size_bound_is_still_respected(self) -> None:
        """Граница предложения не должна превращаться в безразмерный чанк."""
        self.assertTrue(all(len(c) <= 1200 for _, c in self.chunks))

    def test_long_section_is_still_sub_split(self) -> None:
        eq = [c for h, c in self.chunks if h == "Equities"]
        self.assertGreater(len(eq), 1, "длинная секция обязана делиться")

    def test_overlap_starts_at_a_boundary_too(self) -> None:
        """Перекрытие тоже режется по границе, иначе обрыв просто съезжает."""
        eq = [c for h, c in self.chunks if h == "Equities"]
        for piece in eq[1:]:
            self.assertFalse(piece[:1].islower())


class DeepHeadingsAreSectionsTest(unittest.TestCase):
    """Имя секции доезжает до подписи выдержки — оно обязано быть своим."""

    def test_level_four_heading_becomes_its_own_section(self) -> None:
        heads = {h for h, _ in FinancialRAG._chunk_markdown(_report_md())}
        self.assertIn("Sector view", heads,
                      "`#### …` не был точкой разреза, и подраздел уезжал "
                      "в соседнюю секцию с чужим именем")

    def test_all_heading_levels_split(self) -> None:
        md = "".join(f"{'#' * lvl} H{lvl}\nТекст секции уровня {lvl}, достаточно длинный.\n"
                     for lvl in range(1, 7))
        heads = {h for h, _ in FinancialRAG._chunk_markdown(md)}
        self.assertEqual(heads, {f"H{lvl}" for lvl in range(1, 7)})


class ShortSectionsSurviveTest(unittest.TestCase):
    """Порог отсеивает бессодержательное, а не короткое."""

    def test_short_prose_section_is_kept(self) -> None:
        chunks = FinancialRAG._chunk_markdown(_report_md())
        self.assertTrue(
            any("Duration risk is back" in c for _, c in chunks),
            "короткая секция выброшена целиком — у банковского отчёта это "
            "обычно самый резкий тезис")
        self.assertIn("Rates", {h for h, _ in chunks})

    def test_empty_and_numeric_sections_are_dropped(self) -> None:
        heads = {h for h, _ in FinancialRAG._chunk_markdown(_report_md())}
        self.assertNotIn("Appendix", heads, "голый заголовок без тела попал в базу")
        self.assertNotIn("Page footer", heads, "номер страницы попал в базу")

    def test_nothing_readable_is_lost(self) -> None:
        """Каждое предложение исходника обязано найтись хотя бы в одном чанке."""
        md = _report_md()
        joined = " ".join(c for _, c in FinancialRAG._chunk_markdown(md))
        for sentence in ("Published by the research desk",
                         "Duration risk is back",
                         "Equity valuations remain elevated"):
            with self.subTest(sentence=sentence):
                self.assertIn(sentence, joined)


class SnippetCounterIsHonestTest(unittest.TestCase):
    """🔴 «Прочитано 97 отрывков» при шести запрошенных."""

    CTX = (
        "\n--- [2026-01-01] gs_outlook.pdf — Goldman Sachs · Equities "
        "(актуальность: 90%, схожесть: 80%, итог: 85%) ---\n"
        "Первый абзац тела чанка.\n\nВторой абзац.\n\nТретий абзац.\n"
        "\n--- [2026-02-02] citi_note.pdf — Citi · Rates "
        "(актуальность: 91%, схожесть: 70%, итог: 80%) ---\n"
        "Тело второго чанка.\n\nЕщё абзац.\n"
    )

    def test_counts_snippets_not_paragraphs(self) -> None:
        self.assertEqual(count_snippets(self.CTX), 2)

    def test_the_old_formula_really_did_overcount(self) -> None:
        """Контрольный опыт: без него «2» не доказывает ничего."""
        paragraphs = sum(1 for p in self.CTX.split("\n\n") if p.strip())
        self.assertGreater(
            paragraphs, count_snippets(self.CTX),
            "если абзацев не больше, чем отрывков, тест не воспроизводит дефект")

    def test_empty_context_is_zero(self) -> None:
        for empty in ("", None):
            with self.subTest(value=empty):
                self.assertEqual(count_snippets(empty), 0)


class ProvenanceDoesNotInventMethodTest(unittest.TestCase):
    """Панель происхождения не описывает того, чего в коде нет."""

    CTX = (
        "\n--- [2026-01-01] gs.pdf — Goldman Sachs · Equities "
        "(актуальность: 90%, схожесть: 80%, итог: 85%) ---\nтело\n\nабзац\n"
    )

    def test_no_fabricated_threshold_reaches_the_reader(self) -> None:
        """`get_market_sentiment` не отсекает по косинусу ВООБЩЕ.

        Отбор — ре-ранжирование `sem*W_SEMANTIC + rec*W_RECENCY` и срез top-N;
        порога «≥0.72» в коде нет и не было ни дня. Проверяется СТРОКА,
        которую видит читатель, а не исходник: разбор дефекта, написанный в
        комментарии рядом с починкой, не является дефектом (`tests/bundle_text.py`).
        """
        from pdf_payload import _build_integrity_checks
        rows = _build_integrity_checks(
            {}, {"rag_status": "used", "rag_context": self.CTX,
                 "rag_kb_docs": 38, "rag_kb_chunks": 3881,
                 "verdict": "v", "bullets": ["b"]}, {}, {})
        rag = [r for r in rows if "RAG" in str(r.get("label", ""))]
        self.assertTrue(rag, "строка RAG исчезла из панели целостности")
        for row in rag:
            detail = str(row.get("detail", ""))
            with self.subTest(label=row.get("label")):
                self.assertNotRegex(
                    detail, r"cosine\s*[≥>]",
                    f"панель обещает порог отбора, которого нет в коде: {detail!r}")

    def test_reader_sees_the_real_snippet_count(self) -> None:
        """Счётчик в панели — тот же, что считает отрывки по заголовкам."""
        from pdf_payload import _build_integrity_checks
        rows = _build_integrity_checks(
            {}, {"rag_status": "used", "rag_context": self.CTX,
                 "rag_kb_docs": 38, "rag_kb_chunks": 3881,
                 "verdict": "v", "bullets": ["b"]}, {}, {})
        detail = " ".join(str(r.get("detail", "")) for r in rows if "RAG" in str(r.get("label", "")))
        self.assertIn("прочитано 1 отрывков", detail,
                      f"счётчик отрывков разошёлся с данными: {detail!r}")

    def test_ranking_string_follows_the_weights(self) -> None:
        from agent.rag_engine import W_RECENCY, W_SEMANTIC
        from finance.data_lineage import rag_ranking_method
        s = rag_ranking_method()
        self.assertIn(f"{W_SEMANTIC:g}", s)
        self.assertIn(f"{W_RECENCY:g}", s)


class BankShortFormsAreExpandedTest(unittest.TestCase):
    """Прозе — полное имя; сокращение живёт только в теге источника."""

    @staticmethod
    def _f(text: str, held=frozenset()):
        from ai_narrative import _expand_bank_short_forms
        return _expand_bank_short_forms(text, set(held))

    def test_prose_short_form_becomes_the_full_name(self) -> None:
        self.assertEqual(
            self._f("против совета GS делать акцент на качество"),
            "против совета Goldman Sachs делать акцент на качество")

    def test_tag_is_left_alone(self) -> None:
        """Тег — законная короткая форма (`§−14` C-8), трогать его нельзя."""
        self.assertEqual(self._f("вывод [GS][Barclays]"), "вывод [GS][Barclays]")

    def test_rag_file_citation_is_never_rewritten(self) -> None:
        """🔴 Самый дорогой краевой случай.

        `[RAG:gs_outlook_2026.pdf]` — ПРОВЕРЯЕМАЯ ссылка: CoVe сверяет имя
        файла с тем, что реально попало в контекст. Перепишем `gs` внутри
        имени файла — и ссылка перестанет находиться, то есть честная цитата
        превратится в неподтверждённую.
        """
        src = "ссылка [RAG:gs_outlook_2026.pdf] подтверждает вывод"
        self.assertEqual(self._f(src), src)

    def test_short_form_is_expanded_next_to_a_foreign_tag(self) -> None:
        """Соседний тег другого источника не должен глушить разворот."""
        self.assertEqual(
            self._f("против совета GS [Quant Engine]"),
            "против совета Goldman Sachs [Quant Engine]")

    def test_word_boundaries_are_respected(self) -> None:
        """`gs` внутри слова — не банк."""
        self.assertEqual(self._f("magsomething не трогать"), "magsomething не трогать")

    def test_held_ticker_is_never_rewritten(self) -> None:
        """У держателя бумаги `MS` это ПОЗИЦИЯ, а не Morgan Stanley."""
        self.assertEqual(self._f("MS выглядит дорого", {"MS"}), "MS выглядит дорого")

    def test_audit_now_counts_the_expanded_name(self) -> None:
        from ai_narrative import _count_rag_citations
        raw  = "против совета GS и по мнению Barclays"
        self.assertEqual(_count_rag_citations([raw])["bank_cites"], 1)
        fixed = self._f(raw)
        self.assertEqual(_count_rag_citations([fixed])["bank_cites"], 2,
                         "панель провенанса по-прежнему не видит второй банк")

    def test_prompt_states_the_rule_to_the_model(self) -> None:
        """Запрет, о котором не сказано исполнителю, неисполним (`§−90` A-5)."""
        src = (ROOT / "src" / "ai_narrative.py").read_text(encoding="utf-8")
        self.assertIn("ИМЯ БАНКА В ТЕКСТЕ", src)


class RagEngineCopiesStayIdenticalTest(unittest.TestCase):
    """`cloud_function/rag_engine.py` держится идентичным исходнику."""

    def test_copies_match(self) -> None:
        a = ROOT / "src" / "agent" / "rag_engine.py"
        b = ROOT / "cloud_function" / "rag_engine.py"
        if not b.exists():
            self.skipTest("cloud_function/ отсутствует (деплой-образ)")
        self.assertEqual(a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8"),
                         "копии RAG-движка разошлись — правку нарезки надо "
                         "нести в обе, иначе ингест и чтение начнут расходиться")


if __name__ == "__main__":
    unittest.main()
