"""
FinancialRAG — RAG engine с взвешиванием по актуальности (Recency Scoring).

Логика ранжирования:
  Финальный_скор = (1 - distance) * W_sem + recency_score * W_rec
  где:
    distance      — ChromaDB косинусное расстояние (0 = идентично)
    recency_score — нормализованный возраст документа (0.0 = старый, 1.0 = сегодня)
    W_sem = 0.6   — вес семантической схожести
    W_rec = 0.4   — вес актуальности

Дата определяется автоматически:
  1. Из имени файла: goldman_sachs_outlook_Q1_2025.pdf → 2025
  2. Из содержимого PDF (первые 2 страницы): регэксп ищет "January 2025" / "2024-Q3"
  3. Если не нашли — используется дата модификации файла
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone

# chromadb is imported LAZILY inside __init__ (like pymupdf4llm below): the
# pure ingest helpers (_chunk_markdown / _extract_bank / _extract_tickers) and
# module import must not require the heavy vector-store dep — only actually
# opening a collection does.

# pymupdf4llm is needed ONLY for PDF ingestion (ingest_pdf).  The bot's hot
# path only QUERIES ChromaDB, so it is imported LAZILY inside ingest_pdf — a
# missing optional ingest dep must never crash module import and take the
# whole RAG query path down with it (the `No module named 'pymupdf4llm'`
# regression that broke RAG).  It is restored to requirements.txt as well;
# this lazy import is belt-and-suspenders.

logger = logging.getLogger("FinancialRAG")

# ── Ingestion-artifact source names ──────────────────────────────────────────
# Before 2026-07-05 the Cloud Function / boot-ingest passed a NamedTemporaryFile
# path to ingest_pdf without doc_metadata["filename"], so `source` became the
# tmp basename («tmpl3mmhrmf.pdf»).  Those garbage sources survive as duplicates
# of a report that was LATER re-ingested under its proper name → the «база: N
# отчётов» count over-counts (замечание R2#6: 48 shown vs. 29 real) and the
# fragments pollute retrieval.  Detect them so the inventory ignores them and a
# boot purge can remove them.  Pattern = Python's tempfile default: tmp + 6-16
# chars from [A-Za-z0-9_] + .pdf (nothing that looks like a real report name).
_TEMP_SOURCE_RE = re.compile(r"^tmp[A-Za-z0-9_]{5,16}\.pdf$", re.IGNORECASE)


def _is_temp_source(src) -> bool:
    """True for ingestion-artifact source names (tmp file basenames)."""
    return bool(_TEMP_SOURCE_RE.match(str(src or "").strip()))


# ── Идентичность банка — SSOT ФАКТОВ (§−95) ──────────────────────────────────
# До §−95 «как зовут эмитента» было описано ЧЕТЫРЕЖДЫ и по-разному: здесь,
# в `tg_bot._RAG_BANK_REMNANT_RE`, в `ai_narrative` (три структуры) и литералами
# в отчёте.  Добавление одного банка требовало правки четырёх файлов, и ни один
# тест их не связывал — расхождения копились молча:
#   • `Citi` в прозе нарратива не засчитывался (в regex было только
#     Citigroup|Citibank), хотя ингест клал `bank="Citi"`;
#   • `Merrill` не снимался со шапки выдержки, хотя нарратив его знал;
#   • ПЯТЬ эмитентов из одиннадцати становились `Unknown` на собственной
#     конвенции имён проекта (`wells_fargo_2026.pdf`): шаблоны писались через
#     `\s*`, а разделитель в именах файлов — `_`, и он для `\b` СЛОВНЫЙ символ,
#     то есть `\bubs\b` не матчит `ubs_outlook`.
#
# Здесь лежат ФАКТЫ — канон-имя и его написания.  РЕШЕНИЯ у потребителей
# остаются разными и сливать их нельзя (тот же принцип, что у
# `asset_taxonomy`): ингест смотрит имя файла, нарратив — прозу, чистильщик
# выдержки — шапку письма.  Поэтому написания разложены по УВЕРЕННОСТИ:
#   BANK_ALIASES — полные имена, однозначные в любом тексте;
#   BANK_SHORT   — короткие формы, однозначные только в имени файла / в теге;
#   BANK_TAILS   — «хвосты» двусловных имён, остающиеся от разрыва шапки PDF
#                  («…Morgan Stanley» → выдержка начинается со «Stanley …»).
#
# Реестр обязан жить ИМЕННО в этом модуле: `cloud_function/` деплоится
# с `--source ./cloud_function` и `finance/` там нет, а копия обязана быть
# идентичной оригиналу.  Общий L0-модуль потребовал бы ВТОРОЙ синхронной копии.
BANK_ALIASES: dict[str, tuple[str, ...]] = {
    "JPMorgan":        ("j.p. morgan", "jp morgan", "jpmorgan chase"),
    "Morgan Stanley":  ("morgan stanley",),
    "Goldman Sachs":   ("goldman sachs", "goldman"),
    "Bank of America": ("bank of america", "bofa", "merrill lynch", "merrill"),
    "Barclays":        ("barclays",),
    "UBS":             ("ubs",),
    "Citi":            ("citigroup", "citibank", "citi"),
    "Jefferies":       ("jefferies",),
    "Wells Fargo":     ("wells fargo",),
    "Deutsche Bank":   ("deutsche bank", "deutsche"),
    "HSBC":            ("hsbc",),
    # §−111: Amundi — крупнейший управляющий активами Европы. Формально не
    # инвестбанк, но реестр здесь про ИЗДАТЕЛЯ ИССЛЕДОВАНИЯ, а не про лицензию:
    # владелец грузит его аутлуки в ту же базу. До этой строки отчёт Amundi
    # читался и влиял на выводы, а в блоке «ИСТОЧНИКИ» не назывался — метка
    # уходила в `Unknown`, и `_kb_banks` её отбрасывала.
    "Amundi":          ("amundi",),
}

#: Короткие формы. В ПРОЗЕ не ищутся: «MS» — это Microsoft/миллисекунды
#: (`§−14` C-8), «GS» слишком общо. Годятся для имени файла и для тега `[JPM]`.
BANK_SHORT: dict[str, tuple[str, ...]] = {
    "JPMorgan":        ("jpm",),
    "Morgan Stanley":  ("ms",),
    "Goldman Sachs":   ("gs",),
    "Bank of America": ("bac",),
    "Jefferies":       ("jef",),
    "Wells Fargo":     ("wfc",),
}

#: Обрубки имён, остающиеся от разрыва шапки PDF — ТОЛЬКО для чистки выдержки.
#: «Morgan» здесь потому, что это общий обрубок и «J.P. Morgan», и «Morgan
#: Stanley»: в ингесте он был бы двусмысленным, в чистке шапки — нет.
BANK_TAILS: dict[str, tuple[str, ...]] = {
    "Goldman Sachs":   ("sachs",),
    "Morgan Stanley":  ("stanley", "morgan"),
    "JPMorgan":        ("chase",),
}

#: Порядок разбора: сначала длинные/составные имена, иначе «Morgan Stanley»
#: перехватил бы «J.P. Morgan». Питается порядком объявления BANK_ALIASES.
BANK_ORDER: tuple[str, ...] = tuple(BANK_ALIASES)

# Границы, для которых `_` и цифра — РАЗДЕЛИТЕЛИ, а не часть слова.  `\b` здесь
# не годится: в Python `_` — словный символ, поэтому `\bubs\b` не видит
# `ubs_outlook_2026.pdf` (дефект, из-за которого пять банков были `Unknown`).
_L, _R = r"(?<![a-z0-9])", r"(?![a-z0-9])"


def _alias_core(alias: str) -> str:
    """«wells fargo» → `wells[\\s_.\\-]*fargo` — одно написание, любой разделитель.

    Пробел в алиасе означает «здесь может стоять что угодно из разделителей или
    ничего»: так одна запись покрывает и `wells fargo`, и `wells_fargo`, и
    `WellsFargo`, и `j.p. morgan` ⇄ `jpmorgan`."""
    words = [re.escape(w) for w in re.split(r"[\s.]+", alias.strip()) if w]
    return r"[\s_.\-]*".join(words)


def _alternation(aliases) -> str:
    """Альтернация, отсортированная ДЛИННЫМИ ВПЕРЁД.

    Питоновская альтернация берёт ПЕРВОЕ совпадение, а не самое длинное: при
    порядке «jp morgan | jpmorgan chase» шапка «JPMorgan Chase Equities…»
    срезалась до «Chase Equities…» — то есть чистка выдержки оставляла обрубок
    имени ровно того вида, который она и должна была убрать."""
    ordered = sorted(set(aliases), key=lambda a: (-len(a), a))
    return "|".join(_alias_core(a) for a in ordered)


def bank_alias_regex(bank: str) -> str:
    """Regex полных имён банка — безопасен в любом тексте (проза, обложка)."""
    alts = _alternation(BANK_ALIASES.get(bank, ()))
    return f"{_L}(?:{alts}){_R}" if alts else ""


def bank_short_regex(bank: str) -> str:
    """Regex коротких форм — применять ТОЛЬКО к имени файла или тегу."""
    alts = _alternation(BANK_SHORT.get(bank, ()))
    return f"{_L}(?:{alts}){_R}" if alts else ""


def bank_tail_regex(bank: str) -> str:
    """Regex «хвостов» имени — только для чистки шапки письма в выдержке."""
    alts = _alternation(BANK_TAILS.get(bank, ()))
    return f"{_L}(?:{alts}){_R}" if alts else ""


def canonical_bank(raw) -> str:
    """Любое написание → канон-имя; неизвестное возвращается КАК ЕСТЬ.

    Возврат «как есть» намеренный: неизвестный эмитент — это повод завести его
    в реестре, а не потерять имя, которое уже напечатано в отчёте."""
    text = re.sub(r"\s+", " ", str(raw or "").strip()).lower()
    if not text:
        return ""
    for bank in BANK_ORDER:
        if text == bank.lower():
            return bank
        if any(text == a for a in BANK_ALIASES[bank]):
            return bank
        if any(text == s for s in BANK_SHORT.get(bank, ())):
            return bank
    return str(raw or "").strip()


#: Конец предложения: точка/вопрос/восклицание/многоточие, за которым идёт
#: пробел, закрывающая кавычка/скобка или конец текста.  Сокращения вроде
#: «U.S.» ловятся как конец предложения — это ОСОЗНАННО: лишний рез по
#: границе слова безвреден, а пропущенный рез оставляет обрыв в середине.
_SENT_END_RE = re.compile(r"[.!?…](?=[\s\"»)\]]|$)")


#: Заголовок ОДНОГО извлечённого отрывка в собранном контексте.  Он же —
#: единственный признак, по которому отрывки можно пересчитать: тело чанка
#: содержит произвольный markdown с пустыми строками, поэтому «абзац» и
#: «отрывок» — РАЗНЫЕ вещи.
_SNIPPET_HEADER_RE = re.compile(r"^--- \[.*?\] .*? ---$", re.M)


def count_snippets(context: str) -> int:
    """Сколько ОТРЫВКОВ реально прочитано в этом контексте.

    🔴 Считалось иначе и врало в разы. Прежняя формула — «непустые куски после
    split("\n\n")» — опиралась на то, что `_fetch_rag_context` склеивает свои
    ДВА раздела через пустую строку. Но тела чанков это markdown из PDF, и
    пустых строк в них десятки: живой DEEP 17.08 напечатал «прочитано 97
    отрывков», тогда как запрошено было не больше шести (macro 3 + micro 3).

    Место у этой функции ровно здесь: заголовок отрывка пишет
    `get_market_sentiment`, и правило «что считается отрывком» обязано жить
    рядом с тем, кто этот заголовок печатает, а не у двух разных потребителей.
    Панель провенанса — то место, где читатель ПРОВЕРЯЕТ отчёт; завышенный
    счётчик именно там дороже любой другой неточности.
    """
    return len(_SNIPPET_HEADER_RE.findall(str(context or "")))


# ── Recency weights ─────────────────────────────────────────────────────────
W_SEMANTIC = 0.60   # вес семантической близости к запросу
W_RECENCY  = 0.40   # вес свежести документа

# Сколько результатов запрашивать у ChromaDB перед ре-ранжированием
# (берём больше чем нужно, затем отбираем лучшие по combined score)
PREFETCH_MULTIPLIER = 5


class FinancialRAG:
    def __init__(self, db_path: str = "data/chroma_db"):
        import chromadb                                    # lazy (see module top)
        from chromadb.utils import embedding_functions
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.embedding_fn  = embedding_functions.DefaultEmbeddingFunction()
        self.collection    = self.chroma_client.get_or_create_collection(
            name="bank_reports",
            embedding_function=self.embedding_fn,
        )

    # ── Date extraction ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_date_from_filename(filename: str) -> datetime | None:
        """
        Пытается вытащить год (и опционально месяц) из имени файла.
        Примеры: goldman_Q1_2025.pdf → 2025-01-01
                 jpmorgan_outlook_march_2024.pdf → 2024-03-01
        """
        # Паттерн: 4-значный год
        year_match = re.search(r'(?<!\d)(20\d{2})(?!\d)', filename)
        if not year_match:
            return None
        year = int(year_match.group(1))

        # Паттерн: квартал Q1-Q4
        q_match = re.search(r'[Qq]([1-4])', filename)
        if q_match:
            month = (int(q_match.group(1)) - 1) * 3 + 1
            return datetime(year, month, 1, tzinfo=timezone.utc)

        # Паттерн: название месяца
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        }
        lower = filename.lower()
        for abbr, num in month_map.items():
            if abbr in lower:
                return datetime(year, num, 1, tzinfo=timezone.utc)

        return datetime(year, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _extract_date_from_text(md_text: str) -> datetime | None:
        """
        Ищет даты в первых ~3000 символах PDF (обложка/оглавление).
        Паттерны: 'January 2025', 'Q1 2024', 'March 2024', '2025-03'
        """
        sample = md_text[:3000]

        # 'Month YYYY' или 'Month, YYYY'
        month_names = (
            r'(January|February|March|April|May|June|July|August|'
            r'September|October|November|December)'
        )
        m = re.search(rf'{month_names}[,\s]+(\d{{4}})', sample, re.IGNORECASE)
        if m:
            month_str = m.group(1)[:3].capitalize()
            year      = int(m.group(2))
            month_num = datetime.strptime(month_str, '%b').month
            return datetime(year, month_num, 1, tzinfo=timezone.utc)

        # 'Q1 2024'
        m = re.search(r'Q([1-4])[,\s]+(\d{4})', sample, re.IGNORECASE)
        if m:
            month = (int(m.group(1)) - 1) * 3 + 1
            return datetime(int(m.group(2)), month, 1, tzinfo=timezone.utc)

        # 'YYYY-MM' ISO
        m = re.search(r'\b(20\d{2})-(0[1-9]|1[0-2])\b', sample)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), 1, tzinfo=timezone.utc)

        return None

    def _get_doc_date(self, file_path: str, md_text: str,
                      filename: str | None = None) -> tuple[datetime, str]:
        """
        Returns (datetime, method_used).
        Priority: filename → PDF text → file modification time.

        ``filename`` (2026-07-05): the LOGICAL document name when file_path is
        a NamedTemporaryFile download — otherwise the year/quarter in the real
        name is invisible and recency falls back to tmp-file mtime (≈ «сегодня»
        для любого старого отчёта).
        """
        filename = filename or os.path.basename(file_path)

        dt = self._extract_date_from_filename(filename)
        if dt:
            return dt, "filename"

        dt = self._extract_date_from_text(md_text)
        if dt:
            return dt, "pdf_content"

        # Fallback: дата последнего изменения файла
        mtime = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc), "file_mtime"

    # ── Bank / ticker extraction + section-aware chunking (RAG #в) ────────────

    @classmethod
    def _extract_bank(cls, filename: str, md_text: str) -> str:
        """Canonical issuing bank from the filename first, then the cover page.

        Полные имена ищутся и в имени файла, и на обложке; КОРОТКИЕ формы
        (`gs`, `ms`, `jpm`, `jef`…) — ТОЛЬКО в имени файла. Причина — `§−14`
        C-8: «MS» в тексте это обычно Microsoft или миллисекунды, а вот
        `ms_strategy_2026.pdf` двусмысленным не бывает.
        """
        fname = str(filename or "").lower()
        cover = str(md_text or "")[:1500].lower()

        # 🔴 §−112. ИМЯ ФАЙЛА РЕШАЕТ ПЕРВЫМ, и только потом обложка.
        #
        # Прежняя редакция склеивала имя и обложку в одну строку и возвращала
        # первый банк по ПОРЯДКУ ОБЪЯВЛЕНИЯ. Порядок — не свидетельство: он
        # заведён против перехвата «J.P. Morgan» «Morgan Stanley», а решал
        # авторство. Любое упоминание конкурента на обложке перебивало имя
        # файла, если тот банк стоит в реестре раньше.
        #
        # Замер на живой базе 26.08: из 38 отчётов 25 приписаны Goldman Sachs,
        # и минимум 12 из них — чужие (Barclays, HSBC, JPMorgan, Jefferies).
        # У Jefferies при этом НОЛЬ документов при наличии его отчёта. Это
        # хуже `Unknown`: отчёт НАЗЫВАЕТ источником банк, который цитируемое
        # исследование не писал.
        #
        # Имя файла — конвенция оператора, самое надёжное свидетельство.
        # Обложка — запасной вариант, и там побеждает НАИБОЛЕЕ РАННЕЕ
        # упоминание: издатель называет себя первым, а конкурента цитирует
        # ниже. Порядок объявления остаётся тай-брейком при равной позиции —
        # ради него он и заведён.
        found = cls._first_named(fname, include_short=True)
        if found:
            return found
        return cls._first_named(cover, include_short=False) or "Unknown"

    @classmethod
    def _first_named(cls, text: str, *, include_short: bool) -> str:
        """Банк, упомянутый в `text` РАНЬШЕ прочих. `""` — ни одного.

        `include_short` — короткие формы (`gs`, `ms`, `jpm`) ищутся ТОЛЬКО в
        имени файла: «MS» в прозе это Microsoft или миллисекунды (`§−14` C-8).
        """
        if not text:
            return ""
        best: tuple[int, int, str] | None = None
        for order, bank in enumerate(BANK_ORDER):
            spots = []
            m = re.search(bank_alias_regex(bank), text)
            if m:
                spots.append(m.start())
            if include_short:
                short = bank_short_regex(bank)
                if short:
                    m = re.search(short, text)
                    if m:
                        spots.append(m.start())
            if not spots:
                continue
            cand = (min(spots), order, bank)
            if best is None or cand[:2] < best[:2]:
                best = cand
        return best[2] if best else ""

    @staticmethod
    def _extract_tickers(text: str) -> str:
        """Comma-joined uppercase tickers mentioned in a chunk (for filtering).

        ChromaDB metadata must be scalar, so we store a comma-wrapped string
        (",AAPL,MSFT,") that a `$contains` document filter or a substring check
        can match precisely without partial hits."""
        # $NVDA or bare 2–5-letter all-caps tokens; drop common English words.
        cand = set(re.findall(r"\$?([A-Z]{2,5})\b", text))
        _STOP = {"THE","AND","FOR","USD","EPS","GDP","CEO","CFO","ETF","USA",
                 "Q1","Q2","Q3","Q4","YOY","EBIT","FED","ECB","API","PDF","AI"}
        tickers = sorted(t for t in cand if t not in _STOP)[:25]
        return ("," + ",".join(tickers) + ",") if tickers else ""

    @staticmethod
    def _split_point(text: str, limit: int, *, min_frac: float = 0.55) -> int:
        """Где резать окно: индекс конца куска (эксклюзивно), не дальше `limit`.

        Приоритет границ — от самой крупной к самой мелкой: абзац → конец
        предложения → перевод строки → пробел → жёсткий рез. Граница ближе
        `min_frac * limit` не берётся: иначе один ранний перенос строки резал бы
        окно вдвое и чанки мельчали бы без пользы.
        """
        if len(text) <= limit:
            return len(text)
        window = text[:limit]
        floor  = int(limit * min_frac)

        p = window.rfind("\n\n")
        if p >= floor:
            return p + 2
        best = -1
        for m in _SENT_END_RE.finditer(window):
            if m.end() >= floor:
                best = m.end()
        if best > 0:
            return best
        for sep, off in (("\n", 1), (" ", 1)):
            p = window.rfind(sep)
            if p >= floor:
                return p + off
        return limit                      # последнее средство

    @staticmethod
    def _has_prose(text: str) -> bool:
        """Есть ли в куске хоть что-то читаемое, кроме разметки и цифр."""
        return bool(re.search(r"[A-Za-zА-Яа-яЁё]{2,}", text or ""))

    @staticmethod
    def _chunk_markdown(md_text: str, *, max_chars: int = 1200,
                        overlap: int = 150, min_chars: int = 12) -> list[tuple[str, str]]:
        """Section-aware, size-bounded chunks → list of (heading, chunk_text).

        Режет по заголовкам Markdown (заголовок уходит в метаданные секции),
        затем длинные секции — на перекрывающиеся окна ПО ГРАНИЦАМ ТЕКСТА.

        🔴 Три замера 2026-08-18 на разметке банковского PDF (`§−97`):

        1. **Рез шёл по символу.** Окно бралось как `sec[start:start+1200]`,
           поэтому три чанка из пяти начинались с середины предложения
           («history, and we expect…») и столько же им обрывались. Именно это
           чинил постфактум `_clean_rag_excerpt` в слое отчёта: он ищет начало
           предложения и отрезает огрызок — то есть лечил следствие, теряя
           текст, вместо того чтобы резать по границе сразу.
        2. **Заголовки глубже третьего уровня не были точками разреза.**
           `#{1,3}` не видит `#### Sector view`, и подраздел вливался в
           предыдущую секцию с ЧУЖИМ именем — а имя секции доезжает до отчёта
           в подписи выдержки.
        3. **Секции короче 150 символов молча выбрасывались ЦЕЛИКОМ.**
           «### Rates / Duration risk is back.» (тело 22 символа) не попадал
           в базу вообще: у банковских отчётов короткая секция это обычно самый
           резкий тезис. Порог теперь отсеивает НЕ короткое, а бессодержательное:
           голый заголовок, номер страницы, «Source: …» без текста под ним.
           Смещение осознанное — мусорный чанк проигрывает настоящему абзацу по
           семантической близости, а выброшенный тезис не вернуть ничем.
        """
        sections = re.split(r'\n(?=#{1,6}\s)', md_text)
        out: list[tuple[str, str]] = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            first_nl = sec.find("\n")
            heading = (sec[:first_nl] if first_nl > 0 else sec)[:120].lstrip("# ").strip()
            body    = sec[first_nl + 1:].strip() if first_nl > 0 else ""
            # Голый заголовок без тела — оглавление или подпись к таблице.
            if len(body) < min_chars or not FinancialRAG._has_prose(body):
                continue
            if len(sec) <= max_chars:
                out.append((heading, sec))
                continue
            start = 0
            while start < len(sec):
                cut   = start + FinancialRAG._split_point(sec[start:], max_chars)
                piece = sec[start:cut].strip()
                if piece and FinancialRAG._has_prose(piece):
                    out.append((heading, piece))
                if cut >= len(sec):
                    break
                # Перекрытие тоже начинается с границы: иначе следующий чанк
                # снова открывался бы серединой слова.
                back = max(start + 1, cut - overlap)
                nxt  = FinancialRAG._sentence_start(sec, back, cut)
                start = nxt if nxt > start else cut
        return out

    @staticmethod
    def _sentence_start(text: str, lo: int, hi: int) -> int:
        """Первое начало предложения в `[lo, hi)`; иначе граница слова у `lo`."""
        m = _SENT_END_RE.search(text, lo, hi)
        while m is not None:
            j = m.end()
            while j < hi and text[j] in " \t\n\"»)]":
                j += 1
            if j < hi:
                return j
            m = _SENT_END_RE.search(text, m.end(), hi)
        sp = text.rfind(" ", lo, hi)
        return sp + 1 if sp > lo else lo

    # ── Recency scoring ──────────────────────────────────────────────────────

    @staticmethod
    def _recency_score(doc_timestamp: int, half_life_days: int = 365) -> float:
        """
        Экспоненциальный decay: score = exp(-age_days / half_life_days).
        doc_timestamp — Unix epoch seconds (хранится в метаданных).
        Документ сегодняшнего дня → score ~1.0
        Документ год назад → score ~0.37 (1/e)
        Документ 3 года назад → score ~0.08
        """
        now_ts   = time.time()
        age_secs = max(0, now_ts - doc_timestamp)
        age_days = age_secs / 86_400
        import math
        return math.exp(-age_days / half_life_days)

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_pdf(self, file_path: str, doc_metadata: dict | None = None) -> int:
        """
        Reads a PDF, auto-detects its date, and stores all chunks with
        a ``doc_timestamp`` (Unix epoch) in metadata for recency scoring.
        Returns number of chunks ingested.
        """
        if doc_metadata is None:
            doc_metadata = {}

        # 2026-07-05: the Cloud Function / boot-ingest download PDFs into
        # NamedTemporaryFile paths, so basename(file_path) was «tmpl3mmhrmf.pdf»
        # — that garbage became the `source` metadata (leaked into the report's
        # RAG chips) AND broke filename-based date/bank detection, silently
        # degrading recency ranking.  The callers already pass the REAL object
        # name in doc_metadata["filename"] — prefer it.
        filename = os.path.basename(
            str(doc_metadata.get("filename") or os.path.basename(file_path)))
        logger.info("[RAG] Парсинг: %s", filename)
        print(f"[RAG] Парсинг: {filename}")

        try:
            import pymupdf4llm  # lazy: only needed for ingestion (see top)
            md_text = pymupdf4llm.to_markdown(file_path)
        except ImportError as e:
            logger.error("[RAG] pymupdf4llm не установлен — ingest пропущен: %s", e)
            return 0
        except Exception as e:
            logger.error("[RAG] Ошибка чтения PDF %s: %s", filename, e)
            print(f"[RAG] Ошибка: {e}")
            return 0

        # ── Определяем дату документа ──────────────────────────────────────
        doc_dt, method = self._get_doc_date(file_path, md_text, filename=filename)
        doc_ts = int(doc_dt.timestamp())
        print(f"[RAG] Дата документа: {doc_dt.strftime('%Y-%m-%d')} (источник: {method})")

        # ── Section-aware, size-bounded chunking + rich metadata ───────────
        bank = self._extract_bank(filename, md_text)
        sized = self._chunk_markdown(md_text)

        chunks, metadatas, ids = [], [], []
        for i, (heading, chunk) in enumerate(sized):
            meta = doc_metadata.copy()
            meta.update({
                "source":         filename,
                "bank":           doc_metadata.get("bank") or bank,
                "section":        heading or "—",
                "tickers":        self._extract_tickers(chunk),   # ",AAPL,MSFT,"
                "chunk_index":    i,
                "doc_timestamp":  doc_ts,   # ← ключевое поле для ранжирования
                "doc_date_str":   doc_dt.strftime("%Y-%m-%d"),
                "date_method":    method,
            })
            chunks.append(chunk)
            metadatas.append(meta)
            ids.append(f"{filename}_ch{i}")

        if chunks:
            self.collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
            print(f"[RAG] ✅ Загружено {len(chunks)} блоков (дата: {doc_dt.strftime('%Y-%m')})")
        else:
            print("[RAG] ⚠️  Значимый текст не найден.")

        return len(chunks)

    # ── Query with recency re-ranking ────────────────────────────────────────

    def get_market_sentiment(
        self,
        query: str,
        n_results: int = 3,
        half_life_days: int = 365,
        ticker: str | None = None,
    ) -> str:
        """
        Retrieves n_results chunks ranked by:
          score = semantic_similarity * W_SEMANTIC + recency_score * W_RECENCY

        half_life_days (default 365): через сколько дней документ теряет
        половину своей «актуальности». Уменьшите до 180 для быстрых рынков.

        ticker (RAG #в): when given, SOFT-filter to chunks that actually mention
        the ticker (metadata `tickers` OR document text) so a per-holding query
        pulls notes about THAT name — but falls back to the unfiltered top set
        when the ticker has no coverage (never returns empty just because a
        name isn't in the library).
        """
        if self.collection.count() == 0:
            return "NO PDF DATA AVAILABLE. База отчетов пуста."

        # Запрашиваем больше результатов для ре-ранжирования (шире при фильтре).
        mult = PREFETCH_MULTIPLIER * (3 if ticker else 1)
        prefetch = min(n_results * mult, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=prefetch,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]   # 0 = identical, 2 = opposite

        if not docs:
            return "No relevant information found."

        # ── Re-rank ────────────────────────────────────────────────────────
        scored = []
        for doc, meta, dist in zip(docs, metas, distances):
            sem_score  = 1.0 - (dist / 2.0)   # normalize [0, 1]
            doc_ts     = meta.get("doc_timestamp", 0)
            rec_score  = self._recency_score(doc_ts, half_life_days)
            combined   = sem_score * W_SEMANTIC + rec_score * W_RECENCY
            scored.append((combined, sem_score, rec_score, doc, meta))

        # Сортировка: лучшие наверх
        scored.sort(key=lambda x: x[0], reverse=True)

        # Soft ticker filter — keep only chunks mentioning the name, but fall
        # back to the unfiltered ranking when nothing matches.
        if ticker:
            tk = ticker.upper().split(".")[0]
            def _mentions(item):
                _, _, _, doc, meta = item
                return (f",{tk}," in (meta.get("tickers") or "")) or (tk in doc.upper())
            matched = [s for s in scored if _mentions(s)]
            if matched:
                scored = matched

        top = scored[:n_results]

        # ── Build context string ───────────────────────────────────────────
        context = ""
        for combined, sem, rec, doc, meta in top:
            bank_sec = " · ".join(x for x in [meta.get("bank"), meta.get("section")]
                                  if x and x not in ("—", "Unknown"))
            context += (
                f"\n--- [{meta.get('doc_date_str', '?')}] "
                f"{meta.get('source', '?')}"
                f"{(' — ' + bank_sec) if bank_sec else ''} "
                f"(актуальность: {rec:.0%}, схожесть: {sem:.0%}, "
                f"итог: {combined:.0%}) ---\n"
            )
            context += f"{doc}\n"

        return context

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_documents(self, *, include_temp: bool = False) -> list[dict]:
        """Returns a summary of all ingested documents, sorted by date desc.

        Ingestion-artifact sources (tmp basenames, `include_temp=False`) are
        excluded so the «база: N отчётов» inventory counts REAL reports only
        (замечание R2#6).  Pass `include_temp=True` for maintenance/purge paths.
        """
        if self.collection.count() == 0:
            return []

        # Get all metadata (no query needed)
        all_data = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in all_data["metadatas"]:
            src = meta.get("source", "?")
            if not include_temp and _is_temp_source(src):
                continue                        # skip pre-fix tmp-named artifacts
            if src not in seen:
                seen[src] = {
                    "source":     src,
                    "bank":       meta.get("bank", "Unknown"),
                    "date":       meta.get("doc_date_str", "unknown"),
                    "timestamp":  meta.get("doc_timestamp", 0),
                    "method":     meta.get("date_method", "?"),
                    "chunks":     0,
                }
            seen[src]["chunks"] += 1

        return sorted(seen.values(), key=lambda x: x["timestamp"], reverse=True)

    def purge_temp_sources(self) -> int:
        """Delete chunks whose `source` is an ingestion artifact (tmp basename).

        These are stale duplicates of reports since re-ingested under their real
        names (upsert with new ids left the old tmp-id chunks behind).  Removing
        them corrects BOTH the «отчётов» and «чанков» counts and cleans up
        retrieval.  Best-effort; returns the number of chunks deleted."""
        try:
            if self.collection.count() == 0:
                return 0
            got = self.collection.get(include=["metadatas"])
            ids = got.get("ids") or []
            metas = got.get("metadatas") or []
            temp_ids = [i for i, m in zip(ids, metas)
                        if _is_temp_source((m or {}).get("source"))]
            if temp_ids:
                self.collection.delete(ids=temp_ids)
                logger.info("RAG: purged %d chunk(s) from %d tmp-named source(s).",
                            len(temp_ids),
                            len({(m or {}).get("source") for m in metas
                                 if _is_temp_source((m or {}).get("source"))}))
            return len(temp_ids)
        except Exception as exc:
            logger.warning("RAG: purge_temp_sources skipped (%s).", exc)
            return 0
