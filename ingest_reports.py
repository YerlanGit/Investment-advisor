"""
Скрипт для массовой загрузки PDF-отчётов в RAG-базу.

Использование:
    python ingest_reports.py                       # Загрузить все PDF из data/reports/
    python ingest_reports.py путь/к/файлу.pdf      # Загрузить один файл
    python ingest_reports.py --list                # Показать все загруженные отчеты с датами

Соглашение по именам файлов (для лучшего определения даты):
    goldman_sachs_outlook_Q1_2025.pdf      ← квартал
    jpmorgan_equity_strategy_march_2025.pdf ← месяц
    blackrock_2024_annual_outlook.pdf      ← год
    ubs_global_2025-03.pdf                 ← ISO формат
"""
import os
import sys

# Support running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agent.rag_engine import FinancialRAG


def ingest_all(reports_dir: str = "data/reports"):
    rag = FinancialRAG()

    if not os.path.exists(reports_dir):
        print(f"[Ошибка] Папка '{reports_dir}' не найдена.")
        return

    pdf_files = sorted(
        [f for f in os.listdir(reports_dir) if f.lower().endswith(".pdf")]
    )

    if not pdf_files:
        print(f"[Ошибка] В папке '{reports_dir}' нет PDF-файлов.")
        print("Положите туда отчёты банков (Goldman Sachs, JP Morgan, и т.д.) и перезапустите.")
        return

    print(f"═══ Загрузка {len(pdf_files)} отчётов в RAG-базу ═══\n")

    total_chunks = 0
    for i, filename in enumerate(pdf_files, 1):
        filepath = os.path.join(reports_dir, filename)
        print(f"[{i}/{len(pdf_files)}] {filename}")
        n = rag.ingest_pdf(filepath, doc_metadata={"filename": filename})
        total_chunks += n
        print()

    print(f"═══ Готово! Загружено {total_chunks} чанков из {len(pdf_files)} отчётов ═══\n")
    _print_doc_list(rag)


def ingest_single(filepath: str):
    rag = FinancialRAG()

    if not os.path.exists(filepath):
        print(f"[Ошибка] Файл не найден: {filepath}")
        return

    rag.ingest_pdf(filepath, doc_metadata={"filename": os.path.basename(filepath)})
    print(f"\nВсего чанков в базе: {rag.collection.count()}")
    _print_doc_list(rag)


def list_docs():
    rag = FinancialRAG()
    _print_doc_list(rag)


def _print_doc_list(rag: FinancialRAG):
    docs = rag.list_documents()
    if not docs:
        print("База пуста.")
        return

    print("\n📚 Загруженные отчеты (отсортированы по дате, новые сверху):")
    print(f"{'Дата':<12} {'Источник':<50} {'Метод'}")
    print("─" * 75)
    for d in docs:
        print(f"{d['date']:<12} {d['source'][:50]:<50} ({d['method']})")
    print(f"\nИтого документов: {len(docs)}, чанков: {rag.collection.count()}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--list":
            list_docs()
        else:
            ingest_single(arg)
    else:
        ingest_all()
