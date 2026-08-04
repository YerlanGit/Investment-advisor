#!/usr/bin/env python3
"""CLI для golden-фикстуры `analyze_all` (Арх-3.0).

Сама логика живёт в `tests/golden_support.py`, а не здесь, и это НЕ случайность:
деплой-образ копирует `src/` и `tests/`, но НЕ `scripts/`. Если бы драйвер
лежал в `scripts/`, тест в образе пришлось бы пропускать — то есть страховка
разреза не работала бы там, где цена ошибки максимальна. Этот файл — тонкая
обёртка для ручного запуска.

    python scripts/gen_results_fixture.py            # показать sha256 и размер
    python scripts/gen_results_fixture.py --write    # перезаписать эталон
    python scripts/gen_results_fixture.py --check    # 1 при расхождении

Штатный способ обновить эталон из тестов:

    GOLDEN_UPDATE=1 PYTHONPATH=src python -m pytest tests/test_contracts_golden.py

🔴 Во время фазы Арх-3 эталон НЕ обновляют: каждая её подзадача обязана быть
поведенчески пустой, поэтому расхождение — сигнал об ошибке разреза.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import golden_support as gs  # noqa: E402  (после правки sys.path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="перезаписать эталон")
    ap.add_argument("--check", action="store_true", help="код 1 при расхождении")
    args = ap.parse_args()

    produced = gs.fixture_json()
    digest = gs.fixture_sha256(produced)

    if args.write:
        gs.FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        gs.FIXTURE_PATH.write_text(produced, encoding="utf-8")
        print(f"записано: {gs.FIXTURE_PATH} · {len(produced) / 1024:.1f} KB · sha256 {digest}")
        return 0

    if args.check:
        if not gs.FIXTURE_PATH.exists():
            print(f"НЕТ ЭТАЛОНА: {gs.FIXTURE_PATH} — запусти с --write")
            return 1
        expected = gs.FIXTURE_PATH.read_text(encoding="utf-8")
        if produced != expected:
            import json
            got, want = json.loads(produced), json.loads(expected)
            changed = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
            print(f"РАСХОЖДЕНИЕ в ключах ({len(changed)}): {changed}")
            return 1
        print(f"совпадает: sha256 {digest}")
        return 0

    print(f"{gs.FIXTURE_PATH} · {len(produced) / 1024:.1f} KB · sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
