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

    rc = 0
    for scenario in gs.SCENARIOS:
        produced = gs.fixture_json(scenario=scenario)
        digest = gs.fixture_sha256(produced)
        path = gs.fixture_path(scenario)

        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(produced, encoding="utf-8")
            print(f"записано: {path.name} · {len(produced) / 1024:.1f} KB · sha256 {digest}")
            continue

        if args.check:
            if not path.exists():
                print(f"НЕТ ЭТАЛОНА: {path} — запусти с --write")
                rc = 1
                continue
            expected = path.read_text(encoding="utf-8")
            if produced != expected:
                import json
                got, want = json.loads(produced), json.loads(expected)
                changed = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
                print(f"[{scenario}] РАСХОЖДЕНИЕ в ключах ({len(changed)}): {changed}")
                rc = 1
            else:
                print(f"[{scenario}] совпадает: sha256 {digest}")
            continue

        print(f"{path.name} · {len(produced) / 1024:.1f} KB · sha256 {digest}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
