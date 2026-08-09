"""Делает пакеты сервиса импортируемыми при запуске pytest из любого каталога.

Почему это нужно
────────────────
`freedom-etl` запускается как `python main.py` из своего корня, поэтому
`config`, `db` и `services` — пакеты верхнего уровня относительно него.
При запуске тестов из корня репозитория этого каталога в `sys.path` нет.

Почему conftest здесь, а не в корне репозитория
───────────────────────────────────────────────
Корневого `conftest.py` в проекте нет намеренно (корневой `CLAUDE.md`: команда
прогона обязана нести `PYTHONPATH=src`). Файл в этом каталоге действует только
на `freedom-etl/tests/` и никак не влияет на основной прогон `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent

if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))
