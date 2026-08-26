"""Гигиена репозитория: инструкция не разрастается, карта тестов не врёт.

Раунд: архитектурный трек `docs/ARCHITECTURE_FOR_AGENTS.md §4`, фазы Арх-1 и Арх-4.

Почему это ТЕСТ, а не соглашение
--------------------------------
Корневой `CLAUDE.md` уже один раз превратился из инструкции в журнал: за три дня
он вырос на 45% (21 010 → 30 448 символов ≈ 7 600 токенов в КАЖДОЙ сессии), 95%
объёма ушло в мега-строки, а самая длинная строка достигла 18 252 символов —
прочитать её глазами невозможно, и именно поэтому в ней полгода жила неверная
команда запуска тестов (без `PYTHONPATH=src`). Соглашение «пиши коротко» этого
не остановило, потому что нарушать его было нечем.

В этом репозитории уже доказано, что работает обратное: инвариант, закреплённый
исполняемым гейтом. AST-сканер против `int(os.getenv)` (`AUDIT §−50`) поймал
регресс автора того же правила на следующий день; `NoSecondCopyTest` (`§−46`)
не даёт завести вторую копию перечня тикеров. Здесь — тот же приём.

🔴 ГАРДЫ ОБРАЗА. `Dockerfile` копирует только `requirements*`, `src/`,
`SYSTEM_PROMPT.md` и `tests/`. Ни `CLAUDE.md`, ни `docs/`, ни `scripts/`, ни
`design/` в образе НЕТ, а deploy-гейт Cloud Build гоняет этот же набор тестов
внутри образа. Безусловное чтение такого файла даёт зелёный GitHub CI и
падение деплоя с `FileNotFoundError` — так уже было. Поэтому каждый тест ниже
начинается с проверки существования и `skipTest`.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CLAUDE_MD = _ROOT / "CLAUDE.md"
_SCRIPTS = _ROOT / "scripts"
_TEST_MAP = _ROOT / "tests" / "TEST_MAP.md"
_AUDIT_MD = _ROOT / "docs" / "audit" / "AUDIT.md"
#: §−108: индекс раундов переехал из `AUDIT.md §4` сюда — он занимал 23%
#: навигатора, который читается ЦЕЛИКОМ в каждой сессии, будучи при этом
#: данными для ПОИСКА. Гейт полноты переехал ВМЕСТЕ с данными: правило,
#: оставленное без гейта, отрастает обратно (`§−54`, `§−91`).
_ROUNDS_INDEX = _ROOT / "docs" / "audit" / "rounds" / "INDEX.md"


def _indexed_rounds() -> set:
    """Номера раундов, перечисленные в индексе. Нет файла — пустое множество."""
    import re
    if not _ROUNDS_INDEX.exists():
        return set()
    return set(re.findall(r"\|\s*`§−([\d.]+)`\s*\|",
                          _ROUNDS_INDEX.read_text(encoding="utf-8")))

# Пороги — не «красиво», а «читаемо глазами в точке правки».
# Замер после Арх-1 (2026-08-04): 122 строки, максимум 115 символов.
# Запас оставлен намеренно: правила будут добавляться, журнал — нет.
#
# 🔴 `§−99`, потолок поднят 200 → 210 ОДИН раз и по делу. Запас съели не
# пересказы раундов, ради которых гейт и стоит, а две новые ПОДСИСТЕМЫ: гонка
# импорта на потоках (`§−98`, оба бота) и сам бот-загрузчик с его правилами про
# токены и писателя базы. Перед подъёмом файл сжат там, где текст дублировался:
# два правила про `analyze_all` сведены в одно, блок зеркала деплой-образа — с
# семи строк до пяти. Замер после сжатия: 203 строки.
# Гейт от этого не слабеет: дамп истории раунда — это сотни строк, а не десять.
MAX_LINES = 210
MAX_LINE_CHARS = 300


class RootInstructionStaysReadableTest(unittest.TestCase):
    """`CLAUDE.md` — инструкция. Журнал живёт в `docs/audit/AUDIT.md`."""

    def setUp(self) -> None:
        if not _CLAUDE_MD.exists():
            self.skipTest("CLAUDE.md отсутствует (Docker deploy-gate)")
        self.lines = _CLAUDE_MD.read_text(encoding="utf-8").splitlines()

    def test_line_count_within_budget(self) -> None:
        self.assertLessEqual(
            len(self.lines), MAX_LINES,
            f"CLAUDE.md разросся до {len(self.lines)} строк (потолок {MAX_LINES}). "
            "Скорее всего в него дописали историю раунда — её место в docs/audit/AUDIT.md.",
        )

    def test_no_mega_lines(self) -> None:
        """Мега-строка = правило, которое физически нельзя прочитать."""
        offenders = [
            (i + 1, len(ln)) for i, ln in enumerate(self.lines) if len(ln) > MAX_LINE_CHARS
        ]
        self.assertFalse(
            offenders,
            "строки длиннее "
            f"{MAX_LINE_CHARS} символов: {offenders}. Одна мысль = одна строка; "
            "внутри такой строки правило невозможно вычитать глазами (так и жила "
            "неверная команда запуска тестов).",
        )

    def test_verification_command_is_correct(self) -> None:
        """Команда прогона обязана нести `PYTHONPATH=src`.

        Без префикса `import finance…` не находится: в репозитории нет ни
        `conftest.py`, ни `pyproject.toml`. Инструкция, в которой команда не
        работает, хуже отсутствующей.
        """
        text = "\n".join(self.lines)
        self.assertIn("PYTHONPATH=src python -m pytest tests/ -q", text)

    def test_points_at_the_journal(self) -> None:
        """Правило «история → AUDIT.md» должно быть НАПИСАНО, иначе файл отрастёт заново."""
        text = "\n".join(self.lines)
        self.assertIn("docs/audit/AUDIT.md", text)


#: Замер §−91: навигатор после расслоения — 337 строк, максимум 142 символа.
#: Потолки с запасом: доска и правила будут расти, ЖУРНАЛ — нет (он в rounds/).
AUDIT_MAX_LINES = 600
AUDIT_MAX_LINE_CHARS = 400


class AuditJournalStaysNavigableTest(unittest.TestCase):
    """`AUDIT.md` — навигатор; полные записи раундов живут в `rounds/`.

    Гейт написан по ИЗМЕРЕННОЙ деградации (`§−91`), а не на всякий случай:
    журнал дорос до 382 КБ, шапка до 76 КБ, одна строка до 27.6 КБ, а 14
    раундов вообще не получили секции — при том, что на 10 из них ссылается
    боевой код. Дефект был не косметическим: устаревшая доска, утонувшая в
    той самой мега-строке, увела за собой план работ по Manual Portfolio.

    Правило без гейта отрастает обратно — это уже проверено на `CLAUDE.md`.
    """

    def setUp(self) -> None:
        if not _AUDIT_MD.exists():
            self.skipTest("docs/audit/AUDIT.md отсутствует (Docker deploy-gate)")
        self.text = _AUDIT_MD.read_text(encoding="utf-8")
        self.lines = self.text.splitlines()

    def test_navigator_stays_small_enough_to_read_whole(self) -> None:
        self.assertLessEqual(
            len(self.lines), AUDIT_MAX_LINES,
            f"AUDIT.md разросся до {len(self.lines)} строк (потолок "
            f"{AUDIT_MAX_LINES}). Полная запись раунда идёт в "
            "docs/audit/rounds/, сюда — одна строка индекса.")

    def test_no_mega_lines(self) -> None:
        """Строку в 27 КБ невозможно ни ревьюить, ни дифать, ни прочитать."""
        offenders = [(i + 1, len(ln)) for i, ln in enumerate(self.lines)
                     if len(ln) > AUDIT_MAX_LINE_CHARS]
        self.assertFalse(
            offenders,
            f"строки длиннее {AUDIT_MAX_LINE_CHARS} символов: {offenders}. "
            "Так и вырос блок «Версия»: препенд за препендом в одну строку.")

    def test_version_blob_does_not_come_back(self) -> None:
        """🔴 Именно этот блок и был болезнью: препенд-лог, дублирующий раунды.

        Он писал тот же факт дважды, и читаемым первым оказывался худший —
        неструктурированный — экземпляр.
        """
        # Дата в шапке меняется каждый раунд, поэтому сверять её литералом
        # нельзя — гейт сломался бы на первом же обновлении (и сломался).
        # Проверяем ФОРМУ: строка «Версия» допустима РОВНО ОДНА и обязана
        # оставаться однострочной сводкой, а не пересказом раунда.
        version_lines = [ln for ln in self.lines if "**Версия:**" in ln]
        self.assertLessEqual(
            len(version_lines), 1,
            f"строк «Версия» в AUDIT.md: {len(version_lines)}. Так и рос "
            "препенд-лог: раунд за раундом в шапку, пока она не стала 76 КБ. "
            "Запись раунда живёт в rounds/, здесь — одна строка индекса.")
        for ln in version_lines:
            self.assertLessEqual(
                len(ln), 160,
                "строка «Версия» разрослась в пересказ раунда — оставь в ней "
                "только дату, номер последнего раунда и их количество.")

    def test_every_round_has_an_index_line(self) -> None:
        """Индекс — единственный способ найти раунд, не читая 350 КБ."""
        import re
        rounds_dir = _ROOT / "docs" / "audit" / "rounds"
        if not rounds_dir.exists():
            self.skipTest("docs/audit/rounds/ отсутствует")
        have = set()
        for f in rounds_dir.glob("ROUNDS_*.md"):
            have |= set(re.findall(r"^## −([\d.]+)\.",
                                   f.read_text(encoding="utf-8"), re.M))
        indexed = _indexed_rounds()
        missing = sorted(have - indexed, key=float, reverse=True)
        self.assertFalse(
            missing,
            f"раунды есть в rounds/, но их нет в {_ROUNDS_INDEX.name}: {missing}. "
            "Ненайденный раунд — это висячая ссылка из кода (так §−76…§−89 "
            "прожили 14 раундов без секции).")

    def test_index_points_only_at_existing_rounds(self) -> None:
        """Обратная сторона: индекс не должен обещать несуществующее."""
        import re
        rounds_dir = _ROOT / "docs" / "audit" / "rounds"
        if not rounds_dir.exists():
            self.skipTest("docs/audit/rounds/ отсутствует")
        have = set()
        for f in rounds_dir.glob("ROUNDS_*.md"):
            have |= set(re.findall(r"^## −([\d.]+)\.",
                                   f.read_text(encoding="utf-8"), re.M))
        indexed = _indexed_rounds()
        self.assertFalse(sorted(indexed - have, key=float),
                         "индекс ссылается на раунды, которых нет в rounds/")


class RoundReferencesResolveTest(unittest.TestCase):
    """Каждая ссылка `§−N` из кода и доков обязана указывать на живой раунд.

    До `§−91` десять номеров, на которые ссылались `stooq_ingest.py`,
    `stooq_store.py`, `stooq_provider.py`, `data_lineage.py`, `tg_bot.py` и
    тесты, не имели секции вовсе — агент, идущий по ссылке, не находил ничего.
    """

    def test_no_dangling_section_references(self) -> None:
        import re
        rounds_dir = _ROOT / "docs" / "audit" / "rounds"
        if not rounds_dir.exists():
            self.skipTest("docs/audit/rounds/ отсутствует")
        have = set()
        for f in rounds_dir.glob("ROUNDS_*.md"):
            have |= set(re.findall(r"^## −([\d.]+)\.",
                                   f.read_text(encoding="utf-8"), re.M))
        refs: set[str] = set()
        for path in _ROOT.rglob("*"):
            if path.is_dir() or "node_modules" in path.parts or ".git" in path.parts:
                continue
            if path.suffix not in (".py", ".md", ".jsx"):
                continue
            if "docs/audit/rounds/" in path.as_posix():
                continue
            try:
                refs |= set(re.findall(r"§−(\d+(?:\.\d+)?)",
                                       path.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError):
                continue
        missing = sorted(refs - have, key=float, reverse=True)
        self.assertFalse(
            missing,
            f"ссылки на несуществующие раунды: {missing}. Либо раунд не "
            "записан в rounds/, либо номер в ссылке опечатан.")


class TestMapIsGeneratedAndFreshTest(unittest.TestCase):
    """`tests/TEST_MAP.md` собирается скриптом и не разъезжается с кодом."""

    def test_map_exists_and_is_marked_generated(self) -> None:
        if not _TEST_MAP.exists():
            self.skipTest("tests/TEST_MAP.md отсутствует")
        head = _TEST_MAP.read_text(encoding="utf-8")[:600]
        self.assertIn("СГЕНЕРИРОВАН", head,
                      "карта обязана честно говорить, что её не правят руками")

    def test_map_is_up_to_date(self) -> None:
        """Свежесть проверяется тем же генератором (`--check`).

        Скрипт живёт в `scripts/`, которого в образе деплоя нет, — отсюда skip.
        Паттерн тот же, что у `CompiledAssetsTest` для Premium-бандлов.
        """
        if not (_SCRIPTS / "gen_test_map.py").exists():
            self.skipTest("scripts/ отсутствует (Docker deploy-gate)")
        if not _TEST_MAP.exists():
            self.fail("tests/TEST_MAP.md не сгенерирован: python scripts/gen_test_map.py")
        proc = subprocess.run(
            [sys.executable, str(_SCRIPTS / "gen_test_map.py"), "--check"],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(
            proc.returncode, 0,
            "карта тестов устарела — перегенерируй `python scripts/gen_test_map.py`.\n"
            f"{proc.stdout}{proc.stderr}",
        )


class TestsThatReadRepoFilesMustSkipInTheImageTest(unittest.TestCase):
    """Тест, читающий файл ВНЕ образа, обязан сам себя пропустить (`§−99`).

    🔴 Цена ошибки — весь деплой, а не один тест. Гейт `cloudbuild` гоняет
    `unittest discover` ВНУТРИ образа, а образ несёт только то, что копирует
    `Dockerfile`: `src/`, `tests/`, `SYSTEM_PROMPT.md` и requirements. Тест,
    открывающий `cloudbuild.yaml` или `scripts/`, падает там `FileNotFoundError`
    — сборка краснеет, и НЕ ДЕПЛОИТСЯ НИКТО, включая главный бот.

    Так и случилось с `DeployStepTest` бота-загрузчика: пять падений в образе
    при зелёном GitHub CI, потому что CI видит полный чекаут. Правило это в
    `CLAUDE.md` есть, но названо было только про `design/` — а помнить его надо
    про ЛЮБОЙ путь из корня репо. Поэтому здесь оно исполняемое.

    Проверка намеренно ГРУБАЯ: достаточно, чтобы в теле класса или функции,
    где путь строится, БЫЛ ВЫЗОВ `skipTest` (или `raise SkipTest`). Доказать
    срабатывание по всем ветвям статически нельзя, а «есть ли вообще выход»
    ловит ровно ту ошибку, которую люди и совершают, — забыли, а не написали
    неверно.

    🔴 Ищется именно ВЫЗОВ в AST, а не подстрока. Первая редакция этого гейта
    считала слово `skipTest` в тексте узла — и пропустила собственную мутацию:
    слово стояло в КОММЕНТАРИИ рядом. Тот же дефект, что `§−97` E-9, где
    текстовые гейты читали комментарии; повторён здесь через неделю после того,
    как был записан, — поэтому проверка теперь структурная.

    Сам этот тест в образе пропускается: он читает `Dockerfile`, которого там
    нет. Ловить он и должен на полном чекауте — до того, как образ собран.
    """

    #: `Path(...).parents[1] / "имя"` и `parent.parent / "имя"` — обе формы,
    #: которыми тесты этого репозитория добираются до корня.
    _PATH_RE = re.compile(r'(?:parents\[1\]|parent\.parent)\s*/\s*"([^"]+)"')

    @staticmethod
    def _has_skip(node: ast.AST) -> bool:
        """Есть ли в поддереве ВЫЗОВ `skipTest` или `raise SkipTest`."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                if isinstance(func, ast.Attribute) and func.attr == "skipTest":
                    return True
                if isinstance(func, ast.Name) and func.id in ("skipTest",
                                                              "SkipTest"):
                    return True
            if isinstance(sub, ast.Raise):
                exc = sub.exc
                name = getattr(exc, "func", exc)
                if isinstance(name, ast.Attribute) and name.attr == "SkipTest":
                    return True
                if isinstance(name, ast.Name) and name.id == "SkipTest":
                    return True
        return False

    def _image_manifest(self) -> set[str]:
        """Что `Dockerfile` реально кладёт в образ — из него самого, не списком."""
        dockerfile = _ROOT / "Dockerfile"
        if not dockerfile.is_file():
            self.skipTest("Dockerfile отсутствует (сам деплой-гейт)")
        out: set[str] = set()
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.upper().startswith("COPY "):
                continue
            parts = stripped.split()[1:]
            for token in parts[:-1]:                     # последний — назначение
                if token.startswith("--"):
                    continue
                out.add(token.rstrip("/").split("/")[0])
        return out

    def test_every_repo_path_read_by_a_test_is_guarded(self) -> None:
        manifest = self._image_manifest()
        self.assertIn("src", manifest,
                      "разбор Dockerfile сломался: в манифесте нет даже src/")
        offenders: list[str] = []
        for path in sorted((_ROOT / "tests").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                    continue
                body = ast.get_source_segment(source, node) or ""
                names = set(self._PATH_RE.findall(body))
                outside = {n for n in names if n.rstrip("/").split("/")[0]
                           not in manifest}
                if outside and not self._has_skip(node):
                    offenders.append(f"{path.name}::{node.name} → {sorted(outside)}")
        self.assertEqual(
            offenders, [],
            "эти тесты читают файлы, которых в деплой-образе НЕТ, и не умеют "
            "пропуститься — в образе они упадут FileNotFoundError и завалят "
            f"весь деплой (§−99):\n  " + "\n  ".join(offenders))


#: 🔴 Сетевые вызовы, которые НЕ имеют права случиться на импорте модуля.
_NET_ROOTS = {"requests", "httpx", "aiohttp", "urllib", "socket"}


def _reaches_network(node: ast.AST) -> bool:
    """Содержит ли поддерево обращение к сетевой библиотеке."""
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        while isinstance(f, ast.Attribute):
            f = f.value
        if isinstance(f, ast.Name) and f.id in _NET_ROOTS:
            return True
    return False


def _is_main_guard(stmt: ast.AST) -> bool:
    """`if __name__ == "__main__":` — легитимная защита исполняемой части."""
    if not isinstance(stmt, ast.If):
        return False
    return any(isinstance(n, ast.Name) and n.id == "__name__"
               for n in ast.walk(stmt.test))


class NoNetworkAtImportTest(unittest.TestCase):
    """🔴 Импорт модуля из `src/` не имеет права ходить в сеть.

    Найдено аудитом 26.08: `src/test_live_api.py` выполнял
    `result = fetch_live_portfolio(API_KEY)` НА УРОВНЕ МОДУЛЯ — то есть любой
    импорт файла стрелял живым HTTPS-запросом к `tradernet.kz`, с подставным
    ключом `"ваш_api_key"`.

    Почему это не мелочь именно здесь. `Dockerfile` копирует `src/` целиком,
    значит файл уезжает в ПРОД-ОБРАЗ. Запросы с невалидным ключом уходили бы к
    брокеру с боевого исходящего адреса — того самого, который Cloudflare-WAF
    брокера и блокирует (`R-9`, `INFRA_NETWORKING §1`). Проект платит ≈$50/мес
    за статический IP ради обхода этого блока; отправлять с него мусорный
    трафик — работать против собственной задачи.

    `cloudbuild.yaml` при этом ОБЪЯВЛЯЕТ сюиту сетевно-изолированной
    («no live calls leave the builder»). Утверждение держалось на том, что
    шаблон `test_phase*.py` этот файл не подхватывает — то есть на совпадении
    имён, а не на гейте. Голый `pytest src/` подхватывает и падает на
    `ProxyError` прямо при СБОРЕ тестов (проверено).

    Гейт смотрит на верхний уровень модуля, пропуская `if __name__`: там
    исполняемая часть законна.
    """

    def test_no_module_calls_the_network_while_being_imported(self) -> None:
        src = _ROOT / "src"
        if not src.is_dir():
            self.skipTest("src/ отсутствует")
        offenders: list[str] = []
        for path in sorted(src.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            net_fns = {n.name for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and _reaches_network(n)}
            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)) or _is_main_guard(stmt):
                    continue
                direct = _reaches_network(stmt)
                local = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id in net_fns for n in ast.walk(stmt))
                if direct or local:
                    offenders.append(
                        f"{path.relative_to(_ROOT)}:{stmt.lineno}")
        self.assertEqual(
            offenders, [],
            "модуль ходит в сеть на импорте — оберни исполняемую часть в "
            f"`if __name__ == \"__main__\":`. Нарушители: {offenders}")

    def test_the_detector_itself_catches_a_known_call(self) -> None:
        """🔴 Контрольный опыт (`§−68`): «ноль нарушителей» обязан что-то значить.

        Живьём детектор уже сработал — он и нашёл `test_live_api.py:46` до
        починки. Но это доказательство исчезает вместе с дефектом, поэтому
        здесь оно закреплено синтетикой: три образца, два из которых законны.
        """
        caught = ast.parse("import requests\nr = requests.post('http://x')\n")
        indirect = ast.parse("import requests\n"
                             "def go():\n    return requests.get('http://x')\n"
                             "r = go()\n")
        guarded = ast.parse("import requests\n"
                            "def go():\n    return requests.get('http://x')\n"
                            'if __name__ == "__main__":\n    go()\n')
        inside = ast.parse("import requests\n"
                           "def go():\n    return requests.get('http://x')\n")

        def scan(tree):
            net = {n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and _reaches_network(n)}
            out = []
            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)) or _is_main_guard(stmt):
                    continue
                if _reaches_network(stmt) or any(
                        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id in net for n in ast.walk(stmt)):
                    out.append(stmt.lineno)
            return out

        self.assertTrue(scan(caught), "прямой сетевой вызов не пойман")
        self.assertTrue(scan(indirect),
                        "вызов через локальную функцию не пойман — а именно так "
                        "и выглядел настоящий дефект")
        self.assertEqual(scan(guarded), [],
                         "`if __name__` — законная защита, гейт не имеет права "
                         "на неё ругаться")
        self.assertEqual(scan(inside), [],
                         "сетевой вызов ВНУТРИ функции законен: он случится "
                         "только когда функцию позовут")


#: Имена, которые объявляют СЕКРЕТ, а не вычисленный эталон.
_SECRET_NAMES = ("secret", "token", "password", "passwd", "apikey",
                 "api_key", "private_key", "credential")


def _looks_like_a_pasted_secret(value: str) -> bool:
    """Длинная hex-строка — признак ВСТАВЛЕННОГО значения, а не вычисленного."""
    v = value.strip()
    return len(v) >= 32 and all(c in "0123456789abcdefABCDEF" for c in v)


class NoRealSecretsInTestsTest(unittest.TestCase):
    """🔴 «Секрет не печатается» и «секрет не попадает в репозиторий» — РАЗНЫЕ
    утверждения, и `§−110` закрыл гейтом только первое.

    В том же раунде тест маскировки взял образцом НАСТОЯЩИЙ токен из утёкшего
    вывода gcloud. К моменту коммита он был ротирован и мёртв, но попал в
    историю git, где живёт вечно и откуда его не убрать, не переписав общую
    ветку. Тест проверял РЕГУЛЯРКУ — ей безразлично, какие именно 64 hex она
    маскирует, — то есть настоящий секрет не добавлял доказательности, только
    риск.

    Гейт узкий и потому без ложных срабатываний: он ловит не «длинный hex», а
    его ПРИСВАИВАНИЕ переменной с секретным именем. Вычисленные эталоны хешей
    (`expected = "cb50fa13…"` в `test_freedom_auth`) выводятся из видимых рядом
    входов и секретами не являются — гейт их не трогает.

    Обходится осознанно: соберите значение выражением
    (`"0" * 24 + "deadbeef" + …`), и по нему сразу видно, что оно синтетическое.
    """

    def _offenders(self, root: Path) -> list[str]:
        out: list[str] = []
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:                          # pragma: no cover
                continue
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                else:
                    continue
                names = [t.id.lower() for t in targets if isinstance(t, ast.Name)]
                if not any(k in n for n in names for k in _SECRET_NAMES):
                    continue
                v = node.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str) \
                        and _looks_like_a_pasted_secret(v.value):
                    try:
                        where = path.relative_to(_ROOT)
                    except ValueError:       # синтетика контрольного опыта
                        where = path
                    out.append(f"{where}:{node.lineno}")
        return out

    def test_no_secret_shaped_literal_is_committed(self) -> None:
        found: list[str] = []
        for name in ("tests", "src", "scripts", "cloud_function"):
            root = _ROOT / name
            if root.is_dir():
                found += self._offenders(root)
        self.assertEqual(
            found, [],
            "в репозиторий закоммичено значение, похожее на секрет — замените "
            f"синтетическим, собранным выражением: {found}")

    def test_the_detector_itself_catches_a_pasted_secret(self) -> None:
        """🔴 Контрольный опыт: «ноль нарушителей» обязан что-то значить.

        Первая проверка этого класса была `grep` по длинным hex — она нашла
        сначала `node_modules`, а `head` срезал настоящую находку. Проверка,
        чей результат зависит от порядка строк, не проверка.
        """
        import tempfile                                  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "bad.py").write_text(
                'secret = "8ce299ff41c799946b9917f76232eb7e9fe476e6"\n',
                encoding="utf-8")
            (d / "also_bad.py").write_text(
                'API_TOKEN = "cb50fa1321a1574c9564e4441a00429171567b3c"\n',
                encoding="utf-8")
            (d / "ok_expectation.py").write_text(
                'expected = "cb50fa1321a1574c9564e4441a00429171567b3c"\n',
                encoding="utf-8")
            (d / "ok_synthetic.py").write_text(
                'secret = "0" * 24 + "deadbeef"\n', encoding="utf-8")
            (d / "ok_short.py").write_text(
                'token = "abc123"\n', encoding="utf-8")

            found = self._offenders(d)
            names = sorted(f.rsplit("/", 1)[-1].split(":")[0] for f in found)
            self.assertIn("bad.py", names, "прямой секрет не пойман")
            self.assertIn("also_bad.py", names, "секрет под именем TOKEN не пойман")
            self.assertNotIn("ok_expectation.py", names,
                             "вычисленный эталон хеша — не секрет, гейт не имеет "
                             "права на него ругаться")
            self.assertNotIn("ok_synthetic.py", names,
                             "значение, собранное выражением, заведомо синтетическое")
            self.assertNotIn("ok_short.py", names, "короткая строка — не секрет")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
