"""Переезд ГЛАВНОГО бота на имя OMBRI и SSOT имён продукта (`§−101`).

Что здесь пинится и почему именно это
─────────────────────────────────────
Проект переименовывается второй раз. Первый переезд (`§−100`, загрузчик)
показал две вещи, и обе стали тестами ниже.

1. **Имя переменной с токеном нельзя менять «просто так».** `tg_bot` читает
   токен на уровне МОДУЛЯ: нет переменной — падает импорт, то есть прод не
   стартует, и симптом неотличим от гонки импортов (`§−98`). Поэтому новое
   имя обязано иметь фолбэк на прежнее, а фолбэк — предупреждение.

2. **Имя секрета и хэндл бота — ПАРА.** Хэндл едет в отчёт и превращается в
   deep-link «Применить идею»; отчёт статичен, и ссылку в уже выпущенном
   HTML задним числом не поправить. Переключили одно без другого — кнопка
   ведёт к боту, которого не опрашивают. Это единственная необратимая
   ошибка всего переезда, и на неё стоит отдельный тест.

3. **Бренд не размазывается по файлам.** Владелец просил, чтобы имя менялось
   «быстро и безболезненно». Правило без гейта отрастает обратно — уже
   проверено на `CLAUDE.md`, — поэтому здесь запрет на литерал бренда в
   модулях, которые говорят с пользователем.
"""

from __future__ import annotations

import ast
import os
import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

# tg_bot читает токен на уровне модуля — заглушка обязана стоять ДО импорта.
os.environ.setdefault("OMBRI_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")
os.environ.setdefault("FINTECH_MASTER_KEY", "0" * 43 + "=")

# 🔴 Импорт делается ЗДЕСЬ, при живой заглушке токена, а не внутри теста:
# `read_bot_token()` вызывается на уровне модуля, и импорт с очищенным
# окружением упал бы на самом факте импорта — то есть тест мерил бы не то,
# что проверяет. Ниже тесты дёргают саму функцию, окружение им подконтрольно.
try:
    import tg_bot as _tg_bot_module                       # noqa: E402
except ImportError:                                       # pragma: no cover
    _tg_bot_module = None


class BrandingDefaultsTest(unittest.TestCase):
    """Четыре имени, и они РАЗНЫЕ (`branding` — SSOT)."""

    def setUp(self) -> None:
        self._keys = ("BRAND_PROJECT_NAME", "BRAND_BOT_NAME",
                      "BOT_USERNAME", "SUPPORT_CONTACT")
        self._prev = {k: os.environ.get(k) for k in self._keys}

        def _restore() -> None:
            for key, value in self._prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(_restore)
        for key in self._keys:
            os.environ.pop(key, None)

    def test_defaults_are_the_agreed_names(self) -> None:
        import branding                                   # noqa: PLC0415

        self.assertEqual(branding.project_name(), "OMBRIOS")
        self.assertEqual(branding.bot_name(), "OMBRI")
        self.assertEqual(branding.bot_username(), "Ombri_bot")
        self.assertEqual(branding.support_contact(), "@OMBRI_support_bot")

    def test_platform_and_bot_are_not_the_same_name(self) -> None:
        """Решение владельца: платформа OMBRIOS, бот OMBRI — два имени."""
        import branding                                   # noqa: PLC0415

        self.assertNotEqual(branding.project_name(), branding.bot_name())

    def test_every_name_is_overridable_from_env(self) -> None:
        import branding                                   # noqa: PLC0415

        os.environ["BRAND_PROJECT_NAME"] = "ACME"
        os.environ["BRAND_BOT_NAME"] = "ACE"
        os.environ["BOT_USERNAME"] = "@acme_bot"
        os.environ["SUPPORT_CONTACT"] = "acme_help"
        self.assertEqual(branding.project_name(), "ACME")
        self.assertEqual(branding.bot_name(), "ACE")
        # хэндл — БЕЗ «@» (иначе deep-link получит t.me/%40acme_bot)
        self.assertEqual(branding.bot_username(), "acme_bot")
        # контакт — С «@» (его печатают как есть)
        self.assertEqual(branding.support_contact(), "@acme_help")

    def test_empty_value_falls_back_to_the_default(self) -> None:
        """`--set-env-vars` легко оставляет переменную объявленной и пустой.

        Имя продукта — не то место, где пустота лучше дефолта: пустой хэндл
        даёт ссылку `t.me/?start=…`, то есть мёртвую кнопку в отчёте.
        """
        import branding                                   # noqa: PLC0415

        os.environ["BOT_USERNAME"] = "   "
        self.assertEqual(branding.bot_username(), "Ombri_bot")


class MainBotTokenEnvRenameTest(unittest.TestCase):
    """Переезд имени переменной с токеном у ГЛАВНОГО бота (`§−101`)."""

    _KEYS = ("OMBRI_BOT_TOKEN", "RAMP_BOT_TOKEN")

    def setUp(self) -> None:
        self._prev = {k: os.environ.get(k) for k in self._KEYS}

        def _restore() -> None:
            for key, value in self._prev.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.addCleanup(_restore)
        for key in self._KEYS:
            os.environ.pop(key, None)

    def _tg_bot(self):
        if _tg_bot_module is None:                        # pragma: no cover
            self.skipTest("tg_bot не импортируется (нет aiogram?)")
        return _tg_bot_module

    def test_new_name_is_the_primary_one(self) -> None:
        tg_bot = self._tg_bot()

        self.assertEqual(tg_bot.TOKEN_ENV, "OMBRI_BOT_TOKEN")
        os.environ["OMBRI_BOT_TOKEN"] = "111:NEW"
        self.assertEqual(tg_bot.read_bot_token(), "111:NEW")

    def test_legacy_name_still_works_and_says_so(self) -> None:
        """Секрет в Secret Manager ещё под прежним именем — бот обязан встать.

        И обязан сказать, что имя устарело: молчаливый фолбэк означал бы, что
        «работает» перестало значить «настроено», а переезд не закончится
        никогда.
        """
        tg_bot = self._tg_bot()

        self.assertEqual(tg_bot.LEGACY_TOKEN_ENV, "RAMP_BOT_TOKEN")
        os.environ["RAMP_BOT_TOKEN"] = "222:OLD"
        with self.assertLogs("ombri.bot", level="WARNING") as log:
            self.assertEqual(tg_bot.read_bot_token(), "222:OLD")
        self.assertIn("OMBRI_BOT_TOKEN", "".join(log.output),
                      "приём прежнего имени обязан называть новое — иначе "
                      "оператор не узнает, что именно переименовывать")

    def test_new_name_wins_over_the_legacy_one(self) -> None:
        tg_bot = self._tg_bot()

        os.environ["OMBRI_BOT_TOKEN"] = "111:NEW"
        os.environ["RAMP_BOT_TOKEN"] = "222:OLD"
        self.assertEqual(tg_bot.read_bot_token(), "111:NEW")

    def test_neither_name_refuses_loudly(self) -> None:
        """🔴 Fail-closed: бот без токена не «работает частично».

        Прежний код падал голым `KeyError` — по симптому неотличимо от
        `§−98`. Отказ обязан называть ОБА имени и место, откуда токен берётся.
        """
        tg_bot = self._tg_bot()

        with self.assertRaises(RuntimeError) as ctx:
            tg_bot.read_bot_token()
        message = str(ctx.exception)
        self.assertIn("OMBRI_BOT_TOKEN", message)
        self.assertIn("RAMP_BOT_TOKEN", message)

    def test_the_token_is_never_echoed(self) -> None:
        """Секрет не попадает ни в лог, ни в текст отказа."""
        tg_bot = self._tg_bot()

        os.environ["RAMP_BOT_TOKEN"] = "222:SUPER-SECRET-VALUE"
        with self.assertLogs("ombri.bot", level="WARNING") as log:
            tg_bot.read_bot_token()
        self.assertNotIn("SUPER-SECRET-VALUE", "".join(log.output))


class BrandIsNotHardcodedTest(unittest.TestCase):
    """Литерала бренда нет в модулях, которые говорят с пользователем.

    Владелец просил, чтобы имя менялось одной правкой. Держится это на том,
    что тексты берут имя у `branding`, а не носят его в себе. Без гейта
    следующий литерал приедет с первой же новой строкой.

    Гейт смотрит на СТРОКОВЫЕ ЛИТЕРАЛЫ (не на комментарии и не на докстринги:
    там имя объяснимо) и пропускает имена переменных окружения вида
    `OMBRI_BOT_TOKEN` — это не бренд, а ключ конфигурации.
    """

    #: Бренд как отдельное слово: `OMBRI_BOT_TOKEN` не считается (за ним `_`).
    _BRAND = re.compile(r"(?<![A-Za-z0-9_])(OMBRIOS|OMBRI)(?![A-Za-z0-9_])")

    #: Модули, чьи строки видит пользователь.
    _WATCHED = ("tg_bot.py", "profile_manager.py", "pdf_payload.py",
                "premium_payload.py", "ai_narrative.py")

    @staticmethod
    def _docstring_nodes(tree: ast.AST) -> set[int]:
        """id() строковых узлов, которые являются докстрингами."""
        out: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                out.add(id(first.value))
        return out

    def test_no_brand_literals_in_user_facing_modules(self) -> None:
        offenders: list[str] = []
        for name in self._WATCHED:
            path = _SRC / name
            if not path.is_file():                        # pragma: no cover
                self.skipTest(f"{name} отсутствует")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = self._docstring_nodes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                if not isinstance(node.value, str) or id(node) in docstrings:
                    continue
                if self._BRAND.search(node.value):
                    offenders.append(f"{name}:{node.lineno} {node.value[:60]!r}")
        self.assertFalse(
            offenders,
            "имя бренда вписано в строку вместо `branding.project_name()` / "
            "`branding.bot_name()`:\n  " + "\n  ".join(offenders))

    def test_watched_modules_actually_use_the_ssot(self) -> None:
        """Обратная сторона: гейт зелёный и потому, что имени нет ВООБЩЕ.

        Без этой проверки удаление всех упоминаний бренда прошло бы как
        «успех», а пользователь остался бы без названия продукта.
        """
        source = (_SRC / "tg_bot.py").read_text(encoding="utf-8")
        self.assertIn("branding.project_name()", source)
        self.assertIn("branding.bot_name()", source)
        self.assertIn("branding.support_contact()", source)

    def test_support_contact_moved_out_of_the_code(self) -> None:
        """Контакт поддержки — в `branding`, а не в тексте команды."""
        source = (_SRC / "tg_bot.py").read_text(encoding="utf-8")
        self.assertNotIn("@ramp_support_bot", source)
        self.assertNotIn("@OMBRI_support_bot", source,
                         "хэндл поддержки живёт в `branding`, а не здесь")


class StagedSwitchIsAtomicTest(unittest.TestCase):
    """🔴 Имя секрета и хэндл бота переключаются ТОЛЬКО вместе (`§−101`).

    Это единственная НЕОБРАТИМАЯ ошибка переезда. Хэндл вшивается в готовый
    HTML-отчёт и превращается в deep-link «Применить идею»; отчёт лежит в GCS
    статикой, и ссылку в нём задним числом не поправить. Переключили хэндл
    раньше токена — кнопка ведёт к боту, которого ещё не опрашивают; позже —
    к боту, которого уже не опрашивают.
    """

    _LEGACY_SECRET = "RAMP_BOT_TOKEN"
    _LEGACY_HANDLE = "KEN_investment_bot"

    def setUp(self) -> None:
        try:
            import yaml                                   # noqa: PLC0415
        except ImportError:                               # pragma: no cover
            self.skipTest("PyYAML не установлен")
        path = _ROOT / "cloudbuild.yaml"
        # `cloudbuild.yaml` лежит в КОРНЕ РЕПО и в образ не копируется
        # (`Dockerfile` берёт только `src/`, `tests/`, `SYSTEM_PROMPT.md` и
        # requirements). Деплой-гейт гоняет сюиту ВНУТРИ образа — без этой
        # проверки тест валил бы сборку (`§−99` A-4).
        if not path.is_file():
            self.skipTest("cloudbuild.yaml отсутствует (Docker deploy-gate)")
        self.doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.subs = self.doc["substitutions"]
        self.deploy = next(s for s in self.doc["steps"] if s["id"] == "deploy")

    def _arg(self, prefix: str) -> str:
        return next(a for a in self.deploy["args"] if a.startswith(prefix))

    def test_secret_name_is_a_substitution(self) -> None:
        """Слева — переменная В КОНТЕЙНЕРЕ, справа — имя СЕКРЕТА."""
        self.assertIn("_BOT_TOKEN_SECRET", self.subs)
        self.assertIn("OMBRI_BOT_TOKEN=${_BOT_TOKEN_SECRET}:latest",
                      self._arg("--set-secrets"),
                      "переменная в контейнере — уже новая (код читает её), "
                      "а имя секрета живёт в подстановке")

    def test_handle_is_a_substitution_too(self) -> None:
        self.assertIn("_BOT_USERNAME", self.subs)
        self.assertIn("BOT_USERNAME=${_BOT_USERNAME}",
                      self._arg("--set-env-vars"))

    def test_secret_and_handle_flip_together(self) -> None:
        legacy_secret = self.subs["_BOT_TOKEN_SECRET"] == self._LEGACY_SECRET
        legacy_handle = self.subs["_BOT_USERNAME"] == self._LEGACY_HANDLE
        self.assertEqual(
            legacy_secret, legacy_handle,
            "полупереключение: секрет и хэндл разъехались "
            f"(_BOT_TOKEN_SECRET={self.subs['_BOT_TOKEN_SECRET']!r}, "
            f"_BOT_USERNAME={self.subs['_BOT_USERNAME']!r}). Кнопка "
            "«Применить идею» в отчёте поведёт не к тому боту, а отчёт "
            "статичен — задним числом ссылку не поправить.")

    def test_the_deploy_does_not_bind_a_secret_that_may_not_exist(self) -> None:
        """Ступенчатость: дефолт указывает на СУЩЕСТВУЮЩИЙ секрет.

        Привязка к несуществующему секрету роняет ВЕСЬ деплой — прод остаётся
        на старой ревизии. Дефолт в репозитории обязан быть тем именем,
        которое в проекте уже заведено; переключение — правка подстановки
        (здесь или в триггере), а не мерж.
        """
        self.assertEqual(self.subs["_BOT_TOKEN_SECRET"], self._LEGACY_SECRET,
                         "секрет OMBRI_BOT_TOKEN ещё не заведён в проекте — "
                         "дефолт менять только вместе с ним")

    def test_the_loader_secret_is_untouched(self) -> None:
        """Переезд главного бота не задевает загрузчика (`§−100`)."""
        self.assertEqual(self.subs["_INGEST_BOT_TOKEN_SECRET"],
                         "OMBRI_INGEST_BOT_TOKEN")


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
