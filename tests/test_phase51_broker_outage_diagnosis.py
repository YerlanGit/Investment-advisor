"""§−94 — почему «Freedom Broker недоступен» нельзя печатать одним текстом.

Что здесь охраняется
--------------------
**Три РАЗНЫЕ причины печатались одинаково.** `broker_api.fetch_portfolio`
отдаёт fallback-мок из трёх веток, и все три говорили пользователю «обычно это
проходит за 5–15 минут»:

  `waf_block`   Cloudflare отбил ПО IP-адресу. Само не пройдёт — нужен
                статический egress. Рестарт «лечит» его СЛУЧАЙНО (новый инстанс
                берёт другой IP из общего пула) и ровно так же случайно ломает
                работавшего бота.
  `api_error`   транспорт/таймаут. Вот это действительно обычно временное.
  `parse_error` ответ ПРИШЁЛ, разобрать не смогли. Дефект на нашей стороне;
                «подождите 15 минут» — ложь, ждать можно вечно.

Совет по причине — не косметика: он определяет, что человек сделает дальше
(ждать, написать в поддержку или ничего). Тот же класс, что `§−90` A-3:
одинаковый вид у состояний с разной природой.

**Маркеры гейтов трогать нельзя.** `_ramp_is_mock` / `_ramp_is_fallback` держат
и бот-гейт (`cb_confirm` до превью), и движковый `RealPortfolioRequired`.
Причина — ДОБАВКА поверх них, а не замена: потребитель, который её не знает,
обязан работать как раньше.

**Кап потоков у бота.** Бот гоняет ТОТ ЖЕ ONNX-эмбеддинг, что и RAG-функция
(`entrypoint._boot_ingest_from_inbox` на буте), но на 2Gi/1CPU и без
ограничения потоков, тогда как для функции `INFRA_NETWORKING.md §2` предписал
4Gi/2CPU + кап. Правило без гейта отрастает обратно — отсюда тест на deploy-шаг.
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# формат 'id:secret' обязателен: tg_bot валидирует токен на ИМПОРТЕ (`§−85`).
os.environ.setdefault("OMBRI_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")


def _connector_raising(exc):
    """FreedomConnector с клиентом, который падает заданным исключением."""
    import finance.broker_api as ba

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def get_portfolio(self):
            raise exc

    orig = ba.TradernetClient
    ba.TradernetClient = _Boom
    return ba, orig


class FallbackReasonTest(unittest.TestCase):
    """Причина отказа доезжает до вызывающего — иначе диагноз не построить."""

    def _fetch(self, exc):
        ba, orig = _connector_raising(exc)
        self.addCleanup(setattr, ba, "TradernetClient", orig)
        return ba.FreedomConnector("real-key", "real-secret", "login").fetch_portfolio()

    def test_waf_block_is_labelled(self) -> None:
        from freedom_portfolio.client import CloudflareBlockError
        df = self._fetch(CloudflareBlockError("403 HTML challenge"))
        self.assertEqual(df.attrs.get("_ramp_fallback_reason"), "waf_block")

    def test_transport_failure_is_labelled(self) -> None:
        from freedom_portfolio.client import BrokerAPIError
        df = self._fetch(BrokerAPIError("SSL EOF"))
        self.assertEqual(df.attrs.get("_ramp_fallback_reason"), "api_error")

    def test_parse_failure_is_labelled(self) -> None:
        """Ответ пришёл, но не разобрался — это НАШ дефект, не сбой брокера."""
        df = self._fetch(ValueError("malformed json"))
        self.assertEqual(df.attrs.get("_ramp_fallback_reason"), "parse_error")

    def test_gate_markers_survive_every_reason(self) -> None:
        """Регресс-гард: на этих двух маркерах стоят ОБА гейта (`§−34` M-6)."""
        from freedom_portfolio.client import BrokerAPIError, CloudflareBlockError
        for exc in (CloudflareBlockError("x"), BrokerAPIError("y"), ValueError("z")):
            with self.subTest(exc=type(exc).__name__):
                df = self._fetch(exc)
                self.assertTrue(df.attrs.get("_ramp_is_mock"))
                self.assertTrue(df.attrs.get("_ramp_is_fallback"))

    def test_detail_is_captured_and_bounded(self) -> None:
        """Детали нужны логу, но не безразмерные: тело ответа брокера могло бы
        уехать в лог целиком (тот же мотив, что F-6 — не эхать str(exc) юзеру)."""
        from freedom_portfolio.client import BrokerAPIError
        df = self._fetch(BrokerAPIError("q" * 5000))
        detail = df.attrs.get("_ramp_fallback_detail") or ""
        self.assertIn("BrokerAPIError", detail)
        self.assertLessEqual(len(detail), 300)

    def test_demo_has_no_reason(self) -> None:
        """Демо — осознанный выбор, а не отказ; причины у него быть не должно."""
        from finance.broker_api import FreedomConnector
        df = FreedomConnector("demo").fetch_portfolio()
        self.assertIsNone(df.attrs.get("_ramp_fallback_reason"))
        self.assertFalse(df.attrs.get("_ramp_is_fallback", False))


class OutageAdviceTest(unittest.TestCase):
    """Совет обязан соответствовать природе отказа."""

    def test_waf_block_does_not_promise_it_passes(self) -> None:
        from tg_bot import _broker_outage_advice
        text = _broker_outage_advice("waf_block", "abc123")
        self.assertNotIn("5–15 минут", text)
        self.assertIn("не пройдёт сам", text)
        self.assertIn("abc123", text)

    def test_parse_error_names_our_side(self) -> None:
        from tg_bot import _broker_outage_advice
        text = _broker_outage_advice("parse_error", "def456")
        self.assertNotIn("5–15 минут", text)
        self.assertIn("нашей", text)
        self.assertIn("def456", text)

    def test_transport_error_keeps_the_retry_advice(self) -> None:
        from tg_bot import _broker_outage_advice
        text = _broker_outage_advice("api_error", "ghi789")
        self.assertIn("5–15 минут", text)
        self.assertIn("ghi789", text)

    def test_unknown_reason_falls_back_to_transient(self) -> None:
        """Сводки/коннекторы без причины (старый путь) обязаны вести себя как
        раньше — самый мягкий текст, но код ошибки печатается всегда."""
        from tg_bot import _broker_outage_advice
        for unknown in ("", "whatever", None):
            text = _broker_outage_advice(unknown or "", "jkl000")
            self.assertIn("5–15 минут", text)
            self.assertIn("jkl000", text)

    def test_every_reason_has_its_own_advice(self) -> None:
        """Новая причина в коннекторе обязана получить свой текст, а не молча
        свалиться в «подождите» — иначе дефект снова наденет маску сбоя."""
        from finance.broker_api import FreedomConnector
        from tg_bot import _broker_outage_advice
        transient = _broker_outage_advice("api_error", "id")
        for reason in FreedomConnector.FALLBACK_REASONS:
            with self.subTest(reason=reason):
                text = _broker_outage_advice(reason, "id")
                if reason != "api_error":
                    self.assertNotEqual(text, transient,
                                        f"причина {reason} печатается как временный сбой")


class BotThreadCapTest(unittest.TestCase):
    """Бот гоняет ONNX-эмбеддинг в контейнере — значит кап потоков обязателен."""

    def setUp(self) -> None:
        self.cb = _ROOT / "cloudbuild.yaml"
        if not self.cb.exists():
            self.skipTest("cloudbuild.yaml отсутствует (деплой-образ)")
        self.text = self.cb.read_text(encoding="utf-8")

    def test_bot_deploy_carries_thread_caps(self) -> None:
        # Строка env-vars ИМЕННО шага deploy бота (не RAG-функции): у неё есть
        # маркер REPORT_BUCKET_NAME, которого нет у функции.
        line = next((l for l in self.text.splitlines()
                     if "--set-env-vars=" in l and "REPORT_BUCKET_NAME" in l), None)
        self.assertIsNotNone(line, "не нашёл env-строку деплоя бота")
        for cap in ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                    "TOKENIZERS_PARALLELISM=false"):
            self.assertIn(cap, line, f"у бота нет {cap} — тот же ONNX, что у функции")

    def test_in_container_ingest_is_why(self) -> None:
        """Гард держится на ФАКТЕ: если бы бот не ингестил в контейнере, кап был
        бы не нужен. Пропадёт boot-ingest — тест напомнит пересмотреть правило."""
        entry = (_SRC / "entrypoint.py").read_text(encoding="utf-8")
        self.assertIn("_boot_ingest_from_inbox", entry)

    def test_env_vars_are_one_line(self) -> None:
        """`--set-env-vars` ЗАМЕНЯЕТ весь набор (об этом отдельный комментарий в
        cloudbuild). Две строки = вторая молча затрёт первую."""
        lines = [l for l in self.text.splitlines()
                 if "--set-env-vars=" in l and "REPORT_BUCKET_NAME" in l]
        self.assertEqual(len(lines), 1)


class OutageMessageWiringTest(unittest.TestCase):
    """Сообщение собирается ИЗ причины, а не литералом в теле обработчика."""

    def test_handler_calls_the_advice_builder(self) -> None:
        src = (_SRC / "tg_bot.py").read_text(encoding="utf-8")
        block = src.split("Freedom Broker сейчас недоступен", 1)[1][:600]
        self.assertIn("_broker_outage_advice", block)
        self.assertIn("Токен *не списан*", block)

    def test_handler_does_not_hardcode_the_transient_promise(self) -> None:
        """Обещание «пройдёт за 5–15 минут» обязано приходить ИЗ причины.
        Литерал в теле гейта — это ровно тот дефект, который чинится: он
        печатался и при IP-блоке, и при нашей ошибке разбора.

        (Диагностика Шага 1 — отдельное сообщение про цены, там тот же оборот
        уместен и под своим тестом в `test_phase34`; поэтому проверяем БЛОК
        гейта, а не файл целиком.)"""
        src = (_SRC / "tg_bot.py").read_text(encoding="utf-8")
        block = src.split("Freedom Broker сейчас недоступен", 1)[1][:600]
        self.assertNotIn("5–15 минут", block)

    def test_reason_is_read_from_the_portfolio_attrs(self) -> None:
        """Причина берётся из attrs, снятых коннектором, а не выводится в боте:
        иначе диагноз разойдётся с тем, что реально произошло на транспорте."""
        src = (_SRC / "tg_bot.py").read_text(encoding="utf-8")
        gate = src.split("M-6 (2026-07-19)", 1)[1][:1200]
        self.assertIn("_ramp_fallback_reason", gate)
        self.assertLess(gate.index("_ramp_fallback_reason"),
                        gate.index("Freedom Broker сейчас недоступен"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
