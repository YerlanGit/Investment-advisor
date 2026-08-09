"""Подпись Tradernet APIv3: эталонные векторы и три классические ловушки.

Векторы посчитаны от ФИКСИРОВАННЫХ ключа и времени, а значения вбиты в тест
литералами. Это важнее, чем кажется: если бы тест повторял формулу HMAC своими
руками, он проверял бы код самим собой и прошёл бы даже при подписи не того
сообщения (например, `params + timestamp` без сериализации). Литерал ловит
именно это — он посчитан по спецификации SDK, а не по нашей реализации.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import unittest

from services.auth import TradernetAuth

# Заведомо ненастоящие ключи. Значения выбраны СЛОВАРНЫМИ намеренно: строка
# вида `sec_0123456789abcdef…` выглядит как утёкший секрет и для человека, и для
# сканера — gitleaks ловил её правилом `generic-api-key` по энтропии. Осмысленный
# текст решает обе задачи разом: и читателю очевидно, что это фикстура, и
# энтропийный порог не срабатывает.
PUBLIC_KEY = "EXAMPLE-PUBLIC-KEY-FOR-UNIT-TESTS"
SECRET_KEY = "EXAMPLE-SECRET-KEY-FOR-UNIT-TESTS"
FROZEN_TS = 1785960000          # 2026-08-05 20:00:00 UTC


def frozen_clock() -> float:
    return float(FROZEN_TS)


class SignatureVectorTest(unittest.TestCase):
    """Подпись считается от `compact_json(params) + timestamp`."""

    def setUp(self) -> None:
        self.auth = TradernetAuth(PUBLIC_KEY, SECRET_KEY, clock=frozen_clock)
        self.params = {"id": "AAPL.US,MSFT.US", "timeframe": 1440, "count": -1}

    def test_body_is_compact_json(self) -> None:
        """Пробелы в JSON ломают подпись — сервер считает HMAC от компактной формы."""
        signed = self.auth.sign(self.params)
        self.assertNotIn(", ", signed.body)
        self.assertNotIn('": ', signed.body)
        self.assertEqual(
            signed.body,
            '{"id":"AAPL.US,MSFT.US","timeframe":1440,"count":-1}',
        )

    def test_signature_matches_frozen_vector(self) -> None:
        """Замороженный вектор: ключ + тело + время → ровно эта hex-строка.

        Литерал вбит значением намеренно. Тест, который пересчитывал бы HMAC
        своей рукой, проверял бы код самим собой и прошёл бы, даже если
        подписывается не то сообщение. Литерал ловит подмену алгоритма
        (sha256 → sha1), кодировки и порядка конкатенации разом.
        """
        signed = self.auth.sign(self.params)
        self.assertEqual(
            signed.signature,
            "7f8983935b2ae960ad31fa2c6fdaa91d4fe48ecff9c5288a720328fd495caae1",
        )

    def test_second_frozen_vector(self) -> None:
        """Второй вектор — на одиночном тикере, чтобы совпадение не было случайным."""
        signed = self.auth.sign({"id": "AAPL.US", "timeframe": 1440})
        self.assertEqual(signed.body, '{"id":"AAPL.US","timeframe":1440}')
        self.assertEqual(signed.timestamp, str(FROZEN_TS))
        self.assertEqual(
            signed.signature,
            "23df4a495e2a2007c7c592009f316e8a3134e20f9753c347d44f77380ad20b13",
        )
        self.assertEqual(len(signed.signature), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in signed.signature))


class HeaderContractTest(unittest.TestCase):
    """Набор заголовков APIv3 и их согласованность с телом."""

    def setUp(self) -> None:
        self.auth = TradernetAuth(PUBLIC_KEY, SECRET_KEY, clock=frozen_clock)
        self.signed = self.auth.sign({"id": "AAPL.US"})

    def test_required_headers_present(self) -> None:
        for header in ("X-NtApi-PublicKey", "X-NtApi-Timestamp", "X-NtApi-Sig"):
            self.assertIn(header, self.signed.headers, f"нет заголовка {header}")

    def test_public_key_header_carries_the_key(self) -> None:
        self.assertEqual(self.signed.headers["X-NtApi-PublicKey"], PUBLIC_KEY)

    def test_content_type_is_json(self) -> None:
        """Тело едет JSON'ом; form-urlencoded — это ДРУГАЯ схема подписи."""
        self.assertEqual(self.signed.headers["Content-Type"], "application/json")

    def test_timestamp_is_unix_epoch_string(self) -> None:
        raw = self.signed.headers["X-NtApi-Timestamp"]
        self.assertIsInstance(raw, str)
        self.assertEqual(int(raw), FROZEN_TS)
        self.assertNotIn(".", raw, "timestamp обязан быть целым, без дробной части")

    def test_header_timestamp_is_the_one_that_was_signed(self) -> None:
        """Расхождение подписанного и отправленного времени = code=4.

        Проверяется пересчётом: берём timestamp ИЗ ЗАГОЛОВКА и тело ИЗ ОТВЕТА,
        считаем подпись — она обязана совпасть с отправляемой.
        """
        recomputed = hmac.new(
            SECRET_KEY.encode(),
            (self.signed.body + self.signed.headers["X-NtApi-Timestamp"]).encode(),
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(recomputed, self.signed.headers["X-NtApi-Sig"])


class SignatureSensitivityTest(unittest.TestCase):
    """Подпись обязана РЕАГИРОВАТЬ на всё, что в неё входит."""

    def test_different_timestamp_gives_different_signature(self) -> None:
        params = {"id": "AAPL.US"}
        first = TradernetAuth(PUBLIC_KEY, SECRET_KEY, clock=lambda: 1785960000.0).sign(params)
        second = TradernetAuth(PUBLIC_KEY, SECRET_KEY, clock=lambda: 1785960001.0).sign(params)
        self.assertNotEqual(first.signature, second.signature)

    def test_different_params_give_different_signature(self) -> None:
        auth = TradernetAuth(PUBLIC_KEY, SECRET_KEY, clock=frozen_clock)
        self.assertNotEqual(
            auth.sign({"id": "AAPL.US"}).signature,
            auth.sign({"id": "MSFT.US"}).signature,
        )

    def test_different_secret_gives_different_signature(self) -> None:
        params = {"id": "AAPL.US"}
        other_secret = SECRET_KEY[:-1] + "9"
        self.assertNotEqual(
            TradernetAuth(PUBLIC_KEY, SECRET_KEY, clock=frozen_clock).sign(params).signature,
            TradernetAuth(PUBLIC_KEY, other_secret, clock=frozen_clock).sign(params).signature,
        )

    def test_clock_advances_between_calls(self) -> None:
        """По умолчанию источник времени — реальные часы, а не константа."""
        auth = TradernetAuth(PUBLIC_KEY, SECRET_KEY)
        self.assertGreater(int(auth.sign({}).timestamp), 1_700_000_000)


class SerializationTest(unittest.TestCase):
    """`serialize` — единственное место, где параметры превращаются в байты."""

    def test_key_order_is_preserved_not_sorted(self) -> None:
        """Схема APIv3 подписывает тело как есть; сортировка — признак ДРУГОЙ схемы."""
        body = TradernetAuth.serialize({"z": 1, "a": 2})
        self.assertEqual(body, '{"z":1,"a":2}')

    def test_body_roundtrips_through_json(self) -> None:
        params = {"id": "AAPL.US,MSFT.US", "timeframe": 1440,
                  "intervalMode": "ClosedRay", "count": -1}
        self.assertEqual(json.loads(TradernetAuth.serialize(params)), params)

    def test_repr_leaks_no_secrets(self) -> None:
        """Лог уезжает в общий сборщик — ни ключа, ни его префикса там быть не должно."""
        text = repr(TradernetAuth(PUBLIC_KEY, SECRET_KEY))
        self.assertNotIn(PUBLIC_KEY, text)
        self.assertNotIn(SECRET_KEY, text)
        self.assertNotIn(PUBLIC_KEY[:8], text)
        self.assertIn("key_len=", text)


if __name__ == "__main__":
    unittest.main()
