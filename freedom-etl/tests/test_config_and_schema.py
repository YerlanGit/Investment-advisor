"""Конфигурация и DDL: клампы лимитов, fail-soft разбор env, форма схемы.

Почему клампы проверяются тестом, а не «мы же написали в README»
────────────────────────────────────────────────────────────────
`BATCH_DELAY_SECONDS` и `BATCH_SIZE` — это не настройки удобства, а условия
работы с чужим API. Единственная переменная окружения, выставленная «чтобы
быстрее», стоит бана на весь ключ, а ключ у сервиса ОДИН. Поэтому нижняя и
верхняя границы — исполняемый гейт.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from config.config import Settings, env_bool, env_float, env_int, env_str
from db.models import (
    HYPERTABLE_STATEMENTS,
    SCHEMA_STATEMENTS,
    UPSERT_CANDLES_SQL,
    UPSERT_SYNC_STATUS_SQL,
)


def with_env(**values):
    """Подменить окружение только заданными ключами, остальное вычистить."""
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("TRADERNET_", "BATCH_", "DB_", "SCHEDULE_",
                                  "BACKOFF_", "SEED_", "INITIAL_", "RESYNC_",
                                  "INCLUDE_", "MAX_RETRIES", "REQUEST_",
                                  "RUN_ON_", "LOG_", "FREEDOM_"))}
    clean.update(values)
    return mock.patch.dict(os.environ, clean, clear=True)


class EnvPrimitivesTest(unittest.TestCase):

    def test_garbage_int_falls_back_instead_of_crashing(self) -> None:
        """Опечатка в env не должна ронять ИМПОРТ — контейнер не стартовал бы вовсе."""
        with mock.patch.dict(os.environ, {"X": "50шт"}):
            self.assertEqual(env_int("X", 50), 50)

    def test_garbage_float_falls_back(self) -> None:
        with mock.patch.dict(os.environ, {"X": "два"}):
            self.assertEqual(env_float("X", 2.0), 2.0)

    def test_empty_string_is_treated_as_unset(self) -> None:
        with mock.patch.dict(os.environ, {"X": "   "}):
            self.assertEqual(env_int("X", 7), 7)
            self.assertEqual(env_str("X", "def"), "def")

    def test_clamps_are_applied(self) -> None:
        with mock.patch.dict(os.environ, {"X": "999"}):
            self.assertEqual(env_int("X", 5, lo=1, hi=50), 50)
        with mock.patch.dict(os.environ, {"X": "-4"}):
            self.assertEqual(env_int("X", 5, lo=1, hi=50), 1)

    def test_bool_accepts_common_spellings(self) -> None:
        for raw in ("1", "true", "TRUE", "Yes", "on"):
            with mock.patch.dict(os.environ, {"X": raw}):
                self.assertTrue(env_bool("X", False), raw)
        for raw in ("0", "false", "no", "off", "nonsense"):
            with mock.patch.dict(os.environ, {"X": raw}):
                self.assertFalse(env_bool("X", True), raw)


class SettingsTest(unittest.TestCase):

    def test_defaults_match_the_specification(self) -> None:
        with with_env():
            s = Settings.from_env()
        self.assertEqual(s.batch_size, 50)
        self.assertEqual(s.batch_delay_seconds, 2.0)
        self.assertEqual(s.initial_lookback_days, 1825)
        self.assertEqual(s.schedule_hour, 3)
        self.assertEqual(s.schedule_timezone, "America/New_York")
        self.assertFalse(s.include_current_day, "текущий незакрытый день по умолчанию не пишется")

    def test_batch_size_cannot_exceed_fifty(self) -> None:
        """Потолок API. Выше него брокер молча режет ответ — хуже явного 429."""
        with with_env(BATCH_SIZE="500"):
            self.assertEqual(Settings.from_env().batch_size, 50)

    def test_batch_delay_cannot_go_below_two_seconds(self) -> None:
        """🔴 Уменьшить паузу через env невозможно — это условие работы с API."""
        with with_env(BATCH_DELAY_SECONDS="0.01"):
            self.assertEqual(Settings.from_env().batch_delay_seconds, 2.0)

    def test_credentials_accept_both_naming_conventions(self) -> None:
        """В репозитории исторически живут оба имени — принимаем оба."""
        with with_env(FREEDOM_API_KEY="pub", FREEDOM_API_SECRET="sec"):
            s = Settings.from_env()
        self.assertEqual((s.public_key, s.secret_key), ("pub", "sec"))

    def test_tradernet_names_win_over_freedom_names(self) -> None:
        with with_env(TRADERNET_PUBLIC_KEY="a", FREEDOM_API_KEY="b",
                      TRADERNET_SECRET_KEY="c", FREEDOM_API_SECRET="d"):
            s = Settings.from_env()
        self.assertEqual((s.public_key, s.secret_key), ("a", "c"))

    def test_seed_tickers_are_split_and_normalised(self) -> None:
        with with_env(SEED_TICKERS=" aapl.us, msft.us ;tsla.us,, "):
            self.assertEqual(Settings.from_env().seed_tickers,
                             ("AAPL.US", "MSFT.US", "TSLA.US"))

    def test_require_credentials_names_what_is_missing(self) -> None:
        """Без явной проверки сервис ушёл бы в сеть и собрал code=12 по каждому батчу."""
        with with_env():
            with self.assertRaises(RuntimeError) as ctx:
                Settings.from_env().require_credentials()
        self.assertIn("TRADERNET_PUBLIC_KEY", str(ctx.exception))
        self.assertIn("TRADERNET_SECRET_KEY", str(ctx.exception))

    def test_require_credentials_passes_when_set(self) -> None:
        with with_env(TRADERNET_PUBLIC_KEY="p", TRADERNET_SECRET_KEY="s"):
            Settings.from_env().require_credentials()   # не должно бросить

    def test_describe_leaks_no_secrets(self) -> None:
        with with_env(TRADERNET_PUBLIC_KEY="pub_supersecret",
                      TRADERNET_SECRET_KEY="sec_supersecret"):
            text = Settings.from_env().describe()
        self.assertNotIn("pub_supersecret", text)
        self.assertNotIn("sec_supersecret", text)
        self.assertIn("key_len=", text)

    def test_settings_are_immutable(self) -> None:
        """Конфигурация читается один раз на старте и дальше не меняется."""
        with with_env():
            s = Settings.from_env()
        with self.assertRaises(Exception):
            s.batch_size = 1        # type: ignore[misc]


class SchemaTest(unittest.TestCase):
    """DDL проверяется как текст: живой PostgreSQL для этих утверждений не нужен."""

    def test_primary_key_is_the_idempotency_contract(self) -> None:
        ddl = "\n".join(SCHEMA_STATEMENTS)
        self.assertIn("PRIMARY KEY (ticker, trade_date)", ddl)

    def test_close_is_not_null(self) -> None:
        """На цене закрытия держатся все риск-метрики — пустой она быть не может."""
        ddl = "\n".join(SCHEMA_STATEMENTS)
        self.assertIn("close       NUMERIC(12, 4) NOT NULL", ddl)

    def test_required_tables_are_declared(self) -> None:
        ddl = "\n".join(SCHEMA_STATEMENTS)
        for table in ("daily_candles", "sync_status", "tracked_tickers"):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", ddl)

    def test_all_ddl_is_idempotent(self) -> None:
        """Схема применяется на КАЖДОМ старте — повтор не должен ничего ломать."""
        for statement in SCHEMA_STATEMENTS:
            self.assertIn("IF NOT EXISTS", statement,
                          f"неидемпотентный оператор: {statement[:60]}")

    def test_hypertable_partitions_on_trade_date(self) -> None:
        sql = "\n".join(HYPERTABLE_STATEMENTS)
        self.assertIn("create_hypertable", sql)
        self.assertIn("'trade_date'", sql)
        self.assertIn("if_not_exists       => TRUE", sql)

    def test_upsert_updates_close_on_conflict(self) -> None:
        """ТЗ: ON CONFLICT (ticker, trade_date) DO UPDATE SET close = EXCLUDED.close."""
        self.assertIn("ON CONFLICT (ticker, trade_date) DO UPDATE", UPSERT_CANDLES_SQL)
        self.assertIn("close  = EXCLUDED.close", UPSERT_CANDLES_SQL)

    def test_upsert_does_not_erase_existing_ohlc_with_nulls(self) -> None:
        """Повторный проход, отдавший только закрытие, не должен обнулять O/H/L."""
        for column in ("open", "high", "low", "volume"):
            self.assertIn(f"COALESCE(EXCLUDED.{column}", UPSERT_CANDLES_SQL,
                          f"{column} затирается NULL при ресинке")

    def test_sync_cursor_can_only_move_forward(self) -> None:
        """🔴 Откат курсора = бесконечная перезагрузка уже лежащей истории."""
        self.assertIn("GREATEST", UPSERT_SYNC_STATUS_SQL)


if __name__ == "__main__":
    unittest.main()
