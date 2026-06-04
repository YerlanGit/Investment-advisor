"""
Phase 12 — beta-readiness safety harness (whitelist + single-flight +
                                          persistent DB path).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _import_tg_bot():
    """Defer tg_bot import behind a guard — cryptography pyo3 panics in
    some CI sandboxes."""
    try:
        import tg_bot  # noqa: F401
        return tg_bot
    except BaseException:
        return None


class WhitelistEnvParseTest(unittest.TestCase):
    """`_allowed_users` reads TG_ALLOWED_USERS, supports comma & semicolon."""

    def setUp(self) -> None:
        self._snap = os.environ.get("TG_ALLOWED_USERS")
        os.environ.pop("TG_ALLOWED_USERS", None)

    def tearDown(self) -> None:
        if self._snap is None:
            os.environ.pop("TG_ALLOWED_USERS", None)
        else:
            os.environ["TG_ALLOWED_USERS"] = self._snap

    def test_empty_env_returns_none(self) -> None:
        tg = _import_tg_bot()
        if tg is None:
            self.skipTest("tg_bot import unavailable")
        tg._ALLOWED_USERS_CACHE = None
        self.assertIsNone(tg._allowed_users())

    def test_csv_parsed(self) -> None:
        tg = _import_tg_bot()
        if tg is None:
            self.skipTest("tg_bot import unavailable")
        os.environ["TG_ALLOWED_USERS"] = "111, 222 ; 333"
        tg._ALLOWED_USERS_CACHE = None
        self.assertEqual(tg._allowed_users(), {111, 222, 333})


class SingleFlightTest(unittest.TestCase):

    def test_acquire_release_cycle(self) -> None:
        tg = _import_tg_bot()
        if tg is None:
            self.skipTest("tg_bot import unavailable")
        self.assertTrue(tg._try_acquire_user_slot(99999))
        # Second acquire by same user is refused.
        self.assertFalse(tg._try_acquire_user_slot(99999))
        # Different user is OK in parallel.
        self.assertTrue(tg._try_acquire_user_slot(88888))
        tg._release_user_slot(99999)
        self.assertTrue(tg._try_acquire_user_slot(99999))
        tg._release_user_slot(99999)
        tg._release_user_slot(88888)


class DBPathEnvTest(unittest.TestCase):
    """TOKENOMICS_DB_PATH env overrides the default path."""

    def test_env_overrides_default(self) -> None:
        # Re-import with env set.
        os.environ["TOKENOMICS_DB_PATH"] = "/tmp/test-tokenomics.db"
        import importlib, db_tokenomics
        importlib.reload(db_tokenomics)
        self.assertEqual(str(db_tokenomics.DB_PATH), "/tmp/test-tokenomics.db")
        del os.environ["TOKENOMICS_DB_PATH"]
        importlib.reload(db_tokenomics)


if __name__ == "__main__":
    unittest.main()
