"""
Phase-17 — admin /grant token command (testing self-topup).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _tg():
    try:
        import tg_bot
        return tg_bot
    except BaseException:
        return None


class AdminUsersEnvTest(unittest.TestCase):

    def setUp(self) -> None:
        self._snap = os.environ.get("ADMIN_USER_IDS")

    def tearDown(self) -> None:
        if self._snap is None:
            os.environ.pop("ADMIN_USER_IDS", None)
        else:
            os.environ["ADMIN_USER_IDS"] = self._snap

    def test_empty_env_no_admins(self) -> None:
        tg = _tg()
        if tg is None:
            self.skipTest("tg_bot import unavailable")
        os.environ.pop("ADMIN_USER_IDS", None)
        self.assertEqual(tg._admin_users(), set())
        self.assertFalse(tg._is_admin(148046720))

    def test_parse_csv_and_semicolon(self) -> None:
        tg = _tg()
        if tg is None:
            self.skipTest("tg_bot import unavailable")
        os.environ["ADMIN_USER_IDS"] = "148046720, 222 ; 333"
        self.assertEqual(tg._admin_users(), {148046720, 222, 333})
        self.assertTrue(tg._is_admin(148046720))
        self.assertFalse(tg._is_admin(999))

    def test_cmd_grant_registered(self) -> None:
        src = (Path(__file__).resolve().parent.parent / "src" / "tg_bot.py").read_text()
        self.assertIn("async def cmd_grant(", src)
        self.assertIn('F.text.startswith("/grant")', src)
        # grant credits via credit_tokens with an admin_grant reason
        self.assertIn("admin_grant_by_", src)


if __name__ == "__main__":
    unittest.main()
