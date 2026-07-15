"""
Phase 29 — multi-user connection-mode robustness.

Regression tests for the 2026-07-14 «2nd user → demo portfolio» incident.

`save_connection_mode` persists the user's portfolio source ('template' |
'freedom') as a column on the `user_profiles` row.  It used to run a bare
`UPDATE ... WHERE telegram_id = ?` and IGNORE the result: if the row did not
exist yet, the UPDATE matched nothing, the write was lost silently, and
`get_connection_mode` then returned the default 'template' — so a user who
believed they had linked their broker was quietly served a DEMO portfolio.

The fix keeps the normal-flow behaviour identical (row exists → UPDATE →
persisted) but no longer fails silently: it returns True/False and logs a
WARNING when nothing was written.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SaveConnectionModeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "tok.db")
        os.environ["TOKENOMICS_DB_PATH"] = self.db_path
        os.environ.pop("REQUIRE_PERSISTENT_DB", None)
        os.environ.pop("K_SERVICE", None)
        import db_tokenomics
        importlib.reload(db_tokenomics)          # pick up the temp DB_PATH
        self.db = db_tokenomics
        asyncio.run(self.db.init_db())

    def tearDown(self):
        os.environ.pop("TOKENOMICS_DB_PATH", None)

    def _make_profile(self, tid: int):
        asyncio.run(self.db.save_profile(
            telegram_id       = tid,
            score             = 50,
            profile_name      = "Balanced",
            target_volatility = 0.16,
            target_te         = 0.05,
            selected_assets   = ["AAPL"],
            limits_dict       = {"max": 1.0},
            benchmark_ticker  = "SPY.US",
        ))

    # ── normal flow — behaviour is unchanged ────────────────────────────────
    def test_persists_when_profile_exists(self):
        """Row exists (created at onboarding) → mode is saved and readable."""
        tid = 111
        self._make_profile(tid)
        ok = asyncio.run(self.db.save_connection_mode(tid, "freedom"))
        self.assertTrue(ok)
        self.assertEqual(asyncio.run(self.db.get_connection_mode(tid)), "freedom")

    def test_switch_back_to_template(self):
        tid = 112
        self._make_profile(tid)
        asyncio.run(self.db.save_connection_mode(tid, "freedom"))
        ok = asyncio.run(self.db.save_connection_mode(tid, "template"))
        self.assertTrue(ok)
        self.assertEqual(asyncio.run(self.db.get_connection_mode(tid)), "template")

    # ── the incident — a lost write must NOT be silent ──────────────────────
    def test_no_profile_row_is_surfaced_not_silent(self):
        """No profile row → write is lost; must return False + WARN (not silent)."""
        tid = 222  # never onboarded → no user_profiles row
        with self.assertLogs("db_tokenomics", level="WARNING") as cm:
            ok = asyncio.run(self.db.save_connection_mode(tid, "freedom"))
        self.assertFalse(ok)
        # Mode was NOT persisted → default 'template' (exactly the demo state
        # the 2nd user hit on 2026-07-14).
        self.assertEqual(asyncio.run(self.db.get_connection_mode(tid)), "template")
        self.assertTrue(
            any("НЕ сохранён" in line for line in cm.output),
            f"expected a lost-write WARNING, got: {cm.output}",
        )

    def test_get_connection_mode_defaults_to_template(self):
        """Unknown user → 'template' (unchanged default)."""
        self.assertEqual(asyncio.run(self.db.get_connection_mode(999)), "template")


if __name__ == "__main__":
    unittest.main()
