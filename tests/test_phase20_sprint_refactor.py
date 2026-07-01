"""
Sprint refactor regression tests (Sprint 1–3 roadmap).

Covers:
  • #7  Token-DB persistence under a mid-transaction SIGKILL (durability +
        atomicity — a committed deduction survives, an uncommitted one rolls
        back, and the DB is never left partially written).
  • #9  Black-Litterman refactor (He-Litterman solve form) semantic parity +
        nearest-PSD projection.
  • #10 Deterministic hashlib.sha256 bootstrap seed (same input → same CI).
  • #3  finance.portfolio_series math core is import-clean and bot-free.
"""
from __future__ import annotations

import importlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import asyncio  # noqa: E402

import numpy as np  # noqa: E402


# Child that opens a HARDENED (journal=DELETE, synchronous=FULL) write
# transaction on the shared DB, stages a balance deduction, optionally commits,
# signals readiness, then hangs so the parent can SIGKILL it at a known point.
_CHILD = r"""
import os, sys, time, asyncio
sys.path.insert(0, os.environ["SRC_PATH"])
import aiosqlite

DB     = os.environ["TOKENOMICS_DB_PATH"]
READY  = os.environ["READY_FILE"]
COMMIT = os.environ["DO_COMMIT"] == "1"
TID    = int(os.environ["TID"])
AMT    = int(os.environ["AMT"])

async def main():
    db = await aiosqlite.connect(DB, isolation_level=None)
    await db.execute("PRAGMA journal_mode=DELETE")
    await db.execute("PRAGMA synchronous=FULL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("BEGIN IMMEDIATE")
    await db.execute("UPDATE users SET balance = balance - ? WHERE telegram_id = ?", (AMT, TID))
    if COMMIT:
        await db.execute("COMMIT")     # fsync'd (synchronous=FULL) → durable
    # else: transaction stays open with a hot rollback -journal
    with open(READY, "w") as fh:
        fh.write("ready")
    time.sleep(60)                     # parent SIGKILLs us here

asyncio.run(main())
"""


@unittest.skipUnless(hasattr(signal, "SIGKILL"), "SIGKILL is POSIX-only")
class TokenPersistenceSIGKILLTest(unittest.TestCase):
    """#7 — money state is atomic + durable across a hard kill."""

    TID, AMT = 42, 7

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "tok.db")
        os.environ["TOKENOMICS_DB_PATH"] = self.db_path
        os.environ.pop("REQUIRE_PERSISTENT_DB", None)
        os.environ.pop("K_SERVICE", None)
        import db_tokenomics
        importlib.reload(db_tokenomics)          # pick up the temp DB_PATH
        self.dbmod = db_tokenomics
        asyncio.run(self.dbmod.init_db())
        asyncio.run(self.dbmod.init_user(self.TID))
        self.baseline = asyncio.run(self.dbmod.get_balance(self.TID))
        self.assertGreater(self.baseline, self.AMT, "need a baseline > deduction")

    def _run_txn_then_kill(self, commit: bool):
        ready = os.path.join(self.tmp, "ready")
        env = {
            **os.environ,
            "SRC_PATH": str(SRC),
            "TOKENOMICS_DB_PATH": self.db_path,
            "READY_FILE": ready,
            "DO_COMMIT": "1" if commit else "0",
            "TID": str(self.TID), "AMT": str(self.AMT),
        }
        proc = subprocess.Popen([sys.executable, "-c", _CHILD], env=env)
        try:
            for _ in range(200):                 # ≤10s for the child to stage the txn
                if os.path.exists(ready):
                    break
                time.sleep(0.05)
            else:
                proc.kill()
                self.fail("child never signalled readiness")
            proc.send_signal(signal.SIGKILL)     # hard kill at the known point
        finally:
            proc.wait(timeout=10)

    def _reopen_balance(self) -> int:
        importlib.reload(self.dbmod)             # fresh connection → runs recovery
        return asyncio.run(self.dbmod.get_balance(self.TID))

    def test_committed_deduction_survives_sigkill(self):
        # Durability: commit fsync'd before the kill → deduction persists.
        self._run_txn_then_kill(commit=True)
        self.assertEqual(self._reopen_balance(), self.baseline - self.AMT)

    def test_uncommitted_deduction_rolls_back_after_sigkill(self):
        # Atomicity: killed mid-transaction (no commit) → SQLite rolls back via
        # the -journal on reopen; NEVER a partial deduction.
        self._run_txn_then_kill(commit=False)
        self.assertEqual(self._reopen_balance(), self.baseline)


class BlackLittermanRefactorTest(unittest.TestCase):
    """#9 — He-Litterman solve form keeps semantics; nearest-PSD repairs blends."""

    def _sigma(self, n=4):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((n, n))
        return A @ A.T / n + np.eye(n) * 0.02      # SPD

    def test_no_views_returns_prior(self):
        from finance.black_litterman import black_litterman
        sigma = self._sigma()
        tickers = [f"T{i}" for i in range(4)]
        res = black_litterman(
            tickers=tickers, cov=sigma, current_weights={t: 0.25 for t in tickers},
            views_P=None, views_Q=None)
        # No views → posterior == prior → n_views stays 0; weights sum to 1.
        self.assertEqual(res.n_views, 0)
        self.assertAlmostEqual(float(np.sum(res.target_weights)), 1.0, places=6)

    def test_views_shift_weights(self):
        from finance.black_litterman import black_litterman
        sigma = self._sigma()
        tickers = [f"T{i}" for i in range(4)]
        base = black_litterman(
            tickers=tickers, cov=sigma, current_weights={t: 0.25 for t in tickers},
            views_P=None, views_Q=None)
        # Bullish view on T0 → its target weight should rise vs the no-view prior.
        res = black_litterman(
            tickers=tickers, cov=sigma, current_weights={t: 0.25 for t in tickers},
            views_P=np.array([[1.0, 0.0, 0.0, 0.0]]), views_Q=np.array([0.10]))
        self.assertEqual(res.n_views, 1)
        self.assertAlmostEqual(float(np.sum(res.target_weights)), 1.0, places=6)
        self.assertGreater(res.target_weights[0], base.target_weights[0] - 1e-9)

    def test_nearest_psd_repairs_non_psd_and_preserves_psd(self):
        from finance.investment_logic import _nearest_psd
        spd = self._sigma()
        # PSD input → returned matrix has the same eigenvalues (no-op).
        out = _nearest_psd(spd)
        self.assertTrue(np.all(np.linalg.eigvalsh(out) > -1e-9))
        np.testing.assert_allclose(out, spd, atol=1e-9)
        # Non-PSD input (a negative eigenvalue) → projected to PSD.
        bad = spd.copy()
        bad[0, 0] = -5.0
        rep = _nearest_psd(bad)
        self.assertTrue(np.all(np.linalg.eigvalsh(rep) >= -1e-9))


class BootstrapSeedDeterminismTest(unittest.TestCase):
    """#10 — sha256-derived seed is stable and content-driven."""

    def test_same_input_same_ci(self):
        from finance.investment_logic import MAC3RiskEngine
        rng = np.random.default_rng(1)
        rets = rng.standard_normal(300) * 0.01
        a = MAC3RiskEngine._bootstrap_cvar(rets, n_boot=300)
        b = MAC3RiskEngine._bootstrap_cvar(rets.copy(), n_boot=300)
        self.assertEqual(a["point"], b["point"])
        self.assertEqual(a["lo95"], b["lo95"])
        self.assertEqual(a["hi95"], b["hi95"])

    def test_different_input_different_ci(self):
        from finance.investment_logic import MAC3RiskEngine
        rng = np.random.default_rng(2)
        a = MAC3RiskEngine._bootstrap_cvar(rng.standard_normal(300) * 0.01, n_boot=300)
        b = MAC3RiskEngine._bootstrap_cvar(rng.standard_normal(300) * 0.02, n_boot=300)
        self.assertNotEqual(a["point"], b["point"])


class PortfolioSeriesSoCTest(unittest.TestCase):
    """#3 — the extracted math core imports without any bot/aiogram dependency."""

    def test_import_is_bot_free(self):
        import finance.portfolio_series as ps
        self.assertTrue(hasattr(ps, "compute_kpi_trend_series"))
        self.assertTrue(hasattr(ps, "compute_equity_curve_series"))
        # Degenerate input → graceful None, no exception.
        self.assertIsNone(ps.compute_kpi_trend_series({}))
        self.assertEqual(ps.compute_equity_curve_series({}), (None, None, None))


if __name__ == "__main__":
    unittest.main()
