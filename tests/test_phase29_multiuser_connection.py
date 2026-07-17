"""
Phase 29 — multi-user connection-mode robustness (Fixes A1 + B + C).

Regression tests for the 2026-07-14 «2nd user → demo portfolio» incident and
the follow-up hardening (2026-07-16):

* Fix A1 — `connection_mode` lives in its own `user_connection` table written
  via UPSERT, so the write physically cannot be lost to a missing
  `user_profiles` row (the old UPDATE silently no-oped → user defaulted to
  'template' → was served a demo portfolio).  `init_db` idempotently backfills
  legacy 'freedom' values; legacy 'template' is NOT backfilled (it was the old
  column's DEFAULT and proves no explicit choice).
* Fix B — stored vault keys are proof the user linked their broker:
  `SecureVault.has_user` (existence check, никогда не расшифровывает) +
  `tg_bot._resolve_portfolio_source` recovers 'freedom' from keys and
  self-heals the stored mode.  Ключи всегда побеждают (product decision).
* Fix C — billing: демо-отчёты бесплатны (`_effective_cost` → 0); an
  undetermined source ('undetermined') must never produce a paid report.
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


class _DBTestBase(unittest.TestCase):
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

    async def _raw_sql(self, sql: str, params=()):
        import aiosqlite
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(sql, params)
            await conn.commit()


class SaveConnectionModeTest(_DBTestBase):
    """Fix A1 — the mode physically cannot be silently lost."""

    def test_persists_when_profile_exists(self):
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

    def test_persists_without_profile_row(self):
        """THE incident (2026-07-14): no user_profiles row → the old UPDATE
        lost the write and the user silently landed in demo.  The UPSERT into
        user_connection must persist regardless."""
        tid = 222  # never onboarded → no user_profiles row
        ok = asyncio.run(self.db.save_connection_mode(tid, "freedom"))
        self.assertTrue(ok)
        self.assertEqual(asyncio.run(self.db.get_connection_mode(tid)), "freedom")

    def test_explicit_getter_distinguishes_default_from_choice(self):
        """Fix C relies on: None = «не выбирал», 'template' = явный выбор."""
        tid = 333
        self.assertIsNone(asyncio.run(self.db.get_connection_mode_explicit(tid)))
        asyncio.run(self.db.save_connection_mode(tid, "template"))
        self.assertEqual(
            asyncio.run(self.db.get_connection_mode_explicit(tid)), "template")
        asyncio.run(self.db.save_connection_mode(tid, "freedom"))
        self.assertEqual(
            asyncio.run(self.db.get_connection_mode_explicit(tid)), "freedom")

    def test_get_connection_mode_defaults_to_template(self):
        """Compat wrapper: unknown user → 'template' (unchanged contract)."""
        self.assertEqual(asyncio.run(self.db.get_connection_mode(999)), "template")


class BackfillMigrationTest(_DBTestBase):
    """init_db backfill: legacy user_profiles.connection_mode → user_connection."""

    def test_backfill_copies_freedom_only_and_is_idempotent(self):
        tid_free, tid_tmpl = 41, 42
        self._make_profile(tid_free)
        self._make_profile(tid_tmpl)
        # Simulate LEGACY state: mode stored only in the old column, no
        # user_connection rows (pre-fix deployments never wrote that table).
        asyncio.run(self._raw_sql(
            "UPDATE user_profiles SET connection_mode='freedom' WHERE telegram_id=?",
            (tid_free,)))
        asyncio.run(self._raw_sql(
            "UPDATE user_profiles SET connection_mode='template' WHERE telegram_id=?",
            (tid_tmpl,)))
        asyncio.run(self._raw_sql("DELETE FROM user_connection"))

        asyncio.run(self.db.init_db())   # migration run
        self.assertEqual(
            asyncio.run(self.db.get_connection_mode_explicit(tid_free)), "freedom")
        # legacy 'template' was the column DEFAULT → carries no signal → None
        self.assertIsNone(
            asyncio.run(self.db.get_connection_mode_explicit(tid_tmpl)))

        asyncio.run(self.db.init_db())   # idempotent re-run — no dupes/errors
        self.assertEqual(
            asyncio.run(self.db.get_connection_mode_explicit(tid_free)), "freedom")

    def test_backfill_never_overwrites_newer_value(self):
        tid = 43
        self._make_profile(tid)
        asyncio.run(self._raw_sql(
            "UPDATE user_profiles SET connection_mode='freedom' WHERE telegram_id=?",
            (tid,)))
        # The user has ALREADY explicitly switched to template post-fix.
        asyncio.run(self.db.save_connection_mode(tid, "template"))
        asyncio.run(self.db.init_db())   # backfill must not clobber it
        self.assertEqual(
            asyncio.run(self.db.get_connection_mode_explicit(tid)), "template")


class VaultHasUserTest(unittest.TestCase):
    """Fix B — SecureVault.has_user: existence check without decryption."""

    def setUp(self):
        from cryptography.fernet import Fernet
        self.tmp = tempfile.mkdtemp()
        self.vault_path = os.path.join(self.tmp, "vault.db")
        self.key_a = Fernet.generate_key().decode()
        self.key_b = Fernet.generate_key().decode()
        os.environ["FINTECH_MASTER_KEY"] = self.key_a
        import finance.security as security
        importlib.reload(security)
        self.security = security

    def tearDown(self):
        os.environ.pop("FINTECH_MASTER_KEY", None)

    def test_has_user_true_after_save_false_for_unknown(self):
        vault = self.security.SecureVault(db_name=self.vault_path)
        vault.save_user_keys("777", "login", "api-key", "secret")
        self.assertTrue(vault.has_user("777"))
        self.assertFalse(vault.has_user("888"))

    def test_has_user_survives_master_key_rotation(self):
        """has_user must NOT decrypt: it stays True even when the master key
        rotated past its grace window (get_user_keys raises then)."""
        vault_a = self.security.SecureVault(db_name=self.vault_path)
        vault_a.save_user_keys("777", "login", "api-key", "secret")
        # Rotate: only key B remains active.
        os.environ["FINTECH_MASTER_KEY"] = self.key_b
        vault_b = self.security.SecureVault(db_name=self.vault_path)
        self.assertTrue(vault_b.has_user("777"))
        with self.assertRaises(self.security.MasterKeyRotatedError):
            vault_b.get_user_keys("777")


def _import_tg_bot():
    # tg_bot reads RAMP_BOT_TOKEN at import time — stub it for unit tests
    # (no Telegram call is ever made here).
    os.environ.setdefault("RAMP_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")
    try:
        import tg_bot
        return tg_bot
    except Exception:
        return None


class ResolvePortfolioSourceTest(unittest.TestCase):
    """Fix B + C — the source resolver: ключи всегда побеждают; None ≠ демо."""

    @classmethod
    def setUpClass(cls):
        cls.tg = _import_tg_bot()

    def setUp(self):
        if self.tg is None:
            self.skipTest("tg_bot import unavailable")
        self._orig = (self.tg._has_vault_keys_sync,
                      self.tg.get_connection_mode_explicit,
                      self.tg.save_connection_mode)
        self.saved_modes: list[tuple[int, str]] = []

    def tearDown(self):
        if self.tg is not None:
            (self.tg._has_vault_keys_sync,
             self.tg.get_connection_mode_explicit,
             self.tg.save_connection_mode) = self._orig

    def _patch(self, *, has_keys: bool, stored: str | None):
        self.tg._has_vault_keys_sync = lambda uid: has_keys

        async def fake_get(uid):
            return stored

        async def fake_save(uid, mode):
            self.saved_modes.append((uid, mode))
            return True

        self.tg.get_connection_mode_explicit = fake_get
        self.tg.save_connection_mode = fake_save

    def _resolve(self):
        return asyncio.run(self.tg._resolve_portfolio_source(1001))

    def test_keys_present_mode_lost_recovers_freedom(self):
        """The incident shape: keys saved, mode lost → freedom + self-heal."""
        self._patch(has_keys=True, stored=None)
        source, stored = self._resolve()
        self.assertEqual(source, "freedom")
        self.assertIn((1001, "freedom"), self.saved_modes)

    def test_keys_always_win_over_stored_template(self):
        """Product decision 2026-07-16: ключи всегда побеждают."""
        self._patch(has_keys=True, stored="template")
        source, _ = self._resolve()
        self.assertEqual(source, "freedom")
        self.assertIn((1001, "freedom"), self.saved_modes)

    def test_keys_present_mode_freedom_no_rewrite(self):
        self._patch(has_keys=True, stored="freedom")
        source, _ = self._resolve()
        self.assertEqual(source, "freedom")
        self.assertEqual(self.saved_modes, [])   # уже freedom — не переписываем

    def test_no_keys_stored_freedom_stays_freedom(self):
        """Vault пуст, но режим freedom → freedom-ветка покажет re-link
        (НЕ тихое демо)."""
        self._patch(has_keys=False, stored="freedom")
        source, _ = self._resolve()
        self.assertEqual(source, "freedom")

    def test_no_keys_explicit_template_is_demo(self):
        self._patch(has_keys=False, stored="template")
        source, _ = self._resolve()
        self.assertEqual(source, "demo")

    def test_no_keys_no_choice_is_undetermined(self):
        """Fix C: «демо по умолчанию» больше не существует — источник
        неопределён и платный отчёт не строится."""
        self._patch(has_keys=False, stored=None)
        source, _ = self._resolve()
        self.assertEqual(source, "undetermined")


class UndeterminedSourceRecoveryTest(unittest.TestCase):
    """2026-07-16: the undetermined-source guard MUST offer the connection
    keyboard inline — a returning user has no other path to the connection
    screen (/start skips it once a profile exists), so pointing at /start
    created a dead loop (2nd user stuck: /start → меню → «не выбран» → /start…).
    Source-level assertions, phase-15 style."""

    @classmethod
    def setUpClass(cls):
        cls.src = (SRC / "tg_bot.py").read_text(encoding="utf-8")

    def test_cb_confirm_guard_offers_source_keyboard(self):
        marker = "⚠️ *Источник портфеля не выбран.*"   # the message literal
        self.assertIn(marker, self.src)
        block = self.src.split(marker, 1)[1][:600]
        self.assertIn("kb_connect_choice()", block,
                      "the undetermined-source message must attach the "
                      "connection keyboard (one-tap recovery)")

    def test_cmd_start_heals_undetermined_source(self):
        start_body = self.src.split("async def cmd_start", 1)[1].split(
            "\nasync def ", 1)[0]
        self.assertIn("_resolve_portfolio_source", start_body)
        self.assertIn('"undetermined"', start_body)
        self.assertIn("kb_connect_choice()", start_body,
                      "/start for a returning user must offer the source "
                      "choice when the source is undetermined")


class EffectiveCostTest(unittest.TestCase):
    """Fix C — демо бесплатно, живой источник по тарифу."""

    @classmethod
    def setUpClass(cls):
        cls.tg = _import_tg_bot()

    def setUp(self):
        if self.tg is None:
            self.skipTest("tg_bot import unavailable")

    def test_demo_is_free_for_every_tier(self):
        for tier in self.tg.TIER_COST:
            self.assertEqual(self.tg._effective_cost(tier, "demo"), 0)

    def test_freedom_costs_tariff(self):
        for tier, cost in self.tg.TIER_COST.items():
            self.assertEqual(self.tg._effective_cost(tier, "freedom"), cost)


if __name__ == "__main__":
    unittest.main()
