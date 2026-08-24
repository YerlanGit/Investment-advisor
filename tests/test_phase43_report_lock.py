"""A-4 · Межпроцессный single-flight отчётов (аренда в SQLite).

Почему это ПРЕДУСЛОВИЕ снятия `--max-instances=1`, а не косметика:

    токен списывается ПОСЛЕ доставки отчёта (`tg_bot` CHECKPOINT 3).

Значит два инстанса, взявшие одного пользователя параллельно, доставляют ДВА
отчёта и списывают ОДИН токен — второй `deduct_tokens` честно падает
`InsufficientFundsError`, но отчёт уже отправлен. Прежний гард жил в `set()`
внутри процесса и второй инстанс о нём не знал.

Здесь зафиксировано:
  1. Атомарность: из N одновременных претендентов выигрывает РОВНО один.
  2. Аренда: инстанс, умерший посреди расчёта, не запирает пользователя
     навсегда — просроченный слот перехватывается.
  3. Владение: запоздавший release умирающего инстанса НЕ выбивает нового
     держателя.
  4. Деградация: при недоступной БД гард возвращается ровно к прежнему
     внутрипроцессному поведению, а не отказывает пользователю в отчёте.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fresh_db():
    """Изолированная БД на тест — модуль читает путь при импорте."""
    tmp = tempfile.mkdtemp(prefix="ramp-lock-")
    return Path(tmp) / "tokenomics.db"


class ReportLockTestBase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        import db_tokenomics as db
        self.db = db
        self._orig_path = db.DB_PATH
        db.DB_PATH = _fresh_db()
        await db.init_db()

    async def asyncTearDown(self):
        self.db.DB_PATH = self._orig_path


# ──────────────────────────────────────────────────────────────────────────
# 1. Атомарность
# ──────────────────────────────────────────────────────────────────────────
class AtomicityTest(ReportLockTestBase):

    async def test_second_acquire_loses(self):
        self.assertTrue(await self.db.acquire_report_lock(1, "inst-A", "deep"))
        self.assertFalse(await self.db.acquire_report_lock(1, "inst-B", "deep"))

    async def test_exactly_one_winner_among_many(self):
        """Ровно один слот при одновременной гонке — иначе это окно отчётов."""
        results = await asyncio.gather(*[
            self.db.acquire_report_lock(42, f"inst-{i}", "base")
            for i in range(8)
        ])
        self.assertEqual(sum(1 for r in results if r), 1, results)

    async def test_different_users_do_not_block_each_other(self):
        self.assertTrue(await self.db.acquire_report_lock(1, "inst-A"))
        self.assertTrue(await self.db.acquire_report_lock(2, "inst-A"))
        self.assertTrue(await self.db.acquire_report_lock(3, "inst-B"))

    async def test_release_frees_the_slot(self):
        await self.db.acquire_report_lock(7, "inst-A")
        self.assertTrue(await self.db.release_report_lock(7, "inst-A"))
        self.assertTrue(await self.db.acquire_report_lock(7, "inst-B"))

    async def test_release_is_idempotent(self):
        """В обработчике release вызывается из десятка веток выхода."""
        await self.db.acquire_report_lock(7, "inst-A")
        self.assertTrue(await self.db.release_report_lock(7, "inst-A"))
        self.assertFalse(await self.db.release_report_lock(7, "inst-A"))
        self.assertFalse(await self.db.release_report_lock(7, "inst-A"))


# ──────────────────────────────────────────────────────────────────────────
# 2. Аренда и владение
# ──────────────────────────────────────────────────────────────────────────
class LeaseTest(ReportLockTestBase):

    async def test_expired_lease_is_stolen(self):
        """Инстанс умер посреди расчёта — пользователь не заперт навсегда."""
        await self.db.acquire_report_lock(5, "dead-inst", ttl_sec=-1)
        self.assertTrue(await self.db.acquire_report_lock(5, "live-inst"))

    async def test_live_lease_is_not_stolen(self):
        await self.db.acquire_report_lock(5, "inst-A", ttl_sec=600)
        self.assertFalse(await self.db.acquire_report_lock(5, "inst-B"))

    async def test_late_release_does_not_evict_new_holder(self):
        """🔴 Ключевая проверка владения.

        Аренда A просрочилась, её перехватил B, и только теперь A дошёл до
        своего `release`. Без сверки `owner` он выбил бы работающего B — и
        третий инстанс получил бы слот на того же пользователя.
        """
        await self.db.acquire_report_lock(9, "inst-A", ttl_sec=-1)
        self.assertTrue(await self.db.acquire_report_lock(9, "inst-B"))
        self.assertFalse(await self.db.release_report_lock(9, "inst-A"))
        self.assertFalse(await self.db.acquire_report_lock(9, "inst-C"))

    async def test_purge_removes_only_expired(self):
        await self.db.acquire_report_lock(1, "inst-A", ttl_sec=-1)
        await self.db.acquire_report_lock(2, "inst-A", ttl_sec=600)
        self.assertEqual(await self.db.purge_expired_report_locks(), 1)
        self.assertFalse(await self.db.acquire_report_lock(2, "inst-B"))

    async def test_default_ttl_outlives_a_deep_report(self):
        """Аренда обязана переживать самый долгий отчёт (единицы минут)."""
        self.assertGreaterEqual(self.db.REPORT_LOCK_TTL_SEC, 600)

    async def test_boot_purges_stale_leases(self):
        """`init_db` чистит просрочку, чтобы таблица не росла."""
        await self.db.acquire_report_lock(11, "old-inst", ttl_sec=-1)
        await self.db.init_db()
        async with self.db._get_conn() as conn:
            cur = await conn.execute("SELECT COUNT(*) FROM report_locks")
            self.assertEqual((await cur.fetchone())[0], 0)


# ──────────────────────────────────────────────────────────────────────────
# 3. Схема
# ──────────────────────────────────────────────────────────────────────────
class SchemaTest(ReportLockTestBase):

    async def test_table_exists_with_expected_columns(self):
        async with self.db._get_conn() as conn:
            cur = await conn.execute("PRAGMA table_info(report_locks)")
            cols = {r[1] for r in await cur.fetchall()}
        self.assertEqual(cols, {"telegram_id", "owner", "tier",
                                "acquired_at", "expires_at"})

    async def test_one_row_per_user(self):
        """`telegram_id` — PRIMARY KEY: две аренды на одного невозможны."""
        async with self.db._get_conn() as conn:
            cur = await conn.execute("PRAGMA table_info(report_locks)")
            pk = {r[1] for r in await cur.fetchall() if r[5]}
        self.assertEqual(pk, {"telegram_id"})

    async def test_lock_survives_a_reconnect(self):
        """Аренда лежит на диске — новый процесс её увидит."""
        await self.db.acquire_report_lock(3, "inst-A", ttl_sec=600)
        self.assertFalse(await self.db.acquire_report_lock(3, "other-process"))


# ──────────────────────────────────────────────────────────────────────────
# 4. Проводка в боте
# ──────────────────────────────────────────────────────────────────────────
class BotWiringTest(unittest.TestCase):

    SRC = Path("src/tg_bot.py")

    def setUp(self):
        self.src = self.SRC.read_text(encoding="utf-8")

    def test_guard_uses_the_db_lock(self):
        self.assertIn("acquire_report_lock", self.src)
        self.assertIn("release_report_lock", self.src)

    def test_owner_is_process_scoped(self):
        """Владелец — идентификатор ПРОЦЕССА, иначе сверка владения бессмысленна."""
        self.assertIn("_INSTANCE_ID", self.src)
        self.assertIn("os.getpid()", self.src)

    def test_every_call_site_is_awaited(self):
        """Незабытый `await` — иначе гард молча возвращает корутину (=truthy)."""
        import ast
        tree = ast.parse(self.src)
        bad = []

        class V(ast.NodeVisitor):
            def visit_Call(self, node):
                name = getattr(node.func, "id", None)
                if name in ("_try_acquire_user_slot", "_release_user_slot"):
                    bad.append(node.lineno)
                self.generic_visit(node)

        class W(ast.NodeVisitor):
            def visit_Await(self, node):
                # await-обёрнутые вызовы не считаем нарушением
                if isinstance(node.value, ast.Call):
                    nm = getattr(node.value.func, "id", None)
                    if nm in ("_try_acquire_user_slot", "_release_user_slot"):
                        ok.add(node.value.lineno)
                self.generic_visit(node)

        ok = set()
        W().visit(tree)
        V().visit(tree)
        missing = sorted(set(bad) - ok)
        self.assertEqual(missing, [], f"без await в строках {missing}")

    def test_degradation_is_documented_not_silent(self):
        """Сбой БД обязан быть в логе и назван окном — не проглочен."""
        self.assertIn("гард деградировал", self.src)

    def test_deploy_still_pins_single_instance(self):
        """Правка — ПРЕДУСЛОВИЕ снятия max-instances=1, а не само снятие."""
        cb = Path("cloudbuild.yaml")
        if not cb.exists():
            self.skipTest("cloudbuild.yaml отсутствует (деплой-образ)")
        self.assertIn("max-instances=1", cb.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────
# 5. Поведение гарда бота (функционально)
# ──────────────────────────────────────────────────────────────────────────
# tg_bot читает токен при импорте — подставляем на этапе коллекции.
os.environ.setdefault("OMBRI_BOT_TOKEN", "0000000000:TEST-TOKEN-unit")


class BotGuardBehaviourTest(ReportLockTestBase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        import tg_bot
        self.bot = tg_bot
        tg_bot._IN_FLIGHT_USERS.clear()

    async def asyncTearDown(self):
        self.bot._IN_FLIGHT_USERS.clear()
        await super().asyncTearDown()

    async def test_second_tap_is_refused(self):
        self.assertTrue(await self.bot._try_acquire_user_slot(100, tier="deep"))
        self.assertFalse(await self.bot._try_acquire_user_slot(100, tier="deep"))

    async def test_release_allows_the_next_report(self):
        await self.bot._try_acquire_user_slot(101)
        await self.bot._release_user_slot(101)
        self.assertTrue(await self.bot._try_acquire_user_slot(101))

    async def test_lock_row_carries_the_tier(self):
        await self.bot._try_acquire_user_slot(102, tier="scenario")
        async with self.db._get_conn() as conn:
            cur = await conn.execute(
                "SELECT tier, owner FROM report_locks WHERE telegram_id=?", (102,))
            tier, owner = await cur.fetchone()
        self.assertEqual(tier, "scenario")
        self.assertEqual(owner, self.bot._INSTANCE_ID)

    async def test_db_failure_degrades_to_in_process_guard(self):
        """Сбой БД не отказывает в отчёте — гард возвращается к прежнему поведению."""
        with patch.object(self.bot, "acquire_report_lock",
                          side_effect=RuntimeError("disk gone")):
            self.assertTrue(await self.bot._try_acquire_user_slot(103))
            # …и внутри процесса он ВСЁ ЕЩЁ работает
            self.assertFalse(await self.bot._try_acquire_user_slot(103))

    async def test_release_failure_does_not_raise(self):
        """Housekeeping не имеет права уронить доставку отчёта."""
        await self.bot._try_acquire_user_slot(104)
        with patch.object(self.bot, "release_report_lock",
                          side_effect=RuntimeError("disk gone")):
            await self.bot._release_user_slot(104)      # не должно бросить
        self.assertNotIn(104, self.bot._IN_FLIGHT_USERS)


if __name__ == "__main__":
    unittest.main()
