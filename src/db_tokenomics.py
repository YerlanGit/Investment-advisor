"""
Tokenomics DB — async SQLite layer for RAMP user balances.
Pricing: 10 tokens = 5000 KZT (1 token = 500 KZT).
New users receive 10 test tokens on first registration.

Persistence
───────────
Cloud Run's container filesystem is ephemeral — a redeploy or scale-to-
zero wipes anything outside of `/tmp`.  When `TOKENOMICS_DB_PATH` is set
(e.g. `/mnt/data/tokenomics.db` via gcsfuse, or any persistent mount),
the DB lives there and survives restarts.  Otherwise we fall back to the
repo-local `data/tokenomics.db` (good for local dev, NOT for production).
"""

import aiosqlite
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Persistent path takes precedence — set via Cloud Run env var to a
# gcsfuse mount, NFS, or any non-ephemeral volume.  Fallback is the
# repo-local data/ folder (suitable only for local dev).
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "tokenomics.db"
DB_PATH = Path(os.getenv("TOKENOMICS_DB_PATH", str(_DEFAULT_DB_PATH)))
INITIAL_TOKENS = 10


class InsufficientFundsError(Exception):
    """Raised by `deduct_tokens` when the balance cannot cover the amount.

    Carries no money side effects — it is raised *after* the guarded UPDATE
    matched zero rows and the transaction was rolled back, so the balance is
    guaranteed untouched.
    """


# Storage that does NOT survive a Cloud Run restart / scale-to-zero.
_EPHEMERAL_PREFIXES = ("/tmp", "/app", "/var/tmp")


def assert_persistent_state() -> None:
    """Fail-fast if money/credential state would live on ephemeral storage (M-9).

    On Cloud Run the container filesystem is wiped on every restart.  If
    `TOKENOMICS_DB_PATH` / `VAULT_DB_PATH` are unset (or point at /tmp, /app,
    or an in-memory DB) in a production runtime, user balances and encrypted
    broker keys would silently vanish on the next deploy — so we refuse to
    start rather than quietly create a throwaway DB.

    Production is detected via Cloud Run's `K_SERVICE`; set
    `REQUIRE_PERSISTENT_DB=1` to enforce the check in any environment.  Local
    dev (no markers) keeps the repo-local fallback with a loud warning.
    """
    in_prod = bool(os.getenv("K_SERVICE") or os.getenv("REQUIRE_PERSISTENT_DB") == "1")
    if not in_prod:
        if "TOKENOMICS_DB_PATH" not in os.environ:
            logger.warning(
                "TOKENOMICS_DB_PATH unset — using repo-local fallback %s "
                "(DEV ONLY; not safe for production).", DB_PATH)
        return

    problems = []
    for name in ("TOKENOMICS_DB_PATH", "VAULT_DB_PATH"):
        val = (os.getenv(name) or "").strip()
        if not val:
            problems.append(f"{name} is not defined")
        elif val == ":memory:" or val.startswith(_EPHEMERAL_PREFIXES):
            problems.append(f"{name}={val} points at ephemeral storage")
    if problems:
        raise RuntimeError(
            "Persistent DB path is not defined — refusing to start on ephemeral "
            "storage (user balances / broker credentials would be lost on every "
            "restart): " + "; ".join(problems)
        )
    logger.info("Persistent state check passed (M-9): money/credential DBs on durable storage.")


def _cleanup_orphan_journal_files() -> None:
    """Delete stale SQLite ``-wal`` / ``-shm`` sidecars next to the DB.

    gcsfuse (the prod mount) physically cannot do the in-place random rewrites
    SQLite needs for ``-shm`` / ``-wal`` (gcsfuse raises OutOfOrderError), so a
    WAL-mode DB on it can corrupt outright.  We run exclusively in rollback-
    journal (DELETE) mode where these files are NEVER used — so any ``-wal`` /
    ``-shm`` left behind is an ORPHAN from a crashed WAL-mode process or a
    pre-migration database.  Removing them before connecting prevents a stale
    ``-wal`` from poisoning the next open.  (We never touch ``-journal``: in
    DELETE mode that is the *active* rollback journal of a live transaction.)
    """
    try:
        base = str(DB_PATH)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(base + suffix)
            if sidecar.exists():
                sidecar.unlink()
                logger.warning("Removed orphaned SQLite sidecar (gcsfuse-unsafe): %s",
                               sidecar.name)
    except Exception as exc:   # never block startup on cleanup
        logger.warning("Orphan journal cleanup skipped: %s", exc)


async def _harden(db: aiosqlite.Connection) -> None:
    """Force gcsfuse-safe SQLite settings on EVERY connection.

    WAL is categorically forbidden on the gcsfuse mount (it needs ``-shm``
    mmap + byte-range locks GCS can't emulate → OutOfOrderError / corruption).
    ``journal_mode=DELETE`` uses a plain rollback journal that is removed after
    each transaction (no lingering sidecar) and rewrites the DB header off WAL,
    migrating any legacy WAL database on first open.  ``synchronous=FULL``
    fsyncs on commit so a committed token deduction survives a SIGKILL.
    Re-applied on every connect (cheap) so WAL can never silently creep back.
    """
    await db.execute("PRAGMA journal_mode=DELETE")   # NEVER WAL on gcsfuse
    await db.execute("PRAGMA synchronous=FULL")
    await db.execute("PRAGMA busy_timeout=5000")     # wait on the lock, don't fail


@asynccontextmanager
async def _get_conn():
    """Hardened read/write connection.  Cleans orphan WAL sidecars and forces
    journal_mode=DELETE on every open — all 16 call sites use ``async with``,
    so this protects every connection without touching them."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_orphan_journal_files()
    async with aiosqlite.connect(DB_PATH) as db:
        await _harden(db)
        yield db


@asynccontextmanager
async def _get_conn_tx():
    """Hardened autocommit connection for money operations.

    ``isolation_level=None`` disables the driver's implicit transaction
    handling so we own BEGIN/COMMIT/ROLLBACK explicitly (must be passed at
    connect time — aiosqlite runs sqlite3 on a worker thread, so the setter
    can't be called from the event-loop thread).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_orphan_journal_files()
    async with aiosqlite.connect(DB_PATH, isolation_level=None) as db:
        await _harden(db)
        yield db


async def _begin_immediate(db: aiosqlite.Connection) -> None:
    """Open an explicit ``BEGIN IMMEDIATE`` write-transaction.

    BLK-1: balance mutations must serialise.  ``IMMEDIATE`` grabs the SQLite
    write lock up-front (instead of lazily on first write), so two concurrent
    money operations on the same DB block each other deterministically rather
    than racing across ``await`` boundaries.  Journal/sync pragmas are already
    set by ``_harden`` at connect time (must precede BEGIN — journal_mode
    can't change inside a transaction).
    """
    await db.execute("BEGIN IMMEDIATE")


async def init_db() -> None:
    """Create schema if it does not exist. Call once at bot startup.

    Orphan WAL-sidecar cleanup + journal_mode=DELETE are applied by
    ``_get_conn`` on every connection (gcsfuse-safe), so this first connect
    also migrates a legacy WAL database off WAL.
    """
    _cleanup_orphan_journal_files()   # explicit at boot, before the first open
    async with _get_conn() as db:
        # ── One-time purge: set env PURGE_DB_ON_START=1 to wipe all data ──
        import os
        if os.getenv("PURGE_DB_ON_START", "0") == "1":
            logger.warning("PURGE_DB_ON_START=1 — удаляю все таблицы для чистого старта!")
            for table in ("transactions", "user_profiles", "users"):
                await db.execute(f"DROP TABLE IF EXISTS {table}")
            await db.commit()

        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id INTEGER PRIMARY KEY,
                balance     INTEGER NOT NULL DEFAULT 0,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER NOT NULL,
                delta       INTEGER NOT NULL,
                reason      TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (telegram_id) REFERENCES users(telegram_id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                telegram_id       INTEGER PRIMARY KEY,
                score             INTEGER NOT NULL,
                profile_name      TEXT    NOT NULL,
                target_volatility REAL    NOT NULL,
                target_te         REAL    NOT NULL,
                selected_assets   TEXT    NOT NULL,
                limits_dict       TEXT    NOT NULL,
                mandate_approved  INTEGER NOT NULL DEFAULT 0,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS report_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id   INTEGER NOT NULL,
                report_date   TEXT    NOT NULL,
                tier          TEXT    NOT NULL,
                risk_score    INTEGER,
                sharpe        REAL,
                cvar          REAL,
                volatility    REAL,
                total_value   REAL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (telegram_id) REFERENCES users(telegram_id)
            )
        """)
        # Schema migrations: add columns if they don't exist yet.
        for migration in (
            "ALTER TABLE user_profiles ADD COLUMN connection_mode TEXT DEFAULT 'template'",
            "ALTER TABLE user_profiles ADD COLUMN benchmark_ticker TEXT DEFAULT NULL",
        ):
            try:
                await db.execute(migration)
                await db.commit()
            except Exception:
                pass  # Column already present — safe to ignore.

        await db.commit()


async def init_user(telegram_id: int) -> bool:
    """
    Register user and credit INITIAL_TOKENS if first visit.
    Returns True if newly created, False if already existed.

    BLK-1: `INSERT ... ON CONFLICT DO NOTHING` makes the welcome bonus
    exactly-once.  Two concurrent /start calls both run the INSERT, but only
    the one that actually inserts a row (`rowcount == 1`) logs the bonus
    transaction — the loser is a silent no-op, so the user can never be
    double-credited.
    """
    async with _get_conn_tx() as db:
        await _begin_immediate(db)
        try:
            cursor = await db.execute(
                "INSERT INTO users (telegram_id, balance) VALUES (?, ?) "
                "ON CONFLICT(telegram_id) DO NOTHING",
                (telegram_id, INITIAL_TOKENS),
            )
            created = cursor.rowcount == 1
            if created:
                await db.execute(
                    "INSERT INTO transactions (telegram_id, delta, reason) VALUES (?, ?, ?)",
                    (telegram_id, INITIAL_TOKENS, "welcome_bonus"),
                )
        except Exception:
            await db.execute("ROLLBACK")
            raise
        await db.execute("COMMIT")

    if created:
        logger.info("Новый пользователь %s зарегистрирован, начислено %s токенов.", telegram_id, INITIAL_TOKENS)
    return created


async def get_balance(telegram_id: int) -> int:
    """Return current token balance. Returns 0 if user is unknown."""
    async with _get_conn() as db:
        cursor = await db.execute(
            "SELECT balance FROM users WHERE telegram_id = ?", (telegram_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0


async def deduct_tokens(telegram_id: int, amount: int, reason: str = "analysis") -> bool:
    """
    Atomically deduct `amount` tokens.

    BLK-1: the balance check and the decrement are a SINGLE guarded statement
    (`UPDATE ... WHERE balance >= ?`) inside a BEGIN IMMEDIATE transaction.
    There is no read-then-write window in Python, so concurrent deductions for
    the same user cannot both pass a stale check and drive the balance
    negative — the second writer either sees the decremented balance or waits
    on the write lock.

    Returns True on success.  Raises `InsufficientFundsError` when the balance
    cannot cover `amount` (the guarded UPDATE matched zero rows); the
    transaction is rolled back, so the balance is left untouched.
    """
    if amount <= 0:
        raise ValueError("amount must be positive")

    async with _get_conn_tx() as db:
        await _begin_immediate(db)
        try:
            cursor = await db.execute(
                "UPDATE users SET balance = balance - ? "
                "WHERE telegram_id = ? AND balance >= ?",
                (amount, telegram_id, amount),
            )
            if cursor.rowcount == 0:
                # Unknown user OR insufficient balance — either way, no money moved.
                raise InsufficientFundsError(
                    f"user {telegram_id}: balance < {amount} (or user unknown)"
                )
            await db.execute(
                "INSERT INTO transactions (telegram_id, delta, reason) VALUES (?, ?, ?)",
                (telegram_id, -amount, reason),
            )
        except Exception:
            await db.execute("ROLLBACK")
            raise
        await db.execute("COMMIT")

    logger.info("Пользователь %s: списано %s токен(ов) за '%s'.", telegram_id, amount, reason)
    return True


async def credit_tokens(telegram_id: int, amount: int, reason: str = "topup") -> None:
    """Credit tokens (e.g. after a KZT purchase).

    BLK-1: the balance UPDATE and the ledger INSERT commit together inside one
    BEGIN IMMEDIATE transaction, so a credit can never be half-applied (balance
    moved but no audit row, or vice versa) even under concurrent access.
    """
    async with _get_conn_tx() as db:
        await _begin_immediate(db)
        try:
            await db.execute(
                "UPDATE users SET balance = balance + ? WHERE telegram_id = ?",
                (amount, telegram_id),
            )
            await db.execute(
                "INSERT INTO transactions (telegram_id, delta, reason) VALUES (?, ?, ?)",
                (telegram_id, amount, reason),
            )
        except Exception:
            await db.execute("ROLLBACK")
            raise
        await db.execute("COMMIT")

    logger.info("Пользователь %s: зачислено %s токен(ов) за '%s'.", telegram_id, amount, reason)


async def save_profile(
    telegram_id: int,
    score: int,
    profile_name: str,
    target_volatility: float,
    target_te: float,
    selected_assets: list,
    limits_dict: dict,
    benchmark_ticker: str | None = None,
) -> None:
    """
    UPSERT the user's risk profile. mandate_approved is always reset to 0
    so re-runs of onboarding require re-approval.
    """
    async with _get_conn() as db:
        await db.execute(
            """
            INSERT INTO user_profiles
                (telegram_id, score, profile_name, target_volatility, target_te,
                 selected_assets, limits_dict, benchmark_ticker, mandate_approved, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(telegram_id) DO UPDATE SET
                score             = excluded.score,
                profile_name      = excluded.profile_name,
                target_volatility = excluded.target_volatility,
                target_te         = excluded.target_te,
                selected_assets   = excluded.selected_assets,
                limits_dict       = excluded.limits_dict,
                benchmark_ticker  = excluded.benchmark_ticker,
                mandate_approved  = 0,
                updated_at        = CURRENT_TIMESTAMP
            """,
            (
                telegram_id, score, profile_name, target_volatility, target_te,
                json.dumps(selected_assets, ensure_ascii=False),
                json.dumps(limits_dict, ensure_ascii=False),
                benchmark_ticker,
            ),
        )
        await db.commit()
        logger.info("Профиль пользователя %s сохранён: %s (score=%s).", telegram_id, profile_name, score)


async def approve_mandate(telegram_id: int) -> None:
    """Set mandate_approved = 1. No-op if the profile row does not exist."""
    async with _get_conn() as db:
        await db.execute(
            "UPDATE user_profiles SET mandate_approved = 1, updated_at = CURRENT_TIMESTAMP "
            "WHERE telegram_id = ?",
            (telegram_id,),
        )
        await db.commit()
        logger.info("Мандат пользователя %s утверждён.", telegram_id)


async def get_profile(telegram_id: int) -> dict | None:
    """
    Fetch the user_profiles row.
    Returns a dict with selected_assets (list) and limits_dict (dict) deserialised.
    Returns None if no profile exists.
    """
    async with _get_conn() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_profiles WHERE telegram_id = ?", (telegram_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        data = dict(row)
        data["selected_assets"]  = json.loads(data["selected_assets"])
        data["limits_dict"]      = json.loads(data["limits_dict"])
        data["mandate_approved"] = bool(data["mandate_approved"])
        return data


async def get_benchmark_ticker(telegram_id: int) -> str | None:
    """Return the user's selected benchmark ticker, or None."""
    async with _get_conn() as db:
        cursor = await db.execute(
            "SELECT benchmark_ticker FROM user_profiles WHERE telegram_id = ?",
            (telegram_id,),
        )
        row = await cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return row[0]


async def save_connection_mode(telegram_id: int, mode: str) -> None:
    """Persist the user's portfolio source choice ('template' or 'freedom')."""
    async with _get_conn() as db:
        await db.execute(
            "UPDATE user_profiles "
            "SET connection_mode = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE telegram_id = ?",
            (mode, telegram_id),
        )
        await db.commit()
        logger.info("Режим подключения пользователя %s: %s.", telegram_id, mode)


async def get_connection_mode(telegram_id: int) -> str:
    """Return 'template' or 'freedom'. Defaults to 'template' when no profile exists."""
    async with _get_conn() as db:
        cursor = await db.execute(
            "SELECT connection_mode FROM user_profiles WHERE telegram_id = ?",
            (telegram_id,),
        )
        row = await cursor.fetchone()
        if row is None or row[0] is None:
            return "template"
        return row[0]


async def save_benchmark_ticker(telegram_id: int, ticker: str) -> None:
    """Instantly update the user's benchmark without re-approval."""
    async with _get_conn() as db:
        await db.execute(
            "UPDATE user_profiles "
            "SET benchmark_ticker = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE telegram_id = ?",
            (ticker, telegram_id),
        )
        await db.commit()
        logger.info("Бенчмарк пользователя %s: %s.", telegram_id, ticker)


async def save_report_snapshot(
    telegram_id: int,
    tier: str,
    risk_score: int | None,
    sharpe: float | None,
    cvar: float | None,
    volatility: float | None,
    total_value: float | None,
) -> None:
    """Persist key metrics from the just-generated report for month-over-month delta."""
    from datetime import date
    async with _get_conn() as db:
        await db.execute(
            """INSERT INTO report_snapshots
               (telegram_id, report_date, tier, risk_score, sharpe, cvar, volatility, total_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (telegram_id, date.today().isoformat(), tier,
             risk_score, sharpe, cvar, volatility, total_value),
        )
        await db.commit()


async def get_last_report_snapshot(telegram_id: int, tier: str) -> dict | None:
    """Return the previous report snapshot (excluding today) for delta calculation."""
    from datetime import date
    async with _get_conn() as db:
        cursor = await db.execute(
            """SELECT report_date, risk_score, sharpe, cvar, volatility, total_value
               FROM report_snapshots
               WHERE telegram_id = ? AND tier = ? AND report_date < ?
               ORDER BY report_date DESC LIMIT 1""",
            (telegram_id, tier, date.today().isoformat()),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "report_date": row[0],
            "risk_score":  row[1],
            "sharpe":      row[2],
            "cvar":        row[3],
            "volatility":  row[4],
            "total_value": row[5],
        }
