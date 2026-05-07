"""
Tokenomics DB — async SQLite layer for RAMP user balances.
Pricing: 10 tokens = 5000 KZT (1 token = 500 KZT).
New users receive 10 test tokens on first registration.
"""

import aiosqlite
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "tokenomics.db"
INITIAL_TOKENS = 10


def _get_conn() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return aiosqlite.connect(DB_PATH)


async def init_db() -> None:
    """Create schema if it does not exist. Call once at bot startup."""
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
    """
    async with _get_conn() as db:
        cursor = await db.execute(
            "SELECT telegram_id FROM users WHERE telegram_id = ?", (telegram_id,)
        )
        row = await cursor.fetchone()
        if row:
            return False

        await db.execute(
            "INSERT INTO users (telegram_id, balance) VALUES (?, ?)",
            (telegram_id, INITIAL_TOKENS),
        )
        await db.execute(
            "INSERT INTO transactions (telegram_id, delta, reason) VALUES (?, ?, ?)",
            (telegram_id, INITIAL_TOKENS, "welcome_bonus"),
        )
        await db.commit()
        logger.info("Новый пользователь %s зарегистрирован, начислено %s токенов.", telegram_id, INITIAL_TOKENS)
        return True


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
    Returns False without modifying balance if funds are insufficient.
    """
    if amount <= 0:
        raise ValueError("amount must be positive")

    async with _get_conn() as db:
        cursor = await db.execute(
            "SELECT balance FROM users WHERE telegram_id = ?", (telegram_id,)
        )
        row = await cursor.fetchone()
        if not row or row[0] < amount:
            return False

        await db.execute(
            "UPDATE users SET balance = balance - ? WHERE telegram_id = ?",
            (amount, telegram_id),
        )
        await db.execute(
            "INSERT INTO transactions (telegram_id, delta, reason) VALUES (?, ?, ?)",
            (telegram_id, -amount, reason),
        )
        await db.commit()
        logger.info("Пользователь %s: списано %s токен(ов) за '%s'.", telegram_id, amount, reason)
        return True


async def credit_tokens(telegram_id: int, amount: int, reason: str = "topup") -> None:
    """Credit tokens (e.g. after a KZT purchase)."""
    async with _get_conn() as db:
        await db.execute(
            "UPDATE users SET balance = balance + ? WHERE telegram_id = ?",
            (amount, telegram_id),
        )
        await db.execute(
            "INSERT INTO transactions (telegram_id, delta, reason) VALUES (?, ?, ?)",
            (telegram_id, amount, reason),
        )
        await db.commit()
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
