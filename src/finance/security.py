import logging
import os
import re
import sqlite3
from contextlib import closing

from cryptography.fernet import Fernet, MultiFernet, InvalidToken
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("SecureVault")

# User-facing message shown when stored credentials can no longer be decrypted
# (master key rotated past its grace window, or ciphertext corruption).
KEY_ROTATED_USER_MESSAGE = (
    "Мастер-ключ системы был обновлён в целях безопасности. "
    "Пожалуйста, привяжите счёт заново."
)


class MasterKeyRotatedError(Exception):
    """Raised when stored ciphertext cannot be decrypted with any active key.

    H-7: replaces the previously-uncaught `cryptography.fernet.InvalidToken`,
    which used to bubble up as a raw stack trace.  Carries a human-readable
    `user_message` so the bot layer can prompt clean re-onboarding.
    """

    def __init__(self, message: str = KEY_ROTATED_USER_MESSAGE):
        super().__init__(message)
        self.user_message = message


def _load_cipher() -> MultiFernet:
    """Build a MultiFernet from FINTECH_MASTER_KEY (H-7: seamless rotation).

    The env var may hold ONE key or several separated by commas/whitespace.
    The FIRST key is primary (used for new encryption); the rest are legacy
    keys retained only for decryption.  Rotation is then zero-downtime: deploy
    `NEW_KEY,OLD_KEY`, new saves use NEW_KEY while existing credentials still
    decrypt under OLD_KEY; drop OLD_KEY on the following rotation.
    """
    raw = os.getenv("FINTECH_MASTER_KEY")
    if not raw:
        raise ValueError(
            "[Security Critical] FINTECH_MASTER_KEY is missing in environment variables! "
            "Do not run the application without a valid encryption key."
        )
    keys = [k for k in re.split(r"[,\s]+", raw.strip()) if k]
    return MultiFernet([Fernet(k.encode()) for k in keys])


class SecureVault:
    """
    Fernet-encrypted broker-credential store.

    L4: the connection is no longer held open for the object's lifetime.
    Each operation opens a fresh connection inside a `with closing(...)`
    block (auto-close) and an inner `with conn:` transaction (auto-commit
    / auto-rollback).  No leaked file handles even under repeated
    per-request `SecureVault(...)` instantiation.
    """

    def __init__(self, db_name="users_vault.db"):
        # MultiFernet: encrypts with the primary key, decrypts with any
        # configured key → enables seamless FINTECH_MASTER_KEY rotation.
        self.cipher  = _load_cipher()
        self.db_name = db_name
        self._create_table()

    def _create_table(self):
        """Создаем таблицу, если её нет."""
        with closing(sqlite3.connect(self.db_name)) as conn, conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_credentials (
                    user_id TEXT PRIMARY KEY,
                    login TEXT,
                    encrypted_api_key BLOB,
                    encrypted_secret_key BLOB
                )
            ''')

    def save_user_keys(self, user_id, login, api_key, secret_key):
        """Шифруем оба ключа и сохраняем в базу."""
        enc_api    = self.cipher.encrypt(api_key.encode())
        enc_secret = self.cipher.encrypt(secret_key.encode())
        with closing(sqlite3.connect(self.db_name)) as conn, conn:
            conn.execute('''
                INSERT OR REPLACE INTO api_credentials
                (user_id, login, encrypted_api_key, encrypted_secret_key)
                VALUES (?, ?, ?, ?)
            ''', (user_id, login, enc_api, enc_secret))
        # Log only the fact, never the keys.
        logger.info("Credentials for user %s encrypted and stored.", user_id)

    def get_user_keys(self, user_id):
        """Достаём из базы и расшифровываем оба ключа.

        Raises `MasterKeyRotatedError` if the stored ciphertext cannot be
        decrypted with any active master key (post-rotation / corruption), so
        the caller can prompt re-onboarding instead of crashing.
        """
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.execute(
                'SELECT login, encrypted_api_key, encrypted_secret_key '
                'FROM api_credentials WHERE user_id = ?',
                (user_id,),
            )
            row = cursor.fetchone()
        if row:
            try:
                login      = row[0]
                api_key    = self.cipher.decrypt(row[1]).decode()
                secret_key = self.cipher.decrypt(row[2]).decode()
            except InvalidToken as exc:
                logger.warning(
                    "Vault decrypt failed for user %s — master key rotated past "
                    "its grace window or ciphertext corrupt; prompting re-onboarding.",
                    user_id,
                )
                raise MasterKeyRotatedError() from exc
            return login, api_key, secret_key
        return None