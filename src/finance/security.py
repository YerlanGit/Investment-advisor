import logging
import os
import sqlite3
from contextlib import closing

from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("SecureVault")


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
        master_key = os.getenv("FINTECH_MASTER_KEY")
        if not master_key:
            raise ValueError(
                "[Security Critical] FINTECH_MASTER_KEY is missing in environment variables! "
                "Do not run the application without a valid encryption key."
            )
        self.cipher  = Fernet(master_key.encode())
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
        """Достаём из базы и расшифровываем оба ключа."""
        with closing(sqlite3.connect(self.db_name)) as conn:
            cursor = conn.execute(
                'SELECT login, encrypted_api_key, encrypted_secret_key '
                'FROM api_credentials WHERE user_id = ?',
                (user_id,),
            )
            row = cursor.fetchone()
        if row:
            login      = row[0]
            api_key    = self.cipher.decrypt(row[1]).decode()
            secret_key = self.cipher.decrypt(row[2]).decode()
            return login, api_key, secret_key
        return None