import sqlite3
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv

load_dotenv()

class SecureVault:
    def __init__(self, db_name="users_vault.db"):
        master_key = os.getenv("FINTECH_MASTER_KEY")
        if not master_key:
            raise ValueError(
                "[Security Critical] FINTECH_MASTER_KEY is missing in environment variables! "
                "Do not run the application without a valid encryption key."
            )
        
        self.cipher = Fernet(master_key.encode())
            
        self.conn = sqlite3.connect(db_name)
        self._create_table()

    def _create_table(self):
        """Создаем таблицу, если ее нет."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_credentials (
                user_id TEXT PRIMARY KEY,
                login TEXT,
                encrypted_api_key BLOB,
                encrypted_secret_key BLOB
            )
        ''')
        self.conn.commit()

    def save_user_keys(self, user_id, login, api_key, secret_key):
        """Шифруем оба ключа и сохраняем в базу."""
        enc_api    = self.cipher.encrypt(api_key.encode())
        enc_secret = self.cipher.encrypt(secret_key.encode())
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO api_credentials 
            (user_id, login, encrypted_api_key, encrypted_secret_key)
            VALUES (?, ?, ?, ?)
        ''', (user_id, login, enc_api, enc_secret))
        self.conn.commit()
        print(f"[Security] Ключи для пользователя {user_id} надежно зашифрованы и сохранены.")

    def get_user_keys(self, user_id):
        """Достаем из базы и расшифровываем оба ключа."""
        cursor = self.conn.cursor()
        cursor.execute(
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