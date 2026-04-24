import requests
import json
import hashlib
import hmac
import pandas as pd
import logging

logger = logging.getLogger("BrokerAPI")

class FreedomConnector:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://tradernet.kz/api/"

    def fetch_portfolio(self):
        """Реальный запрос к Freedom Finance API."""
        logger.info("Устанавливаю живое соединение с Freedom Broker...")
        
        # 1. Параметры команды
        data = {
            "cmd": "getPortfolio",
            "params": {"is_re_eval": True}
        }
        p_data = json.dumps(data)
        
        # 2. Создание цифровой подписи
        signature = hmac.new(
            self.secret_key.encode('utf-8'), 
            p_data.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()

        # 3. Подготовка финального пакета с ApiKey
        payload = {
            "apiKey": self.api_key,
            "cmd": data["cmd"],
            "params": data["params"]
        }

        headers = {
            'X-Nt-Api-Sig': signature,
            'Content-Type': 'application/json'
        }

        try:
            # 4. РЕАЛЬНЫЙ ВЫЗОВ СЕРВЕРА
            response = requests.post(self.base_url, data=json.dumps(payload), headers=headers)
            response.raise_for_status() # Выдаст ошибку, если ключ неверный
            
            live_data = response.json()
            
            # Проверяем на ошибки в самом ответе API
            if 'error' in live_data:
                raise ValueError(f"API Error: {live_data['error']}")
                
            return self._parse_to_mac3_format(live_data)
            
        except Exception as e:
            logger.error(f"Ошибка при получении живых данных: {e}")
            raise

    def _parse_to_mac3_format(self, raw_json):
        """Парсинг реального ответа Tradernet в таблицу MAC3."""
        # У Freedom данные лежат в ключе result -> pos
        res_data = raw_json.get('result', {})
        positions = res_data.get('pos', [])
        
        if not positions:
            logger.warning("Ваш портфель пуст или API не вернуло открытых позиций.")
            return pd.DataFrame(columns=['Ticker', 'Quantity', 'Purchase_Price'])

        parsed = []
        for item in positions:
            # Маппинг колонок: 'i' - тикер, 'q' - количество, 'p_average' - средняя цена
            ticker = item.get('i')
            if ticker and ticker != 'USD': # Игнорируем денежный остаток
                parsed.append({
                    'Ticker': ticker,
                    'Quantity': float(item.get('q', 0)),
                    'Purchase_Price': float(item.get('p_average', item.get('mkt_p', 0)))
                })
        
        return pd.DataFrame(parsed)