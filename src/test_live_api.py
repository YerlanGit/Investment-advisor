"""
Standalone test script for the Freedom Broker / Tradernet API.

Usage (local only — never commit real keys):
    API_KEY=your_key python src/test_live_api.py
"""

import json
import os

import pandas as pd
import requests

# ВСТАВЬТЕ ВАШ API-КЛЮЧ СЮДА (Только для локального теста!)
API_KEY = os.getenv("FREEDOM_API_KEY", "ваш_api_key")

TRADERNET_URL = "https://tradernet.kz/api/"


def fetch_live_portfolio(api_key: str):
    """
    Form-encoded POST with a ``q`` parameter containing the JSON command.
    The API key is passed inside ``params.apiKey`` — no HMAC signing needed.
    """
    cmd_payload = {
        "cmd": "getPortfolio",
        "params": {"apiKey": api_key, "v": 2},
    }
    q_value = json.dumps(cmd_payload, separators=(",", ":"), ensure_ascii=False)
    form_data = {"q": q_value}

    print(f"[DEBUG] POST {TRADERNET_URL}")
    print(f"[DEBUG] form_data = {form_data}")

    response = requests.post(TRADERNET_URL, data=form_data, timeout=30)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return None


# Запуск теста
result = fetch_live_portfolio(API_KEY)

if result:
    if "error" in result:
        print(f"API вернул ошибку: {result['error']}")
    else:
        res_data = result.get("result", result)
        positions = res_data.get("pos", res_data.get("portfolio", []))

        if not positions:
            print("--- Внимание: Список позиций пуст ---")
            print("Ответ сервера:", json.dumps(res_data, indent=2))
            print(
                "\nПроверьте: выбран ли правильный счет и есть ли на нем купленные активы?"
            )
        else:
            df = pd.DataFrame(positions)
            print("--- Обнаружены следующие колонки в API ---")
            print(df.columns.tolist())

            # Автоматический маппинг: ищем подходящие колонки
            # Обычно: 'i' или 't' - тикер, 'q' - количество, 'mkt_p' - цена
            print("\n--- Первые 2 позиции для анализа ---")
            print(df.head(2))