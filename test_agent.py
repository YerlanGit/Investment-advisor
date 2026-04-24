# test_agent.py — Тест движка MAC3 (обновлен под новый API)
import pandas as pd
from src.finance.investment_logic import MAC3RiskEngine

def main():
    print("Инициализация MAC3 Квант-Движка (Structural Risk)...")
    engine = MAC3RiskEngine()
    
    tickers = ["AAPL", "TSLA"]
    print(f"\nСкачиваем данные для {tickers}...")
    
    # 1. Получаем рыночные данные
    data = engine.get_market_data(tickers, period="1y")
    print(f"Скачано {len(data)} торговых дней.")
    
    # 2. Мок-весов (50/50)
    weights = {t: 0.5 for t in tickers}
    
    # 3. Структурный риск (Euler + CVaR)
    cov_df, exposures_df, metrics = engine.calculate_structural_risk(data, tickers, weights)
    
    print("\n--- Факторные Экспозиции ---")
    print(exposures_df.to_string())
    
    print("\n--- Структурная Ковариационная Матрица ---")
    print(cov_df.to_string())
    
    print("\n--- Портфельные Метрики ---")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

if __name__ == "__main__":
    main()