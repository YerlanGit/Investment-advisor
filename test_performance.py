import pandas as pd
from src.finance.investment_logic import UniversalPortfolioManager

def main():
    print("--- Запуск Универсального Аналитика с поддержкой Кэша (MAC3/RAMP) ---")
    manager = UniversalPortfolioManager()
    
    # Мок DataFrame брокера (включая CASH)
    mock_portfolio = pd.DataFrame([
        {'Ticker': 'AAPL', 'Quantity': 150, 'Purchase_Price': 160.0},
        {'Ticker': 'TSLA', 'Quantity': 50, 'Purchase_Price': 200.0},
        {'Ticker': 'USD', 'Quantity': 25000, 'Purchase_Price': 1.0}
    ])
    
    try:
        report = manager.analyze_all(mock_portfolio)
        
        print("\n1. ТАБЛИЦА ЭФФЕКТИВНОСТИ (Performance):")
        print(report["performance_table"].to_string())
        
        print(f"\n2. ОБЩИЙ PnL ПОРТФЕЛЯ: {report['total_portfolio_pnl']:.2f} USD")
        print(f"3. ОБЩАЯ СТОИМОСТЬ (Включая кэш): {report['total_value']:.2f} USD")
        
        print("\n4. МЕТРИКИ ПОРТФЕЛЯ (Sharpe, CVaR, Volatility):")
        for k, v in report['portfolio_metrics'].items():
            print(f"   {k}: {v:.4f}")
        
        print("\n5. ПОРТФЕЛЬ VS БЕНЧМАРК:")
        bm = report.get("benchmark_comparison", {})
        if bm:
            for name, vals in bm.items():
                beating = "✅ ОБГОНЯЕТ" if vals.get("Beating_Benchmark") else "❌ ОТСТАЁТ"
                print(f"   {name}: Бенчмарк {vals['Benchmark_Return']:.1%}, Портфель {vals['Portfolio_Return']:.1%}, Excess {vals['Excess_Return']:+.1%} {beating}")
                if vals.get("Information_Ratio") is not None:
                    print(f"     Tracking Error: {vals['Tracking_Error']:.1%}, IR: {vals['Information_Ratio']:.2f}")
        else:
            print("   Данные бенчмарков недоступны")
        
    except Exception as e:
        print(f"Ошибка при анализе: {e}")

if __name__ == "__main__":
    main()