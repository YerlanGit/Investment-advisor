from finance.security import SecureVault
from finance.broker_api import FreedomConnector
from finance.investment_logic import UniversalPortfolioManager

import os
import json
import anthropic
from dotenv import load_dotenv
from agent.rag_engine import FinancialRAG
from agent.gatekeeper import run_gatekeeper

load_dotenv()

# 1. Инициализация систем
vault = SecureVault()
manager = UniversalPortfolioManager()
rag = FinancialRAG()

def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SYSTEM_PROMPT.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are an AI assistant. SYSTEM_PROMPT.md not found."

def simulate_user_registration(telegram_id):
    print("--- РЕГИСТРАЦИЯ ПОЛЬЗОВАТЕЛЯ ---")
    user_login = "investor@mail.com"
    fake_api_key = "FRDM_API_123456789_KEY"
    fake_secret  = "FRDM_SECRET_987654321_HASH"
    vault.save_user_keys(telegram_id, user_login, fake_api_key, fake_secret)
    print("Регистрация завершена.\n")

def handle_analyze_request(telegram_id):
    print(f"\n--- АУДИТ ПОРТФЕЛЯ (RAMP) ДЛЯ ПОЛЬЗОВАТЕЛЯ: {telegram_id} ---")
    
    keys = vault.get_user_keys(telegram_id)
    if not keys: return
    login, api_key, secret_key = keys

    connector = FreedomConnector(api_key=api_key, secret_key=secret_key)
    print("\n[Система] Скачивание портфеля из брокера...")
    
    try:
        raw_portfolio_df = connector.fetch_portfolio()
        print("[Система] Запуск квант-модели MAC3 Risk...")
        full_report = manager.analyze_all(source=raw_portfolio_df)
        df_final = full_report["performance_table"]
        port_metrics = full_report.get("portfolio_metrics", {})
        
        # ══════════ GATEKEEPER: Детерминированный аудит ══════════
        print("[Gatekeeper] Запуск предварительной проверки лимитов...")
        gate_result = run_gatekeeper(full_report)
        
        if gate_result["critical"]:
            print("\n⛔ GATEKEEPER ОБНАРУЖИЛ КРИТИЧЕСКИЕ НАРУШЕНИЯ:")
            for v in gate_result["critical"]:
                print(f"   {v}")
        if gate_result["warnings"]:
            print("\n⚠️ ПРЕДУПРЕЖДЕНИЯ:")
            for w in gate_result["warnings"]:
                print(f"   {w}")
        # Gatekeeper summary передается в промпт Claude — чтобы он учёл нарушения
        gatekeeper_context = gate_result["summary"]
        
        # Формируем DUAL-контекст RAG (рисковые тикеры)
        tickers = [t for t in df_final['Ticker'] if t not in ['USD', 'EUR', 'CASH', 'RUB']]
        
        print(f"[RAG] Сканирование PDF-базы (Macro + Micro)...")
        
        macro_query = "Rating upgrades/downgrades, sector anomalies, anomalous fund flows, overweight sectors"
        macro_context = rag.get_market_sentiment(query=macro_query, n_results=3)
        
        micro_query = f"Аналитика и ожидания по компаниям: {', '.join(tickers)}"
        micro_context = rag.get_market_sentiment(query=micro_query, n_results=3)
        
        market_context = f"=== MACRO TRENDS & FUND FLOWS ===\n{macro_context}\n\n=== MICRO INSIGHTS ===\n{micro_context}"

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print("[Ошибка] ANTHROPIC_API_KEY не установлен в .env!")
            return

        client = anthropic.Anthropic(api_key=anthropic_api_key)
        system_instructions = load_system_prompt()
        
        user_limits = {
            "Max_Total_Risk_Appetite": "Moderate",
            "Max_Single_Asset_Weight": "15%",
            "Max_Sector_Weight": "25%",
            "Max_Euler_Risk_Contribution_Per_Asset": "20%"
        }
        
        cols_to_show = ['Ticker', 'Quantity', 'Current_Price', 'PnL']
        if 'Euler_Risk_Contribution_Pct' in df_final.columns:
            cols_to_show.append('Euler_Risk_Contribution_Pct')
        if 'ATR_Pct' in df_final.columns:
            cols_to_show.append('ATR_Pct')
        if 'Fundamental_Score' in df_final.columns:
            cols_to_show.append('Fundamental_Score')
        
        # Бенчмарк-контекст для Claude
        bm_data = full_report.get("benchmark_comparison", {})
        bm_context = ""
        if bm_data:
            bm_lines = []
            for bm_name, bm_vals in bm_data.items():
                beating = "✅ ОБГОНЯЕТ" if bm_vals.get("Beating_Benchmark") else "❌ ОТСТАЁТ"
                bm_lines.append(
                    f"  {bm_name}: Бенчмарк {bm_vals['Benchmark_Return']:.1%}, "
                    f"Портфель {bm_vals['Portfolio_Return']:.1%}, "
                    f"Excess {bm_vals['Excess_Return']:+.1%} {beating}"
                )
                if bm_vals.get("Information_Ratio") is not None:
                    bm_lines.append(f"    Tracking Error: {bm_vals['Tracking_Error']:.1%}, Information Ratio: {bm_vals['Information_Ratio']:.2f}")
            bm_context = "\n".join(bm_lines)

        # ══════════ CoVe: Chain-of-Verification промпт ══════════
        user_message = f"""
ИИ, вот мой портфель из API:
{df_final[cols_to_show].to_json(orient='records', force_ascii=False)}

Вот таблица риск-аналитики MAC3:
Общая стоимость: {full_report['total_value']:.2f} USD
Общий PnL (Доходность): {full_report['total_portfolio_pnl']:.2f} USD
Безрисковая ставка (RFR): {full_report.get('risk_free_rate', 0)*100:.2f}%
Sharpe Ratio: {port_metrics.get('Sharpe_Ratio', 'N/A')}
CVaR 95 (Ожидаемые потери в худшие 5% дней): {port_metrics.get('CVaR_95_Daily', 'N/A')}

=== ПОРТФЕЛЬ VS БЕНЧМАРК (Бьём ли мы рынок?) ===
{bm_context if bm_context else 'Данные бенчмарков недоступны'}

Структура факторов (Бета и Альфа):
{df_final[['Ticker', 'Beta_Market', 'Alpha_Specific', 'Residual_Vol_Ann']].to_string() if 'Beta_Market' in df_final.columns else 'N/A'}

=== РЕЗУЛЬТАТ АУДИТОРА (GATEKEEPER - проверено до тебя) ===
{gatekeeper_context}

Вот аналитика банков из RAG-базы:
{market_context}

Мои риск-лимиты:
{json.dumps(user_limits, ensure_ascii=False)}

ВАЖНО — Протокол Chain-of-Verification (CoVe):
1. Сформулируй свои рекомендации (Health Check, Market Pulse, Action Table).
2. После формулирования — выполни САМОПРОВЕРКУ:
   - Перечисли все числовые утверждения, которые ты сделал (%, цены, рейтинги).
   - Для каждого укажи источник: [MAC3], [RAG: название отчёта], [GATEKEEPER] или [НЕ ПОДТВЕРЖДЕНО].
   - Если факт НЕ ПОДТВЕРЖДЁН данными — удали его или явно пометь: "⚠️ Данные отсутствуют в базе".
3. Выведи финальный отчёт ТОЛЬКО с подтверждёнными фактами.
"""
        print("[ИИ-Агент] Anthropic (Claude) анализирует данные (CoVe протокол)...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.1,
            system=system_instructions,
            messages=[{"role": "user", "content": user_message}]
        )
        
        ai_response = response.content[0].text
        
        # ══════════ ФИНАЛЬНАЯ ПРОВЕРКА: Gatekeeper блокирует? ══════════
        if not gate_result["passed"]:
            print("\n⛔ ════════════ GATEKEEPER: БЛОКИРОВКА ════════════ ⛔")
            print("Портфель нарушает критические лимиты. ИИ-ответ дополнен предупреждением.\n")
            print("КРИТИЧЕСКИЕ НАРУШЕНИЯ (выявлены ДО анализа ИИ):")
            for v in gate_result["critical"]:
                print(f"  {v}")
            print("\n─────────────── ОТВЕТ ИИ-АГЕНТА ───────────────\n")
        else:
            print("\n✅ ════════════ GATEKEEPER: ПРОЙДЕНО ════════════ ✅\n")
        
        print(ai_response)
        print("\n═══════════════════════════════════════════════════\n")
        
    except Exception as e:
        print(f"Критическая ошибка пайплайна: {e}")

if __name__ == "__main__":
    test_user_id = "tg_user_999"
    simulate_user_registration(test_user_id)
    handle_analyze_request(test_user_id)