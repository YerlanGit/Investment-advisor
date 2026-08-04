# TEST_MAP.md — какие тесты защищают какой модуль

<!-- nav | area:tests | code:(все модули src/) | read-before:правка любого модуля — сначала посмотри, что уже защищено -->

> **Файл СГЕНЕРИРОВАН** — не правь руками:
> `python scripts/gen_test_map.py` (проверка свежести — `--check`).
>
> Зачем он есть: 91% тест-файлов названы по номеру фазы, то есть по тому, КОГДА
> написаны. Эта карта отвечает на другой вопрос — ЧТО защищено у модуля,
> который ты собрался править. Связь установлена по ИМПОРТАМ (включая
> отложенные, внутри функций), а не по совпадению строк.
>
> Переименование `test_phase*.py` отклонено осознанно (`AUDIT §−50`): оно рвёт
> ссылки в `AUDIT.md` и доках. Для НОВЫХ файлов конвенция другая —
> `test_<область>_<тема>.py`, номер раунда в докстринге.


**Замер:** 58 тест-файлов · 1253 тест-функций · 49 модулей `src/` под покрытием.

## Модуль → тесты

> **Как читать колонку «тест-функций†».** Это сумма тест-функций В ФАЙЛАХ, которые
> импортируют модуль, — **верхняя оценка, а не счётчик тестов именно этого модуля**:
> один файл обычно проверяет несколько модулей сразу. Колонка отвечает на вопрос
> «куда смотреть», а не «насколько покрыто». Для второго нужен coverage, а не импорты.

| Модуль `src/` | тест-функций† | Файлы (тест-функций в файле) |
|---|---|---|
| `agent.gatekeeper` | 169 | `test_phase18_sprint5.py` (64) · `test_phase32_report_logic_fixes.py` (30) · `test_phase42_instrument_ssot.py` (30) · `test_phase44_live_report_fixes.py` (28) · `test_phase33_effect_reinvest_quality.py` (17) |
| `agent.rag_engine` | 36 | `test_phase21_recos.py` (21) · `test_phase22_rag_boot.py` (15) |
| `ai_narrative` | 558 | `test_phase4_reporting.py` (155) · `test_phase19_block_audit.py` (70) · `test_phase18_sprint5.py` (64) · `test_phase31_benchmark_factor_propagation.py` (33) · `test_phase46_p2_maintainability.py` (32) · `test_phase32_report_logic_fixes.py` (30) · `test_phase44_live_report_fixes.py` (28) · `test_factor_decomposition.py` (22) · `test_phase40_labels_and_boundary.py` (20) · `test_phase36_report_audit_fixes.py` (18) · `test_phase33_effect_reinvest_quality.py` (17) · `test_phase7_logic_wiring.py` (16) · `test_phase5_rag_quality.py` (15) · `test_phase15_phase3.py` (13) · `test_phase13_security.py` (12) · `test_phase8_report_fixes.py` (7) · `test_phase9_fpillar_stress.py` (6) |
| `db_tokenomics` | 56 | `test_phase43_report_lock.py` (24) · `test_phase29_multiuser_connection.py` (19) · `test_phase20_sprint_refactor.py` (8) · `test_phase12_beta_safety.py` (5) |
| `entrypoint` | 15 | `test_phase22_rag_boot.py` (15) |
| `env_config` | 32 | `test_phase46_p2_maintainability.py` (32) |
| `finance` | 215 | `test_phase35_data_checks.py` (57) · `test_phase46_p2_maintainability.py` (32) · `test_phase42_instrument_ssot.py` (30) · `test_phase47_cove_and_fx_honesty.py` (27) · `test_phase45_data_checks_wiring.py` (24) · `test_phase39_report_consistency.py` (23) · `test_phase23_scenario.py` (22) |
| `finance.action_plan` | 131 | `test_phase18_sprint5.py` (64) · `test_phase39_report_consistency.py` (23) · `test_phase36_report_audit_fixes.py` (18) · `test_phase3_modules.py` (14) · `test_phase27_composite_metrics.py` (12) |
| `finance.asset_taxonomy` | 73 | `test_phase32_report_logic_fixes.py` (30) · `test_phase42_instrument_ssot.py` (30) · `test_phase15_phase3.py` (13) |
| `finance.black_litterman` | 86 | `test_phase18_sprint5.py` (64) · `test_phase3_modules.py` (14) · `test_phase20_sprint_refactor.py` (8) |
| `finance.broker_api` | 108 | `test_phase46_p2_maintainability.py` (32) · `test_phase42_instrument_ssot.py` (30) · `test_phase41_fx_base_currency.py` (26) · `test_phase13_security.py` (12) · `test_phase34_broker_outage_honesty.py` (8) |
| `finance.cds_feed` | 179 | `test_phase4_reporting.py` (155) · `test_phase3_modules.py` (14) · `test_phase38_fred_memo.py` (10) |
| `finance.contracts` | 7 | `test_contracts_results.py` (7) |
| `finance.currency` | 56 | `test_phase6_currency_h2.py` (32) · `test_phase25_math_sprint1.py` (13) · `test_phase6_fx_feed.py` (11) |
| `finance.data_checks` | 57 | `test_phase35_data_checks.py` (57) |
| `finance.data_lineage` | 406 | `test_phase4_reporting.py` (155) · `test_phase19_block_audit.py` (70) · `test_phase18_sprint5.py` (64) · `test_phase36_illiquid_proxy.py` (36) · `test_phase28_risk_methodology_audit.py` (30) · `test_phase47_cove_and_fx_honesty.py` (27) · `test_phase45_data_checks_wiring.py` (24) |
| `finance.demo_portfolio` | 148 | `test_phase36_illiquid_proxy.py` (36) · `test_phase46_p2_maintainability.py` (32) · `test_phase35_demo_showcase.py` (29) · `test_phase47_cove_and_fx_honesty.py` (27) · `test_phase45_data_checks_wiring.py` (24) |
| `finance.factor_decomposition` | 22 | `test_factor_decomposition.py` (22) |
| `finance.inference` | 30 | `test_phase28_risk_methodology_audit.py` (30) |
| `finance.investment_logic` | 721 | `test_phase19_block_audit.py` (70) · `test_phase18_sprint5.py` (64) · `test_phase35_price_providers.py` (41) · `test_phase36_illiquid_proxy.py` (36) · `test_phase31_benchmark_factor_propagation.py` (33) · `test_phase46_p2_maintainability.py` (32) · `test_phase6_currency_h2.py` (32) · `test_phase28_risk_methodology_audit.py` (30) · `test_phase32_report_logic_fixes.py` (30) · `test_phase42_instrument_ssot.py` (30) · `test_phase35_demo_showcase.py` (29) · `test_phase47_cove_and_fx_honesty.py` (27) · `test_phase41_fx_base_currency.py` (26) · `test_phase45_data_checks_wiring.py` (24) · `test_phase39_report_consistency.py` (23) · `test_factor_decomposition.py` (22) · `test_phase23_scenario.py` (22) · `test_phase33_effect_reinvest_quality.py` (17) · `test_phase2_modules.py` (16) · `test_phase37_position_contract.py` (16) · `test_phase22_rag_boot.py` (15) · `test_phase26_report_fixes.py` (14) · `test_phase15_phase3.py` (13) · `test_phase25_math_sprint1.py` (13) · `test_phase13_security.py` (12) · `test_phase27_composite_metrics.py` (12) · `test_phase14_refactor.py` (10) · `test_phase20_sprint_refactor.py` (8) · `test_phase1_fixes.py` (4) |
| `finance.leveraged` | 30 | `test_phase28_risk_methodology_audit.py` (30) |
| `finance.period_returns` | 199 | `test_phase4_reporting.py` (155) · `test_phase28_risk_methodology_audit.py` (30) · `test_phase26_report_fixes.py` (14) |
| `finance.portfolio_series` | 24 | `test_phase27_composite_metrics.py` (12) · `test_phase20_sprint_refactor.py` (8) · `test_phase30_benchmark_equity_curve.py` (4) |
| `finance.price_providers` | 61 | `test_phase35_price_providers.py` (41) · `test_phase40_labels_and_boundary.py` (20) |
| `finance.regime` | 153 | `test_phase19_block_audit.py` (70) · `test_phase21_recos.py` (21) · `test_phase36_report_audit_fixes.py` (18) · `test_phase2_modules.py` (16) · `test_phase7_logic_wiring.py` (16) · `test_phase8_report_fixes.py` (7) · `test_phase10_pillar_chip.py` (5) |
| `finance.scenario_engine` | 32 | `test_phase46_p2_maintainability.py` (32) |
| `finance.scenario_report` | 10 | `test_phase24_scenario_report.py` (10) |
| `finance.scoring` | 445 | `test_phase4_reporting.py` (155) · `test_phase19_block_audit.py` (70) · `test_phase18_sprint5.py` (64) · `test_phase32_report_logic_fixes.py` (30) · `test_phase42_instrument_ssot.py` (30) · `test_phase39_report_consistency.py` (23) · `test_phase21_recos.py` (21) · `test_phase2_modules.py` (16) · `test_phase15_phase3.py` (13) · `test_phase16_sprint2.py` (13) · `test_phase14_refactor.py` (10) |
| `finance.scoring_orchestrator` | 158 | `test_phase18_sprint5.py` (64) · `test_phase28_risk_methodology_audit.py` (30) · `test_phase42_instrument_ssot.py` (30) · `test_phase7_logic_wiring.py` (16) · `test_phase8_report_fixes.py` (7) · `test_phase9_fpillar_stress.py` (6) · `test_phase10_pillar_chip.py` (5) |
| `finance.sec_edgar` | 185 | `test_phase4_reporting.py` (155) · `test_phase42_instrument_ssot.py` (30) |
| `finance.security` | 19 | `test_phase29_multiuser_connection.py` (19) |
| `finance.simulate` | 470 | `test_phase4_reporting.py` (155) · `test_phase19_block_audit.py` (70) · `test_phase18_sprint5.py` (64) · `test_phase46_p2_maintainability.py` (32) · `test_phase32_report_logic_fixes.py` (30) · `test_phase42_instrument_ssot.py` (30) · `test_phase44_live_report_fixes.py` (28) · `test_phase36_report_audit_fixes.py` (18) · `test_phase33_effect_reinvest_quality.py` (17) · `test_phase7_logic_wiring.py` (16) · `test_phase14_refactor.py` (10) |
| `finance.smart_money` | 70 | `test_phase19_block_audit.py` (70) |
| `finance.stress` | 230 | `test_phase4_reporting.py` (155) · `test_phase6_currency_h2.py` (32) · `test_phase28_risk_methodology_audit.py` (30) · `test_phase25_math_sprint1.py` (13) |
| `finance.technicals` | 16 | `test_phase2_modules.py` (16) |
| `freedom_portfolio` | 58 | `test_phase35_price_providers.py` (41) · `test_freedom_history.py` (17) |
| `freedom_portfolio.auth` | 10 | `test_freedom_auth.py` (10) |
| `freedom_portfolio.client` | 34 | `test_freedom_history.py` (17) · `test_freedom_client.py` (9) · `test_phase34_broker_outage_honesty.py` (8) |
| `freedom_portfolio.history` | 17 | `test_freedom_history.py` (17) |
| `freedom_portfolio.models` | 3 | `test_freedom_models.py` (3) |
| `html_renderer` | 165 | `test_phase19_block_audit.py` (70) · `test_phase18_sprint5.py` (64) · `test_phase21_recos.py` (21) · `test_phase24_scenario_report.py` (10) |
| `pdf_charts` | 170 | `test_phase4_reporting.py` (155) · `test_phase5_rag_quality.py` (15) |
| `pdf_payload` | 670 | `test_phase4_reporting.py` (155) · `test_phase19_block_audit.py` (70) · `test_phase18_sprint5.py` (64) · `test_phase36_illiquid_proxy.py` (36) · `test_phase31_benchmark_factor_propagation.py` (33) · `test_phase28_risk_methodology_audit.py` (30) · `test_phase44_live_report_fixes.py` (28) · `test_phase47_cove_and_fx_honesty.py` (27) · `test_phase41_fx_base_currency.py` (26) · `test_phase39_report_consistency.py` (23) · `test_factor_decomposition.py` (22) · `test_phase40_labels_and_boundary.py` (20) · `test_phase36_report_audit_fixes.py` (18) · `test_phase33_effect_reinvest_quality.py` (17) · `test_phase37_position_contract.py` (16) · `test_phase7_logic_wiring.py` (16) · `test_phase5_rag_quality.py` (15) · `test_phase16_sprint2.py` (13) · `test_phase25_math_sprint1.py` (13) · `test_phase14_refactor.py` (10) · `test_phase8_report_fixes.py` (7) · `test_phase9_fpillar_stress.py` (6) · `test_phase10_pillar_chip.py` (5) |
| `premium_payload` | 305 | `test_phase19_block_audit.py` (70) · `test_phase36_illiquid_proxy.py` (36) · `test_phase31_benchmark_factor_propagation.py` (33) · `test_phase44_live_report_fixes.py` (28) · `test_phase47_cove_and_fx_honesty.py` (27) · `test_phase39_report_consistency.py` (23) · `test_factor_decomposition.py` (22) · `test_phase21_recos.py` (21) · `test_phase40_labels_and_boundary.py` (20) · `test_phase26_report_fixes.py` (14) · `test_phase37_base_holdings_ai.py` (11) |
| `profile_manager` | 64 | `test_phase18_sprint5.py` (64) |
| `services.fx_feed` | 24 | `test_phase25_math_sprint1.py` (13) · `test_phase6_fx_feed.py` (11) |
| `services.macro_data` | 176 | `test_phase4_reporting.py` (155) · `test_phase21_recos.py` (21) |
| `services.report_storage` | 42 | `test_phase46_p2_maintainability.py` (32) · `test_phase14_refactor.py` (10) |
| `tg_bot` | 140 | `test_phase31_benchmark_factor_propagation.py` (33) · `test_phase43_report_lock.py` (24) · `test_phase29_multiuser_connection.py` (19) · `test_phase5_rag_quality.py` (15) · `test_phase26_report_fixes.py` (14) · `test_phase16_sprint2.py` (13) · `test_phase24_scenario_report.py` (10) · `test_phase12_beta_safety.py` (5) · `test_phase1_fixes.py` (4) · `test_phase17_admin_grant.py` (3) |

## Модули без прямого тест-импорта

> Не обязательно «без покрытия»: модуль может проверяться через вызывающий код. Но правка здесь — без страховки в виде адресного теста.

- `agent`
- `agent.advisor_bot`
- `batch_reports`
- `finance.setup_vault`
- `finance.tool_plugins`
- `freedom_portfolio.__main__`
- `freedom_portfolio.display`
- `freedom_portfolio.websocket`
- `premium_renderer`
- `report_mocks`
- `services`
- `test_live_api`

## Тест-файлы, не привязанные к модулям `src/`

- `test_repo_hygiene.py`
