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
>
> **Почему здесь НЕТ числа тестов на модуль.** Такое число было бы суммой
> тест-функций в файлах, которые модуль импортируют, то есть верхней оценкой,
> а не покрытием: один файл обычно проверяет несколько модулей. Печатать его
> рядом со словом «тесты» — ровно тот класс лукавых чисел, который в проекте
> ловят как дефект (`AUDIT §−52`, D-5). Карта отвечает на «куда смотреть»;
> на «насколько покрыто» отвечает coverage, а не импорты.
>
> Побочная выгода: без счётчиков карта не устаревает от каждого нового теста —
> `--check` краснеет только когда изменилась СТРУКТУРА (новый файл, новый
> импорт), то есть когда карта действительно врёт.


**Замер:** 85 тест-файлов · 60 модулей `src/` имеют хотя бы один адресный тест-импорт.

## Модуль → тесты

| Модуль `src/` | Файлов | Тест-файлы |
|---|---|---|
| `agent.gatekeeper` | 5 | `test_phase18_sprint5.py` · `test_phase32_report_logic_fixes.py` · `test_phase33_effect_reinvest_quality.py` · `test_phase42_instrument_ssot.py` · `test_phase44_live_report_fixes.py` |
| `agent.rag_engine` | 5 | `test_phase21_recos.py` · `test_phase22_rag_boot.py` · `test_phase23_rag_inventory.py` · `test_phase52_bank_identity_ssot.py` · `test_phase54_rag_chunking_and_provenance.py` |
| `ai_narrative` | 21 | `test_factor_decomposition.py` · `test_phase13_security.py` · `test_phase15_phase3.py` · `test_phase18_sprint5.py` · `test_phase19_block_audit.py` · `test_phase23_rag_inventory.py` · `test_phase31_benchmark_factor_propagation.py` · `test_phase32_report_logic_fixes.py` · `test_phase33_effect_reinvest_quality.py` · `test_phase36_report_audit_fixes.py` · `test_phase40_labels_and_boundary.py` · `test_phase44_live_report_fixes.py` · `test_phase46_p2_maintainability.py` · `test_phase48_report_verdict_audit.py` · `test_phase4_reporting.py` · `test_phase52_bank_identity_ssot.py` · `test_phase54_rag_chunking_and_provenance.py` · `test_phase5_rag_quality.py` · `test_phase7_logic_wiring.py` · `test_phase8_report_fixes.py` · `test_phase9_fpillar_stress.py` |
| `branding` | 1 | `test_phase59_brand_rename.py` |
| `db_tokenomics` | 5 | `test_manual_fsm_flow.py` · `test_phase12_beta_safety.py` · `test_phase20_sprint_refactor.py` · `test_phase29_multiuser_connection.py` · `test_phase43_report_lock.py` |
| `entrypoint` | 1 | `test_phase22_rag_boot.py` |
| `env_config` | 1 | `test_phase46_p2_maintainability.py` |
| `finance` | 14 | `test_manual_freedom_parity.py` · `test_manual_preflight_and_locals.py` · `test_phase23_scenario.py` · `test_phase35_data_checks.py` · `test_phase39_report_consistency.py` · `test_phase42_instrument_ssot.py` · `test_phase45_data_checks_wiring.py` · `test_phase46_p2_maintainability.py` · `test_phase47_cove_and_fx_honesty.py` · `test_phase48_report_verdict_audit.py` · `test_phase49_contract_reaches_eyes.py` · `test_phase51_ingest_bot.py` · `test_stooq_price_store.py` · `test_stooq_provider.py` |
| `finance.action_plan` | 6 | `test_phase18_sprint5.py` · `test_phase27_composite_metrics.py` · `test_phase36_report_audit_fixes.py` · `test_phase39_report_consistency.py` · `test_phase3_modules.py` · `test_phase48_report_verdict_audit.py` |
| `finance.asset_taxonomy` | 3 | `test_phase15_phase3.py` · `test_phase32_report_logic_fixes.py` · `test_phase42_instrument_ssot.py` |
| `finance.black_litterman` | 3 | `test_phase18_sprint5.py` · `test_phase20_sprint_refactor.py` · `test_phase3_modules.py` |
| `finance.broker_api` | 7 | `test_phase13_security.py` · `test_phase34_broker_outage_honesty.py` · `test_phase41_fx_base_currency.py` · `test_phase42_instrument_ssot.py` · `test_phase46_p2_maintainability.py` · `test_phase51_broker_outage_diagnosis.py` · `test_stooq_provider.py` |
| `finance.cds_feed` | 3 | `test_phase38_fred_memo.py` · `test_phase3_modules.py` · `test_phase4_reporting.py` |
| `finance.contracts` | 3 | `test_contracts_golden.py` · `test_contracts_results.py` · `test_stooq_provider.py` |
| `finance.currency` | 3 | `test_phase25_math_sprint1.py` · `test_phase6_currency_h2.py` · `test_phase6_fx_feed.py` |
| `finance.data_checks` | 4 | `test_manual_freedom_parity.py` · `test_phase35_data_checks.py` · `test_stooq_price_store.py` · `test_stooq_provider.py` |
| `finance.data_lineage` | 10 | `test_manual_preflight_and_locals.py` · `test_phase18_sprint5.py` · `test_phase19_block_audit.py` · `test_phase23_rag_inventory.py` · `test_phase28_risk_methodology_audit.py` · `test_phase36_illiquid_proxy.py` · `test_phase45_data_checks_wiring.py` · `test_phase47_cove_and_fx_honesty.py` · `test_phase4_reporting.py` · `test_phase54_rag_chunking_and_provenance.py` |
| `finance.demo_portfolio` | 5 | `test_phase35_demo_showcase.py` · `test_phase36_illiquid_proxy.py` · `test_phase45_data_checks_wiring.py` · `test_phase46_p2_maintainability.py` · `test_phase47_cove_and_fx_honesty.py` |
| `finance.factor_decomposition` | 1 | `test_factor_decomposition.py` |
| `finance.inference` | 1 | `test_phase28_risk_methodology_audit.py` |
| `finance.investment_logic` | 39 | `test_contracts_results.py` · `test_engine_orchestrator.py` · `test_factor_decomposition.py` · `test_manual_freedom_parity.py` · `test_manual_fsm_flow.py` · `test_manual_portfolio_identity.py` · `test_manual_portfolio_parser.py` · `test_manual_preflight_and_locals.py` · `test_manual_ticker_synonyms.py` · `test_phase13_security.py` · `test_phase14_refactor.py` · `test_phase15_phase3.py` · `test_phase18_sprint5.py` · `test_phase19_block_audit.py` · `test_phase1_fixes.py` · `test_phase20_sprint_refactor.py` · `test_phase22_rag_boot.py` · `test_phase23_scenario.py` · `test_phase25_math_sprint1.py` · `test_phase26_report_fixes.py` · `test_phase27_composite_metrics.py` · `test_phase28_risk_methodology_audit.py` · `test_phase2_modules.py` · `test_phase31_benchmark_factor_propagation.py` · `test_phase32_report_logic_fixes.py` · `test_phase33_effect_reinvest_quality.py` · `test_phase35_demo_showcase.py` · `test_phase35_price_providers.py` · `test_phase36_illiquid_proxy.py` · `test_phase37_position_contract.py` · `test_phase39_report_consistency.py` · `test_phase41_fx_base_currency.py` · `test_phase42_instrument_ssot.py` · `test_phase45_data_checks_wiring.py` · `test_phase46_p2_maintainability.py` · `test_phase47_cove_and_fx_honesty.py` · `test_phase6_currency_h2.py` · `test_stooq_coverage_probe.py` · `test_stooq_provider.py` |
| `finance.leveraged` | 1 | `test_phase28_risk_methodology_audit.py` |
| `finance.manual_portfolio` | 4 | `test_manual_fsm_flow.py` · `test_manual_portfolio_identity.py` · `test_manual_portfolio_parser.py` · `test_manual_preflight_and_locals.py` |
| `finance.period_returns` | 3 | `test_phase26_report_fixes.py` · `test_phase28_risk_methodology_audit.py` · `test_phase4_reporting.py` |
| `finance.portfolio_series` | 4 | `test_phase20_sprint_refactor.py` · `test_phase27_composite_metrics.py` · `test_phase30_benchmark_equity_curve.py` · `test_phase53_kpi_card_parity.py` |
| `finance.price_providers` | 6 | `test_manual_fsm_flow.py` · `test_phase35_price_providers.py` · `test_phase40_labels_and_boundary.py` · `test_phase4_reporting.py` · `test_stooq_price_store.py` · `test_stooq_provider.py` |
| `finance.regime` | 9 | `test_phase10_pillar_chip.py` · `test_phase19_block_audit.py` · `test_phase21_recos.py` · `test_phase2_modules.py` · `test_phase36_report_audit_fixes.py` · `test_phase48_report_verdict_audit.py` · `test_phase50_report_layer_parity.py` · `test_phase7_logic_wiring.py` · `test_phase8_report_fixes.py` |
| `finance.scenario_engine` | 1 | `test_phase46_p2_maintainability.py` |
| `finance.scenario_report` | 1 | `test_phase24_scenario_report.py` |
| `finance.scoring` | 12 | `test_phase14_refactor.py` · `test_phase15_phase3.py` · `test_phase16_sprint2.py` · `test_phase18_sprint5.py` · `test_phase19_block_audit.py` · `test_phase21_recos.py` · `test_phase2_modules.py` · `test_phase32_report_logic_fixes.py` · `test_phase39_report_consistency.py` · `test_phase42_instrument_ssot.py` · `test_phase48_report_verdict_audit.py` · `test_phase4_reporting.py` |
| `finance.scoring_orchestrator` | 7 | `test_phase10_pillar_chip.py` · `test_phase18_sprint5.py` · `test_phase28_risk_methodology_audit.py` · `test_phase42_instrument_ssot.py` · `test_phase7_logic_wiring.py` · `test_phase8_report_fixes.py` · `test_phase9_fpillar_stress.py` |
| `finance.sec_edgar` | 2 | `test_phase42_instrument_ssot.py` · `test_phase4_reporting.py` |
| `finance.security` | 1 | `test_phase29_multiuser_connection.py` |
| `finance.simulate` | 13 | `test_phase14_refactor.py` · `test_phase18_sprint5.py` · `test_phase19_block_audit.py` · `test_phase32_report_logic_fixes.py` · `test_phase33_effect_reinvest_quality.py` · `test_phase36_report_audit_fixes.py` · `test_phase42_instrument_ssot.py` · `test_phase44_live_report_fixes.py` · `test_phase46_p2_maintainability.py` · `test_phase48_report_verdict_audit.py` · `test_phase4_reporting.py` · `test_phase50_report_layer_parity.py` · `test_phase7_logic_wiring.py` |
| `finance.smart_money` | 1 | `test_phase19_block_audit.py` |
| `finance.stooq_provider` | 4 | `test_manual_freedom_parity.py` · `test_phase35_price_providers.py` · `test_phase50_report_layer_parity.py` · `test_stooq_provider.py` |
| `finance.stooq_store` | 2 | `test_stooq_price_store.py` · `test_stooq_provider.py` |
| `finance.stress` | 4 | `test_phase25_math_sprint1.py` · `test_phase28_risk_methodology_audit.py` · `test_phase4_reporting.py` · `test_phase6_currency_h2.py` |
| `finance.technicals` | 1 | `test_phase2_modules.py` |
| `freedom_portfolio` | 2 | `test_freedom_history.py` · `test_phase35_price_providers.py` |
| `freedom_portfolio.auth` | 1 | `test_freedom_auth.py` |
| `freedom_portfolio.client` | 4 | `test_freedom_client.py` · `test_freedom_history.py` · `test_phase34_broker_outage_honesty.py` · `test_phase51_broker_outage_diagnosis.py` |
| `freedom_portfolio.history` | 3 | `test_freedom_history.py` · `test_stooq_price_store.py` · `test_stooq_provider.py` |
| `freedom_portfolio.models` | 1 | `test_freedom_models.py` |
| `html_renderer` | 5 | `test_phase18_sprint5.py` · `test_phase19_block_audit.py` · `test_phase21_recos.py` · `test_phase24_scenario_report.py` · `test_phase55_mobile_and_dom_gates.py` |
| `ingest_access` | 1 | `test_phase51_ingest_bot.py` |
| `ingest_bot` | 1 | `test_phase51_ingest_bot.py` |
| `ingest_entrypoint` | 1 | `test_phase51_ingest_bot.py` |
| `pdf_charts` | 2 | `test_phase4_reporting.py` · `test_phase5_rag_quality.py` |
| `pdf_payload` | 28 | `test_factor_decomposition.py` · `test_phase10_pillar_chip.py` · `test_phase14_refactor.py` · `test_phase16_sprint2.py` · `test_phase18_sprint5.py` · `test_phase19_block_audit.py` · `test_phase25_math_sprint1.py` · `test_phase28_risk_methodology_audit.py` · `test_phase31_benchmark_factor_propagation.py` · `test_phase33_effect_reinvest_quality.py` · `test_phase36_illiquid_proxy.py` · `test_phase36_report_audit_fixes.py` · `test_phase37_position_contract.py` · `test_phase39_report_consistency.py` · `test_phase40_labels_and_boundary.py` · `test_phase41_fx_base_currency.py` · `test_phase44_live_report_fixes.py` · `test_phase47_cove_and_fx_honesty.py` · `test_phase49_contract_reaches_eyes.py` · `test_phase4_reporting.py` · `test_phase50_report_layer_parity.py` · `test_phase52_bank_identity_ssot.py` · `test_phase54_rag_chunking_and_provenance.py` · `test_phase56_payload_schema.py` · `test_phase5_rag_quality.py` · `test_phase7_logic_wiring.py` · `test_phase8_report_fixes.py` · `test_phase9_fpillar_stress.py` |
| `premium_payload` | 17 | `test_factor_decomposition.py` · `test_phase19_block_audit.py` · `test_phase21_recos.py` · `test_phase26_report_fixes.py` · `test_phase31_benchmark_factor_propagation.py` · `test_phase36_illiquid_proxy.py` · `test_phase37_base_holdings_ai.py` · `test_phase39_report_consistency.py` · `test_phase40_labels_and_boundary.py` · `test_phase44_live_report_fixes.py` · `test_phase47_cove_and_fx_honesty.py` · `test_phase48_report_verdict_audit.py` · `test_phase49_contract_reaches_eyes.py` · `test_phase50_report_layer_parity.py` · `test_phase52_bank_identity_ssot.py` · `test_phase53_kpi_card_parity.py` · `test_phase57_mockup_literal_guard.py` |
| `premium_renderer` | 2 | `test_phase49_contract_reaches_eyes.py` · `test_phase55_mobile_and_dom_gates.py` |
| `profile_manager` | 1 | `test_phase18_sprint5.py` |
| `report_charts` | 1 | `test_phase53_kpi_card_parity.py` |
| `services` | 1 | `test_phase51_ingest_bot.py` |
| `services.fx_feed` | 2 | `test_phase25_math_sprint1.py` · `test_phase6_fx_feed.py` |
| `services.macro_data` | 3 | `test_phase21_recos.py` · `test_phase4_reporting.py` · `test_phase50_report_layer_parity.py` |
| `services.quote_publisher` | 1 | `test_phase51_ingest_bot.py` |
| `services.report_storage` | 2 | `test_phase14_refactor.py` · `test_phase46_p2_maintainability.py` |
| `tg_bot` | 15 | `test_manual_fsm_flow.py` · `test_phase12_beta_safety.py` · `test_phase16_sprint2.py` · `test_phase17_admin_grant.py` · `test_phase1_fixes.py` · `test_phase23_rag_inventory.py` · `test_phase24_scenario_report.py` · `test_phase26_report_fixes.py` · `test_phase29_multiuser_connection.py` · `test_phase31_benchmark_factor_propagation.py` · `test_phase43_report_lock.py` · `test_phase51_broker_outage_diagnosis.py` · `test_phase52_bank_identity_ssot.py` · `test_phase59_brand_rename.py` · `test_phase5_rag_quality.py` |

## Модули без прямого тест-импорта

> Не обязательно «без покрытия»: модуль может проверяться через вызывающий код. Но правка здесь — без страховки в виде адресного теста.

- `agent`
- `agent.advisor_bot`
- `batch_reports`
- `finance.engine`
- `finance.engine.market_preview`
- `finance.engine.portfolio_manager`
- `finance.engine.risk_engine`
- `finance.market_calendar`
- `finance.setup_vault`
- `finance.stooq_ingest`
- `finance.stooq_symbols`
- `finance.tool_plugins`
- `freedom_portfolio.__main__`
- `freedom_portfolio.display`
- `freedom_portfolio.websocket`
- `report_mocks`
- `services.quote_ingest`
- `test_live_api`

## Тест-файлы, не привязанные к модулям `src/`

- `test_engine_benchmark_order.py`
- `test_layering.py`
- `test_phase58_boot_import_race.py`
- `test_repo_hygiene.py`
