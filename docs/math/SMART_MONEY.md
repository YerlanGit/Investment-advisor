# SMART_MONEY.md — слой инсайдеров / Smart Money (SEC Form 4)

> **Статус:** архитектура внедрена и **рендерится в DEEP-отчёте** (секция «Smart Money ·
> Инсайдерские сделки»). Слой **gated**: без флага показывает состояние «источник не
> активирован». Живой парсер Form-4 — следующая фаза (нужен EDGAR-краулер + кэш).
> **Код:** `src/finance/smart_money.py` · билдер `pdf_payload._build_smart_money` ·
> шаблон `report_deep_v3.html` (блок B2.4) · CoVe `data_lineage._smart_money_status`.

---

## 1. Зачем это нужно

«Smart Money» — легальный сигнал убеждённости: корпоративные **инсайдеры** (CEO/CFO/
директора), покупающие/продающие СВОИ акции. **Кластерные покупки** (несколько инсайдеров
в одном окне) — один из немногих публичных сигналов с документированной предсказательной
силой по форвардной доходности 1–3 мес. Покупка CEO/CFO весит сильнее, чем 10%-владельца.

## 2. Архитектура (Data Ingestion → Scoring → UI)

```
SEC EDGAR Form 4 (XML)                                 [ИСТОЧНИК, free]
        │  fetch(ticker) -> {net_flow_usd, distinct_buyers, buy_count,
        │                    sell_count, top_role, as_of}
        ▼
finance.smart_money.build_insider_signals(tickers, fetch=…, market_caps=…)
        │   • gated: SMART_MONEY_INSIDERS=1 (иначе status="disabled")
        │   • per-ticker → InsiderSignal (score_insider_flow)
        ▼
results["smart_money"]  (в analyze_all)                 [ДВИЖОК]
        ▼
pdf_payload._build_smart_money → payload["smart_money"]  [BUILDER]
        │   status ∈ {disabled | active | missing}
        ▼
report_deep_v3.html  «Smart Money» панель                [UI]
        │   • disabled → плашка «источник не активирован» + как включить
        │   • active   → таблица: тикер · нетто-поток · покупки/продажи · кластер · score
        ▼
data_lineage._smart_money_status → строка CoVe           [ВЕРИФИКАЦИЯ]
```

### Детерминированный скорер (tier 1, уже в коде)
`score_insider_flow(net_flow_usd, distinct_buyers, sell_count, market_cap_usd, role_weight)`:
- база = нетто-поток в **bps от рыночной капитализации** (сатурация: ~50 bps ⇒ ~1.0);
- × **role_weight** (CEO/CFO 1.5, President 1.3, Director 1.0, 10%Owner 0.6);
- **+0.5** если кластер (≥3 разных покупателя и нетто > 0);
- **−0.25** если инсайдеры распродают (sells>0, нетто<0);
- клип в **[−2, +2]** — та же шкала, что у пилларов 4-Pillar.

## 3. Как включить

| Шаг | Действие |
|---|---|
| 1 | `SMART_MONEY_INSIDERS=1` в окружении Cloud Run |
| 2 | Реализовать провайдер `fetch(ticker)` поверх EDGAR (`https://data.sec.gov/submissions/CIK<10>.json` → Form-4 XML `ownershipDocument`); соблюдать User-Agent и ≤10 req/s |
| 3 | Кэш в Cloud Function (ночной обход вселенной портфелей), как RAG-ingest |
| 4 | (Опц.) `market_caps` для масштабирования score в bps |

Без шага 2 слой остаётся в состоянии `disabled` — секция **видна** и объясняет статус
(не «пропадает»), что и было целью B2.4.

## 4. Что СЕЙЧАС отложено (и почему)

| Источник | Статус | Причина |
|---|---|---|
| SEC Form 4 (инсайдеры) | 🟡 фундамент + UI готовы, нужен live-провайдер | EDGAR-краулер + кэш — отдельная фаза |
| Сделки политиков (STOCK Act / PTR) | ⏸ дизайн | Чистые данные — только платные агрегаторы (Quiver, Capitol Trades) |
| Госстимулы / контракты (USASpending) | ⏸ дизайн | Крупный ингест + entity-resolution |

Интерфейс `fetch` спроектирован так, что каждый из этих источников добавляется как
ещё один провайдер без изменения скоринга и UI.

## 5. Прогнозные модели на инсайдерских данных

1. **Rule / Event-study (готов):** кластер покупок → форвардная abnormal return 1–3 мес;
   роль-взвешивание (CEO/CFO > 10%-owner). Выход — тилт пиллара A/D, не самостоятельный Buy/Sell.
2. **Cross-sectional ML (след.):** фичи = {нетто/мкап, role-weighted интенсивность, флаг
   10b5-1 vs дискреционная, размер кластера, дни с прошлого кластера, sector-z} →
   gradient-boosted ранкер 21-дн форвардного остаточного (alpha) дохода; walk-forward OOS;
   в отчёт — перцентиль «инсайдерской убеждённости».
3. **Regime-conditioned (позже):** условить ML-голову на макро-режиме (BLOCK 3.4) — покупки
   инсайдеров в Recovery сильнее как long-сигнал, чем в поздней Expansion. Сюда же стыкуются
   UNRATE/Real-GDP и (на след. фазе) госстимулы/контракты как контекст-фичи.

Все тиры — **advisory** (CoVe-тег «не ИИР») и проходят те же LLM-чекеры; ни один выход модели
не минует gatekeeper.

## 6. Тесты

`tests/test_phase19_block_audit.py::SmartMoneyTest` — gated-по-умолчанию, кластер→положительный
score, распродажа→отрицательный, инъекция провайдера. UI-состояния — `_build_smart_money`.
