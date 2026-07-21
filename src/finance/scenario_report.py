# -*- coding: utf-8 -*-
"""
Scenario-tier report builder — превращает уже посчитанный `results`
(`analyze_all`) в самодостаточный payload для шаблона `report_scenario_v3.html`.

Дизайн (docs/roadmap/ROADMAP_SCENARIO_TIER.md):
  • Панель A — «Сценарная диагностика портфеля»: 5 core-метрик через движок
    (ковариационная σ_p, RFR-Sharpe, Σwβ), Euler-MCTR таблица вкладов в риск,
    выживаемость в 3 макро-режимах (группировка существующих 7 шоков),
    funding-кандидаты (слабые звенья по decision-tree).
  • Панель B — «Проверка историей»: walk-forward бэктест трендового правила
    (цена > 200-дневной средней) на крупнейшей позиции, с look-ahead guard
    и жёстко зашитыми дисклеймерами.

Принципы: ZERO LLM-API (детерминизм → 1 токен), переиспользование
`scenario_engine` (математика уже протестирована в test_phase23_scenario) и
Do-No-Harm — модуль ТОЛЬКО ЧИТАЕТ `results`, ничего не мутирует и не считает
финансовую математику мимо `scenario_engine`.

Источник цен: `results["history_result"].data` — полная ценовая матрица
(колонки = RESOLVED тикеры, .US/-USD).  Веса берём из `performance_table`
(DISPLAY тикеры) и мапим на resolved тем же `MAC3RiskEngine.resolve_tickers`,
что строил матрицу, — один и тот же справочник, поэтому колонки совпадают.
Окно истории теперь 5 лет (HISTORY_LOOKBACK_DAYS=1825 = SCENARIO_LOOKBACK_DAYS),
так что диагностика идёт на честной 5-летней выборке.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from finance import scenario_engine as se

logger = logging.getLogger(__name__)


def _weights_and_prices(results: dict) -> Optional[tuple[pd.DataFrame, dict, dict, dict]]:
    """Reconstruct (prices, weights_by_resolved, betas_by_resolved,
    display_by_resolved) из results.  None — если данных не хватает."""
    hist = results.get("history_result")
    prices = getattr(hist, "data", None)
    perf = results.get("performance_table")
    total = float(results.get("total_value") or 0.0)
    if prices is None or getattr(prices, "empty", True) or perf is None \
            or getattr(perf, "empty", True) or total <= 0:
        return None

    display = [str(t) for t in perf["Ticker"].tolist()] if "Ticker" in perf.columns \
        else [str(t) for t in perf.index.tolist()]
    # Resolve DISPLAY→RESOLVED тем же справочником, что строил ценовую матрицу.
    try:
        from finance.investment_logic import MAC3RiskEngine
        resolved = MAC3RiskEngine().resolve_tickers(display)
    except Exception as exc:                       # pragma: no cover - defensive
        logger.warning("scenario: resolve_tickers failed: %s", exc)
        resolved = display

    disp_by_res: dict[str, str] = {}
    w_by_res: dict[str, float] = {}
    beta_by_res: dict[str, float] = {}
    for i, (_, row) in enumerate(perf.iterrows()):
        res = resolved[i] if i < len(resolved) else None
        if res is None or res not in prices.columns:
            continue
        cv = float(row.get("Current_Value") or 0.0)
        w = cv / total
        if w <= 0:                                 # кэш/короткие ноги — не риск-актив
            continue
        w_by_res[res] = w_by_res.get(res, 0.0) + w
        disp_by_res.setdefault(res, str(row.get("Ticker", res)))
        b = row.get("Beta_Market")
        try:
            if b is not None and np.isfinite(float(b)):
                beta_by_res[res] = float(b)
        except (TypeError, ValueError):
            pass
    if not w_by_res:
        return None
    return prices, w_by_res, beta_by_res, disp_by_res


def _trend_signal_factory(ticker: str, window: int = 200):
    """Правило Панели B: цена закрытия выше своей `window`-дневной средней
    (классический тренд-фильтр).  Видит СТРОГО `visible` (look-ahead guard
    обеспечивает walk_forward)."""
    def _sig(visible: pd.DataFrame) -> bool:
        if ticker not in visible.columns:
            return False
        s = visible[ticker].dropna()
        if len(s) < window:
            return False
        return bool(float(s.iloc[-1]) > float(s.tail(window).mean()))
    return _sig


def build_scenario_payload(results: dict) -> dict:
    """Собрать payload сценарного тира.  Всегда возвращает dict со стабильной
    схемой; при нехватке данных — `scenario.available=False` (шаблон покажет
    честную заглушку, отчёт не падает)."""
    payload: dict = {
        "tier": "scenario",
        "scenario": {"available": False,
                     "disclaimers": list(se.DISCLAIMERS),
                     "window_days": se.SCENARIO_LOOKBACK_DAYS},
    }
    core = _weights_and_prices(results)
    if core is None:
        payload["scenario"]["note"] = "Недостаточно ценовой истории для сценарной диагностики."
        return payload
    prices, weights, betas, disp_by_res = core

    sc: dict = {"available": True,
                "disclaimers": list(se.DISCLAIMERS),
                "window_days": se.SCENARIO_LOOKBACK_DAYS,
                "n_obs": int(len(prices)),
                "holdings_display": disp_by_res}

    # ── Панель A — 5 core-метрик (движок) ────────────────────────────────
    metrics = se.five_metrics(prices, weights, betas=betas)
    sc["metrics"] = metrics

    # Euler-MCTR: вклад каждой позиции в риск (ρ·σ, CTRisk, %).  Подписываем
    # DISPLAY-тикером для читаемости.
    mtab = se.mctr_table(prices, weights)
    if mtab:
        rows = sorted(mtab["rows"], key=lambda r: -abs(r.get("pct_ctr") or 0.0))
        for r in rows:
            r["display"] = disp_by_res.get(r["ticker"], r["ticker"])
        sc["mctr_rows"] = rows
        sc["vol_cov"] = mtab["vol_cov"]

    # Funding-кандидаты (слабые звенья).  TRC% берём из основного отчёта
    # (Euler движка) — по DISPLAY-тикерам, мапим на resolved.
    trc_by_res: dict[str, float] = {}
    perf = results.get("performance_table")
    if perf is not None and not perf.empty and "Euler_Risk_Contribution_Pct" in perf.columns:
        res_by_disp = {v: k for k, v in disp_by_res.items()}
        for _, row in perf.iterrows():
            d = str(row.get("Ticker", ""))
            res = res_by_disp.get(d)
            trc = row.get("Euler_Risk_Contribution_Pct")
            if res is not None and trc is not None:
                try:
                    trc_by_res[res] = float(trc)
                except (TypeError, ValueError):
                    pass
    funding = se.funding_candidates(prices, weights, trc_pct=trc_by_res or None)
    for f in funding:
        f["display"] = disp_by_res.get(f["ticker"], f["ticker"])
    sc["funding"] = funding

    # Выживаемость в 3 макро-режимах — группировка существующих стресс-строк.
    # Единицы: движок кладёт stress.port_pct как ДОЛЮ (−0.093), а
    # scenario_engine.regime_survival работает в ПРОЦЕНТАХ (порог −10.0, как в
    # его юнит-тесте).  Нормализуем ДОЛЮ→ПРОЦЕНТ здесь, не трогая контракт движка.
    stress_pct = [{"name": r.get("name", ""),
                   "port_pct": float(r.get("port_pct") or 0.0) * 100.0}
                  for r in (results.get("stress_scenarios") or [])
                  if isinstance(r.get("port_pct"), (int, float))]
    sc["regime_survival"] = se.regime_survival(stress_pct)

    # Молодые активы, исключённые из ковариации (честность CoVe).
    excluded = [disp_by_res.get(t, t) for t in weights
                if t in prices.columns and prices[t].dropna().shape[0] < se._MIN_PERIODS_COV]
    sc["excluded_young"] = excluded

    # ── Панель B — walk-forward на крупнейшей позиции ────────────────────
    top_res = max(weights, key=weights.get)
    bt = se.walk_forward(prices, top_res, _trend_signal_factory(top_res))
    sc["backtest"] = {
        "ticker": disp_by_res.get(top_res, top_res),
        "rule": "Цена выше 200-дневной средней (трендовый фильтр)",
        "summary": bt.summary(),
    }
    payload["scenario"] = sc
    return payload


__all__ = ["build_scenario_payload"]
