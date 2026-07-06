"""
Scenario Analysis engine — Панель A (forward framework) + Панель B (backtest).

Дизайн: docs/ROADMAP_SCENARIO_TIER.md (v2, 7-шаговый фреймворк владельца).
Принципы: детерминизм, ZERO LLM-API, переиспользование конвенций движка и
Do-No-Harm — модуль АДДИТИВЕН: ничего в investment_logic / scoring_orchestrator
не импортирует его, базовый отчёт считается байт-идентично прежнему.

Строже фреймворка (зафиксировано в спеке):
  • волатильность портфеля = √(wᵀΣw) по ковариации дневных доходностей
    (НЕ gross Σwσ — тот показываем только как reference);
  • Sharpe = (E[r] − RFR)/σ_p, RFR из env RISK_FREE_RATE (default 0.045 USD);
  • Панель B: сигнал на дату t видит ТОЛЬКО df[df.index <= t] (look-ahead guard).

Lookback: сценарный бэктест запрашивает СВОЁ окно (SCENARIO_LOOKBACK_DAYS,
default 1825 ≈ 5 лет) — окно ОСНОВНОГО отчёта (HISTORY_LOOKBACK_DAYS=730)
не трогаем.  Молодые активы не роняют матрицу: ковариация считается pairwise
с min_periods (NaN-маскирование), веса недостающих метрик исключаются с
перенормировкой.

Сверка с формульным референсом (Bloomberg-style, разделы 1–9) — что совпадает
и где осознанно отклоняемся:
  • 1.3/1.4  геометрическая годовая доходность (1+R_cum)^(1/N)−1 и Σw·R — ✓;
  • 2.1–2.2  выборочная σ с поправкой Бесселя (ddof=1) ×√252 — ✓;
  • 2.5/2.6  gross Σwσ (reference) vs истинная √(wᵀΣw) — обе, основная = истинная;
  • 5.1–5.3  Euler-разложение: MCTR(i)=ρ(i,P)·σ(i), CTRisk=w·MCTR,
             Σ CTRisk = σ_p ТОЧНО, Σ pct_CTR = 100% — `mctr_table()`;
  • 8.2/8.3  β(P)=Σw·β, DY(P)=Σw·DY — плоская сумма по ВСЕМ позициям,
             отсутствующее значение = вклад 0 (β золота/бондов 0.00), без
             перенормировки на покрытие — иначе β(P)/DY(P) завышаются;
  • 4.1      Sharpe = (E[r]−RFR)/σ_p; референс берёт Rf≈5.3% (3M T-bill) либо 0%
             для gross-сравнения — у нас Rf из env RISK_FREE_RATE (default
             0.045), единая конвенция с основным отчётом (Base Currency + RFR);
  • ОТКЛОНЕНИЕ (оценка Σ): референс — EWMA λ=0.96 (half-life ≈17 дн.);
    сценарный тир СОЗНАТЕЛЬНО берёт равновзвешенную выборочную ковариацию на
    5-летнем окне: дельты «до/после» должны отражать сделку, а не дрожание
    оценщика последнего месяца.  Основной движок (MAC3) использует
    EWMA(hl=63)⊕Ledoit-Wolf — реактивность там, стабильность здесь.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ── Фаза 0/1 · параметры (все — env-переопределяемые) ────────────────────────
SCENARIO_LOOKBACK_DAYS = int(os.environ.get("SCENARIO_LOOKBACK_DAYS", "1825"))
_RFR_DEFAULT           = 0.045          # USD, синхронно с основным отчётом
_MIN_PERIODS_COV       = 252            # ≥1 год общих наблюдений для пары
# Шаг 2 фреймворка — funding decision-tree:
FUND_SHARPE_MAX        = 0.50           # Sharpe ниже → кандидат на продажу
FUND_CORR_DUP          = 0.90           # дублирующийся риск (пара-двойник)
# Шаг 3 — оценка покупок:
BUY_CORR_MAX           = 0.85           # выше к «ядру» портфеля — не диверсифицирует
# Шаг 4 — sizing-капы (MCTR-дисциплина фреймворка):
HIGH_VOL_ANN           = 0.35           # «высоковолатильный» актив
HIGH_VOL_WEIGHT_CAP    = 0.15           # жёсткий потолок доли
MCTR_SHARE_CAP         = 0.35           # пост-трейд доля позиции в риске ≤35%

# ── Фаза 3 · юридический комплаенс (жёстко зашитые строки Панели B) ──────────
DISCLAIMERS: tuple[str, ...] = (
    "Историческая доходность не гарантирует будущих результатов.",
    "Расчёты подвержены искажению выжившего (survivorship bias), так как не "
    "учитывают компании, прошедшие процедуру делистинга или банкротства.",
    "Данный отчёт является результатом математического моделирования и не "
    "является индивидуальной инвестиционной рекомендацией (ИИР).",
)

# Шаг 1 — архетипы (4 существующих бакета + новый high_dividend).
ARCHETYPES: dict[str, dict] = {
    "rebalance":     {"label": "Ребалансировка",   "optimize": "sharpe"},
    "smart_money":   {"label": "Умные деньги",     "optimize": "return"},
    "high_dividend": {"label": "Высокие дивиденды", "optimize": "yield_sharpe"},
    "protect":       {"label": "Защита капитала",  "optimize": "beta_vol_min"},
}

# Шаг 7 — группировка семи параметрических шоков движка в 3 макро-режима.
MACRO_REGIME_GROUPS: dict[str, tuple[str, ...]] = {
    "risk_on":    ("Fed cut", "cut −50"),
    "rate_shock": ("Fed +50", "Credit blow-out", "CPI shock", "+200 bps"),
    "risk_off":   ("Tech sell-off", "Geopolitical", "risk-off", "USD +5%"),
}
MACRO_REGIME_LABELS = {"risk_on": "Risk-On / бычий рынок",
                       "rate_shock": "Шок процентных ставок",
                       "risk_off": "Risk-Off / рецессия"}


def _rfr() -> float:
    try:
        return float(os.environ.get("RISK_FREE_RATE", _RFR_DEFAULT))
    except (TypeError, ValueError):
        return _RFR_DEFAULT


# ── Матричное ядро (self-check: только ЧТЕНИЕ prices; NaN-safe) ──────────────

def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change(fill_method=None)


def _cov_core(prices: pd.DataFrame, weights: dict[str, float],
              min_periods: int) -> Optional[tuple[list[str], np.ndarray, np.ndarray]]:
    """Общее ядро σ_p и Euler-разложения: pairwise-ковариация дневных
    доходностей с NaN-маскированием молодых активов.

    pandas `.cov(min_periods=…)` считает каждую пару на пересечении дат;
    пары без min_periods дают NaN → такие активы исключаются из расчёта с
    перенормировкой весов (портфель не падает, в payload уходит список
    исключённых — честность CoVe).  Возвращает (тикеры, w-нормированные,
    Σ дневная) — один код-путь гарантирует, что Σ CTRisk из `mctr_table`
    сходится с `portfolio_vol_cov` бит-в-бит (тождество Эйлера)."""
    cols = [t for t in weights if t in prices.columns and weights[t] > 0]
    if not cols:
        return None
    rets = _daily_returns(prices[cols])
    cov = rets.cov(min_periods=min_periods)
    ok = [c for c in cols if not cov.loc[c].isna().all()]
    if not ok:
        return None
    w = np.array([weights[c] for c in ok], dtype=float)
    w = w / w.sum()
    return ok, w, cov.loc[ok, ok].fillna(0.0).values


def portfolio_vol_cov(prices: pd.DataFrame, weights: dict[str, float],
                      *, min_periods: int = _MIN_PERIODS_COV) -> Optional[float]:
    """Истинная годовая σ_p = √(wᵀΣw)·√252 (формула 2.6 референса)."""
    core = _cov_core(prices, weights, min_periods)
    if core is None:
        return None
    _, w, sigma = core
    var_d = float(w @ sigma @ w)
    return float(np.sqrt(max(var_d, 0.0)) * np.sqrt(TRADING_DAYS))


def mctr_table(prices: pd.DataFrame, weights: dict[str, float],
               *, min_periods: int = _MIN_PERIODS_COV) -> Optional[dict]:
    """Euler-разложение риска по позициям (формулы 5.1–5.3 референса).

    MCTR(i)   = (Σw)ᵢ/σ_p = ρ(i,P)·σ(i)   — маржинальный вклад в риск;
    CTRisk(i) = w(i)·MCTR(i),  Σ CTRisk(i) = σ_p ТОЧНО (тождество Эйлера);
    pct_CTR(i) = CTRisk(i)/σ_p·100,  Σ = 100%.

    Годовая шкала (×√252).  Хедж с ρ(i,P)<0 даёт ОТРИЦАТЕЛЬНЫЕ MCTR/CTRisk —
    позиция снижает риск портфеля (пример референса: XOM с ρ=−0.15).
    Молодые активы маскируются тем же ядром, что и `portfolio_vol_cov`."""
    core = _cov_core(prices, weights, min_periods)
    if core is None:
        return None
    ok, w, sigma = core
    var_d = float(w @ sigma @ w)
    if var_d <= 0:
        return None
    ann = float(np.sqrt(TRADING_DAYS))
    sd_d = float(np.sqrt(var_d))
    vol_p = sd_d * ann
    mctr = (sigma @ w) / sd_d * ann              # = ρ(i,P)·σᵢ, годовая
    ctrisk = w * mctr                            # Σ = vol_p (Эйлер)
    sig_i = np.sqrt(np.clip(np.diag(sigma), 0.0, None)) * ann
    rows = []
    for j, t in enumerate(ok):
        rho = float(mctr[j] / sig_i[j]) if sig_i[j] > 0 else None
        rows.append({"ticker": t, "weight": float(w[j]),
                     "sigma_i": float(sig_i[j]), "rho_ip": rho,
                     "mctr": float(mctr[j]), "ctrisk": float(ctrisk[j]),
                     "pct_ctr": float(ctrisk[j] / vol_p * 100.0)})
    return {"rows": rows, "vol_cov": vol_p}


def ann_return(prices: pd.DataFrame, ticker: str) -> Optional[float]:
    s = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
    if len(s) < _MIN_PERIODS_COV:
        return None
    years = len(s) / TRADING_DAYS
    total = float(s.iloc[-1] / s.iloc[0])
    if total <= 0:
        return None
    return float(total ** (1.0 / years) - 1.0)


def ann_vol(prices: pd.DataFrame, ticker: str) -> Optional[float]:
    r = _daily_returns(prices[[ticker]])[ticker].dropna() if ticker in prices.columns \
        else pd.Series(dtype=float)
    if len(r) < _MIN_PERIODS_COV:
        return None
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(prices: pd.DataFrame, ticker: str) -> Optional[float]:
    """RFR-adjusted (конвенция движка), НЕ Return/Vol фреймворка."""
    r, v = ann_return(prices, ticker), ann_vol(prices, ticker)
    if r is None or v is None or v <= 0:
        return None
    return float((r - _rfr()) / v)


def corr_to_basket(prices: pd.DataFrame, ticker: str,
                   basket: dict[str, float]) -> Optional[float]:
    """Корреляция кандидата к взвешенному «ядру» текущего портфеля (шаг 3)."""
    core = {t: w for t, w in basket.items() if t in prices.columns and t != ticker}
    if not core or ticker not in prices.columns:
        return None
    rets = _daily_returns(prices[list(core) + [ticker]])
    w = pd.Series(core, dtype=float)
    w = w / w.sum()
    core_ret = (rets[list(core)] * w).sum(axis=1, min_count=1)
    pair = pd.concat([core_ret, rets[ticker]], axis=1).dropna()
    if len(pair) < _MIN_PERIODS_COV:
        return None
    return float(pair.corr().iloc[0, 1])


# ── Шаг 2 · алгоритм фондирования ────────────────────────────────────────────

def funding_candidates(prices: pd.DataFrame, holdings: dict[str, float],
                       *, trc_pct: Optional[dict[str, float]] = None) -> list[dict]:
    """Слабые звенья по decision-tree фреймворка. Возвращает [{ticker, flags,
    sharpe, ann_return, score}] отсортированный: худшие — первыми."""
    out: list[dict] = []
    rets = _daily_returns(prices[[c for c in holdings if c in prices.columns]])
    corr = rets.corr(min_periods=_MIN_PERIODS_COV) if not rets.empty else pd.DataFrame()
    for t, w in holdings.items():
        sh, ar = sharpe(prices, t), ann_return(prices, t)
        flags: list[str] = []
        if sh is not None and sh < FUND_SHARPE_MAX:
            flags.append(f"Sharpe {sh:.2f} < {FUND_SHARPE_MAX}")
        if ar is not None and ar < 0:
            flags.append(f"доходность окна {ar:+.1%} < 0")
        if t in corr.columns:
            dup = corr[t].drop(index=t, errors="ignore")
            twin = dup[dup >= FUND_CORR_DUP]
            if not twin.empty:
                flags.append(f"дублирующийся риск: corr≥{FUND_CORR_DUP} c "
                             + ",".join(twin.index[:2]))
        if trc_pct and t in trc_pct and w > 0:
            ratio = trc_pct[t] / (w * 100 if w <= 1 else w)
            if ratio > 1.6:
                flags.append(f"TRC {trc_pct[t]:.0f}% несоразмерен весу")
        if flags:
            out.append({"ticker": t, "weight": w, "sharpe": sh,
                        "ann_return": ar, "flags": flags,
                        "score": (sh if sh is not None else 9.9)})
    return sorted(out, key=lambda x: x["score"])


# ── Шаг 4 · sizing c Euler-MCTR дисциплиной ──────────────────────────────────

def size_position(prices: pd.DataFrame, after_weights: dict[str, float],
                  ticker: str, proposed_w: float) -> tuple[float, str]:
    """Режет предложенный вес по двум правилам фреймворка: (а) vol>35% → ≤15%;
    (б) пост-трейд доля в риске (ERC%) ≤ MCTR_SHARE_CAP. Возвращает (вес, why)."""
    why = []
    w = float(proposed_w)
    v = ann_vol(prices, ticker)
    if v is not None and v > HIGH_VOL_ANN and w > HIGH_VOL_WEIGHT_CAP:
        w = HIGH_VOL_WEIGHT_CAP
        why.append(f"vol {v:.0%}>35% → потолок {HIGH_VOL_WEIGHT_CAP:.0%}")
    # ERC-доля кандидата в пост-трейд портфеле (тот же Euler, что в движке):
    test = dict(after_weights)
    test[ticker] = w
    cols = [t for t in test if t in prices.columns and test[t] > 0]
    if ticker in cols and len(cols) >= 2:
        rets = _daily_returns(prices[cols])
        cov = rets.cov(min_periods=_MIN_PERIODS_COV).fillna(0.0)
        wv = np.array([test[c] for c in cols])
        wv = wv / wv.sum()
        sig = cov.values
        var_p = float(wv @ sig @ wv)
        if var_p > 0:
            mctr = sig @ wv / np.sqrt(var_p)
            erc = wv * mctr / np.sqrt(var_p)          # доли риска, Σ=1
            share = float(erc[cols.index(ticker)])
            while share > MCTR_SHARE_CAP and w > 0.02:
                w = round(w - 0.01, 4)
                wv = np.array([test[c] if c != ticker else w for c in cols])
                wv = wv / wv.sum()
                var_p = float(wv @ sig @ wv)
                mctr = sig @ wv / np.sqrt(var_p)
                share = float((wv * mctr / np.sqrt(var_p))[cols.index(ticker)])
            if w < proposed_w and not why:
                why.append(f"ERC-доля > {MCTR_SHARE_CAP:.0%} → вес срезан")
    return w, ("; ".join(why) if why else "в пределах капов")


# ── Шаг 6 · метрики «до/после» ───────────────────────────────────────────────

def five_metrics(prices: pd.DataFrame, weights: dict[str, float], *,
                 betas: Optional[dict[str, float]] = None,
                 div_yield: Optional[dict[str, float]] = None,
                 pe: Optional[dict[str, float]] = None) -> dict:
    """Ann.Return (взвеш.), σ_p (КОВАРИАЦИОННАЯ), Sharpe (RFR-adj), Beta Σwβ,
    DY Σw·dy (+ wtd P/E).

    Две агрегации — по референсу:
      • ann_return / gross vol / P/E: перенормировка на покрытие (молодой актив
        без истории ≠ актив с нулевой доходностью — честная оценка по
        покрытым);
      • beta / div_yield (формулы 8.2/8.3): плоская Σ w·x по ВСЕМ позициям —
        отсутствующее значение даёт вклад 0 (β кэша/золота/бондов = 0.00,
        DY неплательщика = 0), веса НЕ перенормируются, иначе β(P) и DY(P)
        завышаются."""
    def _wavg(vals: Optional[dict[str, float]]) -> Optional[float]:
        if not vals:
            return None
        pairs = [(weights[t], vals[t]) for t in weights
                 if t in vals and vals[t] is not None and weights[t] > 0]
        if not pairs:
            return None
        tw = sum(p[0] for p in pairs)
        return float(sum(w * v for w, v in pairs) / tw) if tw else None

    def _wsum(vals: Optional[dict[str, float]]) -> Optional[float]:
        if not vals:
            return None
        tw = sum(w for w in weights.values() if w > 0)
        pairs = [(weights[t], vals[t]) for t in weights
                 if t in vals and vals[t] is not None and weights[t] > 0]
        if not pairs or not tw:
            return None
        return float(sum(w * v for w, v in pairs) / tw)

    rets = {t: ann_return(prices, t) for t in weights}
    ann_r = _wavg({t: r for t, r in rets.items() if r is not None})
    vol_p = portfolio_vol_cov(prices, weights)
    gross = _wavg({t: ann_vol(prices, t) for t in weights})
    shrp = None
    if ann_r is not None and vol_p and vol_p > 0:
        shrp = float((ann_r - _rfr()) / vol_p)
    return {"ann_return": ann_r, "vol_cov": vol_p, "vol_gross_ref": gross,
            "sharpe_rfr": shrp, "beta": _wsum(betas),
            "div_yield": _wsum(div_yield), "pe": _wavg(pe), "rfr": _rfr()}


def delta_metrics(before: dict, after: dict) -> dict:
    out = {}
    for k in ("ann_return", "vol_cov", "sharpe_rfr", "beta", "div_yield", "pe"):
        b, a = before.get(k), after.get(k)
        out[k] = (None if b is None or a is None else float(a - b))
    return out


# ── Шаг 7 · выживаемость сценария в 3 макро-режимах ─────────────────────────

def regime_survival(stress_rows: list[dict]) -> list[dict]:
    """Группирует существующие стресс-строки движка ({name, port_pct|pct}) в
    3 макро-режима: средний Δ портфеля по шокам группы."""
    out = []
    for key, needles in MACRO_REGIME_GROUPS.items():
        vals = []
        for row in stress_rows or []:
            name = str(row.get("name", ""))
            if any(n.lower() in name.lower() for n in needles):
                p = row.get("port_pct", row.get("pct"))
                if isinstance(p, (int, float)):
                    vals.append(float(p))
        avg = float(np.mean(vals)) if vals else None
        out.append({"regime": key, "label": MACRO_REGIME_LABELS[key],
                    "avg_pct": avg, "n_shocks": len(vals),
                    "survives": (avg is None or avg > -10.0)})
    return out


# ── Панель B · walk-forward backtest (look-ahead guard) ──────────────────────

@dataclass
class BacktestResult:
    n_signals: int
    hit_rate_63d: Optional[float]
    fwd_returns: dict = field(default_factory=dict)   # {'21d': [...], ...}
    disclaimers: tuple[str, ...] = DISCLAIMERS

    def summary(self) -> dict:
        def _agg(xs):
            return (None if not xs else
                    {"mean": float(np.mean(xs)), "median": float(np.median(xs)),
                     "worst": float(np.min(xs)), "n": len(xs)})
        return {"n_signals": self.n_signals,
                "hit_rate_63d": self.hit_rate_63d,
                "horizons": {h: _agg(v) for h, v in self.fwd_returns.items()},
                "disclaimers": list(self.disclaimers)}


def walk_forward(prices: pd.DataFrame, ticker: str,
                 signal_fn: Callable[[pd.DataFrame], bool],
                 *, step_days: int = 21,
                 horizons: tuple[int, ...] = (21, 63, 126),
                 warmup: int = _MIN_PERIODS_COV) -> BacktestResult:
    """Помесячная сетка; signal_fn получает СТРОГО df[df.index <= t] (полный
    фрейм — сигнал может смотреть на бенчмарки), исходы — forward-доходности
    self-финансируемой сделки close(t+1)→close(t+1+h)."""
    if ticker not in prices.columns:
        return BacktestResult(0, None)
    idx = prices[ticker].dropna().index
    fwd: dict[str, list[float]] = {f"{h}d": [] for h in horizons}
    n = 0
    max_h = max(horizons)
    for i in range(warmup, len(idx) - max_h - 1, step_days):
        t = idx[i]
        visible = prices.loc[prices.index <= t]          # ← look-ahead guard
        try:
            fired = bool(signal_fn(visible))
        except Exception:
            fired = False
        if not fired:
            continue
        n += 1
        entry = float(prices.loc[idx[i + 1], ticker])    # close следующего дня
        if not np.isfinite(entry) or entry <= 0:
            continue
        for h in horizons:
            exit_ = float(prices.loc[idx[i + 1 + h], ticker])
            if np.isfinite(exit_) and exit_ > 0:
                fwd[f"{h}d"].append(exit_ / entry - 1.0)
    hits = fwd.get("63d") or []
    hr = (float(np.mean([1.0 if x > 0 else 0.0 for x in hits]))
          if hits else None)
    return BacktestResult(n, hr, fwd)


__all__ = [
    "ARCHETYPES", "DISCLAIMERS", "MACRO_REGIME_GROUPS", "SCENARIO_LOOKBACK_DAYS",
    "portfolio_vol_cov", "mctr_table", "ann_return", "ann_vol", "sharpe",
    "corr_to_basket",
    "funding_candidates", "size_position", "five_metrics", "delta_metrics",
    "regime_survival", "walk_forward", "BacktestResult",
]
