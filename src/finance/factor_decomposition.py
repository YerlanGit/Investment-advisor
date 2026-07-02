# -*- coding: utf-8 -*-
"""
Factor-variance decomposition + marginal-overlap analytics (additive layer).

Отвечает на два вопроса, которых нет в базовой структурной модели:

1. «Откуда берётся риск ПО ИСТОЧНИКАМ?» — Euler-декомпозиция дисперсии
   портфеля по ФАКТОРАМ (а не по активам, как TRC):

       σ²_port = wᵀ(B·F·Bᵀ + D)w  =  bᵀFb + wᵀDw,   где b = Bᵀw

   Вклад фактора f (Euler, суммируется точно в систематическую часть):

       c_f = b_f · (F·b)_f

   Идиосинкратическая часть = wᵀDw (специфика бумаг, не объяснённая
   ни одним фактором).  Тождество σ²  =  Σ_f c_f + wᵀDw выполняется
   АЛГЕБРАИЧЕСКИ точно — модуль проверяет его и репортит зазор
   (`identity_gap_pct`, float-шум ≈ 0).

2. «Что позиция добавляет сверх того, что уже есть?» — marginal overlap:
   • systematic-корреляция пары активов из Σ_sys = B·F·Bᵀ — пары с
     corr ≥ TWIN_CORR_THRESHOLD суть «факторные двойники» (одна и та же
     факторная ставка дважды → концентрация, а не диверсификация);
   • `unique_risk_pct` актива = d_i / (Σ_sys,ii + d_i) — доля СОБСТВЕННОГО
     (идиосинкратического) риска в полной дисперсии актива; низкая доля +
     двойник = позиция почти ничего не добавляет к факторному профилю.

Контракт: модуль ЧИСТЫЙ (numpy-only, без I/O), ничего не мутирует и не
меняет существующую математику движка — он читает те же B/F/D/w, которые
структурная модель уже построила.  Движок вызывает его в try/except и при
любой ошибке кладёт {} (тот же graceful-паттерн, что и κ-диагностика 4.6).

Замечание о знаках: c_f может быть ОТРИЦАТЕЛЬНЫМ (фактор-хедж, например
дюрация против акций) — знак сохраняется и подписывается как
«диверсифицирует», доли при этом всё равно суммируются в 100%.
"""

from __future__ import annotations

import numpy as np

# ── Tuning constants (single place, mirrors scoring.HOTSPOT_TRC_PCT style) ───
# Пара активов — «факторные двойники», когда их СИСТЕМАТИЧЕСКАЯ корреляция
# (через общие факторы, идиосинкратика исключена) не ниже порога…
TWIN_CORR_THRESHOLD = 0.90
# …и каждый весит хотя бы столько (иначе перекрытие не влияет на портфель).
TWIN_MIN_WEIGHT_PCT = 2.0
# Максимум пар/драйверов в отчётных списках — держит payload и промпт компактными.
MAX_TWIN_PAIRS = 4
MAX_DRIVERS_PER_FACTOR = 3

# Группировка факторных осей в «источники риска» (терминология отчёта).
# Ключи-оси = имена факторов движка (investment_logic.factor_tickers).
RISK_SOURCE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Рыночная бета",    ("Market",)),
    ("Стилевые наклоны", ("Momentum", "Value", "Quality", "Size", "Volatility")),
    ("EM-риск",          ("EM_Equity", "EM_Bond")),
    ("Ставки / дюрация", ("Rates",)),
    ("Сырьё / золото",   ("Commodities",)),
)
IDIO_LABEL = "Идиосинкратический (специфика бумаг)"


def _round(x: float, nd: int) -> float:
    """round() через float() — глушит numpy-скаляры для JSON-сериализации."""
    return round(float(x), nd)


def variance_decomposition(B: np.ndarray, F: np.ndarray, D: np.ndarray,
                           weights: np.ndarray,
                           factor_names: list[str]) -> dict:
    """
    Euler-декомпозиция дисперсии портфеля по факторам.

    Args:
        B: (N, K) матрица факторных бет (Ridge, по строкам — активы).
        F: (K, K) факторная ковариация (daily; масштаб не влияет на доли).
        D: (N,) вектор специфических дисперсий (диагональ idio-матрицы).
        weights: (N,) веса активов (сумма < 1 при кэше — разбавление честное).
        factor_names: имена K факторов в порядке колонок B / осей F.

    Returns {} когда декомпозиция не определена (нулевая дисперсия / формы
    не согласованы); иначе dict с полями:
        portfolio_betas   — b = Bᵀw (учитывает разбавление кэшем),
        betas_covered     — b / Σ|w| (шкала таблицы отчёта: беты на
                            полностью инвестированный портфель),
        factor_shares     — [{factor, share_pct, beta}] (Euler, знак сохранён),
        group_shares      — [{source, share_pct, factors}] + idio-строка,
        systematic_pct / idio_pct,
        identity_gap_pct  — |wᵀΣw − (bᵀFb + wᵀDw)| / wᵀΣw · 100 (float-шум).
    """
    B = np.asarray(B, dtype=float)
    F = np.asarray(F, dtype=float)
    D = np.asarray(D, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()

    if B.ndim != 2 or B.shape[0] != w.size or B.shape[1] != F.shape[0] \
            or F.shape[0] != F.shape[1] or D.size != w.size \
            or len(factor_names) != B.shape[1]:
        return {}

    b = B.T @ w                                   # (K,) факторная экспозиция портфеля
    sys_var  = float(b @ F @ b)                   # систематическая дисперсия
    idio_var = float(np.sum(w * w * D))           # специфическая дисперсия
    total    = sys_var + idio_var
    if not np.isfinite(total) or total <= 0.0:
        return {}

    # Тождество wᵀΣw = bᵀFb + wᵀDw — алгебраически точное; зазор только float.
    lhs = float(w @ (B @ F @ B.T + np.diag(D)) @ w)
    identity_gap_pct = abs(lhs - total) / total * 100.0

    # Euler по факторам: c_f = b_f (F b)_f;  Σ_f c_f == bᵀFb точно.
    fb = F @ b
    c  = b * fb                                   # (K,) вклады в дисперсию

    factor_shares = [
        {"factor": name,
         "share_pct": _round(c[i] / total * 100.0, 1),
         "beta":      _round(b[i], 3)}
        for i, name in enumerate(factor_names)
    ]

    idx = {name: i for i, name in enumerate(factor_names)}
    group_shares: list[dict] = []
    for label, members in RISK_SOURCE_GROUPS:
        present = [m for m in members if m in idx]
        if not present:
            continue
        share = float(sum(c[idx[m]] for m in present)) / total * 100.0
        group_shares.append({"source": label,
                             "share_pct": _round(share, 1),
                             "factors": present})
    # Факторы вне известных групп (защита от будущего расширения набора).
    grouped = {m for _, members in RISK_SOURCE_GROUPS for m in members}
    orphans = [n for n in factor_names if n not in grouped]
    if orphans:
        share = float(sum(c[idx[n]] for n in orphans)) / total * 100.0
        group_shares.append({"source": "Прочие факторы",
                             "share_pct": _round(share, 1),
                             "factors": orphans})
    group_shares.append({"source": IDIO_LABEL,
                         "share_pct": _round(idio_var / total * 100.0, 1),
                         "factors": []})

    gross = float(np.sum(np.abs(w)))
    return {
        # Absolute daily variance — enables the external cross-check
        # sqrt(total·252) == Total_Volatility_Ann (see tests).
        "total_variance_daily": float(total),
        "portfolio_betas": {n: _round(b[i], 3) for i, n in enumerate(factor_names)},
        "betas_covered":   {n: _round(b[i] / gross, 3) for i, n in enumerate(factor_names)}
                           if gross > 1e-12 else {},
        "factor_shares":   factor_shares,
        "group_shares":    group_shares,
        "systematic_pct":  _round(sys_var / total * 100.0, 1),
        "idio_pct":        _round(idio_var / total * 100.0, 1),
        "identity_gap_pct": _round(identity_gap_pct, 6),
    }


def driven_by(B: np.ndarray, weights: np.ndarray,
              factor_names: list[str], asset_names: list[str],
              top_n: int = MAX_DRIVERS_PER_FACTOR,
              min_abs: float = 0.01) -> dict:
    """
    Атрибуция «кто драйвит фактор»: вклад актива i в портфельную бету
    фактора f равен w_i·B_if (сумма по активам даёт b_f точно).

    Returns {factor: [{ticker, contribution}]} — топ-|top_n| по модулю,
    вклады < min_abs опущены (шум Ridge на мелких весах).
    """
    B = np.asarray(B, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if B.ndim != 2 or B.shape[0] != w.size or B.shape[1] != len(factor_names) \
            or len(asset_names) != w.size:
        return {}
    contrib = w[:, None] * B                      # (N, K)
    out: dict[str, list[dict]] = {}
    for j, fname in enumerate(factor_names):
        col = contrib[:, j]
        order = np.argsort(-np.abs(col))
        rows = [{"ticker": asset_names[i], "contribution": _round(col[i], 3)}
                for i in order[:top_n] if abs(col[i]) >= min_abs]
        if rows:
            out[fname] = rows
    return out


def marginal_overlap(B: np.ndarray, F: np.ndarray, D: np.ndarray,
                     weights: np.ndarray, asset_names: list[str],
                     corr_threshold: float = TWIN_CORR_THRESHOLD,
                     min_weight_pct: float = TWIN_MIN_WEIGHT_PCT) -> dict:
    """
    Marginal contribution: что позиция добавляет сверх уже имеющихся
    факторных ставок.

    • `twins` — пары с систематической корреляцией ≥ corr_threshold
      (через Σ_sys = B·F·Bᵀ, идиосинкратика исключена) и весом каждой
      ноги ≥ min_weight_pct: одна и та же факторная ставка куплена дважды.
    • `unique_risk` — на актив: доля СОБСТВЕННОГО риска d_i/(Σ_sys,ii+d_i)
      в его полной дисперсии.  Низкий процент = актив почти целиком
      объясняется факторами (его "alpha-борт" мал); высокий = позиция
      несёт уникальный риск/потенциал, которого нет у остальных.
    """
    B = np.asarray(B, dtype=float)
    F = np.asarray(F, dtype=float)
    D = np.asarray(D, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    n = w.size
    if B.ndim != 2 or B.shape[0] != n or len(asset_names) != n or D.size != n:
        return {}

    sys_cov = B @ F @ B.T                          # (N, N) систематическая ковариация
    sys_var = np.clip(np.diag(sys_cov), 0.0, None)

    total_var = sys_var + np.clip(D, 0.0, None)
    unique_risk = [
        {"ticker": asset_names[i],
         "weight_pct": _round(w[i] * 100.0, 1),
         "unique_risk_pct": _round(D[i] / total_var[i] * 100.0, 1)
                            if total_var[i] > 1e-18 else None}
        for i in range(n)
    ]

    sd = np.sqrt(np.clip(sys_var, 1e-18, None))
    twins: list[dict] = []
    for i in range(n):
        if abs(w[i]) * 100.0 < min_weight_pct or sys_var[i] <= 1e-18:
            continue
        for j in range(i + 1, n):
            if abs(w[j]) * 100.0 < min_weight_pct or sys_var[j] <= 1e-18:
                continue
            corr = float(sys_cov[i, j] / (sd[i] * sd[j]))
            if corr >= corr_threshold:
                twins.append({
                    "pair": [asset_names[i], asset_names[j]],
                    "systematic_corr": _round(corr, 3),
                    "combined_weight_pct": _round((w[i] + w[j]) * 100.0, 1),
                })
    twins.sort(key=lambda t: (-t["systematic_corr"], -t["combined_weight_pct"]))
    return {"twins": twins[:MAX_TWIN_PAIRS], "unique_risk": unique_risk}


def build_factor_decomposition(B, F, D, weights,
                               factor_names: list[str],
                               asset_names: list[str]) -> dict:
    """
    Оркестратор: полная дополнительная аналитика одним JSON-сериализуемым
    словарём для portfolio_metrics["factor_decomposition"].

    Пустой dict — валидный ответ «декомпозиция не определена»; все
    потребители (payload, промпт, шаблоны) обязаны его переживать.
    """
    var_part = variance_decomposition(B, F, D, weights, factor_names)
    if not var_part:
        return {}
    out = dict(var_part)
    out["driven_by"] = driven_by(B, weights, factor_names, asset_names)
    out.update(marginal_overlap(B, F, D, weights, asset_names))
    return out
