"""
Leveraged / inverse daily-reset ETP registry — Single Source of Truth.

P-1 аудита методологии (docs/audit/risk-methodology-audit.md, C1/C3/C5):
прежний реестр `_LEVERAGED_ETP_BASES` в `investment_logic` хранил только
ИМЕНА — множитель плеча L, базовый актив и expense ratio извлечь было
нельзя, поэтому variance drag моделировался эмпирически через min(α̂, 0).
Этот модуль добавляет параметры и КОНТРАКТНУЮ математику drag, оставляя
эмпирический путь фолбэком для имён без параметров.

Дневное лог-приближение k×-daily-reset продукта:

    r_lev(t) ≈ L·r_u(t) − ½·L·(L−1)·σ_u²          (variance drag, контрактный)

σ_u оценивается из СОБСТВЕННОЙ дневной σ ETP: σ_u = σ_ETP / |L| — не требует
загрузки истории базового актива.  Комиссия (expense ratio) — отдельный
аддитивный источник эрозии: fee_daily = −ER/252.  Оба компонента раскрываются
раздельно (C5: cost-of-leverage ≠ volatility drag).

Модуль dependency-light (stdlib only) — импортируется и sklearn-тяжёлым
`investment_logic`, и sklearn-free `scoring_orchestrator` / `stress` без
циклических импортов (тот же паттерн, что `asset_taxonomy` / `period_returns`).

Расширение без деплоя:
  • LEVERAGED_ETP_EXTRA="TICK1,TICK2"          — только детекция (fallback-drag);
  • LEVERAGED_ETP_PARAMS="TICK:L[:UNDERLYING[:ER]];…"
        пример: "FOO:3:BAR:0.0095;BAZ:-2"      — полные параметры.

Данные реестра (L / underlying / ER) — снимок на 2026-07; ER хранится как
десятичная годовая доля (0.0104 = 1.04%). None = параметр неизвестен: имя
детектируется, но drag остаётся эмпирическим (min(α̂,0)); ER=None → fee 0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

TRADING_DAYS = 252


@dataclass(frozen=True)
class LeveragedETPInfo:
    """Контрактные параметры daily-reset ETP.

    leverage      : дневной множитель L (отрицательный для inverse);
                    None = имя известно, множитель — нет.
    underlying    : тикер базового актива/индекс-прокси (display/QC), опц.
    expense_ratio : годовая комиссия долей (0.0104 = 1.04%), опц.
    """
    leverage:      Optional[float] = None
    underlying:    Optional[str]   = None
    expense_ratio: Optional[float] = None


# ── Реестр (base symbol → параметры) ─────────────────────────────────────────
# Источник L: проспекты эмитентов (GraniteShares/ProShares/Direxion/T-Rex/
# Volatility Shares), снимок 2026-07.  ER — приблизительные публичные значения
# на ту же дату (эрозия от fee на порядок меньше drag, точность до ~10 бп
# достаточна); сомнительные — None.  Имена с leverage=None детектируются, но
# получают ЭМПИРИЧЕСКИЙ drag (min(α̂,0)) — прежнее F-23-поведение.
LEVERAGED_ETP_REGISTRY: dict[str, LeveragedETPInfo] = {
    # Single-stock (held in live books first)
    "CONL": LeveragedETPInfo(+2.0, "COIN", 0.0104),
    "XNDU": LeveragedETPInfo(None, None,   None),    # AIX-нота: L не подтверждён
    "NVDL": LeveragedETPInfo(+2.0, "NVDA", 0.0115),
    "NVDS": LeveragedETPInfo(None, "NVDA", None),    # менял плечо (1.25x→1.5x short)
    "TSLL": LeveragedETPInfo(+2.0, "TSLA", 0.0097),
    "TSLQ": LeveragedETPInfo(None, "TSLA", None),
    "MSTU": LeveragedETPInfo(+2.0, "MSTR", None),
    "MSTZ": LeveragedETPInfo(-2.0, "MSTR", None),
    "MSTX": LeveragedETPInfo(+2.0, "MSTR", None),
    # Broad index
    "TQQQ": LeveragedETPInfo(+3.0, "QQQ",  0.0088),
    "SQQQ": LeveragedETPInfo(-3.0, "QQQ",  0.0095),
    "QLD":  LeveragedETPInfo(+2.0, "QQQ",  0.0095),
    "SSO":  LeveragedETPInfo(+2.0, "SPY",  0.0091),
    "SDS":  LeveragedETPInfo(-2.0, "SPY",  0.0090),
    "SPXL": LeveragedETPInfo(+3.0, "SPY",  0.0091),
    "SPXS": LeveragedETPInfo(-3.0, "SPY",  0.0100),
    "UPRO": LeveragedETPInfo(+3.0, "SPY",  0.0091),
    "SPXU": LeveragedETPInfo(-3.0, "SPY",  0.0090),
    "TNA":  LeveragedETPInfo(+3.0, "IWM",  0.0100),
    "TZA":  LeveragedETPInfo(-3.0, "IWM",  0.0100),
    # Sector / thematic
    "SOXL": LeveragedETPInfo(+3.0, "SOXX", 0.0090),
    "SOXS": LeveragedETPInfo(-3.0, "SOXX", 0.0100),
    "LABU": LeveragedETPInfo(+3.0, "XBI",  0.0100),
    "LABD": LeveragedETPInfo(-3.0, "XBI",  0.0100),
    "TECL": LeveragedETPInfo(+3.0, "XLK",  0.0100),
    "TECS": LeveragedETPInfo(-3.0, "XLK",  0.0100),
    "FAS":  LeveragedETPInfo(+3.0, "XLF",  0.0100),
    "FAZ":  LeveragedETPInfo(-3.0, "XLF",  0.0100),
    "WEBL": LeveragedETPInfo(+3.0, "FDN",  0.0100),
    "YINN": LeveragedETPInfo(+3.0, "FXI",  0.0130),
    "YANG": LeveragedETPInfo(-3.0, "FXI",  0.0130),
    "CWEB": LeveragedETPInfo(+2.0, "KWEB", 0.0130),
    # Rates
    "TMF":  LeveragedETPInfo(+3.0, "TLT",  0.0100),
    "TMV":  LeveragedETPInfo(-3.0, "TLT",  0.0100),
    # Volatility / crypto (underlying — фьючерсные корзины, не тикер)
    "UVXY": LeveragedETPInfo(+1.5, None,   0.0095),
    "SVXY": LeveragedETPInfo(-0.5, None,   0.0095),
    "BITX": LeveragedETPInfo(+2.0, "BTC",  0.0185),
    "ETHU": LeveragedETPInfo(+2.0, "ETH",  None),
}


def _base_symbol(ticker: str) -> str:
    """'CONL.US' → 'CONL' (суффикс биржи отрезается, регистр нормализуется)."""
    return str(ticker or "").split(".")[0].strip().upper()


def _env_extra_names() -> set[str]:
    """LEVERAGED_ETP_EXTRA — детекция-только (прежний F-23 контракт)."""
    return {b.strip().upper()
            for b in os.getenv("LEVERAGED_ETP_EXTRA", "").split(",") if b.strip()}


def _env_param_registry() -> dict[str, LeveragedETPInfo]:
    """LEVERAGED_ETP_PARAMS="TICK:L[:UNDERLYING[:ER]];…" — полные параметры
    без деплоя.  Некорректные записи молча пропускаются (env — не место для
    падений бота); частично заданные поля остаются None."""
    out: dict[str, LeveragedETPInfo] = {}
    raw = os.getenv("LEVERAGED_ETP_PARAMS", "")
    for entry in raw.split(";"):
        parts = [p.strip() for p in entry.split(":")]
        if not parts or not parts[0]:
            continue
        base = parts[0].upper()
        lev: Optional[float] = None
        und: Optional[str] = None
        er:  Optional[float] = None
        try:
            if len(parts) > 1 and parts[1]:
                lev = float(parts[1])
            if len(parts) > 2 and parts[2]:
                und = parts[2].upper()
            if len(parts) > 3 and parts[3]:
                er = float(parts[3])
        except ValueError:
            continue
        out[base] = LeveragedETPInfo(lev, und, er)
    return out


def etp_info(ticker: str) -> Optional[LeveragedETPInfo]:
    """Параметры daily-reset ETP для тикера, или None если имя не в реестре.

    Env-параметры (LEVERAGED_ETP_PARAMS) имеют приоритет над встроенным
    реестром; LEVERAGED_ETP_EXTRA даёт запись без параметров (детекция)."""
    base = _base_symbol(ticker)
    if not base:
        return None
    env_params = _env_param_registry()
    if base in env_params:
        return env_params[base]
    if base in LEVERAGED_ETP_REGISTRY:
        return LEVERAGED_ETP_REGISTRY[base]
    if base in _env_extra_names():
        return LeveragedETPInfo()          # имя известно, параметров нет
    return None


def is_leveraged_etp(ticker: str) -> bool:
    """True для известного daily-reset leveraged/inverse ETP (реестр + env)."""
    return etp_info(ticker) is not None


def leverage_of(ticker: str) -> Optional[float]:
    """Дневной множитель плеча L (отрицательный для inverse), или None."""
    info = etp_info(ticker)
    return info.leverage if info is not None else None


def contractual_drag_daily(leverage: float,
                           sigma_etp_daily: float,
                           expense_ratio: Optional[float] = None,
                           ) -> tuple[float, float]:
    """Контрактные компоненты дневной эрозии k×-daily-reset ETP: (drag, fee).

    drag = −½·L·(L−1)·σ_u², где σ_u = σ_ETP/|L| (собственная дневная σ ETP,
    приведённая к базе) → −½·(L−1)/L·σ_ETP² для L>0; общий вид работает и для
    inverse (L<0 ⇒ L(L−1)>0 ⇒ drag тоже отрицательный — inverse-фонды
    деградируют так же).  fee = −ER/252 (0 при ER=None).

    Оба значения ≤ 0 — поправка может только ПОНИЗИТЬ форвард (сохранён
    anti-218%-guard F-23: положительная «альфа» никогда не добавляется).
    """
    L = float(leverage)
    sig = float(sigma_etp_daily)
    if not (L == L) or L == 0.0 or not (sig == sig) or sig < 0:   # NaN/0 guard
        return 0.0, 0.0
    sigma_u = sig / abs(L)
    drag = -0.5 * L * (L - 1.0) * sigma_u * sigma_u
    drag = min(drag, 0.0)                 # L∈(0,1] даёт drag≥0 → срезаем в 0
    fee = -(float(expense_ratio) / TRADING_DAYS) if expense_ratio else 0.0
    return drag, fee


def path_dependent_period_return(leverage: float,
                                 underlying_period_return: float,
                                 sigma_etp_daily: float,
                                 horizon_days: int,
                                 ) -> float:
    """Ожидаемый ПЕРИОДНЫЙ результат daily-reset ETP при движении базы X за
    период (P-7 аудита, стресс-сценарии):

        R_L ≈ (1 + X_u)^L · exp(−½·L·(L−1)·σ_u²·T) − 1

    — лог-нормальное приближение компаундинга дневного ресета (точное для
    непрерывной ребалансировки под GBM).  Заменяет линейный β·shock, который
    теряет выпуклость: 2×-ETP в базе −25% даёт −46%, а не капнутые −32%.

    X_u клэмпится в (−0.99, +∞), результат ограничен снизу −1.0 (терминальная
    стоимость неотрицательна — свойство самого ресет-механизма).
    """
    import math
    L = float(leverage)
    if L == 0.0:
        return 0.0
    x_u = max(float(underlying_period_return), -0.99)
    sigma_u = float(sigma_etp_daily) / abs(L)
    drag_ln = -0.5 * L * (L - 1.0) * sigma_u * sigma_u * max(int(horizon_days), 0)
    try:
        result = (1.0 + x_u) ** L * math.exp(min(drag_ln, 0.0)) - 1.0
    except (OverflowError, ValueError):
        return -1.0 if L > 0 else 0.0
    return max(result, -1.0)


__all__ = [
    "LeveragedETPInfo",
    "LEVERAGED_ETP_REGISTRY",
    "etp_info",
    "is_leveraged_etp",
    "leverage_of",
    "contractual_drag_daily",
    "path_dependent_period_return",
    "TRADING_DAYS",
]
