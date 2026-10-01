"""Слияние брокерского и ручного состава в ОДИН контрактный фрейм (PR-3 §2).

Что агрегатор делает — и чего НЕ делает
---------------------------------------
Делает ровно три вещи: гейт I-15, проверку «одна бумага — одна валюта» и
конкатенацию с явной разметкой источника. **Ничего не считает.** Склейку
дублей (сумма количества, VWAP цены покупки, кэш по коду валюты) делает
`UniversalPortfolioManager._normalize_positions` — для брокера, ручного ввода
и их смеси одним и тем же кодом (A-1). Второе правило склейки здесь разъехалось
бы с движком, поэтому его нет; математика отчёта от гибрида не меняется.

Почему валюты проверяются ДО склейки
------------------------------------
`_normalize_positions` берёт ПЕРВУЮ непустую валюту группы и молча усредняет
цены разных валют (тот же дефект, который парсер ручного ввода закрывает
внутри одного текста). Между двумя источниками парсер уже не видит, поэтому
отказ — здесь (D-6).

Почему `attrs` пишутся заново
-----------------------------
`pd.concat` фреймов с разными `attrs` теряет их целиком (проверено на pandas
этого репозитория). Унаследованный маркер исчез бы молча, поэтому гейт стоит
ДО слияния, а результат получает ровно один маркер: `_ramp_source=aggregated`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from finance.asset_taxonomy import display_label, from_freedom_metadata
from finance.manual_portfolio import CONTRACT_COLUMNS

from .flags import aggregated_max_positions
from .gate import AGGREGATED_SOURCE, require_live_broker_fetch
from .sources import SourceResult

logger = logging.getLogger(__name__)

#: Колонки, которые у брокерской строки бывают, а у ручной — нет. Пустое
#: значение в них нельзя оставлять `NaN`: `pdf_payload` читает
#: `row.get("Asset_Class_Label") or _classify_asset(ticker)`, а `NaN` истинен,
#: и в отчёте был бы класс актива «nan».
_TAXONOMY_COLUMNS = ("Asset_Class", "Asset_Class_Label")

_CASH_TYPE = "Кэш"


class AggregationRefused(ValueError):
    """Слить нельзя. `reason` — машинный код, `tickers` — что мешает."""

    def __init__(self, reason: str, tickers: tuple[str, ...] = (), *,
                 limit: Optional[int] = None) -> None:
        super().__init__(f"агрегация отклонена: {reason}"
                         + (f" ({', '.join(tickers)})" if tickers else ""))
        self.reason = reason
        self.tickers = tuple(tickers)
        self.limit = limit


@dataclass
class AggregateResult:
    """Смешанный фрейм и всё, что про него обязан узнать пользователь."""

    frame: pd.DataFrame
    #: Бумаги (не кэш), которые есть и на счёте, и в ручном вводе (D-5).
    overlaps: list[str] = field(default_factory=list)
    freedom_positions: int = 0
    manual_positions: int = 0

    def composition(self) -> dict:
        """Провенанс состава для CoVe (`data_lineage._aggregated_source_status`)."""
        return {
            "freedom_positions": self.freedom_positions,
            "manual_positions": self.manual_positions,
            "overlaps": list(self.overlaps),
        }


def _clean_ccy(value) -> str:
    s = "" if value is None else str(value).strip().upper()
    return "" if s in ("", "NAN", "NONE") else s


def _is_cash(row: pd.Series) -> bool:
    return str(row.get("Asset_Type") or "") == _CASH_TYPE


def _risky_currencies(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Тикер бумаги → множество объявленных валют (кэш не участвует)."""
    out: dict[str, set[str]] = {}
    if frame is None or frame.empty or "Ticker" not in frame.columns:
        return out
    for _, row in frame.iterrows():
        if _is_cash(row):
            continue
        ticker = str(row.get("Ticker") or "").strip()
        if not ticker:
            continue
        ccys = out.setdefault(ticker, set())
        ccy = _clean_ccy(row.get("Currency"))
        if ccy:
            ccys.add(ccy)
    return out


def _fill_taxonomy(out: pd.DataFrame) -> pd.DataFrame:
    """Дозаполнить колонки таксономии у строк, где их нет (ручной ввод)."""
    present = [c for c in _TAXONOMY_COLUMNS if c in out.columns]
    if not present:
        return out
    for col in present:
        out[col] = out[col].astype(object)
    for i, row in out.iterrows():
        missing = [c for c in present if pd.isna(row.get(c)) or row.get(c) in ("", None)]
        if not missing:
            continue
        raw = row.get("Raw_Ticker")
        ticker = str(raw if isinstance(raw, str) and raw else row.get("Ticker"))
        aclass = from_freedom_metadata(ticker=ticker, t_field=None, k_field=None,
                                       currency=_clean_ccy(row.get("Currency")) or None)
        if "Asset_Class" in missing:
            out.at[i, "Asset_Class"] = aclass.value
        if "Asset_Class_Label" in missing:
            out.at[i, "Asset_Class_Label"] = display_label(aclass)
    return out


class PortfolioAggregator:
    """`merge(freedom, manual)` — единственная точка, где источники смешиваются."""

    def __init__(self, max_positions: Optional[int] = None) -> None:
        self._max = max_positions

    def merge(self, freedom: SourceResult, manual: SourceResult) -> AggregateResult:
        # ── I-15: гейт ДО слияния ─────────────────────────────────────────
        require_live_broker_fetch(freedom)
        if manual is None or not manual.ok or manual.frame is None or manual.frame.empty:
            raise AggregationRefused("manual_empty")

        broker = freedom.frame.copy()
        hand = manual.frame.copy()
        broker.attrs = {}
        hand.attrs = {}

        # ── D-6: одна бумага — одна валюта ────────────────────────────────
        b_ccy, m_ccy = _risky_currencies(broker), _risky_currencies(hand)
        overlaps = sorted(set(b_ccy) & set(m_ccy))
        conflicts = sorted(t for t in overlaps
                           if b_ccy[t] and m_ccy[t] and b_ccy[t] != m_ccy[t])
        if conflicts:
            raise AggregationRefused("currency_conflict", tuple(conflicts))

        # ── S-7: потолок строк ДО склейки ─────────────────────────────────
        limit = self._max if self._max is not None else aggregated_max_positions()
        if len(broker) + len(hand) > limit:
            raise AggregationRefused("too_many", limit=limit)

        # ── конкатенация: склейку дублей сделает движок (A-1) ─────────────
        # Сборка по записям, а не `pd.concat`: у ручного фрейма колонка цены
        # брокера бывает целиком пустой, и concat (pandas ≥ 2.1) выводил бы её
        # тип по правилу, объявленному устаревшим, — с предупреждением сейчас и
        # с другим dtype потом. Строки и их значения при этом те же.
        columns = list(CONTRACT_COLUMNS) + [
            c for c in list(broker.columns) + list(hand.columns)
            if c not in CONTRACT_COLUMNS]
        columns = list(dict.fromkeys(c for c in columns
                                     if c in broker.columns or c in hand.columns))
        out = pd.DataFrame(broker.to_dict("records") + hand.to_dict("records"),
                           columns=columns)
        out = _fill_taxonomy(out)
        out.attrs = {"_ramp_source": AGGREGATED_SOURCE}

        logger.info("HYBRID: слияние freedom=%d + manual=%d строк, пересечений=%d",
                    len(broker), len(hand), len(overlaps))
        return AggregateResult(frame=out, overlaps=overlaps,
                               freedom_positions=int(len(broker)),
                               manual_positions=int(len(hand)))


__all__ = ["AggregateResult", "AggregationRefused", "PortfolioAggregator"]
