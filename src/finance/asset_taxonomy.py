"""
Asset-class taxonomy — Single Source of Truth for instrument classification
projected from Freedom Broker (Tradernet) raw metadata.

Tradernet position fields (see freedom_portfolio.models.Position):
  • t   — security TYPE   (1=Stock, 2=Bond, 3=Future, 4=Option, 9=Bond alt)
  • k   — security KIND   (broker-specific sub-category; best-effort)
  • curr— position currency (USD / KZT / …)
  • i   — ticker symbol   (suffix + prefix heuristics as last resort)

The broker's own `t` field is the AUTHORITATIVE source when present — it is
the closest thing to ground truth for "is this a bond or a stock".  Ticker
heuristics are a fallback for demo portfolios, proxy ETFs, and rows that
arrive without broker metadata (so we never crash the demo / template path).

This module is dependency-light (stdlib only) so it can be imported by the
broker layer, the adapter, and the bot without pulling sklearn/pandas.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class AssetClass(str, Enum):
    EQUITY       = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    ETF          = "ETF"
    CASH         = "CASH"
    CRYPTO       = "CRYPTO"
    COMMODITY    = "COMMODITY"
    STRUCTURED   = "STRUCTURED"      # AIX / Freedom SPC structured notes
    UNKNOWN      = "UNKNOWN"


# Tradernet `t` (security type) → AssetClass.  Authoritative when present.
_T_FIELD_MAP: dict[int, AssetClass] = {
    1: AssetClass.EQUITY,
    2: AssetClass.FIXED_INCOME,
    9: AssetClass.FIXED_INCOME,      # bond alt-encoding seen on KZ accounts
    # 3 (Future) / 4 (Option) intentionally absent — fall through to heuristic
}

# Russian display labels — kept identical to finance.scoring.classify_asset_class
# vocabulary so templates that branch on the label (e.g. cash filter
# `== "Ден. средства"`) keep working regardless of which path produced it.
_DISPLAY_LABEL: dict[AssetClass, str] = {
    AssetClass.EQUITY:       "Акции США",
    AssetClass.FIXED_INCOME: "Облигации",
    AssetClass.ETF:          "Акции США",   # ETFs render under equities bucket
    AssetClass.CASH:         "Ден. средства",
    AssetClass.CRYPTO:       "Крипто",
    AssetClass.COMMODITY:    "Сырьё",
    AssetClass.STRUCTURED:   "Акции KZ",    # AIX SPC notes shown under KZ bucket
    AssetClass.UNKNOWN:      "Прочее",
}

_CASH_CCYS = frozenset({"USD", "EUR", "RUB", "RUR", "KZT", "GBP", "CHF", "CNY", "JPY"})


def from_freedom_metadata(*,
                          ticker: str,
                          t_field: Optional[int] = None,
                          k_field: Optional[int] = None,
                          currency: Optional[str] = None) -> AssetClass:
    """
    Project a Freedom/Tradernet position's raw metadata onto an AssetClass.

    Priority:
      1. Broker `t` field (authoritative for Stock/Bond).
      2. Ticker shape heuristics (crypto / AIX-structured / commodity ETF /
         bond ETF / cash currency / KZ suffix).
      3. UNKNOWN.
    """
    t = (ticker or "").upper().strip()
    base   = t.split(".")[0] if "." in t else t
    suffix = t.rsplit(".", 1)[-1] if "." in t else ""

    # Cash currencies (a bare currency code as the "ticker", or a cash row).
    if base in _CASH_CCYS or base == "CASH":
        return AssetClass.CASH

    # 1. Trust the broker's security type when it's a clear stock/bond.
    if t_field in _T_FIELD_MAP:
        return _T_FIELD_MAP[t_field]

    # 2. Ticker heuristics (fallback for demo / proxy / metadata-less rows).
    if t.endswith("-USD") or base in {"BTC", "ETH", "SOL", "BNB", "DOGE"}:
        return AssetClass.CRYPTO
    if t.endswith(".AIX") or "FFSPC" in base:
        return AssetClass.STRUCTURED
    if base in {"GLD", "SLV", "GDX", "USO", "DBC", "PDBC", "GOLD", "SILVER", "OIL"}:
        return AssetClass.COMMODITY
    if base in {"TLT", "AGG", "BND", "LQD", "HYG", "IEF", "BIL", "EMB", "SHY"} \
       or "BOND" in base or "OVD" in base \
       or t.startswith(("KZ2P", "KZ1P", "XS", "US912")):
        return AssetClass.FIXED_INCOME
    if suffix in {"KZ", "IL"}:
        return AssetClass.EQUITY        # KZ-listed equity
    if suffix == "US" or len(base) <= 5:
        return AssetClass.EQUITY
    return AssetClass.UNKNOWN


def display_label(asset_class: AssetClass | str) -> str:
    """AssetClass → Russian display label (template-compatible vocabulary)."""
    if isinstance(asset_class, str):
        try:
            asset_class = AssetClass(asset_class)
        except ValueError:
            return "Прочее"
    return _DISPLAY_LABEL.get(asset_class, "Прочее")


def classify_display_from_freedom(*, ticker: str, t_field: Optional[int] = None,
                                  k_field: Optional[int] = None,
                                  currency: Optional[str] = None) -> str:
    """Convenience: broker metadata → ready Russian display label."""
    return display_label(from_freedom_metadata(
        ticker=ticker, t_field=t_field, k_field=k_field, currency=currency))


# ── Sector super-groups (SSOT for combined-sector concentration) ─────────────
# The engine keeps Technology and Semiconductors as DISTINCT sectors (separate
# SOXX factor), but the HEADLINE «tech concentration» must be ONE authoritative
# number so the panel, the AI prose AND the composite-risk aggravator all quote
# the same combined figure (the 59%-single vs 73%-complex mismatch otherwise
# understates the true, CORRELATED concentration).  Kept here — dependency-light
# and finance-level — so both pdf_payload (presentation) and investment_logic
# (risk gauge) import the SAME map without a layering cycle.
SECTOR_SUPERGROUPS: dict[str, tuple[str, ...]] = {
    "Tech-комплекс (Technology+Semiconductors)": ("Technology", "Semiconductors"),
}


def top_sector_concentration_pct(sector_weights: dict) -> float:
    """Largest concentration in the LONG book as a percent (0–100), taking the
    MAX over single sectors AND super-groups (so a Technology+Semiconductors
    book reads its true combined tech share, not just the biggest single
    sector).  `sector_weights` is {sector: weight} on any basis; only positive
    weights count and the result is share-of-long-book.  Returns 0.0 empty."""
    longs = {str(s): float(w) for s, w in (sector_weights or {}).items()
             if _is_pos(w)}
    total = sum(longs.values())
    if total <= 0 or not longs:
        return 0.0
    top = max(longs.values())
    for members in SECTOR_SUPERGROUPS.values():
        grp = sum(longs.get(m, 0.0) for m in members)
        if grp > top:
            top = grp
    return round(top / total * 100.0, 2)


def _is_pos(w) -> bool:
    try:
        return float(w) > 0
    except (TypeError, ValueError):
        return False


__all__ = [
    "AssetClass",
    "from_freedom_metadata",
    "display_label",
    "classify_display_from_freedom",
    "SECTOR_SUPERGROUPS",
    "top_sector_concentration_pct",
]
