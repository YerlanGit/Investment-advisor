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
    # A-5 (2026-08-02): было «Акции KZ» — и живой отчёт печатал ноту как
    # долевую бумагу, тогда как панель мандата считала её «Облигации», а
    # риск-модель проксировала на `VWOB.US` (EM govt bond, §−38).  Один
    # инструмент — один ответ: нота долговая.  Структурная природа при этом
    # НЕ теряется: `broker_api._classify_instrument` держит для неё
    # отдельный `Asset_Type` = «Структ.нота» в своей колонке.
    AssetClass.STRUCTURED:   "Облигации",
    AssetClass.UNKNOWN:      "Прочее",
}

# ═════════════════════════════════════════════════════════════════════════════
# SSOT ФАКТОВ О ТИКЕРЕ (A-5, 2026-08-02)
# ═════════════════════════════════════════════════════════════════════════════
# До этого раздела в проекте жили ШЕСТЬ независимых копий «какой это
# инструмент», и они разошлись.  Замерено: `VWOB`/`VWO`/`SHV`/`VCIT` не знал
# НИКТО (хотя `VWOB`/`VWO` — прокси раунда §−38, они приходят в модель каждым
# прогоном); `EMB` отсутствовал в SEC-skip и в credit-N/A; `GOVT` знал только
# credit-N/A.  В живом отчёте AIX-нота была одновременно «Облигации» (панель
# мандата) и «Акции KZ» (таблица позиций).
#
# ЧТО ЗДЕСЬ ЕСТЬ, А ЧЕГО НЕТ.  Здесь — только ФАКТЫ о тикере («это облигационный
# ETF», «это кэш»).  РЕШЕНИЙ здесь нет: шесть потребителей отвечают на РАЗНЫЕ
# вопросы разными типами ответа —
#
#   asset_taxonomy.from_freedom_metadata   → AssetClass (тип инструмента)
#   scoring.classify_asset_class           → русская подпись для таблицы
#   gatekeeper._classify_to_asset_key      → ключ ЛИМИТА мандата (Check 8)
#   broker_api._classify_instrument        → Asset_Type строки портфеля
#   sec_edgar._should_skip                 → ходить ли в SEC за 10-K
#   scoring_orchestrator._is_credit_not_applicable → применим ли C-пиллар
#
# — поэтому один словарь «тикер → класс» их не заменяет и заменять НЕ должен.
# Общий у них ровно один слой: перечни тикеров ниже.  Каждый потребитель
# читает ЭТИ множества и принимает СВОЁ решение.
#
# Правило пополнения: новый тикер добавляется ЗДЕСЬ и нигде больше.
# `tests/test_phase42_instrument_ssot.py` падает при появлении второй копии.

CASH_CCYS = frozenset({"USD", "EUR", "RUB", "RUR", "KZT", "GBP", "CHF",
                       "CNY", "JPY"})
_CASH_CCYS = CASH_CCYS          # обратная совместимость с прежним именем

CRYPTO_BASES = frozenset({"BTC", "ETH", "SOL", "BNB", "DOGE", "ADA", "DOT",
                          "XRP"})

COMMODITY_ETFS = frozenset({"GLD", "SLV", "GDX", "USO", "UNG", "DBC", "PDBC",
                            "GOLD", "SILVER", "OIL"})

# Облигационные ETF/индексы.  `TNX` — индекс доходности 10Y, не ETF; в
# портфеле не встречается (это факторный тикер движка), оставлен, чтобы не
# менять исторический ответ мандатной панели.
BOND_ETFS = frozenset({
    "TLT", "IEF", "SHY", "SHV", "AGG", "BND", "BNDX", "GOVT", "TIP",
    "LQD", "VCIT", "HYG", "JNK", "EMB", "VWOB", "BIL", "TNX",
})

# Государственные/квази-государственные бумаги РАЗВИТЫХ рынков: у них нет
# корпоративного кредитного риска, поэтому C-пиллар к ним неприменим.
# EM-суверенный долг (`EMB`, `VWOB`) сюда НЕ входит намеренно — там кредитный
# риск реален и пиллар обязан работать.
SOVEREIGN_BOND_ETFS = frozenset({"TLT", "IEF", "SHY", "SHV", "AGG", "BND",
                                 "BIL", "GOVT", "TIP"})

BROAD_EQUITY_ETFS = frozenset({"SPY", "QQQ", "IWM", "EEM", "VWO", "URTH",
                               "VTI", "VOO", "VEA", "EFA", "IEFA"})
#: `SPLV` — факторный тикер движка (low-volatility); до A-5 его не знал никто,
#: и портфель, ДЕРЖАЩИЙ его, уходил в SEC за несуществующим 10-K.
FACTOR_ETFS = frozenset({"MTUM", "VLUE", "QUAL", "USMV", "SIZE", "SPLV"})
SECTOR_ETFS = frozenset({"XLK", "XLF", "XLE", "XLV", "XLP", "XLI", "XLU",
                         "XLY", "XLB", "XLC", "XLRE", "SOXX", "XME"})
EQUITY_ETFS = BROAD_EQUITY_ETFS | FACTOR_ETFS | SECTOR_ETFS

#: Все известные ETF-обёртки — «за ними нет отчётности эмитента-компании».
ETF_BASES = EQUITY_ETFS | BOND_ETFS | COMMODITY_ETFS

KZ_BLUE_CHIPS = frozenset({"KSPI", "HSBK", "KAP", "KZTK", "KCEL", "BAST",
                           "HRGL", "KZAP", "KZTO"})
KZ_SUFFIXES = frozenset({"KZ", "IL"})

#: ISIN-подобные префиксы, за которыми практически всегда долговая бумага.
BOND_ISIN_PREFIXES = ("KZ2P", "KZ1P", "XS", "US912")
#: Слова в тикере, прямо называющие долговую бумагу.
BOND_NAME_MARKERS = ("BOND", "OVD", "EUROBOND", "T-BILL", "TBILL", "TREASURY")
#: Структурные ноты Freedom SPC (AIX).
STRUCTURED_MARKERS = ("FFSPC",)


def ticker_base(ticker: str) -> str:
    """«KMGZ.KZ» → «KMGZ»; «AAPL» → «AAPL». Верхний регистр, без пробелов."""
    t = (ticker or "").upper().strip()
    return t.split(".")[0] if "." in t else t


def ticker_suffix(ticker: str) -> str:
    """«KMGZ.KZ» → «KZ»; «AAPL» → «»."""
    t = (ticker or "").upper().strip()
    return t.rsplit(".", 1)[-1] if "." in t else ""


def is_cash(ticker: str) -> bool:
    b = ticker_base(ticker)
    return b in CASH_CCYS or b == "CASH"


def is_crypto(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    return t.endswith("-USD") or ticker_base(t) in CRYPTO_BASES


def is_commodity(ticker: str) -> bool:
    return ticker_base(ticker) in COMMODITY_ETFS


def is_structured_note(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    return t.endswith(".AIX") or any(m in t for m in STRUCTURED_MARKERS)


def is_bond_like(ticker: str) -> bool:
    """Долговая бумага в ЭКОНОМИЧЕСКОМ смысле.

    Включает структурные ноты AIX: движок моделирует их прокси `VWOB.US`
    (EM govt bond, `AUDIT §−38`), и мандатная панель считает их «Bonds» —
    ответ обязан быть один во всех трёх местах.
    """
    t = (ticker or "").upper().strip()
    b = ticker_base(t)
    return (b in BOND_ETFS
            or any(m in b for m in BOND_NAME_MARKERS)
            or t.startswith(BOND_ISIN_PREFIXES)
            or is_structured_note(t))


def is_sovereign_bond(ticker: str) -> bool:
    """Суверенный долг РАЗВИТОГО рынка — кредитный пиллар неприменим."""
    return ticker_base(ticker) in SOVEREIGN_BOND_ETFS


def is_etf(ticker: str) -> bool:
    """Известная ETF-обёртка (у неё нет собственной отчётности эмитента)."""
    return ticker_base(ticker) in ETF_BASES


def is_kz_listed(ticker: str) -> bool:
    t = (ticker or "").upper().strip()
    return (ticker_suffix(t) in KZ_SUFFIXES
            or t.endswith(".AIX")
            or ticker_base(t) in KZ_BLUE_CHIPS)


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
    base   = ticker_base(t)
    suffix = ticker_suffix(t)

    # Cash currencies (a bare currency code as the "ticker", or a cash row).
    if is_cash(t):
        return AssetClass.CASH

    # 1. Trust the broker's security type when it's a clear stock/bond.
    if t_field in _T_FIELD_MAP:
        return _T_FIELD_MAP[t_field]

    # 2. Ticker heuristics (fallback for demo / proxy / metadata-less rows).
    #    Все перечни — из SSOT выше (A-5); собственных списков здесь нет.
    if is_crypto(t):
        return AssetClass.CRYPTO
    if is_structured_note(t):
        return AssetClass.STRUCTURED
    if is_commodity(t):
        return AssetClass.COMMODITY
    if is_bond_like(t):
        return AssetClass.FIXED_INCOME
    if suffix in KZ_SUFFIXES:
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
    # SSOT фактов о тикере (A-5) — перечни
    "CASH_CCYS", "CRYPTO_BASES", "COMMODITY_ETFS", "BOND_ETFS",
    "SOVEREIGN_BOND_ETFS", "BROAD_EQUITY_ETFS", "FACTOR_ETFS", "SECTOR_ETFS",
    "EQUITY_ETFS", "ETF_BASES", "KZ_BLUE_CHIPS", "KZ_SUFFIXES",
    "BOND_ISIN_PREFIXES", "BOND_NAME_MARKERS", "STRUCTURED_MARKERS",
    # …и предикаты
    "ticker_base", "ticker_suffix", "is_cash", "is_crypto", "is_commodity",
    "is_structured_note", "is_bond_like", "is_sovereign_bond", "is_etf",
    "is_kz_listed",
]
