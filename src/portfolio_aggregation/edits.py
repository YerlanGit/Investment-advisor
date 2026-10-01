"""Правка сохранённого ручного портфеля: Add / Remove / Trim (PR-1 §2).

Слой: L1, чистые функции — без I/O и без aiogram. Состояние портфеля — это
**канонический текст** (одна строка — одна позиция, формат парсера), а не
разобранный фрейм: восстановление всегда идёт через актуальный
`parse_portfolio_text`, ровно как у черновика (правила разбора между версиями
меняются, текст — нет).

Единственное вычисление модуля — VWAP цены покупки при докупке
`(q1·p1 + q2·p2)/(q1+q2)`. Оно разрешено здесь, потому что его результат —
новая СТРОКА ВВОДА пользователя, а не метрика отчёта; та же формула в движке
(`_normalize_positions`) дала бы на склейке двух строк то же число.

Канон тикера — через парсер, то есть `engine.canonical_ticker`, а НЕ
`resolve_tickers` (тот отдаёт прокси, и бумага подменилась бы — `CLAUDE.md`).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from finance.broker_api import strip_exchange_suffix
from finance.manual_portfolio import parse_portfolio_text

from .flags import manual_max_positions

#: Число без экспоненты: парсер принимает только `^-?\d+(\.\d+)?$`.
_AMOUNT_OK = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass
class Entry:
    """Позиция канонического текста."""

    key: str            # «AAPL» для бумаги, «CASH:USD» для кэша
    symbol: str         # что пишется в строку: канон («AAPL.US») или «CASH:USD»
    quantity: float
    price: float
    currency: str
    is_cash: bool

    @property
    def label(self) -> str:
        return self.key


@dataclass
class EditResult:
    """Итог правки. `error` — человеческий текст; тогда `new_text` = старый."""

    new_text: str
    applied: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def fmt_amount(value: float) -> str:
    """Число в форме, которую парсер прочтёт обратно без потерь (без `e`)."""
    s = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _parse_amount(raw: str) -> Optional[float]:
    s = str(raw).strip().replace(" ", "").replace("_", "")
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    if not _AMOUNT_OK.match(s):
        return None
    return float(s)


def _render(entry: Entry) -> str:
    if entry.is_cash:
        return f"{entry.symbol} {fmt_amount(entry.quantity)}"
    return (f"{entry.symbol} {fmt_amount(entry.quantity)} "
            f"{fmt_amount(entry.price)} {entry.currency}")


def render(entries: list[Entry]) -> str:
    return "\n".join(_render(e) for e in entries)


def _entry_from(pos) -> Entry:
    if pos.is_cash:
        ccy = str(pos.currency).upper()
        return Entry(key=f"CASH:{ccy}", symbol=f"CASH:{ccy}", quantity=float(pos.quantity),
                     price=1.0, currency=ccy, is_cash=True)
    return Entry(key=str(pos.ticker), symbol=str(pos.resolved or pos.ticker),
                 quantity=float(pos.quantity), price=float(pos.price),
                 currency=str(pos.currency).upper(), is_cash=False)


def entries_of(text: str, engine) -> list[Entry]:
    """Текст → позиции в порядке ввода. Нераспознанные строки отбрасываются."""
    return [_entry_from(p) for p in parse_portfolio_text(text or "", engine).valid]


def canonical_text(text: str, engine) -> tuple[str, int, int]:
    """Привести ввод к каноническому тексту. → (текст, принято, отвергнуто)."""
    report = parse_portfolio_text(text or "", engine)
    entries = [_entry_from(p) for p in report.valid]
    return render(entries), len(entries), len(report.failed)


def version_tag(text: str) -> str:
    """Короткий хэш версии: старая кнопка «убрать №3» не удалит чужую позицию."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:8]


def _distinct_keys(entries: list[Entry]) -> int:
    return len({e.key for e in entries if not e.is_cash})


def _collapse(entries: list[Entry], key: str) -> tuple[list[Entry], Optional[Entry], int]:
    """Слить все лоты `key` в один (VWAP), вернуть (без них, слитый, индекс)."""
    lots = [e for e in entries if e.key == key]
    if not lots:
        return entries, None, len(entries)
    first = entries.index(lots[0])
    rest = [e for e in entries if e.key != key]
    qty = sum(e.quantity for e in lots)
    if lots[0].is_cash:
        price = 1.0
    elif abs(qty) > 1e-12:
        price = sum(e.quantity * e.price for e in lots) / qty
    else:                                                # pragma: no cover
        price = lots[0].price
    merged = Entry(key=key, symbol=lots[0].symbol, quantity=qty, price=price,
                   currency=lots[0].currency, is_cash=lots[0].is_cash)
    return rest, merged, first


def _cash_key(token: str, engine) -> Optional[str]:
    t = token.upper()
    if t.startswith("CASH:"):
        ccy = t.split(":", 1)[1].strip()
        return f"CASH:{ccy}" if ccy else None
    if t in {str(c).upper() for c in engine.NON_RISK_ASSETS} and t != "CASH":
        return f"CASH:{t}"
    return None


def _apply_plus(entries: list[Entry], body: str, engine, max_positions: int
                ) -> tuple[list[Entry], str]:
    report = parse_portfolio_text(body, engine)
    if report.failed or len(report.valid) != 1:
        reason = report.failed[0][1] if report.failed else "не вижу позиции"
        raise ValueError(f"«+{body}»: {reason}")
    new = _entry_from(report.valid[0])
    rest, cur, at = _collapse(entries, new.key)

    if cur is None:
        if not new.is_cash and _distinct_keys(entries) + 1 > max_positions:
            raise ValueError(f"в ручном портфеле уже {max_positions} позиций — это предел")
        return entries + [new], (f"{new.label}: добавлено {fmt_amount(new.quantity)}"
                                 + ("" if new.is_cash else
                                    f" по {fmt_amount(new.price)} {new.currency}"))

    if cur.currency != new.currency:
        raise ValueError(
            f"«{new.label}» уже записан в {cur.currency}, а здесь — {new.currency}. "
            "Одна бумага — одна валюта: приведите лот к валюте позиции")
    qty = cur.quantity + new.quantity
    if new.is_cash:
        if abs(qty) < 1e-12:
            return rest, f"{cur.label}: обнулён и убран"
        if (qty > 0) != (cur.quantity > 0):
            raise ValueError(
                f"{cur.label}: сумма перешла бы через ноль. Маржа вводится только "
                f"явной строкой, например «CASH:{cur.currency} -1000», после «-{cur.symbol}»")
        merged = Entry(cur.key, cur.symbol, qty, 1.0, cur.currency, True)
        note = f"{cur.label}: {fmt_amount(cur.quantity)} → {fmt_amount(qty)}"
    else:
        price = (cur.quantity * cur.price + new.quantity * new.price) / qty
        merged = Entry(cur.key, cur.symbol, qty, price, cur.currency, False)
        note = (f"{cur.label}: {fmt_amount(cur.quantity)} → {fmt_amount(qty)}, "
                f"цена покупки {fmt_amount(round(cur.price, 6))} → "
                f"{fmt_amount(round(price, 6))} (средневзвешенная)")
    rest.insert(at, merged)
    return rest, note


def _apply_minus(entries: list[Entry], body: str, engine) -> tuple[list[Entry], str]:
    parts = body.split()
    if not parts or len(parts) > 2:
        raise ValueError(f"«-{body}»: ожидалось «-ТИКЕР» или «-ТИКЕР КОЛИЧЕСТВО»")
    token = parts[0]
    key = _cash_key(token, engine)
    if key is None:
        canon = engine.canonical_ticker(token)
        key = strip_exchange_suffix(canon) if canon else token.upper()
    rest, cur, at = _collapse(entries, key)
    if cur is None:
        raise ValueError(f"позиции «{token.upper()}» в портфеле нет")

    if len(parts) == 1:
        return rest, f"{cur.label}: позиция удалена целиком"

    amount = _parse_amount(parts[1])
    if amount is None or amount <= 0:
        raise ValueError(f"«-{body}»: количество должно быть положительным числом")
    qty = cur.quantity - amount
    if cur.is_cash:
        if abs(qty) < 1e-12:
            return rest, f"{cur.label}: обнулён и убран"
        if (qty > 0) != (cur.quantity > 0):
            raise ValueError(
                f"{cur.label}: сумма перешла бы через ноль. Маржа вводится только "
                f"явной строкой, например «CASH:{cur.currency} -1000»")
        rest.insert(at, Entry(cur.key, cur.symbol, qty, 1.0, cur.currency, True))
        return rest, f"{cur.label}: {fmt_amount(cur.quantity)} → {fmt_amount(qty)}"
    if qty <= 1e-12:
        raise ValueError(
            f"{cur.label}: в позиции {fmt_amount(cur.quantity)}, убрать "
            f"{fmt_amount(amount)} нельзя. Чтобы удалить позицию целиком, "
            f"отправьте «-{cur.key}»")
    rest.insert(at, Entry(cur.key, cur.symbol, qty, cur.price, cur.currency, False))
    return rest, (f"{cur.label}: {fmt_amount(cur.quantity)} → {fmt_amount(qty)} "
                  "(цена покупки не меняется)")


def apply_edit(text: str, op_line: str, engine, *,
               max_positions: Optional[int] = None) -> EditResult:
    """Применить одну или несколько строк `+…`/`-…` к портфелю.

    Всё-или-ничего: ошибка в любой строке оставляет портфель нетронутым —
    частично применённая пачка правок хуже понятного отказа.
    """
    limit = max_positions if max_positions is not None else manual_max_positions()
    entries = entries_of(text, engine)
    applied: list[str] = []
    lines = [ln.split("#", 1)[0].strip() for ln in str(op_line or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return EditResult(new_text=text, error="пустая команда")
    try:
        for line in lines:
            sign, body = line[0], line[1:].strip()
            if sign == "+":
                entries, note = _apply_plus(entries, body, engine, limit)
            elif sign in ("-", "−"):
                entries, note = _apply_minus(entries, body, engine)
            else:
                raise ValueError(
                    f"«{line}»: строка правки начинается с «+» (добавить) или «-» (убрать)")
            applied.append(note)
    except ValueError as exc:
        return EditResult(new_text=text, error=str(exc))
    return EditResult(new_text=render(entries), applied=applied)


def remove_at(text: str, index: int, tag: str, engine) -> EditResult:
    """Убрать позицию по номеру с кнопки — только если версия совпала (S-5)."""
    if tag != version_tag(text):
        return EditResult(new_text=text,
                          error="портфель изменился после того, как появилась эта "
                                "кнопка — откройте список заново")
    entries = entries_of(text, engine)
    if not isinstance(index, int) or index < 0 or index >= len(entries):
        return EditResult(new_text=text, error="такой позиции нет")
    gone = entries.pop(index)
    return EditResult(new_text=render(entries),
                      applied=[f"{gone.label}: позиция удалена целиком"])


__all__ = [
    "EditResult",
    "Entry",
    "apply_edit",
    "canonical_text",
    "entries_of",
    "fmt_amount",
    "remove_at",
    "render",
    "version_tag",
]
