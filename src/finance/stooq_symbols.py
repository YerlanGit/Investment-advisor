"""
Форма символа: тикер движка → символ выгрузки Stooq (MP-08.5).

Почему это отдельный модуль, а не ветка в `resolve_tickers`
───────────────────────────────────────────────────────────
Ловушка Т-4 запрещает второй справочник тикеров, и она права — но речь про
справочник ФАКТОВ о бумаге (что это, чем её заменять). Здесь другое: как ОДНА
И ТА ЖЕ бумага записана у конкретного поставщика. `resolve_tickers` служит
брокерскому пути, и правка форм символа в нём сломала бы `freedom` ради Stooq
(`PHASE_08 §2.5`). Поэтому форма живёт рядом с провайдером, который её и
применяет.

Главное правило модуля
──────────────────────
🔴 **Кандидат меняет НОТАЦИЮ, но никогда — ПЛОЩАДКУ.**

Замер вскрыл, чем кончается нарушение (`STOOQ_CONVENTION §3.6`, ловушка №6):
перебор форм для `KSPI.KZ` нашёл `KSPI.US` и счёл это успехом. `KSPI.US`
существует и торгуется — но это **ADR на Nasdaq**, а не листинг на KASE:
другая валюта, другая ликвидность, другой торговый календарь. Портфель
казахстанского инвестора молча посчитался бы по американской бумаге.

Подмена площадки — законное решение, но именно РЕШЕНИЕ: она заводится строкой
в `symbol_map` с `match_kind='venue_substitution'` и показывается пользователю
рядом с прокси-подменами. Автоматически её не делает ничто.
"""

from __future__ import annotations

import logging

from finance import market_calendar as mc

logger = logging.getLogger(__name__)

MATCH_EXACT = "exact"
MATCH_VENUE_SUBSTITUTION = "venue_substitution"

#: Нотация площадки у нас и у Stooq. Это НЕ подмена площадки — та же биржа,
#: другая запись, поэтому кандидат законен.
#:
#: `.IL` у нас означает International Listing (GDR на Лондонской бирже);
#: у Stooq Лондон — `.UK`, а `.IL` занят Израилем. Именно из-за отсутствия
#: этого кандидата замер §3.7 записал `KAP.IL` и `HSBK.IL` как «нет в Stooq»,
#: хотя проверялись формы `kap.il`, `kap`, `kap.us` — ни одна не та
#: (`STOOQ_CONVENTION §3.8.6`).
_NOTATION_ALIASES: dict[str, tuple[str, ...]] = {
    "IL": ("UK",),
}

#: Суффиксы, которых у Stooq нет вовсе. Кандидатов для них не строим — пусть
#: бумага честно окажется непокрытой, чем «найдётся» чужой.
_ABSENT_AT_SOURCE = frozenset({"AIX", "KASE"})


def candidates_for(engine_ticker: str) -> list[str]:
    """Формы символа Stooq для тикера движка, в порядке приоритета.

    Пустой список означает «у этого поставщика такой бумаги быть не может» —
    и это честный ответ, а не неудача перебора.
    """
    text = str(engine_ticker or "").strip().upper()
    if not text:
        return []

    out: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in out:
            out.append(candidate)

    # Крипта: движок держит `BTC-USD`, Stooq — `BTC.V` (замерено: все четыре
    # наши монеты присутствуют именно в этой форме).
    if text.endswith("-USD"):
        add(f"{text[:-4]}.V")
        return out

    base, suffix = mc.split_symbol(text)
    if not suffix:
        add(f"{text}.US")
        return out

    if suffix in _ABSENT_AT_SOURCE:
        return out

    if mc.is_known_suffix(suffix):
        add(text)
        return out

    for alias in _NOTATION_ALIASES.get(suffix, ()):
        add(f"{base}.{alias}")
    if out:
        return out

    # Класс акции: у нас `BRK.B`, у Stooq `brk-b.us` — замерено на 75 тикерах
    # с дефисом, среди них `brk-a.us` и `bf-b.us` (`STOOQ_CONVENTION §3.7`).
    # Условие «суффикс из одной буквы» отделяет класс акции от биржи: биржевых
    # суффиксов длиной в символ у Stooq нет.
    if len(suffix) == 1 and suffix.isalpha():
        add(f"{base}-{suffix}.US")
        return out

    # Суффикс есть, но он нам неизвестен. Единственный отброшенный здесь
    # кандидат — `{base}.US`, и отброшен он СОЗНАТЕЛЬНО: именно он подсунул
    # ADR вместо листинга KASE.
    logger.info("STOOQ: неизвестный суффикс %r у %r — кандидатов нет "
                "(подмена площадки только через symbol_map)", suffix, text)
    return out


def would_change_venue(engine_ticker: str, source_symbol: str) -> bool:
    """True, если сопоставление уводит бумагу на ДРУГУЮ площадку.

    Используется при ручном заведении строки `symbol_map`: оператор вправе
    сопоставить `KSPI.KZ` с `KSPI.US`, но обязан увидеть, что делает.

    🔴 Сначала спрашиваются НАШИ ЖЕ кандидаты (`AUDIT §−80`).  Прежняя редакция
    сравнивала рынки напрямую и противоречила самой себе: `candidates_for`
    отдавал `KAP.IL → KAP.UK` как нотацию (`match_kind='exact'`, Лондон и там,
    и там), а эта функция звала ту же пару подменой площадки — потому что
    суффикса `.IL` нет в карте рынков и он читался как `UNKNOWN`. Две функции
    об одном и том же обязаны отвечать одинаково, иначе раскрытие подмен на
    экране разойдётся с тем, что реально сделал перебор.
    """
    wanted = {c.upper() for c in candidates_for(engine_ticker)}
    if str(source_symbol or "").strip().upper() in wanted:
        return False                       # это НАША форма записи, не подмена
    a = mc.market_of(engine_ticker)
    b = mc.market_of(source_symbol)
    # Неизвестное не объявляем совпадением: `FFSPC6.1028.AIX` и что угодно —
    # это подмена, пока не доказано обратное.
    return a != b
