"""Ручной ввод портфеля: текст → контрактный DataFrame (Manual Portfolio, Фаза 4).

<!-- nav | area:finance | code:src/finance/manual_portfolio.py | read-before:перед правкой разбора ручного ввода -->

Роль : превратить текст пользователя в РОВНО ТОТ фрейм, который отдаёт брокер,
       и назвать всё, что не удалось разобрать. Ничего не считает и ничего не
       конвертирует — числа рождаются в движке.
Слой : L1 (данные). Импортирует только L0/L1; движок не импортирует его.

Что этот модуль НЕ делает — и почему это главное
-----------------------------------------------
**Не распознаёт тикеры сам.** Авторитет — `MAC3RiskEngine.resolve_tickers()` со
всеми его правилами (кэш, неликвид→прокси, `TICKER_MAP`, крипто, `.US`). Своя
регулярка здесь была бы вторым источником правды, который неизбежно разъедется
с движком (ловушка Т-2/Т-4 плана, `PHASE_04 §3.2`).

**Не конвертирует валюту.** Спецификация фазы (§5c.6, написана 2026-07-29)
требовала конвертировать цену в парсере — тогда движок этого не умел. Сегодня
умеет: Base Currency Approach (`§−45`) переводит в базовую валюту КАЖДУЮ строку,
у которой колонка `Currency` отличается от валюты отчёта, включая пиннящуюся
в 1.0 строку кэша. Конвертировать здесь ЗНАЧИЛО БЫ УМНОЖИТЬ НА КУРС ДВАЖДЫ.
Поэтому модуль только ПРОСТАВЛЯЕТ валюту, а пересчёт остаётся за движком —
инвариант «числа считаются только в `finance/engine`» не нарушен.

**Не склеивает дубликаты.** Склейка живёт в `_normalize_positions` и работает
для обоих источников (`§−41`). Дублировать правило нельзя — разъедется.

Единственное, что модуль обязан гарантировать по части валюты
-------------------------------------------------------------
🔴 Склейка дубликатов идёт в стадии 1, а конверсия — в стадии 2, то есть
ПОСЛЕ неё. Для брокера это безопасно: у бумаги одна валюта на все лоты. Для
ручного ввода — нет: пользователь может ввести один тикер и в тенге, и в
долларах. Тогда склейка усреднит числа РАЗНЫХ валют и приложит к результату
один курс — молча и без единого признака ошибки.

Поэтому смешение валют внутри одного тикера — ошибка строки, а не догадка
(A-2, `PHASE_04 §5c`). Требование плана «конверсия ДО склейки» выполнено
единственным способом, который не ломает порядок стадий: до склейки просто
не доезжает то, что склеивать нельзя.

Формат ввода
------------
Одна позиция — одна строка. Разделители полей: пробел, таб, `,`, `;`.

    ТИКЕР  КОЛИЧЕСТВО  ЦЕНА_ПОКУПКИ  [ВАЛЮТА]
    CASH:USD  5000                      # кэш: два поля, цена = 1.0 по определению
    USD 5000                            # то же самое голой валютой
    KSPI 100 104,5 KZT                  # десятичная запятая, явная валюта
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from finance.broker_api import classify_instrument, strip_exchange_suffix
from finance.currency import infer_asset_currency

#: Колонки контрактного фрейма — РОВНО как у `FreedomConnector._to_dataframe`,
#: включая порядок. Расхождение здесь означает, что ручной путь пойдёт по
#: другим веткам `analyze_all`, чем брокерский.
CONTRACT_COLUMNS = ["Ticker", "Quantity", "Purchase_Price",
                    "Broker_Current_Price", "Asset_Type", "Raw_Ticker", "Currency"]

#: Потолок длины тикера на слое ВВОДА. Это не биржевая проверка (её делает
#: провайдер в Фазе 9 своим whitelist), а защита от мусора и управляющих
#: символов: `PHASE_04 §3.2` явно разводит эти два слоя.
MAX_TICKER_LEN = 32

#: Валюты, для которых цену нельзя угадывать. `IL`/`LSE` котируются в ПЕНСАХ
#: (`GBX`), и ошибка в 100 раз не даёт ни одного видимого признака — поэтому
#: для таких бумаг валюта обязательна явно (`PHASE_04 §5c.5`).
AMBIGUOUS_CURRENCIES = frozenset({"GBX"})

_WS_SPLIT = re.compile(r"[;\s]+")        # запятая НЕ разделитель: она десятичная
_ANY_SPLIT = re.compile(r"[,;\s]+")      # фолбэк, когда поля разделены запятыми
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_NUM_OK = re.compile(r"^-?\d+(\.\d+)?$")
#: Группа разрядов: три цифры, возможно с десятичным хвостом («500,50»).
_GROUP3 = re.compile(r"\d{3}([.,]\d+)?")
_NUMERIC_TOKEN = re.compile(r"-?\d+([.,]\d+)?")
#: Код валюты — ровно три буквы. Проверка формы обязательна: без неё
#: «AAPL 10 2 500,50» разбирается как цена 2 и валюта «500,50», а
#: «AAPL,10,150,50» — как цена 150 и валюта «50». Оба случая молча дают
#: неверную стоимость позиции (`PHASE_04 §3.1`).
_CCY_TOKEN = re.compile(r"[A-Za-z]{3}")


def _merge_thousand_groups(fields: list[str]) -> list[str]:
    """«2 500 000», разорванное пробелами, — снова одно число.

    Применяется ТОЛЬКО когда полей больше, чем допускает форма строки: иначе
    `AAPL 10 150` превратилось бы в `AAPL 10150`. Разделитель разрядов и
    разделитель полей — один и тот же пробел, и различить их можно лишь по
    ожидаемой арности, а не по виду токена (`PHASE_04 §3.1`).
    """
    out = [fields[0]]                                  # тикер не трогаем
    for tok in fields[1:]:
        if (len(out) > 1
                and _GROUP3.fullmatch(tok)
                and _NUMERIC_TOKEN.fullmatch(out[-1])
                and out[-1][-1].isdigit()):
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return out


def _split_fields(line: str) -> tuple[list[str], bool]:
    """Поля строки и признак «разделителем была ЗАПЯТАЯ».

    Сначала пробуем пробел/`;`: тогда «104,5» остаётся одним токеном. Если так
    получилось единственное поле, значит поля разделены запятыми.

    Признак нужен дальше: склейка разорванных разрядов допустима ТОЛЬКО когда
    разделителем был пробел. При запятых «AAPL,10,150,50» честно неразрешим, и
    склеивать там — значит угадывать (получилось бы количество 10150).
    """
    fields = [f for f in _WS_SPLIT.split(line) if f]
    if len(fields) == 1 and "," in line:
        return [f for f in _ANY_SPLIT.split(line) if f], True
    return fields, False


@dataclass
class ParsedPosition:
    """Одна принятая строка. Числа — как ввёл пользователь, без пересчёта."""

    raw_line: str
    line_no: int
    ticker: str                  # канон для движка: без биржевого суффикса
    quantity: float
    price: float
    is_cash: bool
    currency: str
    raw_ticker: str              # как написал пользователь
    #: Канон С биржевым суффиксом (`KSPI.KZ`, `FFSPC6.1028.AIX`); у кэша — код
    #: валюты. Хранится, а не выводится заново: `canonical_ticker` от УЖЕ
    #: очищенного имени даёт другой ответ (`FFSPC6.1028` → `FFSPC6.1028`, без
    #: `.AIX`), и подмена прокси перестала бы находиться по этому ключу.
    resolved: str = ""

    def as_row(self) -> dict:
        """Строка контрактного фрейма.

        `Broker_Current_Price` у рисковой позиции — `None` НАМЕРЕННО: ручной
        ввод не знает рыночной цены, и `None` заставляет движок взять её из
        ценовой матрицы. Подставить сюда цену покупки значило бы включить ветку
        `broker_priced_only` — позиция вылетела бы из факторной модели, а P&L
        стал бы тождественно нулевым (`PHASE_04 §2`).
        """
        return {
            "Ticker": self.ticker,
            "Quantity": self.quantity,
            "Purchase_Price": self.price,
            "Broker_Current_Price": 1.0 if self.is_cash else None,
            "Asset_Type": "Кэш" if self.is_cash else classify_instrument(self.ticker),
            "Raw_Ticker": self.raw_ticker,
            "Currency": self.currency,
        }


@dataclass
class ParseReport:
    """Результат разбора: принятое, распознанное и отвергнутое — раздельно.

    Три категории существуют не для красоты. Экран подтверждения (Фаза 5)
    обязан показать пользователю ВСЕ три: что принято, что мы поняли по-своему
    и что потеряно. Молчаливая потеря строки — худший исход ручного ввода:
    портфель посчитается, отчёт построится, а позиции в нём не будет.
    """

    positions: list[ParsedPosition] = field(default_factory=list)
    #: (номер строки, человеческая причина). Никогда не исключение.
    errors: list[tuple[int, str]] = field(default_factory=list)
    #: (как ввёл пользователь, во что превратили) — синонимы: «каспи» → KSPI.KZ.
    #: Показывается ОТДЕЛЬНО от подмен прокси: источник цены у них разный.
    auto_resolved: list[tuple[str, str]] = field(default_factory=list)
    #: (бумага, её факторный прокси) — НЕ переименование, а замена инструмента
    #: в риск-модели. Разводить эти два списка обязательно: у синонима цена
    #: остаётся своей, а проксированная бумага оценивается ценой покупки
    #: (`priced_at_cost`), и пользователь вправе это знать (`§5d` H-1).
    proxied: list[tuple[str, str]] = field(default_factory=list)

    @property
    def valid(self) -> list[ParsedPosition]:
        """Псевдоним `positions` — имя из плана (`ЧК-04.2`)."""
        return self.positions

    @property
    def failed(self) -> list[tuple[int, str]]:
        """Псевдоним `errors` — имя из плана (`ЧК-04.2`)."""
        return self.errors

    def to_dataframe(self) -> pd.DataFrame:
        """Контрактный фрейм. Пустой ввод даёт пустой фрейм, а не исключение.

        `df.attrs["_ramp_source"] = "manual"` — по образцу `"demo"`. Маркеры
        `_ramp_is_mock` / `_ramp_is_fallback` НЕ ставятся: ручной портфель
        реальный, и гейт fallback-мока не должен на него срабатывать.
        """
        rows = [p.as_row() for p in self.positions]
        df = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
        df.attrs["_ramp_source"] = "manual"
        return df


def _to_number(raw: str) -> Optional[float]:
    """«2 500,50» → 2500.50. Порядок операций важен и задан планом.

    Сначала убираются разделители разрядов (пробел, `_`), и только потом
    запятая трактуется как десятичная. Обратный порядок превратил бы
    «2 500,50» в «2500,50» → «2500.50» лишь по счастливой случайности, а
    «2,500.50» сломал бы.
    """
    s = str(raw).strip().replace("_", "").replace(" ", "").replace(" ", "")
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    if not _NUM_OK.match(s):
        return None
    try:
        return float(s)
    except ValueError:                                   # pragma: no cover
        return None


def parse_portfolio_text(text: str, engine) -> ParseReport:
    """Разобрать текст в `ParseReport`. НИКОГДА не бросает на вводе пользователя.

    `engine` — экземпляр `MAC3RiskEngine`: у него спрашиваются и список
    неризиковых валют, и распознавание тикера. Передаётся аргументом, а не
    импортируется, чтобы слой данных не зависел от слоя движка.
    """
    report = ParseReport()
    non_risk = {str(c).upper() for c in engine.NON_RISK_ASSETS}
    # ticker → валюта первой принятой строки: смешение валют внутри одного
    # тикера ловится ДО склейки (см. модульный докстринг).
    seen_currency: dict[str, tuple[str, int]] = {}

    for line_no, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if _CTRL.search(line):
            report.errors.append((line_no, "строка содержит управляющие символы"))
            continue

        fields, comma_separated = _split_fields(line)
        if not fields:
            # Строка из одних разделителей («;;;;»): полей нет вовсе.
            report.errors.append((line_no, "строка не содержит полей"))
            continue
        raw_ticker = fields[0].upper()

        # Форма строки известна по ПЕРВОМУ полю, поэтому ожидаемую арность
        # можно посчитать до разбора остальных — и только затем, при избытке
        # полей, пробовать склейку разрядов.
        _is_cash_form = (raw_ticker.startswith("CASH:")
                         or raw_ticker in {str(c).upper() for c in engine.NON_RISK_ASSETS})
        _max_fields = 3 if _is_cash_form else 4

        # Избыток полей — либо разряды, разорванные пробелом, либо запятая,
        # использованная сразу как разделитель полей и как десятичная.
        if len(fields) > _max_fields and not comma_separated:
            merged_fields = _merge_thousand_groups(fields)
            if len(merged_fields) <= _max_fields:
                fields = merged_fields

        # Полей ровно максимум, но последнее поле не той формы, какую эта
        # арность подразумевает, — значит арность обманчива: на самом деле
        # разряды разорваны пробелом либо случай неразрешим.
        #
        # Формы разные, потому что разные и последние поля. У рисковой строки
        # четвёртое поле — код валюты, и без проверки «AAPL 10 2 500,50» тихо
        # стало бы ценой 2 в валюте «500,50». У строки кэша третье поле — цена,
        # а она у кэша равна 1.0 ПО ОПРЕДЕЛЕНИЮ; «1»/«1.0» под группу разрядов
        # не подходят, поэтому трёхзначный хвост здесь однозначно читается как
        # разорванные разряды. Без этой ветки `CASH:KZT 500 000` отвергался
        # сообщением про цену — при том, что `2 500 000` разбирался (арность 4
        # уходила в ветку выше). Разряды в остатке — норма ровно для тенге.
        if len(fields) == _max_fields:
            _tail_fits = (_GROUP3.fullmatch(fields[-1]) is None if _is_cash_form
                          else _CCY_TOKEN.fullmatch(fields[-1]) is not None)
            if not _tail_fits:
                merged_fields = ([] if comma_separated
                                 else _merge_thousand_groups(fields))
                if merged_fields and len(merged_fields) < len(fields):
                    fields = merged_fields
                elif not _is_cash_form:
                    report.errors.append((
                        line_no,
                        f"последнее поле «{fields[-1]}» не похоже на код валюты "
                        "(три буквы). Если запятая используется и как разделитель "
                        "полей, и как десятичная, различить их невозможно — "
                        "разделяйте поля пробелом или «;»"))
                    continue
                # Кэш со склейкой не справился — падаем в проверку цены ниже,
                # она объяснит про третье поле человеческим языком.

        if len(fields) > _max_fields:
            report.errors.append((
                line_no,
                f"полей {len(fields)}, ожидалось не больше {_max_fields}. "
                "Если запятая используется и как разделитель полей, и как "
                "десятичная, различить их невозможно — разделяйте поля "
                "пробелом или «;»"))
            continue
        if len(raw_ticker) > MAX_TICKER_LEN:
            report.errors.append((line_no, f"тикер длиннее {MAX_TICKER_LEN} символов"))
            continue

        # ── кэш ──────────────────────────────────────────────────────────────
        is_cash = False
        cash_ccy: Optional[str] = None
        if raw_ticker.startswith("CASH:"):
            cash_ccy = raw_ticker.split(":", 1)[1].strip()
            if not cash_ccy:
                report.errors.append((line_no, "после «CASH:» не указана валюта"))
                continue
            is_cash = True
        elif raw_ticker == "CASH":
            # `CASH` есть в NON_RISK_ASSETS, но это не валюта: от брокера такой
            # псевдо-тикер не приходит никогда (`acc` отдаёт коды валют).
            report.errors.append((line_no, "укажите валюту кэша, например «CASH:USD»"))
            continue
        elif raw_ticker in non_risk:
            cash_ccy = raw_ticker
            is_cash = True

        if is_cash:
            if len(fields) < 2:
                report.errors.append((line_no, "не указана сумма"))
                continue
            qty = _to_number(fields[1])
            if qty is None:
                report.errors.append((line_no, f"нечисловая сумма: «{fields[1]}»"))
                continue
            if qty == 0:
                report.errors.append((line_no, "нулевая сумма"))
                continue
            # ОТРИЦАТЕЛЬНАЯ сумма кэша принимается: это маржинальный заём, а не
            # короткая продажа (`PHASE_04 §5a.2`). Брокерский путь отдаёт такие
            # строки штатно, и книги с плечом считаются корректно (Σ весов = 1,
            # плечо 1.07× — замерено на живом отчёте). Запрет здесь означал бы,
            # что ручной ввод не может описать портфель, который freedom-путь
            # обрабатывает без единой оговорки, то есть прямое нарушение I-1.
            # Запрет на короткую продажу — ниже, в РИСКОВОЙ ветке, и он остаётся.
            if len(fields) >= 3:
                price = _to_number(fields[2])
                if price is None or abs(price - 1.0) > 1e-9:
                    report.errors.append((
                        line_no,
                        "у кэша цена равна 1.0 по определению — уберите третье поле"))
                    continue
            report.positions.append(ParsedPosition(
                raw_line=raw_line, line_no=line_no, ticker=cash_ccy,
                quantity=qty, price=1.0, is_cash=True,
                currency=cash_ccy, raw_ticker=raw_ticker, resolved=cash_ccy))
            continue

        # ── рисковая позиция ─────────────────────────────────────────────────
        if len(fields) < 3:
            report.errors.append((
                line_no, "не указана цена покупки (ожидалось: ТИКЕР КОЛИЧЕСТВО ЦЕНА)"))
            continue

        qty = _to_number(fields[1])
        if qty is None:
            report.errors.append((line_no, f"нечисловое количество: «{fields[1]}»"))
            continue
        if qty < 0:
            # Короткие продажи не поддерживаются осознанно: отрицательный вес
            # ломает Euler-декомпозицию и composite_risk_score МОЛЧА.
            report.errors.append((
                line_no, "отрицательное количество: короткие продажи не поддерживаются"))
            continue
        if qty == 0:
            report.errors.append((line_no, "нулевое количество"))
            continue

        price = _to_number(fields[2])
        if price is None:
            report.errors.append((line_no, f"нечисловая цена: «{fields[2]}»"))
            continue
        if price <= 0:
            report.errors.append((line_no, "цена покупки должна быть больше нуля"))
            continue

        # Канон тикера спрашиваем у ДВИЖКА, а не выводим сами. Именно
        # `canonical_ticker`, а НЕ `resolve_tickers`: второй отдаёт прокси, и
        # позиция стала бы прокси-бумагой. Разница видна на AIX-ноте:
        #   canonical_ticker → FFSPC6.1028.AIX   (бумага пользователя)
        #   resolve_tickers  → VWOB.US           (её факторный заменитель)
        # Пока в `Ticker` уезжал второй, нота ИСЧЕЗАЛА из портфеля: цена
        # бралась рыночная от VWOB (вместо цены покупки), `Asset_Type`
        # становился «ETF», а две разные ноты с одним прокси склеивались в одну
        # позицию — то есть дефект `§−38` возвращался через слой ввода.
        canon = engine.canonical_ticker(raw_ticker)
        if not canon:
            report.errors.append((line_no, f"тикер «{raw_ticker}» не распознан"))
            continue

        # Подмена на прокси — ОТДЕЛЬНЫЙ факт, а не переименование: её показывает
        # экран подтверждения своей строкой (`PHASE_04 §5d` H-1). Признак берём
        # сравнением двух публичных ответов движка, чтобы правило подмены
        # осталось в одном месте (`proxy_for`).
        _resolved = engine.resolve_tickers([raw_ticker])
        if _resolved and _resolved[0] != canon:
            report.proxied.append((canon, _resolved[0]))

        # `Ticker` — без биржевого суффикса, как на брокерском пути; полная
        # форма уходит в `Raw_Ticker`.
        ticker = strip_exchange_suffix(canon)
        # В `auto_resolved` попадает только смена САМОГО СИМВОЛА («каспи» →
        # `KSPI.KZ`), а не дописанный суффикс: сравниваются обе стороны в
        # очищенном виде, иначе список заполнился бы парами вида
        # «AAPL.US → AAPL.US», в которых пользователю нечего узнавать.
        if strip_exchange_suffix(raw_ticker) != ticker:
            report.auto_resolved.append((raw_ticker, canon))

        # Валюта: 4-е поле или вывод из тикера.
        if len(fields) == 4:
            currency = fields[3].upper()
        else:
            currency = infer_asset_currency(canon).upper()
            if currency in AMBIGUOUS_CURRENCIES:
                report.errors.append((
                    line_no,
                    f"для «{raw_ticker}» валюта котировки — {currency} (пенсы), "
                    "цену можно понять двояко. Укажите валюту четвёртым полем"))
                continue

        prev = seen_currency.get(ticker)
        if prev is not None and prev[0] != currency:
            report.errors.append((
                line_no,
                f"тикер «{ticker}» уже указан в валюте {prev[0]} (строка {prev[1]}), "
                f"а здесь — {currency}. Склейка дубликатов усреднила бы числа "
                "разных валют: приведите лоты к одной валюте"))
            continue
        seen_currency.setdefault(ticker, (currency, line_no))

        report.positions.append(ParsedPosition(
            raw_line=raw_line, line_no=line_no, ticker=ticker,
            quantity=qty, price=price, is_cash=False,
            currency=currency, raw_ticker=raw_ticker, resolved=canon))

    return report


@dataclass
class PreflightCoverage:
    """Что известно о ценовой истории ДО списания токена (B-2, `ЧК-04.4`).

    Смысл проверки — не «ускорить», а **не взять деньги за отчёт, который
    заведомо получится дырявым**. Токен списывается после доставки, но время
    пользователя и его ожидание тратятся раньше; молча построить книгу, где у
    трети позиций нет истории, — худший из вариантов.
    """

    #: Тикеры, по которым история В КЭШЕ есть (в разрешённом виде движка).
    covered: list[str] = field(default_factory=list)
    #: Тикеры без кэша: их придётся качать, и они МОГУТ не найтись у провайдера.
    unknown: list[str] = field(default_factory=list)
    #: Доля стоимости покупки, покрытая кэшем, ОТДЕЛЬНО ПО ВАЛЮТАМ.
    #: Разные валюты не складываются: на этом шаге курса ещё нет (он потребовал
    #: бы сети), а сложить тенге с долларами «как есть» — ровно тот дефект F-2,
    #: который эта фаза и закрывает. Лучше три честных числа, чем одно ложное.
    cost_share_by_currency: dict[str, float] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return not self.unknown

    def as_dict(self) -> dict:
        return {
            "covered": sorted(self.covered),
            "unknown": sorted(self.unknown),
            "cost_share_by_currency": {
                k: round(v, 4) for k, v in sorted(self.cost_share_by_currency.items())
            },
        }


def preflight_coverage(report: ParseReport, engine, *,
                       days: int, cached_lookup=None) -> PreflightCoverage:
    """Покрытие ценовой историей по КЭШУ, без единого сетевого запроса.

    `cached_lookup` подменяется в тестах; по умолчанию — публичный вход в кэш
    Фазы 1 (`freedom_portfolio.history.cached_observations`).

    Кэш отвечает только «да» или «не знаю»: отсутствие записи НЕ означает, что
    у провайдера истории нет, — её просто ещё не запрашивали. Поэтому вторая
    категория называется `unknown`, а не `missing`: обещать пользователю, что
    бумага не найдётся, мы не вправе.

    Спрашивается кэш ИМЕННО того провайдера, который будет обслуживать ручной
    отчёт (`manual_provider_name`), а не провайдера по умолчанию: кэш Фазы 1
    ключуется провайдером, и ответ брокерского кэша здесь был бы ответом про
    данные, которые ручному портфелю показывать нельзя (I-12). Следствие,
    которое надо принять честно: до Фазы 9 этот кэш пуст всегда, поэтому
    pre-flight отвечает «не знаю» по каждой бумаге — и молчит, вместо того
    чтобы придумывать.
    """
    if cached_lookup is None:                            # pragma: no cover
        from finance.price_providers import manual_provider_name
        from freedom_portfolio.history import cached_observations as _cached

        def cached_lookup(tickers, days_):
            return _cached(tickers, days_, provider=manual_provider_name())

    risky = [p for p in report.positions if not p.is_cash]
    resolved_by_ticker: dict[str, str] = {}
    for pos in risky:
        resolved_list = engine.resolve_tickers([pos.ticker])
        if resolved_list:
            resolved_by_ticker[pos.ticker] = resolved_list[0]

    observations = cached_lookup(sorted(set(resolved_by_ticker.values())), days)

    result = PreflightCoverage()
    cost_total: dict[str, float] = {}
    cost_covered: dict[str, float] = {}
    for pos in risky:
        resolved = resolved_by_ticker.get(pos.ticker)
        obs = observations.get(resolved) if resolved else None
        has_history = obs is not None and obs > 0
        (result.covered if has_history else result.unknown).append(pos.ticker)
        cost = float(pos.quantity) * float(pos.price)
        cost_total[pos.currency] = cost_total.get(pos.currency, 0.0) + cost
        if has_history:
            cost_covered[pos.currency] = cost_covered.get(pos.currency, 0.0) + cost

    result.cost_share_by_currency = {
        ccy: (cost_covered.get(ccy, 0.0) / total if total > 0 else 0.0)
        for ccy, total in cost_total.items()
    }
    # Один тикер мог прийти несколькими лотами — в списках он нужен один раз.
    result.covered = sorted(set(result.covered))
    result.unknown = sorted(set(result.unknown))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Экран подтверждения (`PHASE_05 §4`) — ЧИСЛА. Вид — за слоем доставки.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfirmRow:
    """Одна строка экрана подтверждения — уже посчитанная.

    Слой доставки её только печатает: инвариант «числа считаются в `finance/`»
    действует и здесь, иначе доля позиции окажется посчитанной в слое доставки, а
    потом второй раз, иначе, в отчёте.
    """

    raw_ticker: str              # как ввёл пользователь
    ticker: str                  # что уйдёт в портфель
    resolved: str                # канон с биржевым суффиксом
    asset_type: str
    quantity: float
    price: float                 # цена покупки в валюте сделки
    currency: str
    is_cash: bool
    #: Стоимость в базовой валюте. `None` — курса нет, строка НЕ учтена в итоге.
    value_base: Optional[float]
    #: Доля от итога, 0..1. `None` по той же причине, что и `value_base`.
    share: Optional[float]
    #: Прокси для факторной модели, если бумага неликвидна (H-1).
    proxy: Optional[str] = None


@dataclass
class FxNote:
    """Раскрытие конверсии: валюта, курс, дата наблюдения."""

    currency: str
    rate: float
    as_of: Optional[str]


@dataclass
class ConfirmationView:
    """Полный материал экрана подтверждения.

    Зачем экран вообще (§3.4 задания): опечатка в количестве — лишний ноль —
    не видна в числах отчёта, но полностью искажает веса, TRC и CVaR. Доля
    позиции делает её заметной ГЛАЗОМ, поэтому доля здесь обязательна, а не
    опциональна.
    """

    positions: list[ConfirmRow] = field(default_factory=list)   # по убыванию доли
    cash: list[ConfirmRow] = field(default_factory=list)
    #: Итог в базовой валюте по строкам, которые удалось привести к ней.
    total_base: float = 0.0
    base_currency: str = "USD"
    #: (номер строки, причина) — ровно то, что отвергнул парсер.
    excluded: list[tuple[int, str]] = field(default_factory=list)
    #: Конверсии, которые применены, — с курсом и датой.
    fx_notes: list[FxNote] = field(default_factory=list)
    #: Валюты БЕЗ курса: их строки не учтены в итоге и названы явно (§5.3, Т-5).
    unconvertible: list[str] = field(default_factory=list)
    auto_resolved: list[tuple[str, str]] = field(default_factory=list)
    proxied: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_positions(self) -> bool:
        return bool(self.positions or self.cash)

    @property
    def shares_are_meaningful(self) -> bool:
        """Доли посчитаны только при положительном итоге.

        Книга, у которой маржинальный заём превышает активы, даёт итог ≤ 0 —
        доля от такого знаменателя меняет знак и вводит в заблуждение сильнее,
        чем её отсутствие.
        """
        return self.total_base > 0


def build_confirmation(report: ParseReport, engine) -> ConfirmationView:
    """Материал экрана подтверждения: доли, итог, конверсии, потери.

    Курс берётся у ДВИЖКА (`fx_quote_to_base`) — тем же методом и тем же
    источником, каким потом пересчитает строки сам отчёт. Свой курс здесь
    означал бы, что на экране одно число, а в отчёте другое, и расхождение
    ничем себя не проявит (класс дефекта F-2).

    Валюта без курса **не** приводится к 1.0 (Т-5): её строки честно уходят из
    итога, а валюта называется в `unconvertible` — экран обязан её показать.
    """
    base = str(getattr(engine.reporting_currency, "value", "USD")).upper()

    rates: dict[str, tuple[Optional[float], Optional[str]]] = {}

    def _rate(ccy: str) -> tuple[Optional[float], Optional[str]]:
        key = str(ccy or base).upper()
        if key not in rates:
            try:
                rates[key] = engine.fx_quote_to_base(key)
            except Exception:                            # pragma: no cover
                rates[key] = (None, None)
        return rates[key]

    view = ConfirmationView(base_currency=base, excluded=list(report.errors),
                            auto_resolved=list(report.auto_resolved),
                            proxied=list(report.proxied))
    proxy_by_resolved = dict(report.proxied)

    rows: list[ConfirmRow] = []
    for pos in report.positions:
        rate, as_of = _rate(pos.currency)
        value = (None if rate is None
                 else float(pos.quantity) * float(pos.price) * float(rate))
        resolved = pos.resolved or pos.ticker
        rows.append(ConfirmRow(
            raw_ticker=pos.raw_ticker, ticker=pos.ticker, resolved=resolved,
            asset_type=("Кэш" if pos.is_cash else classify_instrument(pos.ticker)),
            quantity=float(pos.quantity), price=float(pos.price),
            currency=pos.currency, is_cash=pos.is_cash,
            value_base=value, share=None,
            proxy=proxy_by_resolved.get(resolved)))
        if value is None:
            if pos.currency not in view.unconvertible:
                view.unconvertible.append(pos.currency)
        elif rate is not None and pos.currency != base:
            if not any(n.currency == pos.currency for n in view.fx_notes):
                view.fx_notes.append(FxNote(pos.currency, float(rate), as_of))

    view.total_base = sum(r.value_base for r in rows if r.value_base is not None)
    if view.shares_are_meaningful:
        for row in rows:
            if row.value_base is not None:
                row.share = row.value_base / view.total_base

    # Сортировка по ВЕЛИЧИНЕ, а не по знаку: маржинальный заём — крупная строка
    # книги, и прятать её в хвост списка было бы прямым обманом.
    def _weight(row: ConfirmRow) -> float:
        return abs(row.value_base) if row.value_base is not None else -1.0

    view.positions = sorted((r for r in rows if not r.is_cash),
                            key=_weight, reverse=True)
    view.cash = sorted((r for r in rows if r.is_cash), key=_weight, reverse=True)
    return view
