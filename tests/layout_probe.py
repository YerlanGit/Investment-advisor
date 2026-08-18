"""Headless-замер вёрстки отчёта: общий стенд для гейтов R-5 и R-7.

Почему замер, а не глаз. Утверждение «0 px переполнения» уже устаревало молча
(`§−90` §3), и правило проекта требует править мобильную вёрстку ПО ЗАМЕРУ на
320/360/390/414. Здесь этот замер оформлен так, чтобы его гонял CI.

🔴 **Чего стоило доверять прежней метрике.** Таблица стилей ставит
`html, body { overflow-x: hidden }` на ширинах ≤ 640 px — «страница не
прокручивается вбок». Из-за этого `document.scrollWidth` НЕ МОЖЕТ превысить
экран, и метрика «переполнение = scrollWidth − viewport» структурно равна нулю
при любой вёрстке. Замер 2026-08-18 это подтвердил: заведомо широкий блок,
вставленный в страницу, детектором не обнаруживался. Отсечение на уровне
страницы обрезает содержимое, а не помещает его.

Поэтому детектор считает предков ТОЛЬКО до `body`: контейнер со своим
`overflow-x` — осознанная рамка компонента (скролл или кроп декора), а
страничная маска содержимое ТЕРЯЕТ и оправданием быть не может.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["WIDTHS", "chromium_path", "measure", "dom_text"]

#: Ширины телефонов, на которых проект обязуется быть чистым.
WIDTHS = (320, 360, 390, 414)

_PROBE = """
() => {
  const vw = window.innerWidth;
  const body = document.body, root = document.documentElement;
  const over = [];
  const containedByComponent = (el) => {
    for (let p = el.parentElement; p && p !== body && p !== root; p = p.parentElement) {
      if (getComputedStyle(p).overflowX !== 'visible') return true;
    }
    return false;
  };
  const hidden = (el) => {
    if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') return true;
    for (let p = el; p; p = p.parentElement) {
      const s = getComputedStyle(p);
      if (s.display === 'none' || s.visibility === 'hidden') return true;
      if (p.tagName === 'DETAILS' && !p.open) return true;
      const r = p.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return true;
    }
    return false;
  };
  document.querySelectorAll('body *').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    if (r.right <= vw + 0.5 && r.left >= -0.5) return;
    if (containedByComponent(el) || hidden(el)) return;
    over.push({
      tag: el.tagName.toLowerCase(),
      cls: String(el.className && el.className.baseVal !== undefined
                    ? el.className.baseVal : (el.className || '')).slice(0, 70),
      left: Math.round(r.left), right: Math.round(r.right),
      text: (el.textContent || '').trim().slice(0, 48),
    });
  });
  return {vw, offenders: over.slice(0, 15), nodes: document.querySelectorAll('body *').length};
}
"""


def chromium_path() -> str | None:
    """Путь к предустановленному Chromium, если он есть в окружении."""
    root = Path(os.getenv("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"))
    if not root.exists():
        return None
    for d in sorted(root.glob("chromium-*"), reverse=True):
        exe = d / "chrome-linux" / "chrome"
        if exe.exists():
            return str(exe)
    return None


def _launch(pw):
    exe = chromium_path()
    return pw.chromium.launch(executable_path=exe) if exe else pw.chromium.launch()


def measure(html_path: str, widths=WIDTHS) -> dict:
    """{ширина: {offenders, nodes}} — по одному проходу на ширину."""
    from playwright.sync_api import sync_playwright

    out: dict[int, dict] = {}
    with sync_playwright() as pw:
        browser = _launch(pw)
        try:
            for w in widths:
                page = browser.new_page(viewport={"width": w, "height": 900})
                page.goto("file://" + str(Path(html_path).resolve()))
                # Ждём, пока React смонтирует дерево: измерять пустой корень —
                # это получить «ноль переполнений» ни за что.
                page.wait_for_function(
                    "() => document.querySelectorAll('body *').length > 200",
                    timeout=15000)
                out[w] = page.evaluate(_PROBE)
                page.close()
        finally:
            browser.close()
    return out


def dom_text(html_path: str, width: int = 1280) -> str:
    """Весь ВИДИМЫЙ текст отчёта после монтирования — для гейта R-7."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = _launch(pw)
        try:
            page = browser.new_page(viewport={"width": width, "height": 1400})
            page.goto("file://" + str(Path(html_path).resolve()))
            page.wait_for_function(
                "() => document.querySelectorAll('body *').length > 200",
                timeout=15000)
            return page.evaluate("() => document.body.innerText")
        finally:
            browser.close()
