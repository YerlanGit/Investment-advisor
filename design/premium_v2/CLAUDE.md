# CLAUDE.md — `design/premium_v2/` (Premium V2 отчёт)

> Авто-подгружается Claude Code при правке в этом каталоге. Здесь ИСХОДНИКИ
> отчёта (JSX + Tailwind). Скомпилированные бандлы живут в `src/premium_assets/`.

## Прочитай перед правкой
- **`docs/report/PREMIUM_DESIGN.md`** — дизайн-система, контракты данных, сборка.
- `docs/report/REPORT_SECTIONS.md` — какая секция из какого payload-ключа растёт.

## Главный footgun — ОБЯЗАТЕЛЬНАЯ пересборка
Правка `*.jsx` НЕ попадёт в отчёт, пока не пересоберёшь бандлы:
```bash
bash design/premium_v2/build.sh      # → синкает в src/premium_assets/
```
- Tailwind сканирует с **CWD = корень репо** (glob `./design/**/*.jsx` в
  `tailwind.config.js` root-relative); `build.sh` это уже делает — не запускай
  Tailwind из подпапки, иначе получишь пустой reset-only CSS (весь отчёт без стилей).
- Тест `CompiledAssetsTest` (`tests/test_phase31_*`) падает, если исходник `.jsx`
  и артефакт в `src/premium_assets/` разошлись → всегда коммить пересобранные бандлы.
- Расширил DEEP/BASE-контракт (новые ключи) → обнови пин в `test_phase19_block_audit`.

## Цикл
1. Правь `.jsx` → `bash design/premium_v2/build.sh`.
2. `python -m pytest tests/test_phase19_block_audit.py tests/test_phase31_benchmark_factor_propagation.py -q`.
3. Смоук-рендер обоих тиров через `html_renderer.render_report_html`.
