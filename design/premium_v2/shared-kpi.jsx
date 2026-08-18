// ── KPI card — ОДНА реализация на ОБА тира ─────────────────────────────────
//
// 🔴 Почему файл общий, а не скопированный.  До 2026-08-18 карточка KPI
// существовала в двух видах: DEEP рисовал значение + график динамики +
// заметку ИИ, BASE — только значение и подпись.  Данные для полной карточки
// лежали в payload ОБОИХ тиров (`kpi_sparklines`, `ai_*_note`) и в BASE
// молча терялись в маппере.  Две копии одного объекта — ровно тот механизм,
// которым имя эмитента разошлось шестью способами (`§−95`), поэтому здесь
// копии нет: файл включается в оба бандла (см. `build.sh`).
//
// Зависимости — `Icons` и `Sparkline`; обе реализации в бандлах побитово
// одинаковы, поэтому карточка рисуется идентично в BASE и DEEP.

// Методология метрики (по наведению) — делает знаменатель явным (Sprint-3 #8).
const _KPI_METHOD = {
  sharpe: 'Знаменатель — структурная (факторная) волатильность σ = √(w′Σw), Σ = B·F·Bᵀ + D; числитель — геом. годовая доходность − валютно-сопоставленная RFR.',
  cvar:   'Средний убыток в худшие 5% дней (1-дневный горизонт), эмпирически + bootstrap-CI.',
  dd:     'Максимальная просадка пик→дно по реконструированной кривой капитала exp(Σ log-доходностей).',
  vol:    'Годовая σ портфеля; в графике — σ того же скользящего окна, что и у знаменателя Sharpe.',
};

// Аудит 2026-08-12: `k.status` — ВЕРДИКТ ПО ЗНАЧЕНИЮ из движка
// (`finance.scoring.kpi_status` → ok/warn/bad), а не литерал дизайн-макета.
// Прежде Sharpe всегда получал золотой бейдж «good» — даже когда ИИ-заметка
// на той же карточке говорила «отдача на единицу риска слабая».  Старые ключи
// (normal/good/watch) оставлены для payload'ов, собранных до той правки.
const _KPI_BORDER = {
  ok:'#5d7c5c', warn:'#caa01a', bad:'#c47358',
  normal:'#5d7c5c', good:'#caa01a', watch:'#c47358',
};

// R-9 (2026-08-02): у ЧИСЛА и у ГРАФИКА разные окна, и это надо назвать.
// Крупная цифра — по полному окну истории; точки спарклайна — срезы за
// последний год, каждый по СКОЛЬЗЯЩЕМУ окну 60 дней. Отсюда законная, но
// сбивающая с толку картина: Sharpe в заголовке +0.55, а линия целиком ниже
// нуля (последний год был хуже полной истории). Без подписи это читается
// как противоречие в отчёте.
//
// 🔴 Число срезов СЧИТАЕТСЯ, а не пишется словом. Подпись годами обещала
// «12 срезов», тогда как движок отдаёт 11: при окне 252 дня шаг равен 21, и
// первый срез не набирает минимальных 30 наблюдений, поэтому отбрасывается
// ВСЕГДА. Тот же класс, что напечатанные константой счётчики позиций и
// стресс-сценариев (`§−90` A-8, `§−91` B-4).
const _sparkNote = (n) =>
  `${n} срезов за год · каждый по скользящему окну 60 дней ` +
  '(число выше — по полному окну истории)';

const KpiCard = ({ k }) => {
  const border = _KPI_BORDER[k.status] || '#a8a293';
  const pts    = Array.isArray(k.pts) ? k.pts : [];
  const hasPts = pts.length >= 2;
  // Заметка ИИ — ПРОЗА: её отсутствие не заполняется прочерком, а убирает
  // рамку целиком. Пустая рамка «комментарий ИИ» выглядит как доставленный
  // контент, хотя не сообщает ничего; на no-API пути (фолбэк-нарратив вообще
  // не содержит заметок KPI) так было у КАЖДОЙ карточки DEEP.
  const ai = (k.ai || '').trim();
  return (
    <div className="glass-strong rounded-4xl p-5 sm:p-6 shadow-card lift flex flex-col min-w-0"
         style={{ borderTop:`2px solid ${border}` }}>
      <div className="flex items-center justify-between gap-2 mb-2 min-w-0">
        <div className="text-[10px] tracking-widest uppercase text-ink-500 font-mono cursor-help truncate"
             title={_KPI_METHOD[k.key] || undefined}>{k.name}</div>
        {/* Размер числа текучий: DEEP держит полосу в ОДНУ колонку на телефоне,
            BASE — в четыре на десктопе, и фиксированные 40px переполняли бы
            узкую карточку. Верхняя граница равна прежнему размеру DEEP, поэтому
            на широком экране вид не изменился. */}
        <span className="leading-none font-light num text-ink-900 tracking-tight flex-shrink-0"
              style={{ fontSize:'clamp(24px, 5.2vw, 40px)' }}>{k.value}</span>
      </div>
      {/* trend chart — how the metric moved over the last 12 months */}
      <div className="rounded-2xl bg-cream-50/70 border border-ink-900/5 px-3 pt-2 pb-1.5 mb-3">
        <div className="flex items-center justify-between text-[8.5px] tracking-widest uppercase text-ink-400 font-mono mb-0.5">
          <span>Динамика · окно 60 дн.</span><span>сейчас</span>
        </div>
        <div className="h-12">
          {hasPts
            ? <div title={_sparkNote(pts.length)}>
                <Sparkline points={pts} color={k.color} height={44} width={300} gradId={`spk-${k.key}`}/>
              </div>
            : (k.svg
                ? <div className="w-full h-full" dangerouslySetInnerHTML={{ __html: k.svg }}/>
                : <div className="w-full h-full flex items-center justify-center text-[9px] text-ink-300 font-mono">нет истории</div>)}
        </div>
        {/* 🔴 §−92: подпись сделана ВИДИМОЙ, а не всплывающей.
            `R-9` (§−48) уже назвал проблему — у числа и у графика разные окна
            (Max Drawdown −40.3% над линией, не опускающейся ниже −17.2%) — но
            доставлял объяснение через `title=`, то есть ТОЛЬКО по наведению
            мыши. Продукт читают из Telegram, с телефона, где наведения нет:
            для большинства читателей раскрытия просто не существовало. */}
        {hasPts && (
          <div className="mt-1 text-[8px] leading-tight text-ink-400 font-mono">
            число выше — по полному окну истории
          </div>
        )}
      </div>
      {k.sub && k.sub !== '–' && (
        <div className="text-[10.5px] text-ink-400 font-mono leading-snug mb-4 truncate">{k.sub}</div>
      )}
      {ai && (
        <div className="mt-auto flex items-start gap-2.5 rounded-2xl bg-cream-50 border border-ink-900/5 px-3.5 py-3">
          <Icons.Sparkles size={13} className="text-gold-600 mt-0.5 flex-shrink-0" stroke={1.8}/>
          <p className="text-[11.5px] text-ink-700 leading-snug font-light">{ai}</p>
        </div>
      )}
    </div>
  );
};
