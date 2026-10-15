// Inline-SVG charts: multi-series line chart, stacked area chart, bar chart.
// Responsive via viewBox (width 100%, height follows the aspect ratio); axes with "nice"
// ticks; HTML legend; hover crosshair/tooltip. Colours come from CSS tokens
// (--series-1..8) so light and dark themes each use their own validated steps.
import { h, s, clear } from './ui.js';

const W_DEFAULT = 640;
const M = { top: 12, right: 18, bottom: 38, left: 56 };

/** Categorical colour by slot (fixed order). */
export function seriesColor(i) { return `var(--series-${(Math.abs(i) % 8) + 1})`; }

/** Stable colour for a species id (colour follows the entity, never its rank). */
export function speciesColor(id) {
  if (id === null || id === undefined || id === '') return 'var(--fg-faint)';
  const n = Number(id);
  return Number.isFinite(n) ? seriesColor(n) : seriesColor([...String(id)].reduce((a, c) => a + c.charCodeAt(0), 0));
}

/** "Nice" tick values covering [min, max]. */
export function niceTicks(min, max, count = 5, minStep = 0) {
  if (!isFinite(min) || !isFinite(max)) return [0, 1];
  if (min === max) { const pad = Math.abs(min) * 0.1 || 1; min -= pad; max += pad; }
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = Math.max(minStep, (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag);
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(Math.abs(v) < step / 1e6 ? 0 : +v.toPrecision(12));
  return out;
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toFixed(1)}M`;
  if (a >= 1e4) return `${+(v / 1e3).toFixed(1)}k`;
  if (Number.isInteger(v)) return String(v);
  return String(+v.toFixed(a < 1 ? 3 : 2));
}

function legend(series) {
  if (series.length < 2) return null;
  return h('div', { class: 'chart-legend' }, series.map(sr => h('span', { class: 'legend-item' },
    h('span', { class: `legend-swatch ${sr.dash ? 'dashed' : ''}`, style: { '--c': sr.color } }), sr.name)));
}

/** Frame: axes, gridlines and labels. Returns {svg, plot, sx, sy, H, xTicks}. */
function frame({ width: W = W_DEFAULT, xDomain, yDomain, height, xLabel, yLabel, xTickValues, xFormat = fmtTick, yFormat = fmtTick, bottom = M.bottom, integerY = false }) {
  const H = height;
  const mb = bottom;
  const pw = W - M.left - M.right, ph = H - M.top - mb;
  const [x0, x1] = xDomain;
  const yt = niceTicks(yDomain[0], yDomain[1], 5, integerY ? 1 : 0);
  const y0 = yt[0], y1 = yt[yt.length - 1];
  const sx = (x) => M.left + (x1 === x0 ? pw / 2 : ((x - x0) / (x1 - x0)) * pw);
  const sy = (y) => M.top + ph - ((y - y0) / (y1 - y0 || 1)) * ph;
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'img', 'aria-label': yLabel || 'chart' });
  const grid = s('g', { class: 'chart-gridlines' });
  for (const v of yt) {
    grid.appendChild(s('line', { x1: M.left, x2: W - M.right, y1: sy(v), y2: sy(v), class: v === 0 ? 'zero' : '' }));
    grid.appendChild(s('text', { x: M.left - 6, y: sy(v) + 4, 'text-anchor': 'end', class: 'tick', text: yFormat(v) }));
  }
  let xt = xTickValues || niceTicks(x0, x1, Math.min(8, Math.max(1, x1 - x0)));
  xt = xt.filter(v => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  for (const v of xt) {
    grid.appendChild(s('line', { x1: sx(v), x2: sx(v), y1: M.top + ph, y2: M.top + ph + 4, class: 'tickmark' }));
    grid.appendChild(s('text', { x: sx(v), y: M.top + ph + 16, 'text-anchor': 'middle', class: 'tick', text: xFormat(v) }));
  }
  grid.appendChild(s('line', { x1: M.left, x2: W - M.right, y1: M.top + ph, y2: M.top + ph, class: 'axis' }));
  if (xLabel) grid.appendChild(s('text', { x: M.left + pw / 2, y: H - 4, 'text-anchor': 'middle', class: 'axis-label', text: xLabel }));
  if (yLabel) grid.appendChild(s('text', { x: 12, y: M.top + ph / 2, 'text-anchor': 'middle', class: 'axis-label', transform: `rotate(-90 12 ${M.top + ph / 2})`, text: yLabel }));
  svg.appendChild(grid);
  return { svg, sx, sy, W, H, pw, ph, mb, x0, x1, y0, y1 };
}

/** Attach crosshair + tooltip behaviour. `rowsAt(x)` returns [{name,color,value}] for the x. */
function hoverLayer(wrap, f, xs, rowsAt, { xFormat = (x) => `x = ${x}`, valueFormat = fmtTick } = {}) {
  const tip = h('div', { class: 'chart-tip', hidden: true });
  wrap.appendChild(tip);
  const cross = s('line', { class: 'crosshair', y1: M.top, y2: M.top + f.ph, visibility: 'hidden' });
  const dots = s('g', { class: 'hover-dots' });
  const overlay = s('rect', { x: M.left, y: M.top, width: f.pw, height: f.ph, fill: 'transparent', class: 'hover-target' });
  f.svg.append(cross, dots, overlay);
  const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); clear(dots); };
  overlay.addEventListener('mouseleave', hide);
  overlay.addEventListener('mousemove', (e) => {
    if (!xs.length) return;
    const r = f.svg.getBoundingClientRect();
    const vx = (e.clientX - r.left) * (f.W / r.width);
    let best = xs[0];
    for (const x of xs) if (Math.abs(f.sx(x) - vx) < Math.abs(f.sx(best) - vx)) best = x;
    const rows = rowsAt(best);
    const px = f.sx(best);
    cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
    clear(dots);
    for (const row of rows) if (row.y !== undefined && row.y !== null && isFinite(row.y)) dots.appendChild(s('circle', { cx: px, cy: f.sy(row.y), r: 4, style: { fill: row.color } }));
    tip.replaceChildren(h('div', { class: 'chart-tip-head' }, xFormat(best)),
      ...rows.map(row => h('div', { class: 'chart-tip-row' }, h('span', { class: 'legend-swatch', style: { '--c': row.color } }), h('span', { class: 'tip-name' }, row.name),
        h('span', { class: 'tip-val' }, row.value === null || row.value === undefined || !isFinite(row.value) ? '—' : valueFormat(row.value)))));
    tip.hidden = false;
    const wr = wrap.getBoundingClientRect();
    let left = e.clientX - wr.left + 14;
    if (left + tip.offsetWidth > wr.width) left = e.clientX - wr.left - tip.offsetWidth - 14;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, e.clientY - wr.top - 10)}px`;
  });
}

function emptyChart(text) { return h('div', { class: 'chart chart-empty' }, text); }

/**
 * lineChart({series: [{name, color, points: [[x, y]], dash}], xLabel, yLabel, height,
 *   xFormat, valueFormat, yMin}) → element.
 */
export function lineChart({ width = W_DEFAULT, integerY = false, series, xLabel = 'generation', yLabel = '', height = 240, xFormat, valueFormat, yMin = null, integerX = true }) {
  series = (series || []).map((sr, i) => ({ color: seriesColor(i), ...sr, points: (sr.points || []).filter(p => p && p[1] !== null && p[1] !== undefined && isFinite(p[1]) && isFinite(p[0])) }));
  const all = series.flatMap(sr => sr.points);
  if (!all.length) return emptyChart('No data yet.');
  const xs = [...new Set(all.map(p => p[0]))].sort((a, b) => a - b);
  let ymin = Math.min(...all.map(p => p[1])), ymax = Math.max(...all.map(p => p[1]));
  if (yMin !== null) ymin = Math.min(ymin, yMin);
  const xTickValues = integerX ? intTicks(xs[0], xs[xs.length - 1]) : null;
  const f = frame({ width, xDomain: [xs[0], xs[xs.length - 1]], yDomain: [ymin, ymax], height, xLabel, yLabel, xTickValues, integerY });
  const g = s('g', { class: 'chart-lines' });
  for (const sr of series) {
    if (!sr.points.length) continue;
    const d = sr.points.map((p, i) => `${i ? 'L' : 'M'}${f.sx(p[0]).toFixed(1)},${f.sy(p[1]).toFixed(1)}`).join('');
    g.appendChild(s('path', { d, class: 'series-line', style: { stroke: sr.color }, 'stroke-dasharray': sr.dash ? '5 4' : null }));
    if (sr.points.length <= 40) for (const p of sr.points) g.appendChild(s('circle', { cx: f.sx(p[0]), cy: f.sy(p[1]), r: sr.points.length === 1 ? 4 : 2.5, class: 'series-dot', style: { fill: sr.color } }));
  }
  f.svg.appendChild(g);
  const wrap = h('div', { class: 'chart' }, legend(series), f.svg);
  const maps = series.map(sr => new Map(sr.points.map(p => [p[0], p[1]])));
  hoverLayer(wrap, f, xs, (x) => series.map((sr, i) => ({ name: sr.name, color: sr.color, value: maps[i].get(x), y: maps[i].get(x) })),
    { xFormat: xFormat || ((x) => `${xLabel} ${x}`), valueFormat: valueFormat || fmtTick });
  return wrap;
}

function intTicks(a, b) {
  const span = b - a;
  if (span <= 0) return [a];
  const step = Math.max(1, niceTicks(0, span, Math.min(8, span)).reduce((acc, v, i, arr) => i === 1 ? v - arr[0] : acc, 1));
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b; v += step) out.push(v);
  return out;
}

/**
 * stackedArea({x: [x...], series: [{name, color, values: [number per x]}], xLabel, yLabel})
 */
export function stackedArea({ width = W_DEFAULT, x, series, xLabel = 'generation', yLabel = '', height = 260, valueFormat }) {
  x = x || [];
  series = (series || []).map((sr, i) => ({ color: seriesColor(i), ...sr }));
  if (!x.length || !series.length) return emptyChart('No data yet.');
  const totals = x.map((_, i) => series.reduce((a, sr) => a + (Number(sr.values[i]) || 0), 0));
  const f = frame({ width, xDomain: [x[0], x[x.length - 1]], yDomain: [0, Math.max(1, ...totals)], height, xLabel, yLabel, xTickValues: intTicks(x[0], x[x.length - 1]), integerY: true });
  const g = s('g', { class: 'chart-areas' });
  const base = x.map(() => 0);
  for (const sr of series) {
    const top = x.map((_, i) => base[i] + (Number(sr.values[i]) || 0));
    if (top.every((v, i) => v === base[i])) continue;
    const up = x.map((xv, i) => `${f.sx(xv).toFixed(1)},${f.sy(top[i]).toFixed(1)}`);
    const down = x.map((xv, i) => `${f.sx(xv).toFixed(1)},${f.sy(base[i]).toFixed(1)}`).reverse();
    const d = x.length === 1
      ? `M${f.sx(x[0]) - 6},${f.sy(top[0])}H${f.sx(x[0]) + 6}V${f.sy(base[0])}H${f.sx(x[0]) - 6}Z`
      : `M${up.join('L')}L${down.join('L')}Z`;
    g.appendChild(s('path', { d, class: 'series-area', style: { fill: sr.color } }));
    for (let i = 0; i < x.length; i++) base[i] = top[i];
  }
  f.svg.appendChild(g);
  const wrap = h('div', { class: 'chart' }, legend(series), f.svg);
  hoverLayer(wrap, f, x, (xv) => {
    const i = x.indexOf(xv);
    let acc = 0;
    const rows = series.map(sr => { const v = Number(sr.values[i]) || 0; acc += v; return { name: sr.name, color: sr.color, value: v, y: v ? acc : null }; })
      .filter(r => r.value);
    rows.reverse();
    rows.push({ name: 'total', color: 'transparent', value: totals[i] });
    return rows;
  }, { xFormat: (xv) => `${xLabel} ${xv}`, valueFormat: valueFormat || fmtTick });
  return wrap;
}

/**
 * barChart({items: [{label, value, color, detail}], yLabel, valueFormat}) — vertical bars
 * from a zero baseline (negative values go down); per-bar hover tooltip.
 */
export function barChart({ width = W_DEFAULT, integerY = false, items, yLabel = '', height = 260, valueFormat = fmtTick }) {
  items = (items || []).filter(it => it && it.value !== null && it.value !== undefined && isFinite(it.value));
  if (!items.length) return emptyChart('No data.');
  const vals = items.map(it => Number(it.value));
  const f = frame({ width, xDomain: [0, items.length], yDomain: [Math.min(0, ...vals), Math.max(0, ...vals)], height, xLabel: null, yLabel, xTickValues: [], bottom: items.length > 6 ? 70 : M.bottom, integerY });
  const bw = f.pw / items.length;
  const g = s('g', { class: 'chart-bars' });
  const wrap = h('div', { class: 'chart' });
  const tip = h('div', { class: 'chart-tip', hidden: true });
  const rotate = items.length > 6;
  items.forEach((it, i) => {
    const v = Number(it.value);
    const w = Math.max(2, Math.min(96, bw - 2 * Math.min(6, bw * 0.15)));
    const x = M.left + i * bw + (bw - w) / 2;
    const y = f.sy(Math.max(0, v)), y0 = f.sy(Math.min(0, v));
    const color = it.color || seriesColor(0);
    const bar = s('rect', { x, y, width: w, height: Math.max(1, y0 - y), rx: 3, class: 'bar', style: { fill: color } });
    const hit = s('rect', { x: M.left + i * bw, y: M.top, width: bw, height: f.ph, fill: 'transparent', class: 'hover-target' });
    const show = (e) => {
      tip.replaceChildren(h('div', { class: 'chart-tip-head' }, it.label),
        h('div', { class: 'chart-tip-row' }, h('span', { class: 'legend-swatch', style: { '--c': color } }), h('span', { class: 'tip-name' }, yLabel || 'value'), h('span', { class: 'tip-val' }, valueFormat(v))),
        it.detail ? h('div', { class: 'chart-tip-row muted' }, it.detail) : null);
      tip.hidden = false;
      const wr = wrap.getBoundingClientRect();
      let left = e.clientX - wr.left + 14;
      if (left + tip.offsetWidth > wr.width) left = e.clientX - wr.left - tip.offsetWidth - 14;
      tip.style.left = `${Math.max(0, left)}px`; tip.style.top = `${Math.max(0, e.clientY - wr.top - 10)}px`;
      bar.classList.add('hover');
    };
    hit.addEventListener('mousemove', show);
    hit.addEventListener('mouseleave', () => { tip.hidden = true; bar.classList.remove('hover'); });
    const lx = M.left + i * bw + bw / 2, ly = f.H - f.mb + 14;
    const label = it.label.length > 16 ? it.label.slice(0, 15) + '…' : it.label;
    g.append(bar, s('text', { x: lx, y: ly, class: 'tick', 'text-anchor': rotate ? 'end' : 'middle', transform: rotate ? `rotate(-28 ${lx} ${ly})` : null, text: label }), hit);
  });
  f.svg.appendChild(g);
  wrap.append(f.svg, tip);
  return wrap;
}
