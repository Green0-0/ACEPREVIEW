// ACE studio — entry point: theme, hash router and every view of recording spec §12.2.
import { h, clear, replace, toast, fmtTime, fmtSeconds, fmtNum, shortId, prettyJson, truncatedText, badge, stateBlock, link } from './ui.js';
import { api, paths } from './api.js';
import { lineChart, stackedArea, barChart, speciesColor, seriesColor } from './charts.js';
import { renderGraph, typeChip, typeColor, kindColor, WORKER_TYPES } from './graph.js';
import { renderDiff } from './diff.js';
import { renderRollout } from './chat.js';

if (location.search.includes('mock=1')) await import('./dev-mock.js');

const POLL_MS = 3000;
const THEME_KEY = 'ace.theme';
const enc = encodeURIComponent;

// ---------- theme ----------
function currentTheme() {
  const attr = document.documentElement.getAttribute('data-theme');
  if (attr === 'light' || attr === 'dark') return attr;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
document.getElementById('theme-toggle').addEventListener('click', () => {
  const t = currentTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', t);
  try { localStorage.setItem(THEME_KEY, t); } catch { /* storage unavailable */ }
});

// ---------- routes (hash helpers) ----------
export const R = {
  runs: () => '#/runs',
  run: (run) => `#/run/${enc(run)}`,
  generation: (run, g) => `#/run/${enc(run)}/generations/${enc(g)}`,
  species: (run) => `#/run/${enc(run)}/species`,
  genomes: (run, q) => `#/run/${enc(run)}/genomes${hashQuery(q)}`,
  genome: (run, gid) => `#/run/${enc(run)}/genome/${enc(gid)}`,
  evaluation: (run, gid, task) => `#/run/${enc(run)}/eval/${enc(gid)}/${enc(task)}`,
  operators: (run) => `#/run/${enc(run)}/operators`,
  failures: (run) => `#/run/${enc(run)}/failures`,
  compare: (run, a, b) => `#/run/${enc(run)}/compare${hashQuery({ a, b })}`,
  innovations: (run) => `#/run/${enc(run)}/innovations`,
};
function hashQuery(q) {
  const parts = Object.entries(q || {}).filter(([, v]) => v !== null && v !== undefined && v !== '').map(([k, v]) => `${enc(k)}=${enc(String(v))}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

const ROUTES = [
  [/^\/?(?:runs)?$/, viewRuns],
  [/^\/run\/([^/]+)$/, viewOverview],
  [/^\/run\/([^/]+)\/generations\/(-?\d+)$/, viewGeneration],
  [/^\/run\/([^/]+)\/species$/, viewSpecies],
  [/^\/run\/([^/]+)\/genomes$/, viewGenomes],
  [/^\/run\/([^/]+)\/genome\/([^/]+)$/, viewGenome],
  [/^\/run\/([^/]+)\/eval\/([^/]+)\/([^/]+)$/, viewEvaluation],
  [/^\/run\/([^/]+)\/operators$/, viewOperators],
  [/^\/run\/([^/]+)\/failures$/, viewFailures],
  [/^\/run\/([^/]+)\/compare$/, viewCompare],
  [/^\/run\/([^/]+)\/innovations$/, viewInnovations],
];

const root = document.getElementById('app');
let current = null; // {token, cleanups}
let tokenSeq = 0;

function parseHash() {
  const raw = (location.hash || '').replace(/^#/, '');
  const qi = raw.indexOf('?');
  const path = qi >= 0 ? raw.slice(0, qi) : raw;
  const query = new URLSearchParams(qi >= 0 ? raw.slice(qi + 1) : '');
  return { path, query };
}

function dec(v) { try { return decodeURIComponent(v); } catch { return v; } }

async function route() {
  if (current) for (const fn of current.cleanups) { try { fn(); } catch { /* ignore */ } }
  closeModal();
  const token = ++tokenSeq;
  const ctx = {
    token, cleanups: [], query: null,
    alive: () => current === ctx,
    onCleanup(fn) { ctx.cleanups.push(fn); },
    interval(fn, ms) { const t = setInterval(fn, ms); ctx.cleanups.push(() => clearInterval(t)); return t; },
  };
  current = ctx;
  const { path, query } = parseHash();
  ctx.query = query;
  let handler = null, params = [];
  for (const [re, fn] of ROUTES) { const m = re.exec(path); if (m) { handler = fn; params = m.slice(1).map(dec); break; } }
  replace(root, stateBlock('loading', 'Loading…'));
  if (!handler) { replace(root, crumbs([['Runs', R.runs()], ['Not found']]), stateBlock('error', `No page for "#${path}".`)); return; }
  try {
    const node = await handler(ctx, ...params);
    if (ctx.alive() && node) { replace(root, node); window.scrollTo(0, 0); }
  } catch (err) {
    if (!ctx.alive()) return;
    replace(root, params[0] ? runHeader(params[0], [['Error']]) : crumbs([['Runs', R.runs()], ['Error']]),
      stateBlock('error', err && err.status === 404 ? `Not found: ${err.message}` : `Failed to load: ${err && err.message ? err.message : err}`, { label: 'Retry', onClick: route }));
  }
}
window.addEventListener('hashchange', route);

// ---------- shared chrome ----------
function crumbs(items) {
  const nav = h('nav', { class: 'crumbs', 'aria-label': 'Breadcrumb' });
  items.forEach(([label, href], i) => {
    if (i) nav.appendChild(h('span', { class: 'crumb-sep', 'aria-hidden': 'true' }, '›'));
    nav.appendChild(href ? h('a', { href }, label) : h('span', { class: 'crumb-current', 'aria-current': 'page' }, label));
  });
  return nav;
}

const RUN_TABS = [
  ['overview', 'Overview', R.run], ['genomes', 'Genomes', (r) => R.genomes(r)], ['species', 'Species', R.species],
  ['operators', 'Operators', R.operators], ['failures', 'Failures', R.failures], ['innovations', 'Innovations', R.innovations],
  ['compare', 'Compare', (r) => R.compare(r)],
];
const runMeta = new Map(); // run_id → run.json (cached for the header status badge)

/** Breadcrumb + run sub-navigation shown on every run page. */
function runHeader(runId, trail, active = null) {
  const status = h('span', { class: 'run-status' });
  const paint = (run) => replace(status, statusBadge(run.status));
  if (runMeta.has(runId)) paint(runMeta.get(runId));
  else api.run(runId).then(r => { runMeta.set(runId, r); paint(r); }).catch(() => {});
  return h('div', { class: 'page-head' },
    h('div', { class: 'page-head-row' }, crumbs([['Runs', R.runs()], [runId, trail.length ? R.run(runId) : null], ...trail]), status),
    h('nav', { class: 'tabs run-tabs', 'aria-label': 'Run pages' },
      RUN_TABS.map(([key, label, fn]) => h('a', { class: 'tab', href: fn(runId), 'aria-selected': key === active ? 'true' : 'false' }, label))));
}

function statusBadge(status) {
  if (status === 'running') return h('span', { class: 'live-badge', title: 'Run in progress — refreshing every 3 s' }, h('span', { class: 'live-dot' }), 'LIVE');
  return badge(status || 'unknown', status === 'finished' ? 'ok' : status === 'failed' ? 'danger' : '');
}

function card(title, body, { actions = null, cls = '' } = {}) {
  return h('section', { class: `card ${cls}` },
    title ? h('div', { class: 'card-head' }, h('h3', null, title), actions) : null,
    h('div', { class: 'card-body' }, body));
}

function kv(rows) {
  const dl = h('dl', { class: 'kv' });
  for (const [k, v, mono] of rows) {
    if (v === undefined) continue;
    dl.append(h('dt', null, k), h('dd', { class: mono ? 'mono' : '' }, v === null || v === '' ? '—' : v));
  }
  return dl;
}

function statTiles(items) {
  return h('div', { class: 'tiles' }, items.map(([label, value, sub]) => h('div', { class: 'tile' },
    h('div', { class: 'tile-label' }, label), h('div', { class: 'tile-value' }, value ?? '—'), sub ? h('div', { class: 'tile-sub' }, sub) : null)));
}

function speciesChip(id) {
  if (id === null || id === undefined) return h('span', { class: 'muted' }, '—');
  return h('span', { class: 'species-chip', style: { '--c': speciesColor(id) }, title: `species ${id}` }, `S${id}`);
}
function opChip(op) { return h('span', { class: 'op-chip', title: `operator ${op}` }, op); }
function genomeLink(run, gid, label = null) { return gid ? h('a', { class: 'mono', href: R.genome(run, gid), title: gid }, label || shortId(gid)) : h('span', { class: 'muted' }, '—'); }
function workerTypeChips(wt) {
  const entries = Object.entries(wt || {}).filter(([, n]) => n).sort((a, b) => WORKER_TYPES.indexOf(a[0]) - WORKER_TYPES.indexOf(b[0]));
  return h('span', { class: 'chip-list' }, entries.map(([t, n]) => typeChip(t, n)));
}
function fitnessCell(v) { return h('span', { class: 'num' }, fmtNum(v, 3)); }

/**
 * Sortable table. columns: [{key, label, sort?: row → value (false = not sortable), render?: row → node, num?}]
 */
function sortableTable(columns, rows, { sortKey = null, sortDir = 'desc', onRow = null, empty = 'No rows.', rowClass = null } = {}) {
  if (!rows || !rows.length) return stateBlock('empty', empty);
  const state = { key: sortKey, dir: sortDir };
  const tbody = h('tbody');
  const ths = [];
  const valueOf = (c, r) => (c.sort ? c.sort(r) : r[c.key]);
  const paint = () => {
    let list = rows.slice();
    const col = columns.find(c => c.key === state.key);
    if (col && col.sort !== false) {
      list.sort((a, b) => {
        const va = valueOf(col, a), vb = valueOf(col, b);
        const na = va === null || va === undefined, nb = vb === null || vb === undefined;
        if (na || nb) return na === nb ? 0 : na ? 1 : -1;
        const cmp = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb), undefined, { numeric: true });
        return state.dir === 'asc' ? cmp : -cmp;
      });
    }
    clear(tbody);
    for (const r of list) {
      const tr = h('tr', { class: `${onRow ? 'clickable' : ''} ${rowClass ? rowClass(r) || '' : ''}` },
        columns.map(c => h('td', { class: c.num ? 'num' : '' }, c.render ? c.render(r) : (r[c.key] ?? '—'))));
      if (onRow) tr.addEventListener('click', (e) => { if (!e.target.closest('a,button,summary')) onRow(r); });
      tbody.appendChild(tr);
    }
    ths.forEach((th, i) => {
      const c = columns[i];
      th.setAttribute('aria-sort', c.key === state.key ? (state.dir === 'asc' ? 'ascending' : 'descending') : 'none');
    });
  };
  const thead = h('thead', null, h('tr', null, columns.map(c => {
    const sortable = c.sort !== false;
    const th = h('th', { class: `${c.num ? 'num' : ''} ${sortable ? 'sortable' : ''}`, scope: 'col', title: c.title || null },
      sortable ? h('button', { type: 'button', class: 'th-btn', onClick: () => {
        if (state.key === c.key) state.dir = state.dir === 'asc' ? 'desc' : 'asc';
        else { state.key = c.key; state.dir = c.num ? 'desc' : 'asc'; }
        paint();
      } }, c.label, h('span', { class: 'sort-ind', 'aria-hidden': 'true' })) : c.label);
    ths.push(th);
    return th;
  })));
  paint();
  return h('div', { class: 'table-wrap' }, h('table', { class: 'data-table' }, thead, tbody));
}

// ---------- modal ----------
let modalClose = null;
function openModal(title, body) {
  closeModal();
  const prev = document.activeElement;
  const dialog = h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': title, tabindex: '-1' },
    h('div', { class: 'modal-head' }, h('h2', { class: 'modal-title' }, title), h('button', { class: 'icon-btn', 'aria-label': 'Close', onClick: () => closeModal() }, '×')),
    h('div', { class: 'modal-body' }, body));
  const backdrop = h('div', { class: 'modal-backdrop', onMousedown: (e) => { if (e.target === backdrop) closeModal(); } }, dialog);
  const onKey = (e) => { if (e.key === 'Escape') closeModal(); };
  document.addEventListener('keydown', onKey);
  document.getElementById('modal-root').appendChild(backdrop);
  dialog.focus();
  modalClose = () => { document.removeEventListener('keydown', onKey); backdrop.remove(); modalClose = null; if (prev && prev.focus) prev.focus(); };
  return dialog;
}
function closeModal() { if (modalClose) modalClose(); }

// =====================================================================================
// Views
// =====================================================================================

// ---------- #/runs ----------
async function viewRuns(ctx) {
  const runs = await api.runs();
  const body = h('div');
  const paint = (list) => replace(body, sortableTable([
    { key: 'run_id', label: 'Run', render: r => h('a', { class: 'mono', href: R.run(r.run_id) }, r.run_id) },
    { key: 'status', label: 'Status', render: r => statusBadge(r.status) },
    { key: 'evaluator_type', label: 'Evaluator', render: r => r.evaluator_type ? badge(r.evaluator_type, 'mono') : '—' },
    { key: 'population_size', label: 'Population', num: true },
    { key: 'generations_completed', label: 'Generations', num: true },
    { key: 'best_fitness', label: 'Best fitness', num: true, render: r => fitnessCell(r.best_fitness) },
    { key: 'started_at', label: 'Started', render: r => fmtTime(r.started_at) },
    { key: 'finished_at', label: 'Finished', render: r => r.status === 'running' ? h('span', { class: 'muted' }, 'in progress') : fmtTime(r.finished_at) },
  ], list, { sortKey: 'started_at', sortDir: 'desc', onRow: (r) => { location.hash = R.run(r.run_id); }, empty: 'No runs found under the runs root. Start an evolution run with the ACE CLI, then refresh.' }));
  paint(runs);
  if (runs.some(r => r.status === 'running')) {
    ctx.interval(async () => { try { const next = await api.runs({ silent: true }); if (ctx.alive()) paint(next); } catch { /* keep last */ } }, POLL_MS);
  }
  return h('div', { class: 'page' },
    h('div', { class: 'page-head' }, h('div', { class: 'page-head-row' }, crumbs([['Runs']]))),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Evolution runs'), h('span', { class: 'muted' }, `${runs.length} run${runs.length === 1 ? '' : 's'}`)),
    card(null, body));
}

// ---------- #/run/{id} ----------
function fitnessSeries(gens) {
  const pts = (k) => gens.map(g => [g.generation, g[k]]);
  return [
    { name: 'best', points: pts('best_fitness'), color: seriesColor(0) },
    { name: 'mean', points: pts('mean_fitness'), color: seriesColor(1) },
    { name: 'median', points: pts('median_fitness'), color: seriesColor(2) },
    { name: 'best adjusted', points: pts('best_adjusted'), color: seriesColor(0), dash: true },
    { name: 'mean adjusted', points: pts('mean_adjusted'), color: seriesColor(1), dash: true },
  ];
}
function single(name, gens, key, color) { return [{ name, points: gens.map(g => [g.generation, g[key]]), color }]; }

function overviewCharts(gens) {
  return h('div', { class: 'chart-grid' },
    card('Fitness per generation', lineChart({ width: 1280, height: 300, series: fitnessSeries(gens), yLabel: 'fitness', valueFormat: (v) => fmtNum(v, 3) }), { cls: 'span-2' }),
    card('Species count', lineChart({ integerY: true, series: single('species', gens, 'species_count', seriesColor(2)), yLabel: 'species', yMin: 0 })),
    card('Compatibility threshold', lineChart({ series: single('threshold', gens, 'threshold', seriesColor(6)), yLabel: 'threshold', valueFormat: (v) => fmtNum(v, 3) })),
    card('Mean workers per genome', lineChart({ series: single('mean workers', gens, 'mean_workers', seriesColor(3)), yLabel: 'workers', yMin: 0, valueFormat: (v) => fmtNum(v, 2) })),
    card('Evaluations per generation', lineChart({ integerY: true, series: single('evaluated', gens, 'evaluated', seriesColor(4)), yLabel: 'evaluations', yMin: 0 })),
    card('Generation duration', lineChart({ series: single('duration', gens, 'duration_s', seriesColor(5)), yLabel: 'seconds', yMin: 0, valueFormat: fmtSeconds })));
}

function generationsTable(runId, gens) {
  return sortableTable([
    { key: 'generation', label: 'Gen', num: true, render: g => h('a', { href: R.generation(runId, g.generation) }, String(g.generation)) },
    { key: 'evaluated', label: 'Evaluated', num: true },
    { key: 'elites', label: 'Elites', num: true },
    { key: 'species_count', label: 'Species', num: true },
    { key: 'threshold', label: 'Threshold', num: true, render: g => fmtNum(g.threshold, 3) },
    { key: 'best_fitness', label: 'Best', num: true, render: g => fitnessCell(g.best_fitness) },
    { key: 'mean_fitness', label: 'Mean', num: true, render: g => fitnessCell(g.mean_fitness) },
    { key: 'median_fitness', label: 'Median', num: true, render: g => fitnessCell(g.median_fitness) },
    { key: 'best_adjusted', label: 'Best adj.', num: true, render: g => fitnessCell(g.best_adjusted) },
    { key: 'mean_workers', label: 'Workers', num: true, render: g => fmtNum(g.mean_workers, 2) },
    { key: 'duration_s', label: 'Duration', num: true, render: g => fmtSeconds(g.duration_s) },
    { key: 'best_genome_id', label: 'Best genome', render: g => genomeLink(runId, g.best_genome_id) },
  ], gens, { sortKey: 'generation', sortDir: 'desc', onRow: (g) => { location.hash = R.generation(runId, g.generation); }, empty: 'No generation finished yet.' });
}

const EVENT_KIND = { run_start: 'ok', run_end: 'ok', error: 'danger', generation_start: 'running', generation_end: 'running', speciated: '', evaluation_end: '', genome_evaluated: '' };
function eventsList(runId, events) {
  if (!events || !events.length) return stateBlock('empty', 'No events yet.');
  const list = h('ol', { class: 'events' });
  for (const ev of events.slice().reverse()) {
    const extras = [];
    for (const [k, v] of Object.entries(ev)) {
      if (k === 'ts' || k === 'event') continue;
      let val;
      if (k === 'genome_id' && v) val = genomeLink(runId, v);
      else if (k === 'generation' && v !== null && v !== undefined) val = h('a', { href: R.generation(runId, v) }, String(v));
      else if (k === 'task_index' && ev.genome_id && ev.event === 'evaluation_end') val = h('a', { href: R.evaluation(runId, ev.genome_id, v) }, String(v));
      else if (k.endsWith('duration_s')) val = fmtSeconds(v);
      else if (typeof v === 'number') val = fmtNum(v, Number.isInteger(v) ? 0 : 3);
      else val = v === null ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v);
      extras.push(h('span', { class: `ev-field ${k === 'failure' && v ? 'ev-fail' : ''}` }, h('span', { class: 'ev-k' }, k), val));
    }
    list.appendChild(h('li', { class: 'event' },
      h('span', { class: 'ev-ts mono' }, ev.ts ? new Date(ev.ts).toLocaleTimeString(undefined, { hour12: false }) : ''),
      badge(ev.event || '?', EVENT_KIND[ev.event] ?? ''), h('span', { class: 'ev-fields' }, extras)));
  }
  return list;
}

function configTree(value, depth = 0) {
  if (value === null || typeof value !== 'object') return h('span', { class: 'cfg-val mono' }, value === null ? 'null' : JSON.stringify(value));
  if (Array.isArray(value) && value.every(v => v === null || typeof v !== 'object')) return h('span', { class: 'cfg-val mono' }, JSON.stringify(value));
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value);
  const ul = h('ul', { class: 'cfg-tree' });
  for (const [k, v] of entries) {
    if (v !== null && typeof v === 'object' && !(Array.isArray(v) && v.every(x => x === null || typeof x !== 'object'))) {
      ul.appendChild(h('li', null, h('details', { open: depth < 1 }, h('summary', { class: 'cfg-key' }, k, h('span', { class: 'muted' }, Array.isArray(v) ? ` [${v.length}]` : ` {${Object.keys(v).length}}`)), configTree(v, depth + 1))));
    } else ul.appendChild(h('li', { class: 'cfg-leaf' }, h('span', { class: 'cfg-key' }, k), configTree(v, depth + 1)));
  }
  return ul;
}

async function viewOverview(ctx, runId) {
  const [run, progress] = await Promise.all([api.run(runId), api.progress(runId)]);
  runMeta.set(runId, run);
  const counts = run.counts || {};
  const tilesEl = h('div'), chartsEl = h('div'), genTable = h('div'), eventsEl = h('div'), statusEl = h('span'), metaEl = h('div');
  const paint = (r, p) => {
    const gens = (p.generations || []).slice().sort((a, b) => a.generation - b.generation);
    const last = gens[gens.length - 1] || {};
    replace(statusEl, statusBadge(p.status || r.status));
    replace(tilesEl, statTiles([
      ['Status', statusBadge(p.status || r.status), r.status === 'running' ? 'polling every 3 s' : null],
      ['Generations', String(r.generations_completed ?? gens.length), r.config && r.config.generations !== undefined ? `of ${r.config.generations}` : null],
      ['Best fitness', fmtNum(r.best_fitness, 3), r.best_genome_id ? genomeLink(runId, r.best_genome_id) : null],
      ['Genomes', String(counts.genomes ?? '—'), counts.evaluations !== undefined ? `${counts.evaluations} evaluations` : null],
      ['Species', String(last.species_count ?? counts.species ?? '—'), last.threshold !== undefined ? `threshold ${fmtNum(last.threshold, 3)}` : null],
      ['Duration', r.duration_s !== null && r.duration_s !== undefined ? fmtSeconds(r.duration_s) : r.started_at ? fmtSeconds((Date.now() - new Date(r.started_at)) / 1000) : '—', r.status === 'running' ? 'so far' : null],
    ]));
    replace(chartsEl, overviewCharts(gens));
    replace(genTable, generationsTable(runId, gens));
    replace(eventsEl, eventsList(runId, (p.events_tail || []).slice(-50)));
  };
  const model = run.model || {};
  const ev = run.evaluator || {};
  replace(metaEl, kv([
    ['Run id', run.run_id || runId, true], ['ACE version', run.ace_version ?? null, true], ['Status', statusEl],
    ['Started', fmtTime(run.started_at)], ['Finished', fmtTime(run.finished_at)], ['Duration', fmtSeconds(run.duration_s)],
    ['Seed', run.seed ?? null, true], ['Resumed from', run.resumed_from ?? null, true],
    ['Model', model.model ?? null, true], ['API base URL', model.api_base_url ?? null, true], ['API key env var', model.api_key_env_var ?? null, true],
    ['Evaluator', ev.type ?? null, true],
    ['Tasks', ev.tasks && ev.tasks.length ? h('ol', { class: 'task-list' }, ev.tasks.map(t => h('li', null,
      h('div', { class: 'mono' }, `#${t.index} · ${t.project_dir} · scorer ${t.scorer}`), truncatedText(t.prompt, 300)))) : null],
    ['Best genome', run.best_genome_id ? genomeLink(runId, run.best_genome_id, run.best_genome_id) : null],
    ['Counts', Object.keys(counts).length ? Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(' · ') : null],
    ['Error', run.error ? h('pre', { class: 'error-text' }, run.error) : null],
  ]));
  paint(run, progress);
  if ((progress.status || run.status) === 'running') {
    let busy = false;
    ctx.interval(async () => {
      if (busy) return; busy = true;
      try {
        const p = await api.progress(runId, { silent: true });
        if (!ctx.alive()) return;
        let r = run;
        if (p.status !== 'running') { r = await api.run(runId); runMeta.set(runId, r); toast(`Run ${runId} ${p.status}.`, p.status === 'failed' ? 'error' : 'success'); }
        paint(r, p);
        if (p.status !== 'running') { for (const fn of ctx.cleanups) fn(); ctx.cleanups = []; }
      } catch { /* transient; next tick retries */ } finally { busy = false; }
    }, POLL_MS);
  }
  return h('div', { class: 'page' },
    runHeader(runId, [], 'overview'),
    h('div', { class: 'page-title-row' }, h('h1', { class: 'mono' }, runId), h('span', { class: 'muted' }, run.started_at ? `started ${fmtTime(run.started_at)}` : '')),
    tilesEl,
    chartsEl,
    h('div', { class: 'two-col' },
      card('Generations', genTable, { cls: 'grow' }),
      card('Recent events', eventsEl, { actions: (progress.status || run.status) === 'running' ? statusBadge('running') : null })),
    h('div', { class: 'two-col' },
      card('Run metadata', metaEl),
      card('Configuration', run.config ? configTree(run.config) : stateBlock('empty', 'No configuration recorded.'))));
}

// ---------- #/run/{id}/generations/{g} ----------
async function viewGeneration(ctx, runId, g) {
  const gen = await api.generation(runId, g);
  const gi = Number(g);
  const pop = gen.population || [];
  const ops = Object.entries(gen.operator_counts || {}).sort((a, b) => b[1] - a[1]);
  const table = sortableTable([
    { key: 'genome_id', label: 'Genome', render: r => genomeLink(runId, r.genome_id) },
    { key: 'species_id', label: 'Species', num: true, render: r => speciesChip(r.species_id) },
    { key: 'fitness', label: 'Fitness', num: true, render: r => fitnessCell(r.fitness) },
    { key: 'adjusted_fitness', label: 'Adjusted', num: true, render: r => fitnessCell(r.adjusted_fitness) },
    { key: 'num_workers', label: 'Workers', num: true },
    { key: 'worker_types', label: 'Worker types', sort: r => Object.keys(r.worker_types || {}).sort().join(','), render: r => workerTypeChips(r.worker_types) },
    { key: 'operators', label: 'Operators', sort: r => (r.operators || []).join(','), render: r => h('span', { class: 'chip-list' }, (r.operators || []).map(opChip)) },
    { key: 'is_elite', label: 'Elite', sort: r => (r.is_elite ? 1 : 0), render: r => r.is_elite ? badge('elite', 'ok') : '' },
    { key: 'parents', label: 'Parents', sort: r => (r.parents || []).length, render: r => h('span', { class: 'chip-list' }, (r.parents || []).map(p => genomeLink(runId, p))) },
    { key: 'structure_hash', label: 'Structure', render: r => h('span', { class: 'mono muted', title: r.structure_hash }, shortId(r.structure_hash, 7)) },
  ], pop, { sortKey: 'fitness', sortDir: 'desc', onRow: (r) => { location.hash = R.genome(runId, r.genome_id); }, rowClass: r => r.genome_id === gen.best_genome_id ? 'row-best' : '', empty: 'This generation has no population recorded.' });
  return h('div', { class: 'page' },
    runHeader(runId, [[`Generation ${g}`]], null),
    h('div', { class: 'page-title-row' }, h('h1', null, `Generation ${g}`),
      h('span', { class: 'row' },
        gi > 0 ? h('a', { class: 'btn btn-sm', href: R.generation(runId, gi - 1) }, '← previous') : null,
        h('a', { class: 'btn btn-sm', href: R.generation(runId, gi + 1) }, 'next →'))),
    statTiles([
      ['Evaluated', String(gen.evaluated ?? pop.length), `${gen.elites ?? 0} elites`],
      ['Best fitness', fmtNum(gen.best_fitness, 3), gen.best_genome_id ? genomeLink(runId, gen.best_genome_id) : null],
      ['Mean / median', `${fmtNum(gen.mean_fitness, 3)} / ${fmtNum(gen.median_fitness, 3)}`, `adjusted mean ${fmtNum(gen.mean_adjusted, 3)}`],
      ['Species', String(gen.species_count ?? '—'), `threshold ${fmtNum(gen.threshold, 3)}`],
      ['Mean workers', fmtNum(gen.mean_workers, 2), null],
      ['Duration', fmtSeconds(gen.duration_s), fmtTime(gen.started_at)],
    ]),
    ops.length ? card('Operators applied', h('div', { class: 'chip-list' }, ops.map(([op, n]) => h('span', { class: 'op-chip' }, op, h('span', { class: 'type-chip-n' }, `×${n}`))))) : null,
    card(`Population (${pop.length})`, table));
}

// ---------- #/run/{id}/species ----------
async function viewSpecies(ctx, runId) {
  const data = await api.species(runId);
  const timeline = (data.timeline || []).slice().sort((a, b) => a.generation - b.generation);
  const info = data.species || {};
  const ids = [...new Set([...Object.keys(info), ...timeline.flatMap(t => (t.species || []).map(sp => String(sp.id)))])].sort((a, b) => Number(a) - Number(b));
  const xs = timeline.map(t => t.generation);
  const sizeSeries = ids.map(id => ({ name: `S${id}`, color: speciesColor(id), values: timeline.map(t => { const sp = (t.species || []).find(q => String(q.id) === id); return sp ? sp.size : 0; }) }));
  const bestSeries = ids.map(id => ({ name: `S${id}`, color: speciesColor(id), points: timeline.map(t => { const sp = (t.species || []).find(q => String(q.id) === id); return sp ? [t.generation, sp.best_fitness] : null; }).filter(Boolean) }));
  const lastSnap = timeline[timeline.length - 1];
  const rows = ids.map(id => {
    const meta = info[id] || {};
    const cur = lastSnap && (lastSnap.species || []).find(q => String(q.id) === id);
    return { id: Number(id), created: meta.created, last_seen: meta.last_seen, best_fitness: meta.best_fitness, max_size: meta.max_size, size: cur ? cur.size : 0, representative_id: cur ? cur.representative_id : null, alive: !!cur };
  });
  return h('div', { class: 'page' },
    runHeader(runId, [['Species']], 'species'),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Species'), h('span', { class: 'muted' }, `${ids.length} species over ${timeline.length} generations`)),
    h('div', { class: 'chart-grid' },
      card('Species sizes (stacked)', stackedArea({ width: 1280, height: 300, x: xs, series: sizeSeries, yLabel: 'genomes' }), { cls: 'span-2' }),
      card('Best fitness per species', lineChart({ series: bestSeries, yLabel: 'best fitness', valueFormat: (v) => fmtNum(v, 3) })),
      card('Compatibility threshold', lineChart({ series: [{ name: 'threshold', color: seriesColor(6), points: timeline.map(t => [t.generation, t.threshold]) }], yLabel: 'threshold', valueFormat: (v) => fmtNum(v, 3) }))),
    card('All species', sortableTable([
      { key: 'id', label: 'Species', num: true, render: r => speciesChip(r.id) },
      { key: 'alive', label: 'Alive', sort: r => (r.alive ? 1 : 0), render: r => r.alive ? badge('alive', 'ok') : badge('extinct') },
      { key: 'created', label: 'Created (gen)', num: true, render: r => r.created !== undefined && r.created !== null ? h('a', { href: R.generation(runId, r.created) }, String(r.created)) : '—' },
      { key: 'last_seen', label: 'Last seen (gen)', num: true, render: r => r.last_seen !== undefined && r.last_seen !== null ? h('a', { href: R.generation(runId, r.last_seen) }, String(r.last_seen)) : '—' },
      { key: 'size', label: 'Current size', num: true },
      { key: 'max_size', label: 'Max size', num: true },
      { key: 'best_fitness', label: 'Best fitness', num: true, render: r => fitnessCell(r.best_fitness) },
      { key: 'representative_id', label: 'Representative', render: r => genomeLink(runId, r.representative_id) },
      { key: 'members', label: 'Members', sort: false, render: r => h('a', { href: R.genomes(runId, { species: r.id }) }, 'genomes →') },
    ], rows, { sortKey: 'best_fitness', sortDir: 'desc', empty: 'No speciation snapshots recorded.' })));
}

// ---------- #/run/{id}/genomes ----------
const GENOME_SORTS = [
  ['-fitness', 'fitness ↓'], ['fitness', 'fitness ↑'], ['-adjusted_fitness', 'adjusted ↓'], ['adjusted_fitness', 'adjusted ↑'],
  ['-generation', 'generation ↓'], ['generation', 'generation ↑'], ['-num_workers', 'workers ↓'], ['num_workers', 'workers ↑'],
];
const PAGE_SIZE = 50;
async function viewGenomes(ctx, runId) {
  const q = ctx.query;
  const filters = { generation: q.get('generation') || '', species: q.get('species') || '', min_fitness: q.get('min_fitness') || '', sort: q.get('sort') || '-fitness' };
  const page = Math.max(0, parseInt(q.get('page') || '0', 10) || 0);
  const res = await api.genomes(runId, { ...filters, limit: PAGE_SIZE, offset: page * PAGE_SIZE });
  const items = res.items || [];
  const total = res.total ?? items.length;
  const go = (patch) => { location.hash = R.genomes(runId, { ...filters, page, ...patch }); };
  const input = (name, attrs) => h('input', { class: 'input', name, value: filters[name], ...attrs });
  const form = h('form', { class: 'filters', onSubmit: (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    go({ generation: fd.get('generation'), species: fd.get('species'), min_fitness: fd.get('min_fitness'), sort: fd.get('sort'), page: 0 });
  } },
    h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Generation'), input('generation', { type: 'number', min: 0, placeholder: 'any' })),
    h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Species'), input('species', { type: 'number', min: 0, placeholder: 'any' })),
    h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Min fitness'), input('min_fitness', { type: 'number', step: 'any', placeholder: 'any' })),
    h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Sort'),
      (() => { const sel = h('select', { class: 'input', name: 'sort' }, GENOME_SORTS.map(([v, l]) => h('option', { value: v }, l))); sel.value = filters.sort; return sel; })()),
    h('div', { class: 'filter-actions' }, h('button', { class: 'btn btn-primary', type: 'submit' }, 'Apply'),
      h('a', { class: 'btn', href: R.genomes(runId) }, 'Reset')));
  const cols = [
    { key: 'id', label: 'Genome', sort: false, render: r => genomeLink(runId, r.id || r.genome_id) },
    { key: 'generation', label: 'Gen', sort: false, num: true, render: r => h('a', { href: R.generation(runId, r.generation) }, String(r.generation)) },
    { key: 'species_id', label: 'Species', sort: false, render: r => speciesChip(r.species_id) },
    { key: 'fitness', label: 'Fitness', sort: false, num: true, render: r => fitnessCell(r.fitness) },
    { key: 'adjusted_fitness', label: 'Adjusted', sort: false, num: true, render: r => fitnessCell(r.adjusted_fitness) },
    { key: 'num_workers', label: 'Workers', sort: false, num: true },
    { key: 'worker_types', label: 'Worker types', sort: false, render: r => workerTypeChips(r.worker_types) },
    { key: 'operators', label: 'Operators', sort: false, render: r => h('span', { class: 'chip-list' }, (r.operators || []).map(opChip)) },
    { key: 'parents', label: 'Parents', sort: false, render: r => h('span', { class: 'chip-list' }, (r.parents || []).map(p => genomeLink(runId, p))) },
    { key: 'eval_duration_s', label: 'Eval time', sort: false, num: true, render: r => fmtSeconds(r.eval_duration_s) },
  ];
  const from = total ? page * PAGE_SIZE + 1 : 0, to = Math.min(total, page * PAGE_SIZE + items.length);
  const pager = h('div', { class: 'pager' },
    h('button', { class: 'btn btn-sm', type: 'button', disabled: page === 0, onClick: () => go({ page: page - 1 }) }, '← prev'),
    h('span', { class: 'muted' }, `${from}–${to} of ${total}`),
    h('button', { class: 'btn btn-sm', type: 'button', disabled: to >= total, onClick: () => go({ page: page + 1 }) }, 'next →'));
  return h('div', { class: 'page' },
    runHeader(runId, [['Genomes']], 'genomes'),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Genomes'), h('span', { class: 'muted' }, `${total} matching`)),
    card(null, form),
    card(null, [sortableTable(cols, items, { onRow: (r) => { location.hash = R.genome(runId, r.id || r.genome_id); }, empty: 'No genomes match these filters.' }), pager]));
}

// ---------- #/run/{id}/genome/{gid} ----------
function propertiesPanel(node, graphNode, layout) {
  if (!node) return stateBlock('empty', 'Click a node in the graph to see its resolved properties.');
  if (node.end) return h('div', { class: 'stack' }, h('h3', null, 'END'), h('p', { class: 'muted' }, 'The end node. Exactly one EDIT edge enters it; that edit is the graph output.'));
  const resolved = node.properties_resolved || {};
  const indices = (graphNode && graphNode.properties) || {};
  const names = [...new Set([...Object.keys(resolved), ...Object.keys(indices)])];
  const inc = (layout.edges || []).filter(e => String(e.dst) === String(node.id));
  const out = (layout.edges || []).filter(e => String(e.src) === String(node.id));
  const edgeChip = (e, other) => h('span', { class: 'edge-chip', style: { '--c': kindColor(e.kind) } }, `${other} · ${String(e.kind).toUpperCase()}`);
  return h('div', { class: 'stack' },
    h('div', { class: 'row' }, h('h3', { class: 'mono' }, `#${node.id}`), typeChip(node.type), node.level !== undefined ? h('span', { class: 'muted' }, `level ${node.level}`) : null),
    h('div', { class: 'row' }, h('span', { class: 'field-label' }, 'in'), inc.length ? inc.map(e => edgeChip(e, e.src)) : h('span', { class: 'muted' }, 'task prompt only')),
    h('div', { class: 'row' }, h('span', { class: 'field-label' }, 'out'), out.map(e => edgeChip(e, e.dst))),
    names.length ? h('table', { class: 'data-table props-table' }, h('tbody', null, names.map(n => h('tr', null,
      h('th', { scope: 'row', class: 'mono' }, n, indices[n] !== undefined ? h('span', { class: 'muted' }, ` [${indices[n]}]`) : null),
      h('td', null, typeof resolved[n] === 'string' && resolved[n].length > 80 ? truncatedText(resolved[n], 160) : h('span', { class: 'mono' }, resolved[n] === undefined ? '—' : typeof resolved[n] === 'object' ? JSON.stringify(resolved[n]) : String(resolved[n]))))))) : stateBlock('empty', 'No properties.'));
}

function lineageList(runId, genome, lineage) {
  const self = genome.id || genome.genome_id;
  const items = [genome, ...(lineage || []).filter(r => (r.id || r.genome_id) !== self)];
  return h('ol', { class: 'lineage' }, items.map((r, i) => {
    const id = r.id || r.genome_id;
    return h('li', { class: `lineage-item ${i === 0 ? 'self' : ''}` },
      h('div', { class: 'lineage-main' },
        i === 0 ? h('span', { class: 'mono' }, shortId(id), ' (this genome)') : genomeLink(runId, id),
        h('span', { class: 'muted' }, `gen ${r.generation ?? '?'}`), speciesChip(r.species_id),
        h('span', { class: 'num' }, `fit ${fmtNum(r.fitness, 3)}`), h('span', { class: 'muted' }, `${r.num_workers ?? '?'} workers`)),
      h('div', { class: 'lineage-ops' },
        (r.operators || []).length ? ['produced by ', (r.operators || []).map(opChip)] : h('span', { class: 'muted' }, 'initial genome'),
        (r.parents || []).length ? [h('span', { class: 'muted' }, ' from '), (r.parents || []).map(p => genomeLink(runId, p))] : null));
  }));
}

function evaluationsTable(runId, gid, evals) {
  return sortableTable([
    { key: 'task_index', label: 'Task', num: true, render: e => h('a', { href: R.evaluation(runId, e.genome_id || gid, e.task_index) }, `#${e.task_index}`) },
    { key: 'score', label: 'Score', num: true, render: e => fitnessCell(e.score) },
    { key: 'failure', label: 'Failure', sort: e => e.failure_type || (e.failure ? 'x' : ''), render: e => e.failure ? h('span', { class: 'row' }, badge(e.failure_type || 'failed', 'danger'), h('span', { class: 'clip', title: e.failure }, e.failure)) : e.mock ? badge('mock') : badge('ok', 'ok') },
    { key: 'failure_node_id', label: 'Failed node', num: true, render: e => e.failure_node_id ?? '' },
    { key: 'duration_s', label: 'Duration', num: true, render: e => fmtSeconds(e.duration_s) },
    { key: 'diff_chars', label: 'Diff chars', num: true },
    { key: 'touched_paths', label: 'Touched', sort: e => (e.touched_paths || []).length, render: e => h('span', { class: 'mono clip', title: (e.touched_paths || []).join('\n') }, (e.touched_paths || []).join(', ')) },
    { key: 'open', label: '', sort: false, render: e => h('a', { href: R.evaluation(runId, e.genome_id || gid, e.task_index) }, 'open →') },
  ], evals, { sortKey: 'task_index', sortDir: 'asc', onRow: (e) => { location.hash = R.evaluation(runId, e.genome_id || gid, e.task_index); }, empty: 'No evaluations recorded for this genome.' });
}

async function viewGenome(ctx, runId, gid) {
  const g = await api.genome(runId, gid);
  const layout = g.layout || { nodes: [], edges: [] };
  const graphNodes = new Map(((g.graph && g.graph.nodes) || []).map(n => [String(n.id), n]));
  const panel = h('div', { class: 'props-panel' });
  const selectNode = (n) => replace(panel, propertiesPanel(n, n ? graphNodes.get(String(n.id)) : null, layout));
  const dag = renderGraph(layout, { onSelect: selectNode });
  const firstWorker = [...dag.layout.nodes.values()].find(n => !n.end);
  selectNode(null);
  if (firstWorker) { dag.select(firstWorker.id); selectNode(firstWorker); }
  const options = [...new Set([...(g.parents || []), ...(g.children || []), ...((g.lineage || []).map(r => r.id || r.genome_id))])].filter(x => x && x !== gid);
  const cmpInput = h('input', { class: 'input mono', placeholder: 'other genome id', list: 'cmp-options', required: true });
  const compareForm = h('form', { class: 'row compare-form', onSubmit: (e) => { e.preventDefault(); const b = cmpInput.value.trim(); if (b) location.hash = R.compare(runId, gid, b); } },
    h('label', { class: 'field-label' }, 'Compare with…'), cmpInput,
    h('datalist', { id: 'cmp-options' }, options.map(o => h('option', { value: o }))),
    h('button', { class: 'btn btn-sm', type: 'submit' }, 'Compare'));
  const evals = g.evaluations || [];
  return h('div', { class: 'page' },
    runHeader(runId, [['Generation ' + (g.generation ?? '?'), g.generation !== undefined && g.generation !== null ? R.generation(runId, g.generation) : null], [`Genome ${shortId(gid)}`]], null),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Genome ', h('span', { class: 'mono' }, gid)),
      h('span', { class: 'row' }, speciesChip(g.species_id),
        h('a', { class: 'btn btn-sm', href: paths.dot(runId, gid), download: `${gid}.dot` }, 'Download DOT'))),
    statTiles([
      ['Fitness', fmtNum(g.fitness, 3), `adjusted ${fmtNum(g.adjusted_fitness, 3)}`],
      ['Workers', String(g.num_workers ?? dag.layout.nodes.size - 1), workerTypeChips(g.worker_types)],
      ['Generation', g.generation ?? '—', (g.operators || []).length ? h('span', { class: 'chip-list' }, g.operators.map(opChip)) : 'initial'],
      ['Task scores', (g.task_scores || []).length ? g.task_scores.map(v => fmtNum(v, 2)).join(' · ') : '—', `${evals.length} evaluation${evals.length === 1 ? '' : 's'}`],
      ['Eval time', fmtSeconds(g.eval_duration_s), g.evaluated_at ? fmtTime(g.evaluated_at) : null],
      ['Structure', h('span', { class: 'mono', title: g.structure_hash || '' }, shortId(g.structure_hash || '—', 10)), g.created_at ? `created ${fmtTime(g.created_at)}` : null],
    ]),
    h('div', { class: 'genome-grid' },
      card('Worker graph', dag.el, { cls: 'grow' }),
      card('Node properties', panel)),
    card(null, compareForm),
    h('div', { class: 'two-col' },
      card('Lineage', lineageList(runId, g, g.lineage)),
      card(`Children (${(g.children || []).length})`, (g.children || []).length ? h('ul', { class: 'link-list' }, g.children.map(c => h('li', null, genomeLink(runId, c, c)))) : stateBlock('empty', 'No recorded children.'))),
    card('Evaluations', evaluationsTable(runId, gid, evals)),
    g.genome ? card(null, h('details', { class: 'collapsible' }, h('summary', null, 'Raw genome genes (JSON)'), h('pre', { class: 'json-pre' }, prettyJson(g.genome)))) : null);
}

// ---------- #/run/{id}/eval/{gid}/{task} ----------
const LEVEL_NAMES = { 1: 'COLLAPSED', 2: 'FULL' };
function viewSummary(view) {
  const files = (view && view.files) || {};
  const names = Object.keys(files);
  if (!names.length) return stateBlock('empty', 'Empty view.');
  return h('table', { class: 'data-table' },
    h('thead', null, h('tr', null, h('th', null, 'File'), h('th', null, 'Elements (level)'), h('th', null, 'Line ranges'))),
    h('tbody', null, names.map(p => {
      const fv = files[p] || {};
      const lv = Object.entries(fv.levels || {});
      return h('tr', null, h('td', { class: 'mono' }, p),
        h('td', null, h('span', { class: 'chip-list' }, lv.map(([el, l]) => h('span', { class: `level-chip level-${LEVEL_NAMES[l] || l}` }, el === '' ? '(file)' : el, ' · ', LEVEL_NAMES[l] || String(l))))),
        h('td', { class: 'mono' }, (fv.line_ranges || []).map(([a, b]) => a === b ? `${a}` : `${a}–${b}`).join(', ') || '—'));
    })));
}

function workerPanel(runId, gid, task, w, inputs, isFailedNode, idx) {
  const viewBox = h('div', { class: 'view-box' }, stateBlock('loading', 'Loading view…'));
  let loaded = false;
  const loadView = async () => {
    if (loaded) return; loaded = true;
    try {
      const res = await api.workerView(runId, gid, task, w.node_id);
      if (!res.view && (res.rendered === null || res.rendered === undefined)) replace(viewBox, stateBlock('empty', 'This worker forwarded no view.'));
      else if (res.rendered !== null && res.rendered !== undefined) replace(viewBox, h('div', { class: 'muted small' }, 'Rendered through render_view against the task project.'), truncatedText(res.rendered, 8000, 'pre code-block'));
      else replace(viewBox, h('div', { class: 'muted small' }, 'Project directory unavailable — showing the view as JSON.'), viewSummary(res.view),
        h('details', { class: 'collapsible' }, h('summary', null, 'Raw view JSON'), h('pre', { class: 'json-pre' }, prettyJson(res.view))));
    } catch (err) { loaded = false; replace(viewBox, stateBlock('error', `Could not load view: ${err.message}`, { label: 'Retry', onClick: loadView })); }
  };
  const inp = inputs || {};
  const rolloutList = (w.rollout_ids || []).length ? h('ul', { class: 'rollout-list' }, w.rollout_ids.map(rid => h('li', null,
    h('button', { class: 'link-btn mono', type: 'button', onClick: () => openRollout(runId, gid, task, rid, w) }, rid)))) : h('span', { class: 'muted' }, 'No rollouts (deterministic or skipped worker).');
  const vs = w.view_summary || {};
  const st = w.status || 'ok';
  const details = h('details', { class: `worker-panel status-${st} ${isFailedNode ? 'failed-node' : ''}`, open: isFailedNode || null },
    h('summary', { class: 'worker-head' },
      h('span', { class: 'worker-order' }, idx !== null ? `${idx + 1}.` : '–'),
      h('span', { class: 'mono strong' }, `#${w.node_id}`), typeChip(w.type),
      badge(st, st === 'ok' ? 'ok' : st === 'failed' ? 'danger' : ''),
      h('span', { class: 'muted' }, fmtSeconds(w.duration_s)),
      h('span', { class: 'spacer' }),
      h('span', { class: 'worker-counts' }, `${w.llm_calls ?? 0} LLM · ${w.tool_calls ?? 0} tool`)),
    h('div', { class: 'worker-body' },
      w.error ? h('div', { class: 'banner banner-danger' }, h('strong', null, 'Error: '), truncatedText(w.error, 1200)) : null,
      h('div', { class: 'two-col' },
        kv([
          ['Started', fmtTime(w.started_at)], ['Finished', fmtTime(w.finished_at)], ['Duration', fmtSeconds(w.duration_s)],
          ['LLM calls', String(w.llm_calls ?? 0)], ['Tool calls', String(w.tool_calls ?? 0)],
          ['Views from', (inp.views_from || []).join(', ') || null, true], ['Edits from', (inp.edits_from || []).join(', ') || null, true], ['Reports from', (inp.reports_from || []).join(', ') || null, true],
          ['View', `${vs.files ?? 0} files · ${vs.entries ?? 0} entries · ${(vs.rendered_chars ?? 0).toLocaleString()} chars`],
          ['Edit', (w.edit_touched_paths || []).length ? `${w.edit_touched_paths.join(', ')} (${(w.edit_chars ?? 0).toLocaleString()} chars)` : null, true],
          ['Rollout dir', w.rollout_dir || null, true],
        ]),
        h('div', { class: 'stack' }, h('h4', null, 'Resolved properties'),
          Object.keys(w.properties || {}).length ? h('table', { class: 'data-table props-table' }, h('tbody', null, Object.entries(w.properties).map(([k, v]) =>
            h('tr', null, h('th', { scope: 'row', class: 'mono' }, k), h('td', null, typeof v === 'string' && v.length > 80 ? truncatedText(v, 160) : h('span', { class: 'mono' }, typeof v === 'object' ? JSON.stringify(v) : String(v))))))) : h('span', { class: 'muted' }, '—'))),
      h('h4', null, 'Report'), w.report ? truncatedText(w.report, 3000) : h('span', { class: 'muted' }, ['vieweditor', 'critic', 'pickmerger', 'judgemerger'].includes(w.type) ? 'No report (this worker type emits none).' : 'No report recorded.'),
      h('h4', null, 'Rollouts'), rolloutList,
      h('details', { class: 'collapsible view-details', onToggle: (e) => { if (e.target.open) loadView(); } }, h('summary', null, 'Forwarded view'), viewBox)));
  return details;
}

async function openRollout(runId, gid, task, rid, w) {
  const body = h('div', null, stateBlock('loading', 'Loading rollout…'));
  openModal(`Rollout ${rid} · node #${w.node_id} (${w.type})`, body);
  try { replace(body, renderRollout(await api.rollout(runId, gid, task, rid))); }
  catch (err) { replace(body, stateBlock('error', `Could not load rollout: ${err.message}`)); }
}

async function viewEvaluation(ctx, runId, gid, task) {
  const rec = await api.evaluation(runId, gid, task);
  const workers = (rec.workers || []).slice();
  const order = (rec.order || []).map(String);
  const pos = (w) => { const i = order.indexOf(String(w.node_id)); return i >= 0 ? i : order.length + (w.status === 'skipped' ? 1000 : 0) + (w.started_at ? Date.parse(w.started_at) / 1e13 : 0); };
  workers.sort((a, b) => pos(a) - pos(b));
  const failedWorker = rec.failure_node_id !== null && rec.failure_node_id !== undefined ? workers.find(w => String(w.node_id) === String(rec.failure_node_id)) : null;
  const t = rec.task || {};
  return h('div', { class: 'page' },
    runHeader(runId, [[`Genome ${shortId(gid)}`, R.genome(runId, gid)], [`Task ${task}`]], null),
    h('div', { class: 'page-title-row' }, h('h1', null, `Evaluation · task #${task}`), h('span', { class: 'row' }, 'genome ', genomeLink(runId, gid, gid))),
    rec.failure ? h('div', { class: 'banner banner-danger', role: 'alert' },
      h('div', { class: 'banner-title' }, badge(rec.failure_type || 'failure', 'danger'),
        rec.failure_node_id !== null && rec.failure_node_id !== undefined ? h('span', null, ` at node #${rec.failure_node_id}`, failedWorker ? [' ', typeChip(failedWorker.type)] : null) : null),
      truncatedText(rec.failure, 1500)) : null,
    statTiles([
      ['Score', fmtNum(rec.score, 3), rec.failure ? 'failed → scored 0' : rec.mock ? 'mock evaluator' : null],
      ['Duration', fmtSeconds(rec.duration_s), fmtTime(rec.started_at)],
      ['Generation', rec.generation ?? '—', null],
      ['Workers', String(workers.length), `${workers.filter(w => w.status === 'ok').length} ok · ${workers.filter(w => w.status === 'failed').length} failed · ${workers.filter(w => w.status === 'skipped').length} skipped`],
      ['Diff', `${(rec.diff_chars ?? 0).toLocaleString()} chars`, `${(rec.touched_paths || []).length} files`],
    ]),
    rec.mock ? h('div', { class: 'banner' }, 'This record was written by the MockEvaluator: no task, diff or workers.') : null,
    t.prompt || t.project_dir ? card('Task', h('div', { class: 'stack' }, kv([['Project dir', t.project_dir ?? null, true], ['Scorer', t.scorer ?? null, true]]), truncatedText(t.prompt, 2000))) : null,
    rec.mock ? null : card('Git diff', renderDiff(rec.diff, { emptyText: rec.failure ? 'No diff — the graph failed before producing an edit.' : 'The output edit changed nothing.' })),
    workers.length ? card(`Workers in execution order (${workers.length})`, h('div', { class: 'stack' },
      workers.map(w => workerPanel(runId, gid, task, w, (rec.inputs || {})[String(w.node_id)], failedWorker === w, order.indexOf(String(w.node_id)) >= 0 ? order.indexOf(String(w.node_id)) : null)))) : null);
}

// ---------- #/run/{id}/operators ----------
async function viewOperators(ctx, runId) {
  const ops = await api.operators(runId);
  const sorted = ops.slice().sort((a, b) => (b.mean_delta ?? -Infinity) - (a.mean_delta ?? -Infinity));
  return h('div', { class: 'page' },
    runHeader(runId, [['Operators']], 'operators'),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Operators'), h('span', { class: 'muted' }, 'delta = child adjusted − best parent adjusted')),
    h('div', { class: 'chart-grid' },
      card('Mean fitness delta per operator', barChart({ items: sorted.map(o => ({ label: o.operator, value: o.mean_delta, color: seriesColor(0), detail: `${o.count} children · success ${fmtNum((o.success_rate ?? 0) * 100, 0)}%` })), yLabel: 'mean delta', valueFormat: (v) => fmtNum(v, 3) })),
      card('Children per operator', barChart({ integerY: true, items: sorted.map(o => ({ label: o.operator, value: o.count, color: seriesColor(2) })), yLabel: 'children', valueFormat: (v) => String(v) }))),
    card('Operator statistics', sortableTable([
      { key: 'operator', label: 'Operator', render: o => opChip(o.operator) },
      { key: 'count', label: 'Count', num: true },
      { key: 'mean_parent_fitness', label: 'Mean parent', num: true, render: o => fitnessCell(o.mean_parent_fitness) },
      { key: 'mean_child_fitness', label: 'Mean child', num: true, render: o => fitnessCell(o.mean_child_fitness) },
      { key: 'mean_delta', label: 'Mean delta', num: true, render: o => h('span', { class: `num ${o.mean_delta > 0 ? 'pos' : o.mean_delta < 0 ? 'neg' : ''}` }, (o.mean_delta > 0 ? '+' : '') + fmtNum(o.mean_delta, 3)) },
      { key: 'success_rate', label: 'Success rate', num: true, render: o => o.success_rate === null || o.success_rate === undefined ? '—' : h('span', { class: 'rate' }, h('span', { class: 'rate-track' }, h('span', { class: 'rate-bar', style: { width: `${Math.round(o.success_rate * 100)}%` } })), `${fmtNum(o.success_rate * 100, 0)}%`) },
    ], ops, { sortKey: 'mean_delta', sortDir: 'desc', empty: 'No operator statistics yet (needs at least one generation of offspring).' })));
}

// ---------- #/run/{id}/failures ----------
async function viewFailures(ctx, runId) {
  const rows = await api.failures(runId);
  const groups = new Map();
  for (const r of rows) { const k = r.failure_type || 'unknown'; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r); }
  const totals = [...groups.entries()].map(([k, list]) => ({ label: k, value: list.reduce((a, r) => a + (r.count || 0), 0) })).sort((a, b) => b.value - a.value);
  const total = totals.reduce((a, t) => a + t.value, 0);
  return h('div', { class: 'page' },
    runHeader(runId, [['Failures']], 'failures'),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Failures'), h('span', { class: 'muted' }, `${total} failed evaluation${total === 1 ? '' : 's'}`)),
    rows.length ? card('Failures by type', barChart({ width: 1280, height: 220, integerY: true, items: totals.map(t => ({ ...t, color: seriesColor(7) })), yLabel: 'failures', valueFormat: (v) => String(v) })) : null,
    rows.length ? totals.map(({ label }) => card(`${label}`, sortableTable([
      { key: 'node_type', label: 'Node type', render: r => r.node_type ? typeChip(r.node_type) : h('span', { class: 'muted' }, '(graph-level)') },
      { key: 'count', label: 'Count', num: true },
      { key: 'examples', label: 'Examples', sort: false, render: r => h('ul', { class: 'examples' }, (r.examples || []).map(ex => h('li', null,
        h('a', { class: 'mono', href: R.evaluation(runId, ex.genome_id, ex.task_index) }, `${shortId(ex.genome_id)} · task ${ex.task_index}`),
        h('span', { class: 'clip muted', title: ex.message || '' }, ex.message || '')))) },
    ], groups.get(label), { sortKey: 'count', sortDir: 'desc' }), { actions: badge(String(groups.get(label).reduce((a, r) => a + (r.count || 0), 0)), 'danger') }))
      : card(null, stateBlock('empty', 'No failed evaluations in this run.')));
}

// ---------- #/run/{id}/compare?a=&b= ----------
/** Normalise the compare endpoint's property diffs into [{node, type, name, a, b}]. */
function normaliseDiffs(res) {
  const raw = res.property_diffs ?? res.diffs ?? res.node_diffs ?? [];
  const out = [];
  const push = (node, type, name, a, b) => out.push({ node, type, name, a, b });
  const entries = Array.isArray(raw) ? raw : Object.entries(raw).map(([node, v]) => ({ node_id: node, ...(Array.isArray(v) ? { diffs: v } : v) }));
  for (const d of entries) {
    const node = d.node ?? d.node_id ?? d.id;
    const type = d.type ?? d.node_type ?? null;
    if (d.name !== undefined) { push(node, type, d.name, d.a, d.b); continue; }
    const props = d.diffs ?? d.properties ?? d.changes ?? {};
    if (Array.isArray(props)) for (const p of props) push(node, type, p.name ?? p.property, p.a, p.b);
    else for (const [name, v] of Object.entries(props)) {
      if (v && typeof v === 'object' && !Array.isArray(v)) out.push({ node, type, name, a: 'a_resolved' in v ? v.a_resolved : v.a, b: 'b_resolved' in v ? v.b_resolved : v.b, ia: v.a, ib: v.b });
      else push(node, type, name, Array.isArray(v) ? v[0] : v, Array.isArray(v) ? v[1] : undefined);
    }
  }
  return out;
}

async function viewCompare(ctx, runId) {
  const a = ctx.query.get('a') || '', b = ctx.query.get('b') || '';
  const ia = h('input', { class: 'input mono', value: a, placeholder: 'genome A id', required: true });
  const ib = h('input', { class: 'input mono', value: b, placeholder: 'genome B id', required: true });
  const form = h('form', { class: 'row compare-form', onSubmit: (e) => { e.preventDefault(); location.hash = R.compare(runId, ia.value.trim(), ib.value.trim()); } },
    h('span', { class: 'compare-letter' }, 'A'), ia, h('span', { class: 'compare-letter' }, 'B'), ib,
    h('button', { class: 'btn btn-primary', type: 'submit' }, 'Compare'),
    a && b ? h('button', { class: 'btn', type: 'button', onClick: () => { location.hash = R.compare(runId, b, a); } }, 'Swap') : null);
  const head = [runHeader(runId, [['Compare']], 'compare'), h('div', { class: 'page-title-row' }, h('h1', null, 'Compare genomes')), card(null, form)];
  if (!a || !b) return h('div', { class: 'page' }, head, card(null, stateBlock('empty', 'Enter two genome ids (or use "Compare with…" on a genome page).')));
  const res = await api.compare(runId, a, b);
  const ga = res.a || {}, gb = res.b || {};
  const diffs = normaliseDiffs(res);
  const diffNodes = new Set(diffs.map(d => String(d.node)));
  const layoutIds = (g) => new Set(((g.layout || {}).nodes || []).map(n => String(n.id)).filter(x => x !== 'END'));
  const idsA = layoutIds(ga), idsB = layoutIds(gb);
  const onlyA = (res.only_a_nodes || [...idsA].filter(x => !idsB.has(x))).map(String);
  const onlyB = (res.only_b_nodes || [...idsB].filter(x => !idsA.has(x))).map(String);
  const matched = (res.matched_nodes || [...idsA].filter(x => idsB.has(x))).map(String);
  const cls = (only) => (n) => only.includes(String(n.id)) ? 'only-node' : diffNodes.has(String(n.id)) ? 'diff-node' : '';
  const side = (letter, g, only, fallbackId) => {
    const rec = g.genome || g;
    const id = rec.id || rec.genome_id || fallbackId;
    const dag = renderGraph(g.layout || { nodes: [], edges: [] }, { nodeClass: cls(only) });
    return card(null, h('div', { class: 'stack' },
      h('div', { class: 'row' }, h('span', { class: 'compare-letter' }, letter), genomeLink(runId, id, id), speciesChip(rec.species_id),
        h('span', { class: 'muted' }, `gen ${rec.generation ?? '?'} · fitness ${fmtNum(rec.fitness, 3)} · adjusted ${fmtNum(rec.adjusted_fitness, 3)} · ${rec.num_workers ?? '?'} workers`)),
      dag.el));
  };
  // connection lists are connection-gene innovation numbers (see the Innovations page)
  const connList = (list) => (list || []).length ? h('div', { class: 'chip-list' }, list.map(c => h('span', { class: 'badge badge-mono', title: 'connection innovation' }, typeof c === 'object' ? JSON.stringify(c) : `#${c}`))) : h('span', { class: 'muted' }, 'none');
  return h('div', { class: 'page' }, head,
    statTiles([
      ['Distance', fmtNum(res.distance, 3), 'speciation.distance with the run config'],
      ['Excess / disjoint', `${res.excess ?? '—'} / ${res.disjoint ?? '—'}`, res.property_difference !== undefined ? `property difference ${fmtNum(res.property_difference, 3)}` : null],
      ['Matched nodes', String(matched.length), `${diffs.length ? new Set(diffs.map(d => d.node)).size : 0} with property differences`],
      ['Only in A', String(onlyA.length), onlyA.map(x => `#${x}`).join(', ') || null],
      ['Only in B', String(onlyB.length), onlyB.map(x => `#${x}`).join(', ') || null],
    ]),
    h('div', { class: 'legend-row' }, h('span', { class: 'legend-item' }, h('span', { class: 'legend-node diff' }), 'matched node with property differences'),
      h('span', { class: 'legend-item' }, h('span', { class: 'legend-node only' }), 'node only in this genome')),
    h('div', { class: 'compare-grid' }, side('A', ga, onlyA, a), side('B', gb, onlyB, b)),
    card('Property differences on matched nodes', sortableTable([
      { key: 'node', label: 'Node', num: true, sort: d => Number(d.node), render: d => h('span', { class: 'mono' }, `#${d.node}`) },
      { key: 'type', label: 'Type', render: d => d.type ? typeChip(d.type) : '—' },
      { key: 'name', label: 'Property', render: d => h('span', { class: 'mono' }, d.name) },
      { key: 'a', label: 'A', sort: false, render: d => valueCell(d.a, d.ia) },
      { key: 'b', label: 'B', sort: false, render: d => valueCell(d.b, d.ib) },
    ], diffs, { sortKey: 'node', sortDir: 'asc', empty: 'Matched nodes have identical properties.' })),
    res.matched_connections || res.only_a_connections || res.only_b_connections ? h('div', { class: 'three-col' },
      card(`Connections only in A (${(res.only_a_connections || []).length})`, connList(res.only_a_connections)),
      card(`Connections only in B (${(res.only_b_connections || []).length})`, connList(res.only_b_connections)),
      card(`Matched connections (${(res.matched_connections || []).length})`, connList(res.matched_connections))) : null);
}
function valueCell(v, index) {
  const idx = index !== undefined && index !== null && index !== v ? h('span', { class: 'muted mono' }, `[${index}] `) : null;
  if (v === undefined || v === null) return h('span', null, idx, h('span', { class: 'muted' }, '—'));
  if (typeof v === 'string' && v.length > 80) return h('div', null, idx, truncatedText(v, 160));
  return h('span', null, idx, h('span', { class: 'mono' }, typeof v === 'object' ? JSON.stringify(v) : String(v)));
}

// ---------- #/run/{id}/innovations ----------
async function viewInnovations(ctx, runId) {
  const inn = await api.innovations(runId);
  const part = (e) => Array.isArray(e) && e.length === 2 && typeof e[0] === 'string' ? String(e[1]) : e !== null && typeof e === 'object' ? JSON.stringify(e) : String(e);
  const keyText = (k) => Array.isArray(k) ? k.map(part).join(' · ') : part(k);
  return h('div', { class: 'page' },
    runHeader(runId, [['Innovations']], 'innovations'),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Innovation database'), h('span', { class: 'muted' }, `${(inn.nodes || []).length} node and ${(inn.connections || []).length} connection innovations`)),
    h('div', { class: 'two-col' },
      card('Node innovations', sortableTable([
        { key: 'id', label: 'Id', num: true },
        { key: 'type', label: 'Type', render: n => typeChip(String(n.type || '').toLowerCase()) },
        { key: 'key', label: 'Key', sort: n => keyText(n.key), render: n => h('span', { class: 'mono', title: JSON.stringify(n.key) }, keyText(n.key)) },
      ], inn.nodes || [], { sortKey: 'id', sortDir: 'asc', empty: 'No node innovations.' })),
      card('Connection innovations', sortableTable([
        { key: 'id', label: 'Id', num: true },
        { key: 'src', label: 'Source', num: true, render: c => h('span', { class: 'mono' }, String(c.src)) },
        { key: 'dst', label: 'Destination', render: c => h('span', { class: 'mono' }, String(c.dst)) },
        { key: 'kind', label: 'Kind', render: c => h('span', { class: 'edge-chip', style: { '--c': kindColor(c.kind) } }, String(c.kind).toUpperCase()) },
      ], inn.connections || [], { sortKey: 'id', sortDir: 'asc', empty: 'No connection innovations.' }))));
}

// ---------- health chip + boot ----------
async function renderHealth() {
  const el = document.getElementById('health-chip');
  try {
    const hres = await api.health({ silent: true });
    replace(el, h('span', { class: 'chip chip-ok chip-mono', title: `runs root: ${hres.runs_root}` }, hres.runs_root || 'connected'));
  } catch (err) {
    replace(el, h('span', { class: 'chip chip-bad', title: err.message }, 'backend unreachable'));
    toast(`Cannot reach the ACE WebUI backend: ${err.message}. Start it with "python scripts/run_webui.py".`, 'error', 8000);
  }
}

if (!location.hash) history.replaceState(null, '', '#/runs');
renderHealth();
route();

// exposed for debugging in the console
window.ace = { R, route, typeColor };
