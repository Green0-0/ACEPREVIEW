// Demo runs: a standalone page per genome (prompt form, settings, previous demos) and the
// live view of one demo (status bar, graph with worker statuses, a node pane with
// Transcript / Diff / Details / View tabs, and a bottom panel with the final result and
// the log). The server runs the graph; this page polls its state while it is active.
import { h, replace, toast, fmtTime, fmtSeconds, shortId, prettyJson, truncatedText, badge, stateBlock } from './ui.js';
import { api } from './api.js';
import { renderGraph, typeChip } from './graph.js';
import { renderDiff } from './diff.js';
import { renderChat } from './chat.js';
import { R, kv, statTiles, runHeader, genomeLink, viewSummary } from './app.js';

const POLL_MS = 1500;
const ACTIVE = new Set(['running']);
const BADGE_KIND = { ok: 'ok', done: 'ok', running: 'running', failed: 'danger', cancelled: 'warn', skipped: '', pending: '' };
const sbadge = (status) => badge(status || 'pending', BADGE_KIND[status] ?? '');
const NO_REPORT = new Set(['vieweditor', 'critic', 'pickmerger', 'judgemerger']);

function lockBanner() {
  const icon = h('span', { class: 'banner-icon', 'aria-hidden': 'true' });
  icon.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="7" width="10" height="7" rx="1.5"/><path d="M5 7V5a3 3 0 0 1 6 0v2"/></svg>';
  return h('div', { class: 'demo-banner', role: 'note' }, icon,
    h('span', null, 'Read-only run — the workers edit private workspaces of the project; nothing on disk is modified. Only their rollouts are written, into a temporary folder. You get to see the diffs.'));
}

function elapsed(startIso, endIso) {
  const a = startIso ? Date.parse(startIso) : NaN;
  if (isNaN(a)) return '—';
  const b = endIso ? Date.parse(endIso) : Date.now();
  return fmtSeconds(Math.max(0, (b - a) / 1000));
}

function tabStrip(items, active, onPick) {
  return h('nav', { class: 'tabs pane-tabs', 'aria-label': 'Panel tabs' }, items.map(([id, label]) =>
    h('a', { class: 'tab', href: '#', 'aria-selected': id === active ? 'true' : 'false', onClick: (e) => { e.preventDefault(); onPick(id); } }, label)));
}

function settingsRows(cfg) {
  return kv([
    ['Model', cfg.model || '—', true],
    ['Thinking', cfg.thinking || 'off'],
    ['Max tokens per call', cfg.max_tokens != null ? String(cfg.max_tokens) : '—'],
    ['API base', cfg.api_base_url || '—', true],
    ['API key', cfg.key_env ? h('span', null, h('span', { class: 'mono' }, cfg.key_env), ' ', cfg.key_set ? badge('set', 'ok') : badge('not set', 'danger')) : 'none needed'],
    ['Project', cfg.project_dir || '—', true],
  ]);
}

function historyTable(list, { empty = 'No demos of this genome yet.' } = {}) {
  if (!list.length) return stateBlock('empty', empty);
  return h('table', { class: 'data-table demo-history' },
    h('thead', null, h('tr', null, h('th', null, 'Status'), h('th', null, 'Started'), h('th', null, 'Duration'), h('th', null, 'LLM calls'), h('th', null, 'Workers'), h('th', null, 'Prompt'), h('th', null, ''))),
    h('tbody', null, list.map(d => h('tr', { onClick: () => { location.hash = R.demo(d.id); } },
      h('td', null, sbadge(d.status)), h('td', null, fmtTime(d.started_at)), h('td', { class: 'num' }, fmtSeconds(d.duration_s)),
      h('td', { class: 'num' }, String(d.llm_calls ?? 0)), h('td', { class: 'num' }, `${d.workers_done ?? 0}/${d.workers_total ?? 0}`),
      h('td', { class: 'prompt-cell', title: d.prompt || '' }, d.prompt || ''),
      h('td', null, h('a', { href: R.demo(d.id) }, 'open →'))))));
}

// ---------- #/run/{run}/genome/{gid}/demo ----------
export async function viewDemoPage(ctx, runId, gid) {
  const [g, cfg, list] = await Promise.all([
    api.genome(runId, gid),
    api.demoInfo({ silent: true }).catch(err => ({ enabled: false, key_set: false, reason: `could not read the demo settings: ${err.message}` })),
    api.demos({ run: runId, genome: gid }, { silent: true }).catch(() => []),
  ]);
  const available = !!(cfg.enabled && cfg.key_set);
  const layout = g.layout || { nodes: [], edges: [] };
  const prefill = (ctx.query && ctx.query.prompt) || '';
  const prompt = h('textarea', { class: 'input demo-prompt', rows: 6, placeholder: 'What should the graph do to the project? For example: "Add a clear_tag(tag) helper to schedule/__init__.py that cancels every job carrying that tag. Keep it short, with a one-line docstring."' });
  prompt.value = prefill;
  const startBtn = h('button', { class: 'btn btn-primary', type: 'submit', disabled: !available }, 'Start demo');
  const note = h('span', { class: 'field-help' }, 'The graph runs server-side; the next page follows it live.');
  const form = h('form', { class: 'stack', onSubmit: async (e) => {
    e.preventDefault();
    const text = prompt.value.trim();
    if (!text) { toast('Type a prompt first.', 'error'); prompt.focus(); return; }
    startBtn.disabled = true; startBtn.textContent = 'Starting…';
    try { const d = await api.demoStart(runId, gid, text); location.hash = R.demo(d.id); }
    catch { startBtn.disabled = !available; startBtn.textContent = 'Start demo'; }
  } },
    h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'Prompt'), prompt,
      h('span', { class: 'field-help' }, 'Given to every worker as the task; the first-level workers see only this, downstream workers also receive the views, edits and reports carried by their incoming edges.')),
    h('div', { class: 'row' }, startBtn, note));
  const unavailable = !available ? h('div', { class: 'banner banner-danger', role: 'alert' }, h('div', { class: 'banner-title' }, 'Demo runs are unavailable'), cfg.reason || 'not configured') : null;
  const busy = cfg.running ? h('div', { class: 'banner' }, 'Another demo is running: ', h('a', { href: R.demo(cfg.running) }, shortId(cfg.running, 12)), '. Only one demo runs at a time.') : null;
  const dag = renderGraph(layout, { showLegend: true });
  return h('div', { class: 'page' },
    runHeader(runId, [[`Genome ${shortId(gid)}`, R.genome(runId, gid)], ['Demo']], null),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Demo · genome ', h('span', { class: 'mono' }, gid)),
      h('span', { class: 'row' }, h('a', { class: 'btn btn-sm', href: R.genome(runId, gid) }, '← Genome page'))),
    lockBanner(), unavailable, busy,
    h('div', { class: 'demo-form' },
      h('div', { class: 'card' }, h('div', { class: 'pane-head' }, h('h3', null, 'New demo')), h('div', { class: 'pane-body' }, form)),
      h('div', { class: 'card' }, h('div', { class: 'pane-head' }, h('h3', null, 'Settings')), h('div', { class: 'pane-body' }, settingsRows(cfg)))),
    h('div', { class: 'card' }, h('div', { class: 'pane-head' }, h('h3', null, 'Graph that will run'), h('span', { class: 'muted small' }, `${(g.num_workers ?? layout.nodes.length - 1)} workers · fitness ${g.fitness != null ? g.fitness.toFixed(3) : '—'} · generation ${g.generation ?? '?'}`),
      h('span', { class: 'spacer' }), h('a', { class: 'link-btn', href: R.genome(runId, gid) }, 'node properties on the genome page →')),
      h('div', { class: 'pane-body' }, dag.el)),
    h('div', { class: 'card' }, h('div', { class: 'pane-head' }, h('h3', null, `Previous demos of this genome (${list.length})`),
      h('span', { class: 'spacer' }), h('button', { class: 'btn btn-sm', type: 'button', onClick: () => { location.hash = R.demoPage(runId, gid); location.reload(); } }, 'Refresh')),
      h('div', { class: 'pane-body' }, historyTable(list))));
}

// ---------- #/demo/{id} ----------
export async function viewDemo(ctx, demoId) {
  let d = await api.demo(demoId);
  const g = await api.genome(d.run_id, d.genome_id).catch(() => null);
  const layout = (g && g.layout) || { nodes: [], edges: [] };
  const state = { d, selected: null, tab: 'transcript', bottomTab: 'final', transcripts: new Map(), views: new Map() };
  const workersById = () => new Map((state.d.workers || []).map(w => [String(w.node_id), w]));
  const isActive = () => ACTIVE.has(state.d.status);

  const statusBar = h('div', { class: 'demo-status-bar' });
  const graphBox = h('div', { class: 'stack' });
  const nodeHead = h('div', { class: 'pane-head' });
  const nodeBody = h('div', { class: 'pane-body' });
  const bottomHead = h('div', { class: 'pane-head' });
  const bottomBody = h('div', { class: 'pane-body' });
  const page = h('div', { class: 'page' },
    runHeader(d.run_id, [[`Genome ${shortId(d.genome_id)}`, R.genome(d.run_id, d.genome_id)], ['Demo', R.demoPage(d.run_id, d.genome_id)], [shortId(demoId)]], null),
    h('div', { class: 'page-title-row' }, h('h1', null, 'Demo ', h('span', { class: 'mono' }, demoId)),
      h('span', { class: 'row' }, 'genome ', genomeLink(d.run_id, d.genome_id, d.genome_id))),
    lockBanner(), statusBar,
    h('div', { class: 'demo-live' },
      h('div', { class: 'card demo-left' }, h('div', { class: 'pane-head' }, h('h3', null, 'Worker graph'), h('span', { class: 'muted small' }, 'click a worker to inspect it')), h('div', { class: 'pane-body' }, graphBox)),
      h('div', { class: 'card demo-right' }, nodeHead, nodeBody)),
    h('div', { class: 'card demo-bottom' }, bottomHead, bottomBody));

  // pick a worker to show: the running one, else the first in execution order, else the first node
  function defaultSelection() {
    const ws = state.d.workers || [];
    const running = ws.find(w => w.status === 'running');
    if (running) return String(running.node_id);
    if ((state.d.order || []).length) return String(state.d.order[0]);
    return ws.length ? String(ws[0].node_id) : null;
  }

  // ---------- status bar ----------
  async function cancel() {
    if (!window.confirm('Cancel this demo? Running workers are stopped.')) return;
    try { await api.demoCancel(demoId); } catch { /* toast */ }
  }
  function renderStatusBar() {
    const s = state.d;
    replace(statusBar,
      h('a', { class: 'btn btn-sm', href: R.demoPage(s.run_id, s.genome_id), title: 'Start another demo of this genome' }, '← New demo'),
      sbadge(s.status),
      h('span', { class: 'mono muted', title: 'demo id' }, shortId(s.id, 12)),
      h('span', { class: 'chip', title: 'LLM calls across all workers' }, `${s.llm_calls ?? 0} LLM calls`),
      h('span', { class: 'chip', title: 'workers finished / total' }, `workers ${s.workers_done ?? 0}/${s.workers_total ?? 0}`),
      h('span', { class: 'chip', title: 'rollout files written so far' }, `${s.rollouts ?? 0} rollouts`),
      h('span', { class: 'elapsed', title: 'elapsed' }, elapsed(s.started_at, s.finished_at)),
      h('span', { class: 'chip chip-mono', title: 'model' }, s.model),
      isActive() ? h('span', { class: 'live-badge' }, h('span', { class: 'live-dot' }), 'LIVE') : null,
      h('span', { class: 'spacer' }),
      h('a', { class: 'btn btn-sm', href: R.demoPage(s.run_id, s.genome_id, { prompt: s.prompt }), title: 'Open the form with this prompt filled in' }, 'Re-run with same prompt'),
      isActive() ? h('button', { class: 'btn btn-sm', type: 'button', onClick: cancel }, 'Cancel') : null,
      s.error && s.status !== 'cancelled' ? h('div', { class: 'banner banner-danger', style: { width: '100%' } }, truncatedText(s.error, 600)) : null);
  }

  // ---------- graph ----------
  let dag = null;
  function renderGraphBox() {
    const ws = workersById();
    dag = renderGraph(layout, {
      selected: state.selected,
      onSelect: (n) => { state.selected = String(n.id); renderNode(); },
      nodeClass: (n) => `st-${(ws.get(String(n.id)) || {}).status || 'pending'}`,
      nodeBadge: (n) => { const w = ws.get(String(n.id)); return w && (w.llm_calls || w.status !== 'pending') ? `${w.llm_calls ?? 0} LLM` : ''; },
    });
    const legend = h('div', { class: 'demo-legend' }, [['pending', 'var(--fg-faint)'], ['running', 'var(--accent)'], ['ok', 'var(--success)'], ['failed', 'var(--danger)'], ['skipped', 'var(--border-strong)']]
      .map(([k, c]) => h('span', { style: { '--c': c } }, k)));
    replace(graphBox, dag.el, legend);
  }

  // ---------- node pane ----------
  async function loadTranscripts(nid, force = false) {
    const w = workersById().get(nid);
    const stale = !state.transcripts.has(nid) || force || (w && w.status === 'running');
    if (!stale) return state.transcripts.get(nid);
    try { const rows = await api.demoWorkerRollouts(demoId, nid, { silent: true }); state.transcripts.set(nid, rows); return rows; }
    catch (err) { return state.transcripts.get(nid) || { error: err.message }; }
  }
  function transcriptPanel(w, rows) {
    if (rows && rows.error) return stateBlock('error', `Could not load the transcript: ${rows.error}`);
    const parts = [];
    if (!rows || !rows.length) {
      parts.push(stateBlock(w.status === 'running' ? 'loading' : 'empty',
        w.status === 'pending' ? 'Waiting for upstream workers.' : w.status === 'running' ? 'Waiting for the first model response…' : w.status === 'skipped' ? 'This worker did not run.' : 'No transcript recorded (deterministic worker or the agent never saved a rollout).'));
    } else {
      rows.forEach((r, i) => {
        const chat = r.raw_chat || [];
        const steps = chat.filter(m => m && m.role === 'assistant').length;
        const tools = chat.reduce((n, m) => n + ((m && m.tool_calls) || []).length, 0);
        parts.push(h('div', { class: 'transcript-head' }, h('span', { class: 'strong' }, rows.length > 1 ? `Rollout ${i + 1} of ${rows.length}` : 'Rollout'), h('span', { class: 'mono muted' }, r.id || ''),
          h('span', { class: 'chip' }, `${steps} LLM step${steps === 1 ? '' : 's'}`), h('span', { class: 'chip' }, `${tools} tool call${tools === 1 ? '' : 's'}`), h('span', { class: 'chip' }, `${chat.length} messages`)));
        parts.push(renderChat(chat, { emptyText: 'No messages yet.' }));
      });
    }
    if (w.status === 'running') parts.push(h('div', { class: 'row muted small' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), 'Model is thinking… the transcript refreshes after each step.'));
    return h('div', { class: 'stack' }, parts);
  }
  function detailsPanel(w) {
    const inp = (state.d.inputs || {})[String(w.node_id)] || {};
    const vs = w.view_summary || null;
    const rows = [
      ['Node', `#${w.node_id}`, true], ['Type', w.type], ['Status', sbadge(w.status)],
      ['Started', fmtTime(w.started_at)], ['Finished', fmtTime(w.finished_at)], ['Duration', w.started_at ? elapsed(w.started_at, w.finished_at) : '—'],
      ['Last activity', fmtTime(w.last_activity)],
      ['LLM calls', String(w.llm_calls ?? 0)], ['Steps in rollouts', String(w.steps ?? 0)], ['Tool calls', String(w.tool_calls ?? 0)],
      ['Rollouts', (w.rollout_ids || []).join(', ') || null, true],
      ['Views from', (inp.views_from || []).map(x => `#${x}`).join(', ') || 'task prompt only', true],
      ['Edits from', (inp.edits_from || []).map(x => `#${x}`).join(', ') || null, true],
      ['Reports from', (inp.reports_from || []).map(x => `#${x}`).join(', ') || null, true],
      ['Edit', (w.edit_touched_paths || []).length ? `${w.edit_touched_paths.join(', ')} (${(w.diff_chars ?? 0).toLocaleString()} diff chars)` : (w.status === 'ok' ? 'none' : null), true],
      ['View', vs ? `${vs.files ?? 0} files · ${vs.entries ?? 0} entries · ${(vs.rendered_chars ?? 0).toLocaleString()} chars` : null],
      ['Rollout dir', w.rollout_dir || null, true],
    ];
    const props = w.properties || {};
    return h('div', { class: 'stack' },
      w.error ? h('div', { class: 'banner banner-danger' }, h('strong', null, 'Error: '), truncatedText(w.error, 1200)) : null,
      kv(rows),
      h('h4', null, 'Report'), w.report ? h('pre', { class: 'final-answer' }, w.report) : h('span', { class: 'muted' }, NO_REPORT.has(w.type) ? 'This worker type emits no report.' : w.status === 'ok' ? 'No report recorded.' : 'Available when the worker finishes.'),
      h('h4', null, 'Resolved properties'),
      Object.keys(props).length ? h('table', { class: 'data-table props-table' }, h('tbody', null, Object.entries(props).map(([k, v]) =>
        h('tr', null, h('th', { scope: 'row', class: 'mono' }, k), h('td', null, typeof v === 'string' && v.length > 80 ? truncatedText(v, 240) : h('span', { class: 'mono' }, typeof v === 'object' ? JSON.stringify(v) : String(v))))))) : h('span', { class: 'muted' }, '—'));
  }
  async function viewPanel(w) {
    if (isActive() || w.status === 'pending' || w.status === 'running') return stateBlock('empty', 'The forwarded view is rendered once the demo has finished.');
    if (!state.views.has(String(w.node_id))) {
      try { state.views.set(String(w.node_id), await api.demoWorkerView(demoId, w.node_id)); }
      catch (err) { return stateBlock('error', `Could not load the view: ${err.message}`); }
    }
    const res = state.views.get(String(w.node_id));
    if (!res.view && (res.rendered === null || res.rendered === undefined)) return stateBlock('empty', 'This worker forwarded no view.');
    if (res.rendered !== null && res.rendered !== undefined) return h('div', { class: 'stack' }, h('div', { class: 'muted small' }, 'Rendered through render_view against the demo project.'), truncatedText(res.rendered, 12000, 'pre code-block'));
    return h('div', { class: 'stack' }, viewSummary(res.view), h('details', { class: 'collapsible' }, h('summary', null, 'Raw view JSON'), h('pre', { class: 'json-pre' }, prettyJson(res.view))));
  }
  async function renderNode() {
    if (state.selected === null || !workersById().has(state.selected)) state.selected = defaultSelection();
    const nid = state.selected;
    const w = nid !== null ? workersById().get(nid) : null;
    if (dag) dag.select(nid);
    if (!w) {
      replace(nodeHead, h('h3', null, 'Worker'));
      replace(nodeBody, stateBlock('empty', 'Click a worker in the graph to inspect its transcript, diff and details.'));
      return;
    }
    const tabs = tabStrip([
      ['transcript', `Transcript (${w.steps ?? 0})`], ['diff', `Diff (${(w.edit_touched_paths || []).length})`], ['details', 'Details'], ['view', 'View'],
    ], state.tab, (t) => { state.tab = t; renderNode(); });
    replace(nodeHead, h('h3', null, `#${w.node_id}`), typeChip(w.type), sbadge(w.status),
      h('span', { class: 'muted small' }, `${w.llm_calls ?? 0} LLM · ${w.tool_calls ?? 0} tool · ${w.started_at ? elapsed(w.started_at, w.finished_at) : 'not started'}`),
      tabs);
    const sameTarget = nodeBody.dataset.node === nid && nodeBody.dataset.tab === state.tab;
    const atBottom = nodeBody.scrollHeight - nodeBody.scrollTop - nodeBody.clientHeight < 40;
    const prevTop = nodeBody.scrollTop;
    let content;
    if (state.tab === 'transcript') content = transcriptPanel(w, await loadTranscripts(nid));
    else if (state.tab === 'diff') content = renderDiff(w.diff, { emptyText: w.status === 'pending' ? 'Not started yet.' : w.status === 'running' ? 'The diff appears when the worker finishes.' : w.status === 'skipped' ? 'This worker did not run.' : 'This worker produced no edit.' });
    else if (state.tab === 'details') content = detailsPanel(w);
    else content = await viewPanel(w);
    if (!ctx.alive() || state.selected !== nid) return;
    replace(nodeBody, content);
    nodeBody.dataset.node = nid; nodeBody.dataset.tab = state.tab;
    if (sameTarget && state.tab === 'transcript') nodeBody.scrollTop = atBottom && w.status === 'running' ? nodeBody.scrollHeight : prevTop;
    else if (state.tab === 'transcript' && w.status === 'running') nodeBody.scrollTop = nodeBody.scrollHeight;
    else nodeBody.scrollTop = sameTarget ? prevTop : 0;
  }

  // ---------- bottom panel ----------
  function finalPanel() {
    const s = state.d;
    const ws = workersById();
    const finalWorker = s.final_node !== null && s.final_node !== undefined ? ws.get(String(s.final_node)) : null;
    const tiles = statTiles([
      ['Status', sbadge(s.status), s.model],
      ['Duration', elapsed(s.started_at, s.finished_at), fmtTime(s.started_at)],
      ['LLM calls', String(s.llm_calls ?? 0), `${s.rollouts ?? 0} rollout${s.rollouts === 1 ? '' : 's'} written`],
      ['Workers', `${s.workers_done ?? 0}/${s.workers_total ?? 0}`, s.worker_counts ? `${s.worker_counts.ok} ok · ${s.worker_counts.failed} failed · ${s.worker_counts.skipped} skipped` : null],
      ['Diff', isActive() ? '…' : `${(s.diff_chars ?? 0).toLocaleString()} chars`, isActive() ? 'updates when the graph finishes' : `${(s.touched_paths || []).length} file${(s.touched_paths || []).length === 1 ? '' : 's'}`],
    ]);
    const parts = [tiles];
    if (s.failure) parts.push(h('div', { class: 'banner banner-danger', role: 'alert' }, h('div', { class: 'banner-title' }, badge(s.failure_type || 'failure', 'danger'),
      s.failure_node_id !== null && s.failure_node_id !== undefined ? h('span', null, ` at worker #${s.failure_node_id}`, ' ', h('button', { class: 'link-btn', type: 'button', onClick: () => { state.selected = String(s.failure_node_id); state.tab = 'details'; renderNode(); } }, 'inspect →')) : null), truncatedText(s.failure, 1500)));
    if (s.status === 'cancelled') parts.push(h('div', { class: 'banner' }, 'The demo was cancelled before the graph finished.'));
    if (isActive()) parts.push(stateBlock('loading', `Running… the edit of worker #${s.final_node ?? '?'} is the graph output and appears here when the graph finishes.`));
    else {
      parts.push(h('div', { class: 'final-node' },
        h('div', { class: 'final-head' }, 'Output edit', finalWorker ? [' from worker ', h('span', { class: 'mono' }, `#${finalWorker.node_id}`), ' ', typeChip(finalWorker.type), ' ', sbadge(finalWorker.status)] : null,
          h('span', { class: 'spacer' }), finalWorker ? h('button', { class: 'link-btn', type: 'button', onClick: () => { state.selected = String(finalWorker.node_id); state.tab = 'transcript'; renderNode(); } }, 'inspect worker →') : null),
        h('div', { class: 'final-body' },
          h('div', null, h('h4', null, 'Diff'), renderDiff(s.diff, { emptyText: s.failure ? 'No diff — the graph failed before producing an edit.' : s.status === 'cancelled' ? 'No diff — the run was cancelled.' : 'The output edit changed nothing.' })),
          h('div', null, h('h4', null, 'Report of the output worker'), finalWorker && finalWorker.report ? h('pre', { class: 'final-answer' }, finalWorker.report) : h('span', { class: 'muted' }, finalWorker && NO_REPORT.has(finalWorker.type) ? 'This worker type emits no report; see its transcript for the reasoning.' : 'No report.')))));
      const others = (s.workers || []).filter(w => w.report && (!finalWorker || w.node_id !== finalWorker.node_id));
      if (others.length) parts.push(h('details', { class: 'collapsible' }, h('summary', null, `Reports of the other workers (${others.length})`),
        h('div', { class: 'stack' }, others.map(w => h('div', null, h('div', { class: 'row' }, h('span', { class: 'mono strong' }, `#${w.node_id}`), typeChip(w.type)), h('pre', { class: 'final-answer' }, w.report))))));
    }
    return h('div', { class: 'stack' }, parts);
  }
  function logPanel() {
    const lines = state.d.log || [];
    if (!lines.length) return stateBlock('empty', 'No log lines yet.');
    const el = h('div', { class: 'log-lines' }, lines.map(l => {
      const t = l.ts ? new Date(l.ts) : null;
      return h('div', { class: `log-line lvl-${l.level || 'info'}` }, h('span', { class: 'ts' }, t && !isNaN(t) ? t.toLocaleTimeString(undefined, { hour12: false }) : ''), h('span', { class: 'lvl' }, l.level || 'info'), h('span', null, l.text || ''));
    }));
    return el;
  }
  function promptPanel() {
    return h('div', { class: 'stack' }, h('pre', { class: 'final-answer' }, state.d.prompt || ''), kv([['Project', state.d.project_dir, true], ['Model', state.d.model, true], ['Rollouts', (state.d.workers || [])[0] ? (state.d.workers[0].rollout_dir || null) : null, true]]));
  }
  function renderBottom() {
    const n = (state.d.log || []).length;
    const tabs = tabStrip([['final', 'Final result'], ['log', `Log (${n})`], ['prompt', 'Prompt']], state.bottomTab, (t) => { state.bottomTab = t; renderBottom(); });
    replace(bottomHead, h('h3', null, state.bottomTab === 'final' ? 'Final result' : state.bottomTab === 'log' ? 'Log' : 'Prompt'), h('span', { class: 'spacer' }), tabs);
    const atBottom = bottomBody.scrollHeight - bottomBody.scrollTop - bottomBody.clientHeight < 40;
    replace(bottomBody, state.bottomTab === 'final' ? finalPanel() : state.bottomTab === 'log' ? logPanel() : promptPanel());
    if (state.bottomTab === 'log' && (atBottom || isActive())) bottomBody.scrollTop = bottomBody.scrollHeight;
  }

  function renderAll() { renderStatusBar(); renderGraphBox(); renderNode(); renderBottom(); }
  renderAll();

  // polling while active; a 1 s tick keeps the elapsed times moving
  let polling = false;
  ctx.interval(async () => {
    if (!isActive() || polling) return;
    polling = true;
    try {
      const nd = await api.demo(demoId, { silent: true });
      if (!ctx.alive()) return;
      state.d = nd;
      renderAll();
    } catch { /* keep the last state; try again on the next tick */ }
    finally { polling = false; }
  }, POLL_MS);
  ctx.interval(() => { if (isActive()) { renderStatusBar(); } }, 1000);
  return page;
}
