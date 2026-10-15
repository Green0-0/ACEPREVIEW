// SVG DAG renderer for worker graphs. Input is the `layout` JSON of the genome endpoint:
//   {nodes: [{id, type, level, x, y, properties_resolved}], edges: [{src, dst, kind}]}
// Columns are topological levels (left → right); rows follow the backend's x order within
// a level (the backend normalises x to 0..1 within a level and y = level / END level).
// Nodes are coloured by worker type; edges by payload kind (VIEW / EDIT / REPORT) with
// arrowheads; the END node is drawn as a distinct dark pill. Parallel edges between the
// same pair (different kinds) are fanned out so each kind stays visible.
import { h, s } from './ui.js';

export const WORKER_TYPES = ['viewer', 'listviewer', 'editor', 'vieweditor', 'critic', 'judgemerger', 'pickmerger'];
export const KINDS = ['view', 'edit', 'report'];
export const END = 'END';

const NODE_W = 150, NODE_H = 50, END_W = 64, END_H = 34;
const COL_GAP = 90, ROW_GAP = 26, PAD = 24;
let uid = 0;

export const isEnd = (id) => String(id) === END;
export function typeColor(type) { return WORKER_TYPES.includes(type) ? `var(--wt-${type})` : 'var(--fg-faint)'; }
export function kindColor(kind) { return KINDS.includes(String(kind).toLowerCase()) ? `var(--kind-${String(kind).toLowerCase()})` : 'var(--fg-faint)'; }

/** Small coloured chip for a worker type. */
export function typeChip(type, count = null) {
  return h('span', { class: 'type-chip', style: { '--c': typeColor(type) }, title: type }, type, count !== null ? h('span', { class: 'type-chip-n' }, `×${count}`) : null);
}

/** Normalise layout JSON into positioned nodes and edges. Exported for tests/inspection. */
export function computeLayout(layout) {
  const rawNodes = (layout && layout.nodes) || [];
  const edges = ((layout && layout.edges) || []).map(e => ({ src: normId(e.src), dst: normId(e.dst), kind: String(e.kind || '').toLowerCase() }));
  const nodes = new Map();
  for (const n of rawNodes) {
    const id = normId(n.id);
    nodes.set(String(id), { ...n, id, type: isEnd(id) ? END : String(n.type || '').toLowerCase(), end: isEnd(id) || String(n.type).toUpperCase() === END });
  }
  for (const e of edges) for (const id of [e.src, e.dst]) if (!nodes.has(String(id))) nodes.set(String(id), { id, type: isEnd(id) ? END : '?', end: isEnd(id) });
  if (!nodes.has(END) && nodes.size) nodes.set(END, { id: END, type: END, end: true });
  // levels: backend `level` when present, else longest path from the sources
  const level = new Map();
  const missing = [...nodes.values()].some(n => !n.end && typeof n.level !== 'number');
  if (missing) {
    const indeg = new Map([...nodes.keys()].map(k => [k, 0]));
    for (const e of edges) indeg.set(String(e.dst), (indeg.get(String(e.dst)) || 0) + 1);
    let frontier = [...nodes.keys()].filter(k => !indeg.get(k));
    for (const k of frontier) level.set(k, 0);
    let guard = 0;
    while (frontier.length && guard++ < 1000) {
      const next = [];
      for (const k of frontier) for (const e of edges) if (String(e.src) === k) {
        const d = String(e.dst);
        level.set(d, Math.max(level.get(d) || 0, (level.get(k) || 0) + 1));
        indeg.set(d, indeg.get(d) - 1);
        if (indeg.get(d) === 0) next.push(d);
      }
      frontier = next;
    }
  } else for (const n of nodes.values()) if (!n.end) level.set(String(n.id), n.level);
  const maxWorker = Math.max(0, ...[...nodes.values()].filter(n => !n.end).map(n => level.get(String(n.id)) ?? 0));
  for (const n of nodes.values()) {
    if (n.end) level.set(String(n.id), Math.max(maxWorker + 1, level.get(String(n.id)) ?? 0));
    else if (!level.has(String(n.id))) level.set(String(n.id), 0);
  }
  const cols = new Map();
  for (const n of nodes.values()) {
    const l = level.get(String(n.id));
    if (!cols.has(l)) cols.set(l, []);
    cols.get(l).push(n);
  }
  const levels = [...cols.keys()].sort((a, b) => a - b);
  const maxRows = Math.max(1, ...[...cols.values()].map(c => c.length));
  const height = PAD * 2 + maxRows * NODE_H + (maxRows - 1) * ROW_GAP;
  levels.forEach((l, ci) => {
    // within a level the backend's x (0..1, barycentric order) gives the row order
    const col = cols.get(l).sort((a, b) => (num(a.x) - num(b.x)) || (num(a.id) - num(b.id)));
    const colH = col.length * NODE_H + (col.length - 1) * ROW_GAP;
    col.forEach((n, ri) => {
      const w = n.end ? END_W : NODE_W, hh = n.end ? END_H : NODE_H;
      n.px = PAD + ci * (NODE_W + COL_GAP) + (NODE_W - w) / 2;
      n.py = (height - colH) / 2 + ri * (NODE_H + ROW_GAP) + (NODE_H - hh) / 2;
      n.w = w; n.h = hh; n.col = ci;
    });
  });
  const width = PAD * 2 + levels.length * NODE_W + Math.max(0, levels.length - 1) * COL_GAP;
  return { nodes, edges, width, height };
}

function normId(id) {
  if (id === null || id === undefined) return id;
  if (isEnd(id)) return END;
  const n = Number(id);
  return Number.isFinite(n) && String(id).trim() !== '' ? n : id;
}
function num(v) { const n = Number(v); return Number.isFinite(n) ? n : 0; }

/**
 * renderGraph(layout, {onSelect(node), selected, nodeClass(node) → extra class, legend})
 * → {el, select(id)}
 */
export function renderGraph(layout, { onSelect = null, selected = null, nodeClass = null, showLegend = true } = {}) {
  const L = computeLayout(layout);
  const id = `g${++uid}`;
  if (!L.nodes.size) return { el: h('div', { class: 'state' }, 'Empty graph.'), select() {} };
  const svg = s('svg', { viewBox: `0 0 ${L.width} ${L.height}`, class: 'dag', role: 'img', 'aria-label': 'worker graph', style: { minWidth: `${Math.round(L.width * 0.8)}px`, maxWidth: `${Math.round(Math.max(L.width, 320) * 1.15)}px` } });
  const defs = s('defs');
  for (const k of KINDS) {
    defs.appendChild(s('marker', { id: `${id}-arrow-${k}`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
      s('path', { d: 'M0,0 L10,5 L0,10 z', style: { fill: kindColor(k) } })));
  }
  svg.appendChild(defs);
  // edges, fanned per (src, dst) pair
  const pairs = new Map();
  for (const e of L.edges) { const k = `${e.src}->${e.dst}`; if (!pairs.has(k)) pairs.set(k, []); pairs.get(k).push(e); }
  const eg = s('g', { class: 'dag-edges' });
  for (const group of pairs.values()) {
    group.sort((a, b) => KINDS.indexOf(a.kind) - KINDS.indexOf(b.kind));
    group.forEach((e, i) => {
      const a = L.nodes.get(String(e.src)), b = L.nodes.get(String(e.dst));
      if (!a || !b) return;
      const off = (i - (group.length - 1) / 2) * 9;
      const x1 = a.px + a.w, y1 = a.py + a.h / 2 + off;
      const x2 = b.px - 2, y2 = b.py + b.h / 2 + off;
      const dx = Math.max(30, (x2 - x1) / 2);
      const backward = x2 <= x1;
      const d = backward
        ? `M${x1},${y1} C${x1 + 60},${y1 - 80} ${x2 - 60},${y2 - 80} ${x2},${y2}`
        : `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
      eg.appendChild(s('path', { d, class: `dag-edge edge-${e.kind}`, style: { stroke: kindColor(e.kind) }, 'marker-end': `url(#${id}-arrow-${KINDS.includes(e.kind) ? e.kind : 'view'})` },
        s('title', { text: `${e.src} → ${e.dst} [${e.kind}]` })));
    });
  }
  svg.appendChild(eg);
  // nodes
  const ng = s('g', { class: 'dag-nodes' });
  const els = new Map();
  for (const n of L.nodes.values()) {
    let g;
    if (n.end) {
      g = s('g', { class: 'dag-node dag-end', transform: `translate(${n.px},${n.py})` },
        s('rect', { width: n.w, height: n.h, rx: n.h / 2 }),
        s('text', { x: n.w / 2, y: n.h / 2 + 4, 'text-anchor': 'middle', text: 'END' }),
        s('title', { text: 'END — the single EDIT edge entering it is the graph output' }));
    } else {
      const extra = nodeClass ? nodeClass(n) || '' : '';
      g = s('g', { class: `dag-node ${extra}`, transform: `translate(${n.px},${n.py})`, tabindex: onSelect ? 0 : null, role: onSelect ? 'button' : null, style: { '--c': typeColor(n.type) } },
        s('rect', { class: 'node-box', width: n.w, height: n.h, rx: 7 }),
        s('rect', { class: 'node-stripe', width: 6, height: n.h, rx: 3 }),
        s('text', { x: 16, y: 21, class: 'node-id', text: `#${n.id}` }),
        s('text', { x: 16, y: 39, class: 'node-type', text: n.type }),
        s('title', { text: `node ${n.id} (${n.type}) — level ${n.level ?? n.col}` }));
      if (onSelect) {
        const pick = () => { select(n.id); onSelect(n); };
        g.addEventListener('click', pick);
        g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); } });
      }
    }
    els.set(String(n.id), g);
    ng.appendChild(g);
  }
  svg.appendChild(ng);
  function select(nid) {
    for (const [k, g] of els) g.classList.toggle('selected', nid !== null && nid !== undefined && k === String(nid));
  }
  if (selected !== null && selected !== undefined) select(selected);
  const types = [...new Set([...L.nodes.values()].filter(n => !n.end).map(n => n.type))];
  const kinds = [...new Set(L.edges.map(e => e.kind))];
  const legendEl = showLegend ? h('div', { class: 'dag-legend' },
    KINDS.filter(k => kinds.includes(k)).map(k => h('span', { class: 'legend-item' }, h('span', { class: 'legend-edge', style: { '--c': kindColor(k) } }), k.toUpperCase())),
    h('span', { class: 'legend-sep' }),
    WORKER_TYPES.filter(t => types.includes(t)).map(t => typeChip(t))) : null;
  return { el: h('div', { class: 'dag-wrap' }, h('div', { class: 'dag-scroll' }, svg), legendEl), select, layout: L };
}
