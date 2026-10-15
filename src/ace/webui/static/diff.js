// Unified (git) diff parsing and rendering: per-file blocks with headers, hunk headers,
// +/- line colouring, old/new line numbers and light syntax colouring.
import { h, stateBlock } from './ui.js';

/**
 * Split a full `git diff` text into files: [{path, oldPath, newPath, headers: [str], lines}].
 * Works with or without "diff --git" lines (plain "---/+++" pairs start a new file too).
 */
export function splitFiles(text) {
  const files = [];
  let cur = null;
  const start = (header) => { cur = { path: '', oldPath: null, newPath: null, headers: header ? [header] : [], lines: [] }; files.push(cur); };
  for (const line of String(text ?? '').split('\n')) {
    if (line.startsWith('diff --git ')) {
      start(line);
      const m = /^diff --git a\/(.+?) b\/(.+)$/.exec(line);
      if (m) cur.path = m[2];
      continue;
    }
    if (line.startsWith('--- ') && (!cur || cur.lines.length)) start(null);
    if (!cur) start(null);
    if (!cur.lines.length && (line.startsWith('--- ') || line.startsWith('+++ ') || /^(index|new file|deleted file|similarity|rename|old mode|new mode|Binary)/.test(line))) {
      cur.headers.push(line);
      if (line.startsWith('--- ')) cur.oldPath = stripPrefix(line.slice(4));
      if (line.startsWith('+++ ')) { cur.newPath = stripPrefix(line.slice(4)); }
      continue;
    }
    cur.lines.push(line);
  }
  for (const f of files) {
    if (!f.path) f.path = (f.newPath && f.newPath !== '/dev/null') ? f.newPath : (f.oldPath || '(unknown file)');
    while (f.lines.length && f.lines[f.lines.length - 1] === '') f.lines.pop();
  }
  return files.filter(f => f.headers.length || f.lines.length);
}

function stripPrefix(p) {
  p = p.replace(/\t.*$/, '');
  return p.replace(/^[ab]\//, '');
}

/** Parse hunk lines into [{type, text, oldNo, newNo}]. */
export function parseHunks(lines) {
  const out = [];
  // null = unknown (hunk header without line numbers): numbers are then left blank
  let oldNo = null, newNo = null;
  const inc = (n) => (n === null ? null : n + 1);
  for (const line of lines) {
    if (line.startsWith('@@')) {
      const m = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
      if (m) { oldNo = parseInt(m[1], 10); newNo = parseInt(m[2], 10); } else { oldNo = null; newNo = null; }
      out.push({ type: 'hunk', text: line });
    } else if (line.startsWith('+')) { out.push({ type: 'add', text: line, newNo }); newNo = inc(newNo); }
    else if (line.startsWith('-')) { out.push({ type: 'del', text: line, oldNo }); oldNo = inc(oldNo); }
    else if (line.startsWith('\\')) out.push({ type: 'meta', text: line });
    else { out.push({ type: 'context', text: line, oldNo, newNo }); oldNo = inc(oldNo); newNo = inc(newNo); }
  }
  return out;
}

export function diffStats(text) {
  let add = 0, del = 0;
  for (const f of splitFiles(text)) for (const l of parseHunks(f.lines)) { if (l.type === 'add') add++; else if (l.type === 'del') del++; }
  return { add, del };
}

const KEYWORDS = new Set(('def class return if elif else for while in not and or is None True False import from as with try except finally raise ' +
  'lambda yield pass break continue async await global nonlocal assert del function const let var new this export default switch case null undefined true false').split(' '));
const TOKEN_RE = /(#.*$|\/\/.*$)|("""|'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)/g;

/** Light syntax colouring of one code line → array of text/span nodes. */
export function highlight(code) {
  const out = [];
  let last = 0;
  for (const m of code.matchAll(TOKEN_RE)) {
    let cls = null;
    if (m[1]) cls = 'tok-com';
    else if (m[2]) cls = 'tok-str';
    else if (m[3]) cls = 'tok-num';
    else if (m[4] && KEYWORDS.has(m[4])) cls = 'tok-kw';
    if (!cls) continue;
    if (m.index > last) out.push(code.slice(last, m.index));
    out.push(h('span', { class: cls }, m[0]));
    last = m.index + m[0].length;
  }
  if (last < code.length) out.push(code.slice(last));
  return out;
}

function renderLine(l) {
  const sign = l.type === 'add' ? '+' : l.type === 'del' ? '-' : l.type === 'context' ? ' ' : '';
  const body = (l.type === 'add' || l.type === 'del' || l.type === 'context') ? l.text.slice(1) : l.text;
  return h('div', { class: `diff-line diff-${l.type}` },
    h('span', { class: 'ln' }, l.oldNo !== undefined && l.oldNo !== null && l.type !== 'add' ? String(l.oldNo) : ''),
    h('span', { class: 'ln' }, l.newNo !== undefined && l.newNo !== null && l.type !== 'del' ? String(l.newNo) : ''),
    h('span', { class: 'sign' }, sign),
    h('span', { class: 'txt' }, l.type === 'hunk' || l.type === 'meta' ? body : highlight(body)));
}

/** Render a unified diff string into a file list + per-file coloured blocks. */
export function renderDiff(text, { emptyText = 'No changes.' } = {}) {
  const files = splitFiles(text);
  if (!text || !files.length) return stateBlock('empty', emptyText);
  const wrap = h('div', { class: 'diff' });
  const list = h('div', { class: 'diff-files' });
  let totAdd = 0, totDel = 0;
  const blocks = files.map((f) => {
    const lines = parseHunks(f.lines);
    const add = lines.filter(l => l.type === 'add').length, del = lines.filter(l => l.type === 'del').length;
    totAdd += add; totDel += del;
    const body = h('div', { class: 'diff-body' });
    for (const hd of f.headers) body.appendChild(h('div', { class: 'diff-line diff-header' }, h('span', { class: 'ln' }), h('span', { class: 'ln' }), h('span', { class: 'sign' }), h('span', { class: 'txt' }, hd)));
    for (const l of lines) body.appendChild(renderLine(l));
    const block = h('details', { class: 'diff-file', open: true },
      h('summary', { class: 'diff-file-head' }, h('span', null, f.path),
        h('span', { class: 'counts' }, h('span', { class: 'plus' }, `+${add} `), h('span', { class: 'minus' }, `−${del}`))),
      body);
    list.appendChild(h('button', { class: 'diff-file-link', type: 'button', onClick: () => { block.open = true; block.scrollIntoView({ behavior: 'smooth', block: 'start' }); } },
      h('span', null, f.path), h('span', { class: 'plus' }, `+${add}`), h('span', { class: 'minus' }, `−${del}`)));
    return block;
  });
  wrap.appendChild(h('div', { class: 'row diff-summary' },
    `${files.length} file${files.length === 1 ? '' : 's'} changed`,
    h('span', { class: 'plus' }, `+${totAdd}`), h('span', { class: 'minus' }, `−${totDel}`)));
  if (files.length > 1) wrap.appendChild(list);
  for (const b of blocks) wrap.appendChild(b);
  return wrap;
}
