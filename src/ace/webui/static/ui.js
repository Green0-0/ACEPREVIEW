// Small DOM + formatting helpers shared by every module. All text is inserted via
// textContent (never innerHTML with untrusted strings). Mirrors agent2/demo/web/ui.js.

/**
 * h(tag, attrs?, ...children) — create an element.
 * attrs: { class, dataset:{}, style:{}, on<Event>: fn, text, ...anything else => setAttribute }
 * children: string | Node | null | false | array (nested)
 */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs && typeof attrs === 'object' && !(attrs instanceof Node) && !Array.isArray(attrs)) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k === 'style' && typeof v === 'object') setStyle(el, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'value') el.value = v;
      else if (k === 'checked' || k === 'disabled' || k === 'selected' || k === 'hidden' || k === 'open') el[k] = !!v;
      else if (k === 'text') el.textContent = v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  } else if (attrs !== undefined && attrs !== null) {
    children.unshift(attrs);
  }
  append(el, children);
  return el;
}

/** Assign inline styles; custom properties ("--c") need setProperty. */
export function setStyle(el, styles) {
  for (const [k, v] of Object.entries(styles)) {
    if (v === null || v === undefined) continue;
    if (k.startsWith('--')) el.style.setProperty(k, String(v));
    else el.style[k] = v;
  }
  return el;
}

export function append(el, children) {
  for (const c of children) {
    if (c === null || c === undefined || c === false || c === true) continue;
    if (Array.isArray(c)) { append(el, c); continue; }
    if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function replace(el, ...children) {
  clear(el);
  append(el, children);
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
/** s(tag, attrs, ...children) — create an SVG element. */
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'text') el.textContent = v;
    else if (k === 'style' && typeof v === 'object') setStyle(el, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// ---------- Toasts ----------
let toastRoot = null;
export function toast(message, kind = 'info', ms = 4500) {
  if (!toastRoot) toastRoot = document.getElementById('toasts');
  if (!toastRoot) return () => {};
  const text = Array.isArray(message) ? message.join('\n') : String(message ?? '');
  const el = h('div', { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    h('div', { class: 'toast-body' }, text),
    h('button', { class: 'toast-close', 'aria-label': 'Dismiss', onClick: () => dismiss() }, '×'));
  let timer = null;
  const dismiss = () => { clearTimeout(timer); el.classList.add('leaving'); setTimeout(() => el.remove(), 160); };
  toastRoot.appendChild(el);
  timer = setTimeout(dismiss, ms);
  return dismiss;
}

// ---------- Formatting ----------
export function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return String(iso);
  return d.toLocaleString(undefined, { hour12: false });
}

/** Seconds → "1.2s" / "3m 04s" / "2h 05m". */
export function fmtSeconds(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '—';
  if (sec < 1) return `${Math.round(sec * 1000)}ms`;
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 2 : 1)}s`;
  const m = Math.floor(sec / 60), r = Math.round(sec % 60);
  if (m < 60) return `${m}m ${String(r).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

/** Number with a fixed number of decimals, or an em dash. */
export function fmtNum(v, digits = 3) {
  if (v === null || v === undefined || v === '' || !isFinite(Number(v))) return '—';
  const n = Number(v);
  return Number.isInteger(n) && digits > 0 && Math.abs(n) >= 1000 ? n.toLocaleString() : n.toFixed(digits);
}

export function shortId(id, n = 8) {
  const s = String(id ?? '');
  return s.length > n ? s.slice(0, n) : s;
}

export function prettyJson(value) {
  if (typeof value === 'string') {
    try { return JSON.stringify(JSON.parse(value), null, 2); } catch { return value; }
  }
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

export function countLines(text) { return text ? String(text).split('\n').length : 0; }

/** A text block that truncates after `limit` chars with an expand toggle. */
export function truncatedText(text, limit = 1600, cls = 'pre') {
  text = String(text ?? '');
  const pre = h('pre', { class: cls });
  if (text.length <= limit) { pre.textContent = text; return pre; }
  let expanded = false;
  const btn = h('button', { class: 'link-btn', type: 'button' });
  const render = () => {
    pre.textContent = expanded ? text : text.slice(0, limit) + '\n…';
    btn.textContent = expanded ? 'Show less' : `Show all (${text.length.toLocaleString()} chars)`;
  };
  btn.addEventListener('click', () => { expanded = !expanded; render(); });
  render();
  return h('div', { class: 'truncatable' }, pre, btn);
}

export function badge(text, kind = '', title = null) {
  return h('span', { class: `badge ${kind ? 'badge-' + kind : ''}`, title }, text);
}

/** Empty/loading/error state block. */
export function stateBlock(kind, text, action) {
  return h('div', { class: `state state-${kind}` },
    kind === 'loading' ? h('span', { class: 'spinner', 'aria-hidden': 'true' }) : null,
    h('div', { class: 'state-text' }, text),
    action ? h('button', { class: 'btn btn-sm', type: 'button', onClick: action.onClick }, action.label) : null);
}

/** <a> for an in-app hash route. */
export function link(hash, ...children) {
  return h('a', { href: hash }, ...children);
}

export function debounce(fn, ms) {
  let t = null;
  const d = (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  d.cancel = () => clearTimeout(t);
  return d;
}
