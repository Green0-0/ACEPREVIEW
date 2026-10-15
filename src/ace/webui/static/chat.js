// Rollout viewer: renders an agent2 rollout (OpenAI-format messages in `raw_chat`) as a
// chat transcript. Tool calls and tool results are collapsible; long text is truncated.
import { h, stateBlock, countLines, prettyJson, truncatedText, badge } from './ui.js';

/** Flatten a message `content` (string | parts[] | object) into text. */
export function contentText(content) {
  if (content === null || content === undefined) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map(p => typeof p === 'string' ? p : (p && p.text) || (p && p.type ? `[${p.type}]` : '')).join('\n');
  return prettyJson(content);
}

function isEditTool(name) { return /^(replace_|open_|move_|make_)/.test(String(name || '')); }

/** One collapsible tool call block: name, id, pretty-printed arguments. */
export function renderToolCall(tc) {
  const fn = tc.function || tc;
  const name = fn.name || '?';
  const args = prettyJson(fn.arguments ?? {});
  const edit = isEditTool(name);
  return h('details', { class: `tool-call ${edit ? 'tool-call-edit' : ''}`, open: args.length < 300 },
    h('summary', { class: 'tc-name' }, edit ? '✎ ' : '⚙ ', name,
      edit ? badge('edit', 'warn') : null,
      tc.id ? h('span', { class: 'tc-id' }, tc.id) : null,
      h('span', { class: 'tc-meta' }, `${args.length.toLocaleString()} chars`)),
    truncatedText(args, 3000));
}

/** Basic statistics of a chat. */
export function chatStats(chat) {
  let assistant = 0, toolCalls = 0, toolResults = 0, chars = 0;
  for (const m of chat || []) {
    if (!m || typeof m !== 'object') continue;
    chars += contentText(m.content).length;
    if (m.role === 'assistant') { assistant++; toolCalls += (m.tool_calls || []).length; }
    if (m.role === 'tool') toolResults++;
  }
  return { messages: (chat || []).length, assistant, toolCalls, toolResults, chars };
}

/** renderChat(messages) → Node. */
export function renderChat(chat, { emptyText = 'No messages recorded.' } = {}) {
  chat = Array.isArray(chat) ? chat : [];
  if (!chat.length) return stateBlock('empty', emptyText);
  const wrap = h('div', { class: 'chat' });
  const toolNames = new Map();
  for (const m of chat) for (const tc of (m && m.tool_calls) || []) if (tc && tc.id) toolNames.set(tc.id, (tc.function && tc.function.name) || tc.name || '?');
  chat.forEach((m, i) => {
    if (!m || typeof m !== 'object') return;
    const role = m.role || 'unknown';
    const content = contentText(m.content);
    const toolName = role === 'tool' ? (toolNames.get(m.tool_call_id) || m.name || 'tool') : null;
    const body = h('div', { class: 'msg-body' });
    if (role === 'tool') {
      body.appendChild(h('details', { class: 'collapsible', open: content.length < 400 },
        h('summary', null, `Result of ${toolName} (${countLines(content)} line${countLines(content) === 1 ? '' : 's'}, ${content.length.toLocaleString()} chars)`),
        truncatedText(content, 4000)));
    } else if (role === 'system') {
      body.appendChild(h('details', { class: 'collapsible' },
        h('summary', null, `System prompt (${content.length.toLocaleString()} chars)`),
        truncatedText(content, 6000)));
    } else {
      if (content) body.appendChild(truncatedText(content, 2400));
      else if (!(m.tool_calls && m.tool_calls.length)) body.appendChild(h('span', { class: 'field-help' }, '(empty)'));
      for (const tc of m.tool_calls || []) if (tc) body.appendChild(renderToolCall(tc));
    }
    wrap.appendChild(h('div', { class: `msg msg-${role}` },
      h('div', { class: 'msg-head' }, role,
        role === 'assistant' && m.tool_calls && m.tool_calls.length ? badge(`${m.tool_calls.length} tool call${m.tool_calls.length === 1 ? '' : 's'}`) : null,
        h('span', { class: 'meta' }, role === 'tool' ? `${toolName} · #${i + 1}` : `#${i + 1}`)),
      body));
  });
  return wrap;
}

/** Full rollout panel: header (id, dates, parent/children, stats) + chat. */
export function renderRollout(rollout) {
  if (!rollout || typeof rollout !== 'object') return stateBlock('empty', 'Rollout not found.');
  const chat = rollout.raw_chat || rollout.messages || [];
  const st = chatStats(chat);
  const kv = h('dl', { class: 'kv' },
    h('dt', null, 'Rollout'), h('dd', { class: 'mono' }, rollout.id || '—'),
    rollout.date_deployed ? [h('dt', null, 'Deployed'), h('dd', null, rollout.date_deployed)] : null,
    rollout.config_path ? [h('dt', null, 'Config'), h('dd', { class: 'mono' }, rollout.config_path)] : null,
    rollout.parent ? [h('dt', null, 'Parent'), h('dd', { class: 'mono' }, rollout.parent)] : null,
    rollout.children && rollout.children.length ? [h('dt', null, 'Children'), h('dd', { class: 'mono' }, rollout.children.join(', '))] : null);
  const stats = h('div', { class: 'stat-row' },
    [['messages', st.messages], ['assistant', st.assistant], ['tool calls', st.toolCalls], ['chars', st.chars.toLocaleString()]]
      .map(([l, n]) => h('div', { class: 'stat' }, h('span', { class: 'stat-n' }, String(n)), h('span', { class: 'stat-l' }, l))));
  return h('div', { class: 'rollout stack' }, kv, stats, renderChat(chat));
}
