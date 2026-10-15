// Development-only fetch stub. Loaded by app.js ONLY when the page is opened with ?mock=1.
// Returns deterministic fixtures for every endpoint of recording spec §12.1 (shapes as
// implemented by src/ace/webui/app.py). Run "r-live" grows by one generation per poll so
// the LIVE badge and 3 s polling can be exercised.
/* eslint-disable no-console */

let seed = 7;
const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
const pick = (a) => a[Math.floor(rand() * a.length)];
const hex = () => Array.from({ length: 32 }, () => '0123456789abcdef'[Math.floor(rand() * 16)]).join('');
const iso = (msAgo) => new Date(Date.now() - msAgo).toISOString();

const OPTIONS = {
  role: ['You are a careful senior engineer.', 'You are a fast pragmatic coder.', 'You are a meticulous reviewer.', 'You are a minimalist.'],
  strategy: ['Read the relevant code before editing; keep diffs small.', 'Search broadly first, then narrow down.', 'Prefer structural edits over line edits.', 'Work test-first where possible.'],
  report_focus: ['Summarise findings and open questions.', 'List touched files and why.', 'Explain the root cause.', 'Describe risks.'],
  instruction: ['Complete the task below.', 'Investigate, then implement.', 'Solve the task with minimal changes.', 'Fix the issue described.'],
  context_budget: [12000, 6000, 24000, 48000], max_steps: [30, 15, 60], tool_set: ['editor_standard', 'editor_minimal', 'editor_interactive', 'editor_full'],
  ace_format: ['element', 'element_file', 'element_file_lines', 'file_lines'], list_budget: [8000, 4000, 16000], view_full_weight: [2, 1, 3],
};
const PROPS = {
  viewer: ['role', 'strategy', 'report_focus', 'instruction', 'max_steps', 'tool_set'],
  listviewer: ['role', 'strategy', 'report_focus', 'instruction', 'context_budget', 'list_budget', 'view_full_weight'],
  editor: ['role', 'strategy', 'report_focus', 'instruction', 'context_budget', 'max_steps', 'tool_set'],
  vieweditor: ['role', 'strategy', 'instruction', 'context_budget', 'ace_format', 'view_full_weight'],
  critic: ['role', 'strategy', 'instruction', 'context_budget', 'ace_format', 'view_full_weight'],
  judgemerger: ['role', 'strategy', 'instruction', 'context_budget', 'view_full_weight'],
  pickmerger: ['role', 'strategy', 'instruction', 'context_budget', 'ace_format', 'view_full_weight'],
};
const resolve = (type, props) => Object.fromEntries(Object.entries(props).map(([k, i]) => [k, OPTIONS[k][i]]));
const randomProps = (type) => Object.fromEntries(PROPS[type].map(k => [k, rand() < 0.6 ? 0 : Math.floor(rand() * OPTIONS[k].length)]));

// graph templates: [nodes [[id,type]], edges [[src,dst,kind]]]
const TEMPLATES = [
  [[[0, 'editor']], [[0, 'END', 'edit']]],
  [[[0, 'editor'], [3, 'viewer']], [[3, 0, 'view'], [3, 0, 'report'], [0, 'END', 'edit']]],
  [[[0, 'editor'], [3, 'viewer'], [8, 'critic']], [[3, 0, 'view'], [0, 8, 'edit'], [0, 8, 'view'], [3, 8, 'view'], [8, 'END', 'edit']]],
  [[[0, 'editor'], [3, 'viewer'], [11, 'editor'], [14, 'judgemerger']], [[3, 0, 'view'], [3, 11, 'view'], [3, 11, 'report'], [0, 14, 'edit'], [11, 14, 'edit'], [3, 14, 'view'], [14, 'END', 'edit']]],
  [[[0, 'editor'], [3, 'viewer'], [5, 'viewer'], [6, 'listviewer'], [11, 'vieweditor'], [14, 'pickmerger'], [8, 'critic']],
    [[3, 6, 'view'], [5, 6, 'view'], [6, 0, 'view'], [6, 0, 'report'], [6, 11, 'view'], [5, 0, 'report'], [0, 14, 'edit'], [11, 14, 'edit'], [6, 14, 'view'], [14, 8, 'edit'], [14, 8, 'view'], [3, 8, 'view'], [8, 'END', 'edit']]],
];
function mkGraph(ti) {
  const [nodes, edges] = TEMPLATES[ti];
  return { nodes: nodes.map(([id, type]) => ({ id, type, properties: randomProps(type) })), edges: edges.map(([src, dst, kind]) => ({ src, dst, kind })) };
}
function layoutOf(graph) {
  const ids = [...graph.nodes.map(n => n.id), 'END'];
  const level = new Map(ids.map(i => [i, 0]));
  for (let k = 0; k < ids.length; k++) for (const e of graph.edges) level.set(e.dst, Math.max(level.get(e.dst), level.get(e.src) + 1));
  const endLevel = level.get('END');
  const byLevel = new Map();
  for (const i of ids) { const l = level.get(i); if (!byLevel.has(l)) byLevel.set(l, []); byLevel.get(l).push(i); }
  const xs = new Map();
  for (const [, members] of byLevel) members.forEach((m, i) => xs.set(m, (i + 1) / (members.length + 1)));
  return {
    nodes: ids.map(i => {
      const n = graph.nodes.find(q => q.id === i);
      return { id: i, type: n ? n.type : 'end', level: level.get(i), x: xs.get(i), y: endLevel ? level.get(i) / endLevel : 0, properties_resolved: n ? resolve(n.type, n.properties) : {} };
    }),
    edges: graph.edges.map(e => ({ ...e })),
  };
}

const DIFF = `diff --git a/calc/ops.py b/calc/ops.py
index 3b18e51..a9c2f4d 100644
--- a/calc/ops.py
+++ b/calc/ops.py
@@ -10,7 +10,10 @@ def add(a, b):
     return a + b
 
 
-def divide(a, b):
-    return a / b
+def divide(a, b):
+    """Divide a by b, raising on zero."""
+    if b == 0:
+        raise ZeroDivisionError("division by zero")  # explicit
+    return a / b
 
 
 def mul(a, b):
diff --git a/tests/test_ops.py b/tests/test_ops.py
new file mode 100644
--- /dev/null
+++ b/tests/test_ops.py
@@ -0,0 +1,6 @@
+import pytest
+from calc.ops import divide
+
+
+def test_divide_by_zero():
+    with pytest.raises(ZeroDivisionError):
+        divide(1, 0)`;

const TASKS = [
  { index: 0, project_dir: '/home/user/agent2/smoke_repos/calc', prompt: 'Make divide() raise ZeroDivisionError with a clear message when b == 0 and add a test.', scorer: 'pytest' },
  { index: 1, project_dir: '/home/user/agent2/smoke_repos/calc', prompt: 'Add a pow(a, b) operation with integer exponent validation.', scorer: 'pytest' },
];
const OPS = ['mutate_properties', 'add_connection', 'add_worker', 'insert_merger', 'remove_connection', 'remove_worker', 'toggle_enable'];
const FAILS = [['WorkerError', 'model returned no edit'], ['AceParseError', 'could not parse ACE block: unterminated code fence'], ['Timeout', 'worker exceeded max_steps']];

function buildRun(runId, { gens, pop, status, evaluator }) {
  const R = { genomes: new Map(), generations: [], evaluations: [], species: [], events: [], rollouts: new Map() };
  const t0 = gens * 90_000 + 60_000;
  let prev = [];
  let threshold = 3.0;
  for (let g = 0; g < gens; g++) {
    const popRows = [];
    const started = iso(t0 - g * 90_000);
    for (let i = 0; i < pop; i++) {
      const elite = g > 0 && i < 2;
      const parentA = prev.length ? pick(prev) : null;
      let parents = [], operators = [];
      if (g > 0) {
        if (elite) { parents = [prev[i].genome_id]; operators = []; }
        else if (rand() < 0.4) { parents = [parentA.genome_id, pick(prev).genome_id]; operators = ['crossover', pick(OPS)]; }
        else { parents = [parentA.genome_id]; operators = [pick(OPS)]; if (rand() < 0.3) operators.push(pick(OPS)); }
      }
      const ti = Math.min(TEMPLATES.length - 1, Math.floor(rand() * (2 + g / 2)));
      const graph = elite ? R.genomes.get(prev[i].genome_id)._graph : mkGraph(ti);
      const workerTypes = {};
      for (const n of graph.nodes) workerTypes[n.type] = (workerTypes[n.type] || 0) + 1;
      const failed = rand() < 0.15;
      const taskScores = evaluator === 'task' ? TASKS.map(() => failed && rand() < 0.6 ? 0 : +(0.3 + 0.08 * g + rand() * 0.3).toFixed(3)) : [];
      const fitness = evaluator === 'task' ? +(taskScores.reduce((a, b) => a + b, 0) / taskScores.length).toFixed(4) : +(0.2 + 0.07 * g + rand() * 0.25).toFixed(4);
      const adjusted = +(fitness - 0.02 * graph.nodes.length).toFixed(4);
      const id = elite ? prev[i].genome_id : hex();
      const sp = Math.floor(rand() * Math.min(5, 1 + g));
      const row = { genome_id: id, species_id: sp, fitness, adjusted_fitness: adjusted, num_workers: graph.nodes.length, structure_hash: hex().slice(0, 40), is_elite: elite, parents, operators, worker_types: workerTypes };
      popRows.push(row);
      if (!R.genomes.has(id)) {
        R.genomes.set(id, {
          id, generation: g, parents, operators, species_id: sp, fitness, adjusted_fitness: adjusted, structure_hash: row.structure_hash, num_workers: graph.nodes.length,
          created_at: started, evaluated_at: iso(t0 - g * 90_000 - 40_000), eval_duration_s: 20 + rand() * 60, worker_types: workerTypes, task_scores: taskScores,
          genome: { id, generation: g, parents, operators, fitness, adjusted_fitness: adjusted, species_id: sp,
            nodes: graph.nodes.map(n => ({ innovation: n.id, type: n.type, properties: n.properties })),
            connections: graph.edges.map((e, k) => ({ innovation: 100 + k, src: e.src, dst: e.dst, kind: e.kind, enabled: true })) },
          _graph: graph,
        });
        if (evaluator === 'task') for (const task of TASKS) R.evaluations.push(mkEval(R, id, g, task, graph, taskScores[task.index], failed && taskScores[task.index] === 0));
        else R.evaluations.push({ genome_id: id, generation: g, task_index: 0, score: fitness, mock: true, started_at: started, finished_at: started, duration_s: 0.01 });
      }
    }
    popRows.sort((a, b) => b.fitness - a.fitness);
    const fits = popRows.map(r => r.fitness).sort((a, b) => a - b);
    const speciesIds = [...new Set(popRows.map(r => r.species_id))];
    threshold = +(threshold * (speciesIds.length < 3 ? 0.9 : 1.05)).toFixed(3);
    const opCounts = {};
    for (const r of popRows) for (const o of r.operators) opCounts[o] = (opCounts[o] || 0) + 1;
    R.generations.push({
      generation: g, started_at: started, finished_at: iso(t0 - g * 90_000 - 80_000), duration_s: 70 + rand() * 30, evaluated: pop, elites: g ? 2 : 0,
      species_count: speciesIds.length, threshold, best_fitness: fits[fits.length - 1], mean_fitness: +(fits.reduce((a, b) => a + b, 0) / fits.length).toFixed(4),
      median_fitness: fits[Math.floor(fits.length / 2)], best_adjusted: Math.max(...popRows.map(r => r.adjusted_fitness)),
      mean_adjusted: +(popRows.reduce((a, r) => a + r.adjusted_fitness, 0) / pop).toFixed(4), mean_workers: popRows.reduce((a, r) => a + r.num_workers, 0) / pop,
      best_genome_id: popRows[0].genome_id, operator_counts: opCounts, population: popRows,
    });
    R.species.push({ generation: g, threshold, species: speciesIds.map(sid => { const m = popRows.filter(r => r.species_id === sid); return { id: sid, size: m.length, best_fitness: Math.max(...m.map(r => r.fitness)), representative_id: m[0].genome_id }; }) });
    R.events.push({ ts: started, event: 'generation_start', generation: g });
    for (const r of popRows.slice(0, 3)) R.events.push({ ts: started, event: 'genome_evaluated', genome_id: r.genome_id, fitness: r.fitness, adjusted_fitness: r.adjusted_fitness, duration_s: 31.2 });
    R.events.push({ ts: started, event: 'speciated', generation: g, species_count: speciesIds.length, threshold });
    R.events.push({ ts: started, event: 'generation_end', generation: g, duration_s: 80.5, best_fitness: fits[fits.length - 1] });
    prev = popRows;
  }
  const all = R.generations.flatMap(g => g.population);
  const best = all.reduce((a, b) => (b.fitness > a.fitness ? b : a), all[0]);
  R.run = {
    run_id: runId, ace_version: '0.3.0', status, started_at: iso(t0), finished_at: status === 'running' ? null : iso(10_000), duration_s: status === 'running' ? null : t0 / 1000,
    resumed_from: null, seed: 0,
    config: { population_size: pop, generations: status === 'running' ? 20 : gens, node_cost: 0.02, crossover_rate: 0.75, initial_min_workers: 1, initial_max_workers: 5, max_workers: 12, seed: 0,
      speciation: { enabled: true, target_species: 5, c_excess: 1, c_disjoint: 1, c_properties: 0.5, initial_threshold: 3, threshold_step: 0.15, stagnation_generations: 15, elitism_per_species: 1, survival_fraction: 0.4 },
      mutation: { p_property: 0.8, p_property_per_gene: 0.1, p_add_connection: 0.3, p_add_worker: 0.15, p_insert_merger: 0.05, p_remove_connection: 0.05, p_remove_worker: 0.05, p_toggle_enable: 0.02, max_attempts: 20 },
      history_dir: `.ace/neat/${runId}` },
    model: { model: 'claude-haiku-5-5', api_base_url: 'https://api.anthropic.com/v1/', api_key_env_var: 'ANTHROPIC_API_KEY' },
    evaluator: { type: evaluator, tasks: evaluator === 'task' ? TASKS : [] },
    generations_completed: gens, best_genome_id: best.genome_id, best_fitness: best.fitness, error: null,
  };
  R.events.unshift({ ts: iso(t0), event: 'run_start', run_id: runId });
  if (status !== 'running') R.events.push({ ts: iso(10_000), event: 'run_end', status, duration_s: t0 / 1000 });
  return R;
}

function mkEval(R, gid, g, task, graph, score, failed) {
  const order = layoutOf(graph).nodes.filter(n => n.id !== 'END').sort((a, b) => a.level - b.level || a.id - b.id).map(n => n.id);
  const failAt = failed ? order[Math.floor(rand() * order.length)] : null;
  const [ftype, fmsg] = pick(FAILS);
  let seenFail = false;
  const workers = order.map((nid, k) => {
    const n = graph.nodes.find(q => q.id === nid);
    const status = seenFail ? 'skipped' : nid === failAt ? 'failed' : 'ok';
    if (nid === failAt) seenFail = true;
    const tools = n.type === 'viewer' || n.type === 'editor';
    const rids = status === 'skipped' ? [] : [`r_${gid.slice(0, 6)}_${task.index}_${nid}`];
    for (const rid of rids) R.rollouts.set(rid, mkRollout(rid, n.type, task));
    return {
      node_id: nid, type: n.type, properties: resolve(n.type, n.properties), started_at: iso(50_000 - k * 5000), finished_at: iso(45_000 - k * 5000), duration_s: status === 'skipped' ? 0 : 3 + rand() * 20,
      status, error: status === 'failed' ? `${ftype}: ${fmsg}` : null, rollout_ids: rids, rollout_dir: `rollouts/${gid}/${task.index}`,
      llm_calls: status === 'skipped' ? 0 : 1 + Math.floor(rand() * 8), tool_calls: tools && status !== 'skipped' ? Math.floor(rand() * 12) : 0,
      report: ['viewer', 'listviewer', 'editor'].includes(n.type) && status === 'ok' ? 'Looked at calc/ops.py: divide() has no zero check. Proposed raising ZeroDivisionError and added a regression test in tests/test_ops.py.' : null,
      view: status === 'skipped' ? null : { files: { 'calc/ops.py': { levels: { '': 1, divide: 2, add: 1 }, line_ranges: [[1, 12]] }, 'tests/test_ops.py': { levels: { '': 2 }, line_ranges: [] } } },
      view_summary: { files: 2, entries: 4, rendered_chars: 1830 }, edit_touched_paths: ['editor', 'vieweditor', 'critic', 'pickmerger', 'judgemerger'].includes(n.type) ? ['calc/ops.py', 'tests/test_ops.py'] : [], edit_chars: 412,
    };
  });
  const inputs = {};
  for (const n of graph.nodes) {
    const inc = graph.edges.filter(e => e.dst === n.id);
    inputs[n.id] = { views_from: inc.filter(e => e.kind === 'view').map(e => e.src), edits_from: inc.filter(e => e.kind === 'edit').map(e => e.src), reports_from: inc.filter(e => e.kind === 'report').map(e => e.src) };
  }
  return {
    genome_id: gid, generation: g, task_index: task.index, task: { project_dir: task.project_dir, prompt: task.prompt, scorer: task.scorer },
    started_at: iso(60_000), finished_at: iso(30_000), duration_s: 30, score: failed ? 0 : score,
    failure: failed ? fmsg : null, failure_node_id: failAt, failure_type: failed ? ftype : null,
    diff: failed ? null : DIFF, touched_paths: failed ? [] : ['calc/ops.py', 'tests/test_ops.py'], diff_chars: failed ? 0 : DIFF.length, order, workers, inputs,
  };
}

function mkRollout(id, type, task) {
  return {
    config_path: `<ace:${type}>`, id, date_deployed: iso(55_000), parent: null, children: [], action_history: [],
    raw_chat: [
      { role: 'system', content: 'You are a careful senior engineer.\n\nRead the relevant code before editing; keep diffs small.\n\n' + 'Output format: ...\n'.repeat(40) },
      { role: 'user', content: `Complete the task below.\n\n# Task\n${task.prompt}` },
      { role: 'assistant', content: 'Let me look at the operations module first.', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'view', arguments: JSON.stringify({ path: 'calc/ops.py' }) } }] },
      { role: 'tool', tool_call_id: 'call_1', content: 'def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n'.repeat(30) },
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_2', type: 'function', function: { name: 'replace_element', arguments: JSON.stringify({ path: 'calc/ops.py', element: 'divide', content: 'def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError("division by zero")\n    return a / b\n' }) } }] },
      { role: 'tool', tool_call_id: 'call_2', content: 'Replaced calc/ops.py:divide.' },
      { role: 'assistant', content: 'Done. divide() now raises ZeroDivisionError on b == 0 and a regression test was added.' },
    ],
  };
}

const RUNS = new Map([
  ['r-demo', buildRun('r-demo', { gens: 8, pop: 12, status: 'finished', evaluator: 'task' })],
  ['r-mock', buildRun('r-mock', { gens: 5, pop: 10, status: 'failed', evaluator: 'mock' })],
]);
RUNS.get('r-mock').run.error = 'KeyboardInterrupt';
const LIVE = buildRun('r-live', { gens: 12, pop: 8, status: 'running', evaluator: 'task' });
let liveGens = 3;
RUNS.set('r-live', LIVE);
const liveView = () => {
  const gens = LIVE.generations.slice(0, liveGens);
  return { ...LIVE, generations: gens, run: { ...LIVE.run, generations_completed: gens.length, status: liveGens >= LIVE.generations.length ? 'finished' : 'running' } };
};

function view(runId) { return runId === 'r-live' ? liveView() : RUNS.get(runId); }
function stripBody(r) { const { genome: _g, _graph, ...rest } = r; return rest; }
function lineage(R, gid) {
  const out = [], seen = new Set([gid]), q = [gid];
  while (q.length) { const cur = R.genomes.get(q.shift()); for (const p of cur.parents) if (!seen.has(p) && R.genomes.has(p)) { seen.add(p); q.push(p); out.push(stripBody(R.genomes.get(p))); } }
  return out.sort((a, b) => b.generation - a.generation);
}

const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
const err = (detail, status = 404) => json({ detail }, status);

const realFetch = window.fetch.bind(window);
window.fetch = async (input, init = {}) => {
  const url = new URL(typeof input === 'string' ? input : input.url, location.origin);
  const path = url.pathname;
  if (!path.startsWith('/api/')) return realFetch(input, init);
  const p = url.searchParams;
  let m;
  if (path === '/api/health') return json({ ok: true, runs_root: '.ace/neat (mock)' });
  if (path === '/api/runs') return json([...RUNS.keys()].map(id => { const r = view(id).run; return { run_id: id, status: r.status, started_at: r.started_at, finished_at: r.finished_at, generations_completed: r.generations_completed, best_fitness: r.best_fitness, population_size: r.config.population_size, evaluator_type: r.evaluator.type }; }));
  if (!(m = /^\/api\/runs\/([^/]+)(\/.*)?$/.exec(path))) return err(`mock: no route for ${path}`);
  const runId = decodeURIComponent(m[1]);
  const rest = m[2] || '';
  if (!RUNS.has(runId)) return err(`run ${runId} not found`);
  const R = view(runId);
  if (rest === '') return json({ ...R.run, counts: { genomes: R.genomes.size, evaluations: R.evaluations.length, generations: R.generations.length, species: new Set(R.species.flatMap(s => s.species.map(x => x.id))).size } });
  if (rest === '/progress') {
    if (runId === 'r-live' && liveGens < LIVE.generations.length) liveGens++;
    const V = view(runId);
    return json({ generations: V.generations.map(({ population: _p, ...g }) => g), events_tail: V.events.filter(e => e.generation === undefined || e.generation < liveGens || runId !== 'r-live').slice(-50), status: V.run.status });
  }
  if ((m = /^\/generations\/(-?\d+)$/.exec(rest))) { const g = R.generations.find(x => x.generation === Number(m[1])); return g ? json(g) : err(`generation ${m[1]} not found`); }
  if (rest === '/species') {
    const summary = {};
    for (const snap of R.species.slice(0, R.generations.length)) for (const sp of snap.species) {
      const e = summary[String(sp.id)] || (summary[String(sp.id)] = { created: snap.generation, last_seen: snap.generation, best_fitness: null, max_size: 0 });
      e.last_seen = snap.generation; e.best_fitness = Math.max(e.best_fitness ?? -Infinity, sp.best_fitness); e.max_size = Math.max(e.max_size, sp.size);
    }
    return json({ timeline: R.species.slice(0, R.generations.length), species: summary });
  }
  if (rest === '/genomes') {
    let items = [...R.genomes.values()];
    if (p.get('generation')) items = items.filter(r => r.generation === Number(p.get('generation')));
    if (p.get('species')) items = items.filter(r => r.species_id === Number(p.get('species')));
    if (p.get('min_fitness')) items = items.filter(r => r.fitness >= Number(p.get('min_fitness')));
    const sort = p.get('sort');
    if (sort) { const desc = sort.startsWith('-'), key = sort.replace(/^[-+]/, ''); items.sort((a, b) => (a[key] - b[key]) * (desc ? -1 : 1)); }
    const off = Number(p.get('offset') || 0), lim = Number(p.get('limit') || 100);
    return json({ total: items.length, items: items.slice(off, off + lim).map(stripBody) });
  }
  if ((m = /^\/genomes\/([^/]+)(\/dot)?$/.exec(rest))) {
    const gid = decodeURIComponent(m[1]);
    const g = R.genomes.get(gid);
    if (!g) return err(`genome ${gid} not found`);
    if (m[2]) return new Response(`digraph g_${gid} {\n  rankdir=TB;\n${g._graph.nodes.map(n => `  n${n.id} [label="${n.id}\\n${n.type}"];`).join('\n')}\n  END [shape=doublecircle];\n${g._graph.edges.map(e => `  ${e.src === 'END' ? 'END' : 'n' + e.src} -> ${e.dst === 'END' ? 'END' : 'n' + e.dst} [label="${e.kind}"];`).join('\n')}\n}\n`, { headers: { 'Content-Type': 'text/vnd.graphviz' } });
    const children = [...R.genomes.values()].filter(c => c.parents.includes(gid) && c.id !== gid).map(c => c.id);
    const evals = R.evaluations.filter(e => e.genome_id === gid).map(e => ({ ...e, workers: (e.workers || []).map(({ view: _v, ...w }) => w) }));
    const { _graph, ...rec } = g;
    return json({ ...rec, graph: _graph, layout: layoutOf(_graph), lineage: lineage(R, gid), children, evaluations: evals });
  }
  if (rest === '/evaluations') {
    let items = R.evaluations.slice();
    if (p.get('genome')) items = items.filter(e => e.genome_id === p.get('genome'));
    if (p.get('failed')) items = items.filter(e => (e.failure !== null && e.failure !== undefined) === (p.get('failed') === 'true'));
    return json({ total: items.length, items: items.map(({ workers: _w, ...e }) => e) });
  }
  if ((m = /^\/evaluations\/([^/]+)\/(\d+)(?:\/workers\/(\d+)\/view)?$/.exec(rest))) {
    const ev = R.evaluations.find(e => e.genome_id === decodeURIComponent(m[1]) && e.task_index === Number(m[2]));
    if (!ev) return err('evaluation not found');
    if (m[3] === undefined) return json(ev);
    const w = (ev.workers || []).find(x => x.node_id === Number(m[3]));
    if (!w) return err(`worker ${m[3]} not found`);
    const rendered = w.view && Number(m[3]) % 2 === 0 ? '# calc/ops.py\ndef add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n' : null;
    return json({ view: w.view, rendered });
  }
  if ((m = /^\/rollouts\/([^/]+)\/(\d+)\/([^/]+)$/.exec(rest))) { const r = R.rollouts.get(decodeURIComponent(m[3])); return r ? json(r) : err('rollout not found'); }
  if (rest === '/operators') {
    const acc = {};
    for (const c of R.genomes.values()) for (const op of new Set(c.operators)) {
      const parents = c.parents.map(x => R.genomes.get(x)).filter(Boolean);
      const e = acc[op] || (acc[op] = { operator: op, count: 0, child: [], parent: [], ok: 0 });
      e.count++;
      if (parents.length) { const bp = Math.max(...parents.map(x => x.adjusted_fitness)); e.child.push(c.adjusted_fitness); e.parent.push(bp); if (c.adjusted_fitness >= bp) e.ok++; }
    }
    const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
    return json(Object.values(acc).map(e => ({ operator: e.operator, count: e.count, mean_child_fitness: mean(e.child), mean_parent_fitness: mean(e.parent), mean_delta: e.child.length ? mean(e.child) - mean(e.parent) : null, success_rate: e.child.length ? e.ok / e.child.length : null })).sort((a, b) => b.count - a.count));
  }
  if (rest === '/compare') {
    const A = R.genomes.get(p.get('a')), B = R.genomes.get(p.get('b'));
    if (!A || !B) return err('genome not found');
    const na = new Map(A._graph.nodes.map(n => [n.id, n])), nb = new Map(B._graph.nodes.map(n => [n.id, n]));
    const matched = [...na.keys()].filter(k => nb.has(k)).sort((x, y) => x - y);
    const diffs = matched.map(k => {
      const x = na.get(k), y = nb.get(k), ch = {};
      for (const prop of new Set([...Object.keys(x.properties), ...Object.keys(y.properties)])) if (x.properties[prop] !== y.properties[prop]) ch[prop] = { a: x.properties[prop], b: y.properties[prop], a_resolved: OPTIONS[prop][x.properties[prop]], b_resolved: OPTIONS[prop][y.properties[prop]] };
      return { node: k, type: x.type, changed: Object.keys(ch).length, diffs: ch };
    });
    const ca = new Set(A.genome.connections.map(c => c.innovation)), cb = new Set(B.genome.connections.map(c => c.innovation));
    const side = (G) => ({ genome: stripBody(G), graph: G._graph, layout: layoutOf(G._graph) });
    return json({ a: side(A), b: side(B), matched_nodes: matched, only_a_nodes: [...na.keys()].filter(k => !nb.has(k)), only_b_nodes: [...nb.keys()].filter(k => !na.has(k)),
      matched_connections: [...ca].filter(c => cb.has(c)), only_a_connections: [...ca].filter(c => !cb.has(c)), only_b_connections: [...cb].filter(c => !ca.has(c)),
      property_diffs: diffs, excess: 1, disjoint: 2, property_difference: 0.5, distance: 2.75 });
  }
  if (rest === '/innovations') return json({
    nodes: [[0, ['init', 0, 'editor'], 'editor'], [3, ['source', 0, 'view', 'viewer'], 'viewer'], [5, ['source', 6, 'view', 'viewer'], 'viewer'], [6, ['split', 4, 'listviewer'], 'listviewer'], [8, ['critic', 1, 'critic'], 'critic'], [11, ['branch', 3, 'editor'], 'editor'], [14, ['merge', 0, 11, 'judgemerger'], 'judgemerger']].map(([id, key, type]) => ({ id, key, type })),
    connections: [[1, 0, 'END', 'edit'], [2, 3, 0, 'view'], [4, 3, 0, 'report'], [7, 0, 8, 'edit'], [9, 8, 'END', 'edit'], [10, 3, 8, 'view']].map(([id, src, dst, kind]) => ({ id, src, dst, kind })),
  });
  if (rest === '/failures') {
    const groups = new Map();
    for (const e of R.evaluations) {
      if (!e.failure) continue;
      const w = (e.workers || []).find(x => x.node_id === e.failure_node_id);
      const key = `${e.failure_type}|${w ? w.type : ''}`;
      if (!groups.has(key)) groups.set(key, { failure_type: e.failure_type, node_type: w ? w.type : null, count: 0, examples: [] });
      const g = groups.get(key); g.count++;
      if (g.examples.length < 5) g.examples.push({ genome_id: e.genome_id, task_index: e.task_index, message: e.failure });
    }
    return json([...groups.values()].sort((a, b) => b.count - a.count));
  }
  return err(`mock: no route for ${path}`);
};

console.info('[mock] ACE studio dev mock active — all /api traffic is simulated in the browser.');
{
  const b = document.createElement('div');
  b.textContent = 'MOCK MODE — data is simulated in the browser (?mock=1)';
  b.setAttribute('style', 'position:fixed;left:50%;bottom:8px;transform:translateX(-50%);background:#7a4df6;color:#fff;font:600 11px system-ui;padding:3px 10px;border-radius:999px;z-index:99;opacity:.9;pointer-events:none');
  document.body.appendChild(b);
}
