// Thin fetch wrapper for the ACE analysis API (recording spec §12.1). Every API path the
// frontend uses is built here so the contract lives in one place.
import { toast } from './ui.js';

export class ApiError extends Error {
  constructor(status, detail, raw) {
    super(typeof detail === 'string' ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
    this.raw = raw;
  }
}

function detailText(detail) {
  if (detail === null || detail === undefined) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(d => (d && d.msg) ? `${(d.loc || []).join('.')}: ${d.msg}` : JSON.stringify(d)).join('\n');
  return JSON.stringify(detail);
}

/**
 * get(path, {silent, text}) → parsed JSON (or text when `text`). Throws ApiError and shows an
 * error toast unless `silent`.
 */
export async function get(path, opts = {}) { return request('GET', path, opts); }

/** post(path, body) / del(path): JSON requests with the same error handling as get(). */
export async function post(path, body, opts = {}) { return request('POST', path, { ...opts, body }); }
export async function del(path, opts = {}) { return request('DELETE', path, opts); }

export async function request(method, path, { silent = false, text = false, body = undefined } = {}) {
  let res;
  try {
    const init = { method, headers: { Accept: text ? 'text/plain' : 'application/json' } };
    if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
    res = await fetch(path, init);
  } catch (err) {
    const e = new ApiError(0, `Network error: ${err.message || err}`);
    if (!silent) toast(e.message, 'error');
    throw e;
  }
  const payload = await res.text();
  let data = payload;
  if (!text && payload) { try { data = JSON.parse(payload); } catch { data = payload; } }
  if (!res.ok) {
    const detail = data && typeof data === 'object' && 'detail' in data ? data.detail : (data || `${res.status} ${res.statusText}`);
    const e = new ApiError(res.status, detail, data);
    if (!silent) toast(`${res.status}: ${detailText(detail) || e.message}`, 'error', 7000);
    throw e;
  }
  return data;
}

const enc = encodeURIComponent;

/** Build "?a=1&b=2" from an object, skipping empty values. */
export function qs(params) {
  const parts = [];
  for (const [k, v] of Object.entries(params || {})) {
    if (v === null || v === undefined || v === '') continue;
    parts.push(`${enc(k)}=${enc(String(v))}`);
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

/** Path builders, one per endpoint in the §12.1 table. */
export const paths = {
  health: () => `/api/health`,
  runs: () => `/api/runs`,
  run: (run) => `/api/runs/${enc(run)}`,
  progress: (run) => `/api/runs/${enc(run)}/progress`,
  generation: (run, g) => `/api/runs/${enc(run)}/generations/${enc(g)}`,
  species: (run) => `/api/runs/${enc(run)}/species`,
  genomes: (run, params) => `/api/runs/${enc(run)}/genomes${qs(params)}`,
  genome: (run, id) => `/api/runs/${enc(run)}/genomes/${enc(id)}`,
  dot: (run, id) => `/api/runs/${enc(run)}/genomes/${enc(id)}/dot`,
  evaluations: (run, params) => `/api/runs/${enc(run)}/evaluations${qs(params)}`,
  evaluation: (run, genome, task) => `/api/runs/${enc(run)}/evaluations/${enc(genome)}/${enc(task)}`,
  workerView: (run, genome, task, node) => `/api/runs/${enc(run)}/evaluations/${enc(genome)}/${enc(task)}/workers/${enc(node)}/view`,
  rollout: (run, genome, task, rollout) => `/api/runs/${enc(run)}/rollouts/${enc(genome)}/${enc(task)}/${enc(rollout)}`,
  operators: (run) => `/api/runs/${enc(run)}/operators`,
  compare: (run, a, b) => `/api/runs/${enc(run)}/compare${qs({ a, b })}`,
  innovations: (run) => `/api/runs/${enc(run)}/innovations`,
  failures: (run) => `/api/runs/${enc(run)}/failures`,
  demoInfo: () => `/api/demo`,
  demoStart: (run, id) => `/api/runs/${enc(run)}/genomes/${enc(id)}/demo`,
  demos: (params) => `/api/demos${qs(params)}`,
  demo: (id) => `/api/demos/${enc(id)}`,
  demoRollout: (id, rollout) => `/api/demos/${enc(id)}/rollouts/${enc(rollout)}`,
  demoWorkerRollouts: (id, node) => `/api/demos/${enc(id)}/workers/${enc(node)}/rollouts`,
  demoWorkerView: (id, node) => `/api/demos/${enc(id)}/workers/${enc(node)}/view`,
};

export const api = {
  health: (opts) => get(paths.health(), opts),
  runs: (opts) => get(paths.runs(), opts),
  run: (run) => get(paths.run(run)),
  progress: (run, opts) => get(paths.progress(run), opts),
  generation: (run, g) => get(paths.generation(run, g)),
  species: (run) => get(paths.species(run)),
  genomes: (run, params) => get(paths.genomes(run, params)),
  genome: (run, id) => get(paths.genome(run, id)),
  evaluations: (run, params) => get(paths.evaluations(run, params)),
  evaluation: (run, genome, task) => get(paths.evaluation(run, genome, task)),
  workerView: (run, genome, task, node) => get(paths.workerView(run, genome, task, node)),
  rollout: (run, genome, task, rollout) => get(paths.rollout(run, genome, task, rollout)),
  operators: (run) => get(paths.operators(run)),
  compare: (run, a, b) => get(paths.compare(run, a, b)),
  innovations: (run) => get(paths.innovations(run)),
  failures: (run) => get(paths.failures(run)),
  demoInfo: (opts) => get(paths.demoInfo(), opts),
  demoStart: (run, id, prompt) => post(paths.demoStart(run, id), { prompt }),
  demos: (params, opts) => get(paths.demos(params), opts),
  demo: (id, opts) => get(paths.demo(id), opts),
  demoCancel: (id) => del(paths.demo(id)),
  demoRollout: (id, rollout) => get(paths.demoRollout(id, rollout)),
  demoWorkerRollouts: (id, node, opts) => get(paths.demoWorkerRollouts(id, node), opts),
  demoWorkerView: (id, node) => get(paths.demoWorkerView(id, node)),
};
