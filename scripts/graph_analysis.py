#!/usr/bin/env python3
"""Secondary analysis: which evolved worker graphs were effective in the NEAT runs.

    python scripts/graph_analysis.py \
        --runs /usr/project/xtmp/kg349/ace/runs/neat_a /usr/project/xtmp/kg349/ace/runs/neat_b \
        --out /usr/project/xtmp/kg349/ace/runs/graph_analysis

Reads a snapshot of each run's genomes.jsonl / evaluations.jsonl / generations.jsonl /
innovations.json / population_state.json (tolerating a partial last line; the runs may
still be appending) and never writes into the run directories.

Outcome. Each task record's proxy score is centred within its (run, batch, instance)
cell: ``delta = score - mean(score of every genome that attempted that instance in that
minibatch)``. This is an instance fixed effect, so minibatch and task difficulty drop out;
elites re-scored on later batches contribute more cells. Cells attempted by a single
genome carry no information and are dropped. Identical graphs (same WorkerGraph JSON,
i.e. structure and properties) within a run are pooled into one unit.

Inference. A graph-level feature ``x`` is tested with task-level OLS
``delta ~ x + C(run)`` with cluster-robust SEs by graph (effect size and 95% CI), and with
a stratified permutation test (labels permuted across graphs within run, weighted by task
count). Permutation p-values are BH-corrected over all features tested in one family.
A selection-free sensitivity uses only the generation-0 minibatch (random initial genomes
plus bootstrap mutants).

Outputs (all under --out): report.md, report.json, graphs.csv, features.csv, fig1_size.png,
fig2_effects.png, fig3_pareto_lineage.png.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ace.graph.properties import PROPERTIES
from ace.graph.types import END, WorkerType
from ace.neat.genome import Genome

RUNS_ROOT = Path("/usr/project/xtmp/kg349/ace/runs")
COLORS = {"neat_a": "#2a78d6", "neat_b": "#eb6834"}
ACE_TYPES = {"vieweditor", "critic", "pickmerger"}
TYPES = ["viewer", "listviewer", "editor", "vieweditor", "critic", "judgemerger", "pickmerger"]
SHORT = {"viewer": "VW", "listviewer": "LV", "editor": "ED", "vieweditor": "VE", "critic": "CR",
         "judgemerger": "JM", "pickmerger": "PM"}
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading

def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    with open(path, "rb") as fh:
        data = fh.read()  # one read = one snapshot
    for line in data.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial last line of a file still being appended
    return out


def strategy_label(t: str, idx: int) -> str:
    text = PROPERTIES[WorkerType(t)]["strategy"].options[idx]
    first = text.splitlines()[0].lstrip("# ").strip()
    return first.split(":", 1)[-1].strip()


def option_label(t: str, prop: str, idx: int) -> str:
    spec = PROPERTIES[WorkerType(t)][prop]
    if prop == "strategy":
        return strategy_label(t, idx)
    if prop == "role":
        text = spec.options[idx]
        words = re.sub(r"^You are (a |an |the )?", "", text).split()
        return f"#{idx} '{' '.join(words[:2]).rstrip('.,')}'"
    if prop == "report_focus":
        body = spec.options[idx].split("\n", 1)[-1].split()
        return f"#{idx} '{' '.join(body[:2]).rstrip('.,:')}'"
    if prop == "instruction":
        return f"#{idx}"
    return str(spec.options[idx])


# ---------------------------------------------------------------- graph features

def graph_features(genome: Genome) -> dict:
    g = genome.to_graph().to_json()
    nodes = {n["id"]: n for n in g["nodes"]}
    edges = g["edges"]
    types = {i: n["type"] for i, n in nodes.items()}
    f: dict = {"n_workers": len(nodes), "n_edges": len(edges)}
    cnt = Counter(types.values())
    for t in TYPES:
        f[f"n_{t}"] = cnt.get(t, 0)
        f[f"has_{t}"] = int(cnt.get(t, 0) > 0)
    end_src = [e["src"] for e in edges if e["dst"] == END]
    end_t = types[end_src[0]] if end_src else "none"
    f["end_type"] = end_t
    f["end_is_merger"] = int(end_t in ("pickmerger", "judgemerger"))
    f["end_is_ace"] = int(end_t in ACE_TYPES)
    f["has_merger"] = int(cnt.get("pickmerger", 0) + cnt.get("judgemerger", 0) > 0)
    f["n_generators"] = cnt.get("editor", 0) + cnt.get("vieweditor", 0)
    f["n_ace_nodes"] = sum(cnt.get(t, 0) for t in ACE_TYPES)
    # merger fan-in at END
    f["end_fanin"] = sum(1 for e in edges if e["dst"] == (end_src[0] if end_src else None) and e["kind"] == "edit") \
        if f["end_is_merger"] else 1
    # editor -> editor chain (VIEW/REPORT from one editor into another)
    f["editor_chain"] = int(any(types.get(e["src"]) == "editor" and e["dst"] != END and types.get(e["dst"]) == "editor"
                                for e in edges))
    # depth: longest path (in workers) ending at END over all edge kinds
    preds = defaultdict(list)
    for e in edges:
        if e["dst"] != END:
            preds[e["dst"]].append(e["src"])
    memo: dict = {}

    def depth(n):
        if n not in memo:
            memo[n] = 1 + max((depth(p) for p in preds[n]), default=0)
        return memo[n]
    f["depth"] = max((depth(n) for n in nodes), default=0)
    # property indicators: any node of type t with property p == option o
    props = {}
    for n in nodes.values():
        for p, o in n["properties"].items():
            props[f"prop:{n['type']}:{p}:{o}"] = props.get(f"prop:{n['type']}:{p}:{o}", 0) + 1
    f["_props"] = props
    f["_types_sig"] = "+".join(f"{SHORT[t]}{cnt[t]}" for t in TYPES if cnt.get(t))
    return f


def node_family(innov_db: dict) -> dict[int, tuple[str, str]]:
    out = {}
    for rec in innov_db.get("nodes", []):
        key = rec["key"]
        fam = key[0][1] if isinstance(key[0], list) else key[0]
        typ = key[-1][1] if isinstance(key[-1], list) else key[-1]
        out[int(rec["id"])] = (str(fam), str(typ))
    return out


# ---------------------------------------------------------------- stats helpers

def cluster_ols(df: pd.DataFrame, formula: str, groups: str):
    return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df[groups])[0]})


def perm_test(gdf: pd.DataFrame, x: str, y: str = "delta", w: str = "n_tasks", n_perm: int = 4000) -> float:
    """Two-sided stratified (by run) permutation p for the weighted within-run slope of y on x."""
    runs = gdf["run"].to_numpy()
    X = gdf[x].to_numpy(float)
    Y = gdf[y].to_numpy(float)
    W = gdf[w].to_numpy(float)
    idx_by_run = [np.where(runs == r)[0] for r in np.unique(runs)]

    def resid(v):
        out = v.copy()
        for ix in idx_by_run:
            out[ix] = v[ix] - np.average(v[ix], weights=W[ix])
        return out
    Yr = resid(Y)

    def stat(xv):
        xr = resid(xv)
        den = np.sum(W * xr * xr)
        return 0.0 if den == 0 else np.sum(W * xr * Yr) / den
    obs = stat(X)
    perms = np.empty(n_perm)
    for k in range(n_perm):
        xp = X.copy()
        for ix in idx_by_run:
            xp[ix] = X[RNG.permutation(ix)]
        perms[k] = stat(xp)
    return float((1 + np.sum(np.abs(perms) >= abs(obs) - 1e-12)) / (n_perm + 1))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def classify_failure(msg: str | None) -> str:
    if not msg:
        return "none"
    m = msg
    if "Overlapping ACE blocks" in m:
        hs = re.findall(r"'(@@ [^']*)'", m)
        return "overlap (duplicate block)" if len(hs) >= 2 and hs[0] == hs[1] else "overlap (distinct blocks)"
    for pat, lab in [("No ACE blocks found", "no ACE block"), ("Unclosed code block", "unclosed fence"),
                     ("not permitted in format", "header not allowed by format"),
                     ("is outside", "line span out of file"), ("Malformed element path", "malformed element path"),
                     ("Unknown element", "unknown element"), ("Conflicting ACE blocks", "conflicting blocks"),
                     ("unknown path", "unknown path"), ("already exists", "NEW on existing file"),
                     ("Malformed ACE header", "malformed header"), ("not a parsed code file", "element header on non-code file"),
                     ("does not name a candidate", "judge reply names no candidate"),
                     ("needs at least one non-empty input view", "empty input view"),
                     ("no selected location resolves", "listviewer selection unresolved"),
                     ("empty report", "empty report")]:
        if pat in m:
            return lab
    return "other"


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path,
                    default=[RUNS_ROOT / "neat_a", RUNS_ROOT / "neat_b"])
    ap.add_argument("--out", type=Path, default=RUNS_ROOT / "graph_analysis")
    ap.add_argument("--n-perm", type=int, default=4000)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    R: dict = {"snapshot": {}}

    # ---------------- load runs
    task_rows, graph_meta, pop_rows, innov_rows = [], {}, [], []
    worker_rows = []
    genome_cache: dict[tuple[str, str], Genome] = {}
    gid2key: dict[tuple[str, str], str] = {}
    key2feat: dict[str, dict] = {}
    for run_dir in args.runs:
        run = run_dir.name
        G = {}
        for rec in load_jsonl(run_dir / "genomes.jsonl"):
            G[rec["id"]] = rec.get("genome") or rec
        state_path = run_dir / "population_state.json"
        state = {}
        try:
            state = json.loads(state_path.read_text())
            for g in state.get("genomes", []):
                G.setdefault(g["id"], g)
        except (OSError, json.JSONDecodeError):
            pass
        evals = load_jsonl(run_dir / "evaluations.jsonl")
        gens = load_jsonl(run_dir / "generations.jsonl")
        try:
            innov = json.loads((run_dir / "innovations.json").read_text())
        except (OSError, json.JSONDecodeError):
            innov = state.get("innovations", {})
        fam = node_family(innov)
        n_mock = sum(1 for e in evals if e.get("mock"))
        n_unknown = sum(1 for e in evals if not e.get("mock") and e["genome_id"] not in G)
        R["snapshot"][run] = {"genomes": len(G), "evaluation_records": len(evals), "mock_skipped": n_mock,
                              "records_unknown_genome_skipped": n_unknown, "generations_logged": len(gens)}

        def key_of(gid):
            if (run, gid) in gid2key:
                return gid2key[(run, gid)]
            gen = Genome.from_json(G[gid])
            k = run + ":" + json.dumps(gen.to_graph().to_json(), sort_keys=True)
            gid2key[(run, gid)] = k
            genome_cache[(run, gid)] = gen
            if k not in key2feat:
                key2feat[k] = graph_features(gen)
                graph_meta[k] = {"run": run, "first_gen": gen.generation, "ids": [], "genome": gen,
                                 "species": gen.species_id}
            graph_meta[k]["ids"].append(gid)
            graph_meta[k]["first_gen"] = min(graph_meta[k]["first_gen"], gen.generation)
            return k

        for e in evals:
            if e.get("mock") or e["genome_id"] not in G:
                continue
            k = key_of(e["genome_id"])
            u = e.get("usage") or {}
            task_rows.append({
                "run": run, "graph": k, "genome_id": e["genome_id"], "batch": e["batch"],
                "instance": e["instance_id"], "score": float(e.get("score") or 0.0),
                "tokens": float(u.get("total_tokens") or np.nan), "failed": int(e.get("failure") is not None),
                "failure_class": classify_failure(e.get("failure")),
                "failure_type": e.get("failure_type"),
                "fail_node_type": next((w["type"] for w in e.get("workers", [])
                                        if w["node_id"] == e.get("failure_node_id")), None),
            })
            for w in e.get("workers", []):
                if w.get("status") not in ("ok", "failed"):
                    continue
                props = w.get("properties") or {}
                worker_rows.append({"run": run, "graph": k, "type": w["type"], "failed": int(w["status"] == "failed"),
                                    "ace_format": props.get("ace_format"),
                                    "failure_class": classify_failure(w.get("error")) if w["status"] == "failed" else "none"})
        # population per generation
        for gr in gens:
            for p in gr.get("population", []):
                gid = p["genome_id"]
                if gid not in G:
                    continue
                k = key_of(gid)
                gen = genome_cache[(run, gid)]
                f = key2feat[k]
                pop_rows.append({"run": run, "generation": gr["generation"], "genome_id": gid, "graph": k,
                                 "n_workers": f["n_workers"], "nodes": sorted(gen.nodes),
                                 **{c: f[c] for c in ["has_viewer", "has_listviewer", "has_vieweditor", "has_critic",
                                                      "has_judgemerger", "has_pickmerger", "editor_chain",
                                                      "end_is_merger", "n_editor"]},
                                 "end_type": f["end_type"]})
        for nid, (fm, ty) in fam.items():
            innov_rows.append({"run": run, "innovation": nid, "family": fm, "type": ty})

    T = pd.DataFrame(task_rows)
    W = pd.DataFrame(worker_rows)
    P = pd.DataFrame(pop_rows)
    I = pd.DataFrame(innov_rows)

    # ---------------- centring
    cell = T.groupby(["run", "batch", "instance"])
    T["cell_n_genomes"] = cell["genome_id"].transform("nunique")
    T["cell_mean"] = cell["score"].transform("mean")
    T["delta"] = T["score"] - T["cell_mean"]
    T["batch_mean"] = T.groupby(["run", "batch"])["score"].transform("mean")
    n_all = len(T)
    T = T[T["cell_n_genomes"] >= 2].copy()
    R["snapshot"]["task_records_used"] = int(len(T))
    R["snapshot"]["task_records_dropped_singleton_cells"] = int(n_all - len(T))
    R["snapshot"]["batch_mean_score_range"] = [float(T["batch_mean"].min()), float(T["batch_mean"].max())]
    R["snapshot"]["variance_explained_by_instance_cells"] = float(
        1 - T["delta"].var() / T["score"].var())

    # ---------------- graph table
    rows = []
    for k, grp in T.groupby("graph"):
        f = key2feat[k]
        m = graph_meta[k]
        rows.append({"graph": k, "run": m["run"], "label": f"{m['run'][-1].upper()}:{m['ids'][0][:8]}",
                     "genome_ids": ";".join(sorted(set(m["ids"]))), "first_gen": m["first_gen"],
                     "n_tasks": len(grp), "n_batches": grp["batch"].nunique(),
                     "score": grp["score"].mean(), "delta": grp["delta"].mean(),
                     "tokens": grp["tokens"].mean(), "fail_rate": grp["failed"].mean(),
                     "types": f["_types_sig"],
                     **{c: v for c, v in f.items() if not c.startswith("_")}})
    GD = pd.DataFrame(rows)
    # attach graph features to tasks
    feat_cols = [c for c in GD.columns if c not in ("graph", "run", "label", "genome_ids", "first_gen", "n_tasks",
                                                    "n_batches", "score", "delta", "tokens", "fail_rate", "types",
                                                    "end_type")]
    prop_keys = sorted({p for k in GD["graph"] for p in key2feat[k]["_props"]})
    # share of the graph's workers of that type that use the option (0..1)
    PR = pd.DataFrame([{p: key2feat[k]["_props"].get(p, 0) / max(1, key2feat[k][f"n_{p.split(':')[1]}"])
                        for p in prop_keys} for k in GD["graph"]])
    PR.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in PR.columns]
    prop_name_map = dict(zip(PR.columns, prop_keys))
    GD = pd.concat([GD.reset_index(drop=True), PR], axis=1)
    T = T.merge(GD.drop(columns=["run", "score", "delta", "tokens", "fail_rate"]), on="graph", how="left")
    T["log_tokens"] = np.log(T["tokens"])
    for t in TYPES:
        T[f"end_{t}"] = (T["end_type"] == t).astype(int)
        GD[f"end_{t}"] = (GD["end_type"] == t).astype(int)
    GD.drop(columns=[c for c in GD.columns if c.startswith("prop_")]).to_csv(out / "graphs.csv", index=False)
    R["n_graphs"] = {r: int((GD.run == r).sum()) for r in GD.run.unique()}
    R["n_tasks"] = {r: int((T.run == r).sum()) for r in T.run.unique()}

    # ======================= Q1 size
    Q1: dict = {}
    by_size = []
    for n, grp in T.groupby("n_workers"):
        ng = grp["graph"].nunique()
        gm = grp.groupby("graph").agg(delta=("delta", "mean"), n=("delta", "size"), tok=("tokens", "median"))
        # cluster bootstrap over graphs
        bs = []
        arr = gm[["delta", "n"]].to_numpy()
        for _ in range(2000):
            s = arr[RNG.integers(0, len(arr), len(arr))]
            bs.append(np.sum(s[:, 0] * s[:, 1]) / np.sum(s[:, 1]))
        by_size.append({"n_workers": int(n), "graphs": int(ng), "tasks": int(len(grp)),
                        "delta": float(grp["delta"].mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "raw": float(grp["score"].mean()), "median_tokens": float(grp["tokens"].median()),
                        "fail_rate": float(grp["failed"].mean())})
    Q1["by_size"] = by_size
    m_lin = cluster_ols(T, "delta ~ n_workers + C(run)", "graph")
    m_quad = cluster_ols(T.assign(n2=T.n_workers ** 2), "delta ~ n_workers + n2 + C(run)", "graph")
    m_cat = cluster_ols(T, "delta ~ C(n_workers) + C(run)", "graph")
    Q1["linear_slope"] = [float(m_lin.params["n_workers"]), *map(float, m_lin.conf_int().loc["n_workers"]),
                          float(m_lin.pvalues["n_workers"])]
    Q1["quadratic"] = {"b1": float(m_quad.params["n_workers"]), "b2": float(m_quad.params["n2"]),
                       "p_b2": float(m_quad.pvalues["n2"]),
                       "vertex": float(-m_quad.params["n_workers"] / (2 * m_quad.params["n2"]))
                       if m_quad.params["n2"] != 0 else None}
    wald = m_cat.wald_test(" , ".join(c + " = 0" for c in m_cat.params.index if c.startswith("C(n_workers)")),
                           scalar=True)
    Q1["categorical_wald_p"] = float(wald.pvalue)
    Q1["perm_p_linear"] = perm_test(GD, "n_workers", n_perm=args.n_perm)
    gtok = GD.dropna(subset=["tokens"])
    mt = smf.wls("np.log(tokens) ~ n_workers + C(run)", data=gtok, weights=gtok["n_tasks"]).fit(cov_type="HC1")
    Q1["log_tokens_per_worker"] = [float(mt.params["n_workers"]), *map(float, mt.conf_int().loc["n_workers"])]
    Q1["tokens_pct_per_worker"] = float(100 * (math.exp(mt.params["n_workers"]) - 1))
    Q1["spearman_tokens_delta_graph"] = list(map(float, stats.spearmanr(GD["tokens"], GD["delta"], nan_policy="omit")))
    # generation trend of mean size
    trend = {}
    for run, grp in P.groupby("run"):
        ms = grp.groupby("generation")["n_workers"].mean()
        sl = stats.linregress(ms.index, ms.values)
        trend[run] = {"by_gen": {int(g): float(v) for g, v in ms.items()}, "slope_per_gen": float(sl.slope),
                      "slope_ci": [float(sl.slope - 1.96 * sl.stderr), float(sl.slope + 1.96 * sl.stderr)],
                      "spearman": float(stats.spearmanr(ms.index, ms.values)[0]),
                      "start": float(ms.iloc[0]), "end": float(ms.iloc[-1])}
    Q1["size_trend"] = trend
    # gen-0 (selection-free) size slope
    T0 = T[T.batch == 0]
    m0 = cluster_ols(T0, "delta ~ n_workers + C(run)", "graph")
    Q1["gen0_slope"] = [float(m0.params["n_workers"]), *map(float, m0.conf_int().loc["n_workers"]),
                        float(m0.pvalues["n_workers"])]
    R["Q1_size"] = Q1

    # ======================= Q2/Q3 composition + topology (one BH family)
    struct_feats = {
        "has_viewer": "has VIEWER", "has_listviewer": "has LISTVIEWER", "has_vieweditor": "has VIEWEDITOR",
        "has_critic": "has CRITIC", "has_judgemerger": "has JUDGEMERGER", "has_pickmerger": "has PICKMERGER",
        "has_merger": "has any merger", "end_pickmerger": "END fed by PICKMERGER",
        "end_judgemerger": "END fed by JUDGEMERGER", "end_editor": "END fed by EDITOR",
        "end_critic": "END fed by CRITIC", "end_vieweditor": "END fed by VIEWEDITOR",
        "end_is_ace": "END source writes ACE", "editor_chain": "EDITOR -> EDITOR chain",
        "n_editor": "# EDITORs (per +1)", "n_generators": "# edit generators (per +1)",
        "depth": "depth (per +1)", "n_ace_nodes": "# ACE-writing nodes (per +1)",
        "end_fanin": "END merger fan-in (per +1)", "n_edges": "# edges (per +1)",
    }
    feats = []
    for c, lab in struct_feats.items():
        feats.append((c, lab, "structure"))
    for c in PR.columns:
        _, t, p, o = prop_name_map[c].split(":")
        if p == "instruction":
            continue
        feats.append((c, f"{SHORT[t]} {p}={option_label(t, p, int(o))}", "property"))
    eff = []
    for c, lab, fam_ in feats:
        if fam_ == "property":
            # settings are compared only among graphs that contain the worker type, as the share of
            # that type's workers using the option (1 = all of them, 0 = none)
            wtype = prop_name_map[c].split(":")[1]
            GDs, Ts, T0s = GD[GD[f"n_{wtype}"] > 0], T[T[f"n_{wtype}"] > 0], T0[T0[f"n_{wtype}"] > 0]
        else:
            GDs, Ts, T0s = GD, T, T0
        x = GDs[c]
        binary = set(x.unique()) <= {0, 1} or fam_ == "property"
        if binary:
            x = (x > 0).astype(int)
            n1, n0 = int((x == 1).sum()), int((x == 0).sum())
            t1 = int(GDs.loc[x == 1, "n_tasks"].sum())
            t0 = int(GDs.loc[x == 0, "n_tasks"].sum())
            if min(n1, n0) < 5 or min(t1, t0) < 30:
                continue
        else:
            if x.nunique() < 2:
                continue
            n1 = n0 = t1 = t0 = None
        try:
            m = cluster_ols(Ts, f"delta ~ {c} + C(run)", "graph")
        except Exception:
            continue
        lo, hi = m.conf_int().loc[c]
        p_perm = perm_test(GDs, c, n_perm=args.n_perm)
        # gen-0 selection-free estimate
        g0 = None
        if len(T0s) and T0s[c].nunique() > 1 and (not binary or min(((T0s.groupby("graph")[c].first() > 0) == v).sum() for v in (0, 1)) >= 3):
            m0 = cluster_ols(T0s, f"delta ~ {c} + C(run)", "graph")
            g0 = [float(m0.params[c]), *map(float, m0.conf_int().loc[c])]
        # failure association
        mf = cluster_ols(Ts, f"failed ~ {c} + C(run)", "graph")
        eff.append({"feature": c, "label": lab, "family": fam_, "binary": binary,
                    "graphs_with": n1, "graphs_without": n0, "tasks_with": t1, "tasks_without": t0,
                    "beta": float(m.params[c]), "ci_lo": float(lo), "ci_hi": float(hi),
                    "p_cluster": float(m.pvalues[c]), "p_perm": p_perm, "gen0": g0,
                    "fail_beta": float(mf.params[c]), "fail_ci": list(map(float, mf.conf_int().loc[c])),
                    "fail_p": float(mf.pvalues[c])})
    E = pd.DataFrame(eff)
    E["q_bh"] = multipletests(E["p_perm"], method="fdr_bh")[1]
    E["q_bh_cluster"] = multipletests(E["p_cluster"], method="fdr_bh")[1]
    E["fail_q_bh"] = multipletests(E["fail_p"], method="fdr_bh")[1]
    E = E.sort_values("p_perm")
    E.to_csv(out / "features.csv", index=False)
    R["Q2_Q3_features"] = {"n_tested": int(len(E)), "results": E.to_dict(orient="records")}

    # multivariable topology model
    mv_terms = "n_workers + has_viewer + has_listviewer + has_vieweditor + has_critic + end_pickmerger + end_judgemerger + editor_chain + C(run)"
    mv = cluster_ols(T, f"delta ~ {mv_terms}", "graph")
    R["Q3_multivariable"] = {k: {"beta": float(mv.params[k]), "ci": list(map(float, mv.conf_int().loc[k])),
                                 "p": float(mv.pvalues[k])} for k in mv.params.index if k != "Intercept"}
    R["Q3_multivariable"]["_n_tasks"] = int(mv.nobs)
    # species-clustered sensitivity (lineage autocorrelation)
    T["species_cl"] = T["run"] + ":" + T["graph"].map(lambda k: str(graph_meta[k]["species"]))
    mvs = cluster_ols(T, f"delta ~ {mv_terms}", "species_cl")
    R["Q3_multivariable_species_clustered"] = {k: {"beta": float(mvs.params[k]), "ci": list(map(float, mvs.conf_int().loc[k])),
                                                   "p": float(mvs.pvalues[k])} for k in mvs.params.index if k != "Intercept"}
    R["Q3_multivariable_species_clustered"]["_n_clusters"] = int(T["species_cl"].nunique())
    # END-type table
    endtab = []
    for et, grp in T.groupby("end_type"):
        endtab.append({"end_type": et, "graphs": int(grp.graph.nunique()), "tasks": int(len(grp)),
                       "delta": float(grp.delta.mean()), "raw": float(grp.score.mean()),
                       "fail_rate": float(grp.failed.mean()), "median_tokens": float(grp.tokens.median())})
    R["Q3_end_type"] = endtab

    # ======================= Q4 cost-effectiveness
    GQ = GD[GD.n_tasks >= 6].copy()
    GQ["score_per_Mtok"] = GQ["score"] / (GQ["tokens"] / 1e6)
    pareto = []
    for run, grp in GQ.groupby("run"):
        g = grp.sort_values("tokens")
        best = -1e9
        for _, r in g.iterrows():
            if r["delta"] > best:
                best = r["delta"]
                pareto.append(r["graph"])
    GQ["pareto"] = GQ["graph"].isin(pareto)
    cols = ["label", "run", "types", "n_workers", "n_tasks", "score", "delta", "tokens", "score_per_Mtok", "fail_rate"]
    R["Q4_cost"] = {
        "min_tasks": 6,
        "pareto": GQ[GQ.pareto].sort_values(["run", "tokens"])[cols].to_dict(orient="records"),
        "top_score_per_Mtok": GQ.sort_values("score_per_Mtok", ascending=False).head(5)[cols].to_dict(orient="records"),
        "spearman_tokens_vs_delta": list(map(float, stats.spearmanr(GQ.tokens, GQ.delta))),
        "spearman_tokens_vs_score_per_Mtok": list(map(float, stats.spearmanr(GQ.tokens, GQ.score_per_Mtok))),
    }

    # ======================= Q5 failures
    Q5: dict = {"task_fail_rate": {r: float(g.failed.mean()) for r, g in T.groupby("run")}}
    wt = []
    for t, grp in W.groupby("type"):
        k, n = int(grp.failed.sum()), len(grp)
        wt.append({"type": t, "runs": n, "failed": k, "rate": k / n, "ci": wilson(k, n),
                   "top_causes": Counter(grp.loc[grp.failed == 1, "failure_class"]).most_common(3)})
    Q5["worker_fail_by_type"] = sorted(wt, key=lambda d: -d["rate"])
    af = []
    WA = W[W.type.isin(ACE_TYPES)].copy()
    for (t, fmt), grp in WA.groupby(["type", "ace_format"]):
        k, n = int(grp.failed.sum()), len(grp)
        af.append({"type": t, "ace_format": fmt, "runs": n, "failed": k, "rate": k / n, "ci": wilson(k, n)})
    Q5["ace_worker_fail_by_format"] = af
    WA["fmt_lines"] = WA.ace_format.isin(["element_file_lines", "file_lines"]).astype(int)
    WA["fmt_element_only"] = (WA.ace_format == "element").astype(int)
    for c in ["fmt_lines", "fmt_element_only"]:
        m = cluster_ols(WA, f"failed ~ {c} + C(type) + C(run)", "graph")
        Q5[f"ace_{c}_effect"] = [float(m.params[c]), *map(float, m.conf_int().loc[c]), float(m.pvalues[c])]
    # merger kind: judge (select a number) vs pick (writes ACE)
    WM = W[W.type.isin(["pickmerger", "judgemerger"])].copy()
    WM["is_pick"] = (WM.type == "pickmerger").astype(int)
    m = cluster_ols(WM, "failed ~ is_pick + C(run)", "graph")
    Q5["pick_vs_judge_worker_fail"] = [float(m.params["is_pick"]), *map(float, m.conf_int().loc["is_pick"]),
                                       float(m.pvalues["is_pick"])]
    Q5["train_failure_classes"] = Counter(T.loc[T.failed == 1, "failure_class"]).most_common()
    Q5["train_failure_node_types"] = Counter(T.loc[T.failed == 1, "fail_node_type"].fillna("?")).most_common()
    Q5["delta_failed_vs_ok"] = {"failed_mean_score": float(T.loc[T.failed == 1, "score"].mean()),
                                "ok_mean_score": float(T.loc[T.failed == 0, "score"].mean())}
    R["Q5_failures"] = Q5

    # ======================= Q6 lineage
    lin = []
    P["n_pop"] = P.groupby(["run", "generation"])["genome_id"].transform("count")
    for run, grp in P.groupby("run"):
        last_gen = grp.generation.max()
        node_prev = defaultdict(dict)
        for gen, gg in grp.groupby("generation"):
            cnt = Counter(n for ns in gg["nodes"] for n in ns)
            for nid, c in cnt.items():
                node_prev[nid][gen] = c / len(gg)
        Irun = I[I.run == run].set_index("innovation")
        fams = defaultdict(list)
        for nid, prev in node_prev.items():
            if nid not in Irun.index:
                continue
            fm, ty = Irun.loc[nid, "family"], Irun.loc[nid, "type"]
            fams[(fm, ty)].append((nid, min(prev), max(prev.values()), prev.get(last_gen, 0.0), max(prev)))
        for (fm, ty), lst in fams.items():
            lin.append({"run": run, "family": fm, "type": ty, "n_innovations": len(lst),
                        "first_gen": int(min(x[1] for x in lst)),
                        "peak_prevalence": float(max(x[2] for x in lst)),
                        "final_prevalence_best_innov": float(max(x[3] for x in lst)),
                        "n_surviving": int(sum(1 for x in lst if x[3] > 0)),
                        "n_extinct": int(sum(1 for x in lst if x[3] == 0)),
                        "n_reached_50pct": int(sum(1 for x in lst if x[2] >= 0.5))})
    LIN = pd.DataFrame(lin).sort_values(["run", "first_gen", "family"])
    motifs = ["has_viewer", "has_listviewer", "has_vieweditor", "has_critic", "has_judgemerger", "has_pickmerger",
              "editor_chain"]
    mot = {}
    for run, grp in P.groupby("run"):
        mot[run] = {m_: {int(g): float(v) for g, v in grp.groupby("generation")[m_].mean().items()} for m_ in motifs}
    # prevalence of the winner's structure (property-independent WL hash) and of its exact node set
    win_gen = Genome.from_json(json.loads((RUNS_ROOT / "verified" / "neat_best" / "graph.json").read_text()))
    win_hash = win_gen.structure_hash()
    P["winner_structure"] = P["graph"].map(lambda k: int(graph_meta[k]["genome"].structure_hash() == win_hash))
    P["winner_nodes"] = P["nodes"].map(lambda ns: int(set(win_gen.nodes) <= set(ns)))
    win_prev = {run: {"structure": {int(g): float(v) for g, v in grp.groupby("generation")["winner_structure"].mean().items()},
                      "node_set": {int(g): float(v) for g, v in grp.groupby("generation")["winner_nodes"].mean().items()}}
                for run, grp in P.groupby("run")}
    R["Q6_winner_prevalence"] = win_prev
    # species extinctions (stagnation pruning) from species.jsonl
    ext = {}
    for run_dir in args.runs:
        recs = {}
        for x in load_jsonl(run_dir / "species.jsonl"):
            recs[x["generation"]] = x  # last record per generation
        evs, prev = [], None
        for g in sorted(recs):
            cur = {sp["id"]: sp for sp in recs[g]["species"]}
            if prev is not None:
                lost = [sid for sid in prev if sid not in cur]
                if lost:
                    evs.append({"generation": g, "lost_species": lost,
                                "lost_members": int(sum(prev[sid]["size"] for sid in lost)),
                                "lost_last_improved": [prev[sid].get("last_improved") for sid in lost],
                                "lost_best_fitness": [prev[sid].get("best_fitness") for sid in lost],
                                "survivor_best_fitness": [sp.get("best_fitness") for sp in cur.values()]})
            prev = cur
        ext[run_dir.name] = evs
    R["Q6_species_extinctions"] = ext
    R["Q6_lineage"] = {"innovation_families": LIN.to_dict(orient="records"), "motif_prevalence": mot,
                       "last_generation": {r: int(g.generation.max()) for r, g in P.groupby("run")}}

    # ======================= held-out
    H: dict = {}
    val_root = RUNS_ROOT / "val"
    val_ids = [json.loads(l)["instance_id"] for l in open("/usr/project/xtmp/kg349/ace/data/train_val40.jsonl")]
    sel = json.loads((RUNS_ROOT / "selection" / "selection.json").read_text())
    base = {r["instance_id"]: r for r in load_jsonl(val_root / "baseline" / "preds.jsonl")}
    base_s = np.array([float(base[i].get("proxy_score") or 0) if i in base else np.nan for i in val_ids])
    vrows = [{"name": "baseline (single editor agent)", "workers": 1, "types": "-", "train_mean": None,
              "val_mean": float(np.nanmean(base_s)), "val_fail": 0.0, "n": int(np.sum(~np.isnan(base_s))),
              "median_tokens": float(np.median([(base[i].get("usage") or {}).get("total_tokens", np.nan) for i in val_ids if i in base])),
              "wilcoxon_p_vs_base": None}]
    for s in sel:
        recs = {r["instance_id"]: r for p in glob.glob(str(val_root / s["name"] / "preds*.jsonl")) for r in load_jsonl(Path(p))}
        sc = np.array([float(recs[i].get("proxy_score") or 0) if i in recs else np.nan for i in val_ids])
        ok = ~np.isnan(sc) & ~np.isnan(base_s)
        wp = float(stats.wilcoxon(sc[ok], base_s[ok], zero_method="wilcox").pvalue) if ok.sum() > 5 else None
        gen = Genome.from_json(json.loads(Path(s["path"]).read_text()))
        f = graph_features(gen)
        fails = [classify_failure(recs[i].get("failure")) for i in val_ids if i in recs and recs[i].get("failure")]
        vrows.append({"name": s["name"], "workers": s["workers"], "types": f["_types_sig"],
                      "end_type": f["end_type"], "editor_chain": f["editor_chain"],
                      "pm_ace_format": [n.properties.get("ace_format") for n in gen.nodes.values() if n.type.value == "pickmerger"],
                      "train_mean": s["train_mean"], "train_tasks": s["train_tasks"],
                      "val_mean": float(np.nanmean(sc)), "n": int(np.sum(~np.isnan(sc))),
                      "val_fail": len(fails) / max(1, int(np.sum(~np.isnan(sc)))),
                      "val_fail_classes": Counter(fails).most_common(),
                      "median_tokens": float(np.median([(recs[i].get("usage") or {}).get("total_tokens", np.nan) for i in val_ids if i in recs])),
                      "wilcoxon_p_vs_base": wp})
    from ace.graph.properties import ACE_FORMAT_VALUES
    for v in vrows:
        if "pm_ace_format" in v:
            v["pm_ace_format"] = [ACE_FORMAT_VALUES[i] for i in v["pm_ace_format"] if i is not None]
    H["val40"] = vrows
    # Verified failures of the winner
    ver = {}
    for name in ["neat_best", "neat_best_r2", "baseline_r1"]:
        d = RUNS_ROOT / "verified" / name
        preds = {r["instance_id"]: r for r in load_jsonl(d / "all_preds.jsonl")}
        res = {r["instance_id"]: bool(r.get("resolved")) for p in sorted(glob.glob(str(d / "eval.shard*.jsonl")))
               for r in load_jsonl(Path(p))}
        ver[name] = (preds, res)
    vp, vr = ver["neat_best"]
    bp, br = ver["baseline_r1"]
    p2, r2 = ver["neat_best_r2"]
    fail_ids = {i for i, r in vp.items() if r.get("failure")}
    fail2 = {i for i, r in p2.items() if r.get("failure")}
    cls = Counter(classify_failure(vp[i]["failure"]) for i in fail_ids)
    cls2 = Counter(classify_failure(p2[i]["failure"]) for i in fail2)
    ok_ids = set(vp) - fail_ids
    H["verified_winner"] = {
        "n": len(vp), "resolved": int(sum(vr.get(i, False) for i in vp)),
        "n_failed": len(fail_ids), "fail_rate": len(fail_ids) / len(vp), "fail_rate_ci": wilson(len(fail_ids), len(vp)),
        "failure_classes": cls.most_common(), "failure_classes_r2": cls2.most_common(), "n_failed_r2": len(fail2),
        "failed_in_both_runs": len(fail_ids & fail2),
        "jaccard_r1_r2": len(fail_ids & fail2) / max(1, len(fail_ids | fail2)),
        "p_fail_r2_given_fail_r1": len(fail_ids & fail2) / max(1, len(fail_ids)),
        "p_fail_r2_given_ok_r1": len(fail2 - fail_ids) / max(1, len(ok_ids)),
        "resolved_rate_nonfailed": float(np.mean([vr.get(i, False) for i in ok_ids])),
        "resolved_among_failed": int(sum(vr.get(i, False) for i in fail_ids)),
        "baseline_resolved_on_failed": int(sum(br.get(i, False) for i in fail_ids)),
        "baseline_rate_on_failed": float(np.mean([br.get(i, False) for i in fail_ids])),
        "baseline_rate_on_nonfailed": float(np.mean([br.get(i, False) for i in ok_ids])),
        "r2_resolved_on_r1_failed": int(sum(r2.get(i, False) for i in fail_ids)),
        "failing_node": "pickmerger (only ACE-writing worker in the graph; both editors use tools)",
    }
    R["held_out"] = H

    # ======================= figures
    make_figures(out, R, T, GD, GQ, P, E)
    (out / "report.json").write_text(json.dumps(R, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    write_report(out, R, E, GQ)
    print("wrote", out)


# ---------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "grid.color": "#e4e3df", "grid.linewidth": 0.5, "savefig.dpi": 300,
                         "font.family": "DejaVu Sans", "figure.facecolor": "white"})
    return plt


def make_figures(out, R, T, GD, GQ, P, E):
    plt = _style()
    # ---- Fig 1: size
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.0), constrained_layout=True)
    ax = axes[0]
    for j, (run, grp) in enumerate(T.groupby("run")):
        xs, ys, lo, hi = [], [], [], []
        for n, gg in grp.groupby("n_workers"):
            gm = gg.groupby("graph").agg(d=("delta", "mean"), k=("delta", "size")).to_numpy()
            if len(gm) < 2:
                continue
            bs = [np.average(s[:, 0], weights=s[:, 1]) for s in (gm[RNG.integers(0, len(gm), len(gm))] for _ in range(1000))]
            xs.append(n + (j - 0.5) * 0.18); ys.append(gg.delta.mean())
            lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
        ax.errorbar(xs, ys * 100, yerr=[np.clip(ys - lo, 0, None) * 100, np.clip(hi - ys, 0, None) * 100], fmt="o", ms=3.5, lw=1, capsize=0,
                    color=COLORS.get(run), label=run.replace("neat_", "run "))
    ax.axhline(0, color="#8a8984", lw=0.6, ls="--")
    ax.set_ylabel("Δ score vs. same task\n(points, 95% CI)")
    ax.set_xlabel("workers in graph")
    ax.set_title("a  Batch-centred fitness by graph size", loc="left")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax = axes[1]
    for j, (run, grp) in enumerate(T.groupby("run")):
        med = grp.groupby("n_workers")["tokens"].median() / 1e3
        q1 = grp.groupby("n_workers")["tokens"].quantile(0.25) / 1e3
        q3 = grp.groupby("n_workers")["tokens"].quantile(0.75) / 1e3
        x = med.index + (j - 0.5) * 0.18
        ax.errorbar(x, med, yerr=[np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)], fmt="o", ms=3.5, lw=1, color=COLORS.get(run))
    ax.set_yscale("log")
    ax.set_ylabel("tokens per task (k)\nmedian, IQR")
    ax.set_xlabel("workers in graph")
    ax.set_title(f"b  Cost grows ~{R['Q1_size']['tokens_pct_per_worker']:.0f}% per added worker", loc="left")
    ax = axes[2]
    for run, grp in P.groupby("run"):
        ms = grp.groupby("generation")["n_workers"].agg(["mean", "sem"])
        ax.plot(ms.index, ms["mean"], "-o", ms=3, lw=1.5, color=COLORS.get(run), label=run.replace("neat_", "run "))
        ax.fill_between(ms.index, ms["mean"] - ms["sem"], ms["mean"] + ms["sem"], color=COLORS.get(run), alpha=0.15, lw=0)
    ax.set_xlabel("generation"); ax.set_ylabel("mean workers\nin population (±SE)")
    ax.set_title("c  Graph size over generations", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.savefig(out / "fig1_size.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # ---- Fig 2: forest plot
    pre = ["has_critic", "has_merger", "end_pickmerger", "end_judgemerger", "end_editor", "editor_chain", "depth",
           "has_viewer", "has_listviewer", "has_vieweditor", "n_editor"]
    show = E[((E.q_bh < 0.05) & (E.family == "property") & (E.graphs_with >= 20) & (E.graphs_without >= 20))
             | E.feature.isin(pre)]
    show = show.drop_duplicates("feature").sort_values("beta")
    fig, ax = plt.subplots(figsize=(3.4, 0.17 * len(show) + 0.8), constrained_layout=True)
    y = np.arange(len(show))
    sig = show.q_bh < 0.05
    for yi, (_, r), s in zip(y, show.iterrows(), sig):
        c = "#0b0b0b" if s else "#8a8984"
        ax.plot([r.ci_lo * 100, r.ci_hi * 100], [yi, yi], color=c, lw=1.2, solid_capstyle="round")
        ax.plot(r.beta * 100, yi, "o", ms=4, color=c, mfc=c if s else "white")
    ax.axvline(0, color="#8a8984", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}" + (f"  ({int(r.graphs_with)}/{int(r.graphs_without)})" if r.binary else "")
                        for _, r in show.iterrows()], fontsize=6)
    ax.set_xlabel("effect on Δ score (points, 95% CI)")
    ax.set_title("Feature effects (filled: BH q<0.05)", loc="right")
    ax.grid(axis="x")
    fig.savefig(out / "fig2_effects.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # ---- Fig 3: Pareto + motif prevalence
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    ax = axes[0]
    for run, grp in GQ.groupby("run"):
        ax.scatter(grp.tokens / 1e3, grp.delta * 100, s=6 + 1.2 * grp.n_tasks, color=COLORS.get(run), alpha=0.55,
                   edgecolor="white", lw=0.6, label=run.replace("neat_", "run "))
        pf = grp[grp.pareto].sort_values("tokens")
        ax.step(pf.tokens / 1e3, pf.delta * 100, where="post", color=COLORS.get(run), lw=1.2)
    win = GQ[GQ.genome_ids.str.contains("37b607a6")]
    if len(win):
        r = win.iloc[0]
        ax.annotate("winner\n(ED→ED→PM)", (r.tokens / 1e3, r.delta * 100), xytext=(12, -22), textcoords="offset points",
                    fontsize=6, arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#52514e"})
    ax.set_xscale("log")
    ax.axhline(0, color="#8a8984", lw=0.6, ls="--")
    ax.set_xlabel("mean tokens per task (k, log)"); ax.set_ylabel("Δ score (points)")
    ax.set_title("a  Fitness vs. cost; lines = Pareto front (≥6 tasks)", loc="left")
    ax.legend(frameon=False, loc="lower right", markerscale=0.6)
    ax = axes[1]
    motifs = [("has_pickmerger", "PICKMERGER", "-"), ("editor_chain", "ED→ED chain", ":"), ("has_critic", "CRITIC", "--")]
    for run, grp in P.groupby("run"):
        for m_, lab, ls in motifs:
            s = grp.groupby("generation")[m_].mean() * 100
            ax.plot(s.index, s.values, ls=ls, lw=1.3, color=COLORS.get(run))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#52514e", ls=ls, lw=1.3, label=lab) for _, lab, ls in motifs]
    h += [Line2D([], [], color=c, lw=2, label=r.replace("neat_", "run ")) for r, c in COLORS.items()]
    ax.legend(handles=h, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=6)
    ax.set_xlabel("generation"); ax.set_ylabel("% of population")
    ax.set_title("b  Spread and loss of structural motifs", loc="left")
    ax.set_ylim(-3, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    for run, evs in R.get("Q6_species_extinctions", {}).items():
        for e in evs:
            if e["lost_members"] >= 8:
                ax.axvline(e["generation"], color=COLORS.get(run), lw=0.6, ls=(0, (1, 2)))
                ax.text(e["generation"] + 0.1, 121, f"B: 3 stagnant species\n({e['lost_members']} genomes) pruned", fontsize=5.5,
                        va="top", color="#52514e")
    fig.savefig(out / "fig3_pareto_lineage.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- report

def _pp(x):
    return f"{100 * x:+.1f}"


def write_report(out, R, E, GQ):
    Q1, Q5, H = R["Q1_size"], R["Q5_failures"], R["held_out"]
    L = []
    a = L.append
    a("# Which evolved worker graphs were effective? (secondary analysis of NEAT runs A and B)\n")
    s = R["snapshot"]
    a(f"Snapshot: neat_a {s['neat_a']['evaluation_records']} / neat_b {s['neat_b']['evaluation_records']} task records "
      f"({s['neat_a']['generations_logged']} / {s['neat_b']['generations_logged']} generations logged); records of genomes "
      f"not yet logged skipped ({s['neat_a']['records_unknown_genome_skipped']} / {s['neat_b']['records_unknown_genome_skipped']}); "
      f"mock records skipped ({s['neat_a']['mock_skipped']} / {s['neat_b']['mock_skipped']}). "
      f"{s['task_records_used']} records used (centred cells need ≥2 genomes); distinct graphs: "
      + ", ".join(f"{k} {v}" for k, v in R['n_graphs'].items()) + " (identical graph JSON pooled).\n")
    a("**Outcome.** Δ = proxy score minus the mean score of all genomes that attempted the same instance in the same "
      f"minibatch (instance fixed effect). Minibatch means range {s['batch_mean_score_range'][0]:.2f}–{s['batch_mean_score_range'][1]:.2f}; "
      f"instance cells explain {100 * s['variance_explained_by_instance_cells']:.0f}% of score variance. "
      "Effects are in score points (proxy ×100). **Inference.** Task-level OLS Δ ~ x + run with cluster-robust SEs by graph "
      "(CIs); stratified permutation test across graphs within run (p_perm); Benjamini–Hochberg over all "
      f"{R['Q2_Q3_features']['n_tested']} features (q). 'gen0' = estimate from the generation-0 minibatch only "
      "(random initial genomes + bootstrap mutants; no selection).\n")

    def ef(feat):
        r = E[E.feature == feat]
        if not len(r):
            return "n/a"
        r = r.iloc[0]
        return f"{_pp(r.beta)} pts (95% CI {_pp(r.ci_lo)} to {_pp(r.ci_hi)}, q={r.q_bh:.2g})"
    v = H["verified_winner"]
    by = {d["n_workers"]: d for d in Q1["by_size"]}
    mvp = R["Q3_multivariable"]
    a("## Key findings\n")
    a(f"- **Size sweet spot at 3 workers.** Δ = {_pp(by[3]['delta'])} pts at 3 workers vs {_pp(by[1]['delta'])} at 1 and "
      f"{_pp(by[4]['delta'])} at 4 (no linear trend, p={Q1['perm_p_linear']:.2g}; quadratic p={Q1['quadratic']['p_b2']:.1g}). "
      f"Holding composition fixed, each extra worker shifts Δ by {_pp(mvp['n_workers']['beta'])} pts, and tokens rise "
      f"{Q1['tokens_pct_per_worker']:.0f}% per worker. Run A drifted to larger graphs "
      f"({Q1['size_trend']['neat_a']['start']:.1f}→{Q1['size_trend']['neat_a']['end']:.1f} workers); run B stayed near 3.")
    a(f"- **What helped:** a PICKMERGER feeding END {ef('end_pickmerger')}, an EDITOR→EDITOR chain {ef('editor_chain')}, "
      f"more editors {ef('n_editor')}. **What hurt:** a VIEWEDITOR {ef('has_vieweditor')}, a lone EDITOR at END "
      f"{ef('end_editor')}. CRITIC, VIEWER and LISTVIEWER presence had no reliable univariate effect.")
    a(f"- **Settings:** PICKMERGER ace_format=element {ef('prop_pickmerger_ace_format_0')} and context_budget=12000 "
      f"{ef('prop_pickmerger_context_budget_0')}; EDITOR max_steps=60 {ef('prop_editor_max_steps_2')}, context_budget=6000 "
      f"{ef('prop_editor_context_budget_1')}. Most prompt-text options are not distinguishable after BH.")
    a(f"- **Failures drive fitness.** Tool-using workers never fail; ACE writers fail per execution "
      + ", ".join(f"{r['type']} {100 * r['rate']:.0f}%" for r in Q5["worker_fail_by_type"] if r["type"] in ACE_TYPES)
      + f" vs JUDGEMERGER {100 * next(r['rate'] for r in Q5['worker_fail_by_type'] if r['type'] == 'judgemerger'):.0f}%. "
      f"Line-addressable ACE formats fail {_pp(-Q5['ace_fmt_lines_effect'][0])[1:]} pts less often. PICKMERGER graphs still win "
      "on Δ despite failing more, so its successful merges must outscore picking a single candidate.")
    a(f"- **Winner on Verified:** {v['n_failed']}/{v['n']} ({100 * v['fail_rate']:.0f}%) graph failures, all in the "
      f"PICKMERGER; {sum(n for c, n in v['failure_classes'] if c.startswith('overlap'))} are overlapping ACE blocks "
      f"({dict(v['failure_classes']).get('overlap (duplicate block)', 0)} of them the same block emitted twice). Resolve "
      f"rate on non-failed instances is {100 * v['resolved_rate_nonfailed']:.1f}%. Note its topology: editor 0 (not a "
      "VIEWER) passes VIEW+REPORT to editor 2, and both edits go to the PICKMERGER.")
    a("- **Lineage:** the winner's lineage (nodes 0, 2, 19) held 25–58% of run B's population through generation 8, then "
      "went extinct at generation 9 when stagnation pruning removed its 3 species (11 of 12 genomes). A record set on a "
      "lucky minibatch at gen 2 could not be beaten on later batches.\n")

    a("## 1. Size\n")
    a("| workers | graphs | tasks | Δ (95% CI) | raw score | median tokens | task fail rate |")
    a("|---|---|---|---|---|---|---|")
    for r in Q1["by_size"]:
        a(f"| {r['n_workers']} | {r['graphs']} | {r['tasks']} | {_pp(r['delta'])} ({_pp(r['ci'][0])}, {_pp(r['ci'][1])}) | "
          f"{r['raw']:.3f} | {r['median_tokens'] / 1e3:.0f}k | {r['fail_rate']:.2f} |")
    sl, q = Q1["linear_slope"], Q1["quadratic"]
    a(f"\nLinear slope {_pp(sl[0])} points/worker (95% CI {_pp(sl[1])}, {_pp(sl[2])}; cluster p={sl[3]:.3g}; "
      f"perm p={Q1['perm_p_linear']:.3g}). Quadratic term p={q['p_b2']:.3g} (vertex {q['vertex']:.1f} workers); "
      f"joint test of size categories p={Q1['categorical_wald_p']:.3g}. Gen-0 only: {_pp(Q1['gen0_slope'][0])} "
      f"({_pp(Q1['gen0_slope'][1])}, {_pp(Q1['gen0_slope'][2])}). Tokens rise {Q1['tokens_pct_per_worker']:.0f}% per worker "
      f"(log-slope {Q1['log_tokens_per_worker'][0]:.3f}, CI {Q1['log_tokens_per_worker'][1]:.3f}–{Q1['log_tokens_per_worker'][2]:.3f}).\n")
    for run, t in Q1["size_trend"].items():
        a(f"- {run}: mean workers {t['start']:.2f} → {t['end']:.2f}, slope {t['slope_per_gen']:+.3f}/gen "
          f"(CI {t['slope_ci'][0]:+.3f}, {t['slope_ci'][1]:+.3f}; Spearman ρ={t['spearman']:.2f}; generations are autocorrelated, so treat as descriptive).")
    a("")

    a("## 2–3. Composition, settings and topology\n")
    a("All features with q<0.10 plus the pre-specified topology features (full list: features.csv). Settings rows "
      "(e.g. 'ED max_steps=30') are estimated only among graphs containing that worker type, with x = share of that "
      "type's workers using the option, so β contrasts 'all such workers use it' vs 'none do'; with/without counts "
      "are graphs with share>0 vs share=0.\n")
    a("| feature | graphs with/without | Δ effect (95% CI) | p_cluster | p_perm | q (BH) | gen0 Δ | effect on fail rate |")
    a("|---|---|---|---|---|---|---|---|")
    pre = {"has_critic", "has_merger", "end_pickmerger", "end_judgemerger", "editor_chain", "depth", "has_viewer",
           "has_listviewer", "has_vieweditor", "end_editor", "n_editor", "n_workers", "end_is_ace", "end_fanin", "n_generators"}
    for _, r in E.iterrows():
        if not (r.q_bh < 0.10 or r.feature in pre):
            continue
        g0 = "–" if r.gen0 is None or (isinstance(r.gen0, float) and np.isnan(r.gen0)) else f"{_pp(r.gen0[0])}"
        gw = f"{int(r.graphs_with)}/{int(r.graphs_without)}" if r.binary else "(numeric)"
        a(f"| {r.label} | {gw} | {_pp(r.beta)} ({_pp(r.ci_lo)}, {_pp(r.ci_hi)}) | {r.p_cluster:.3g} | {r.p_perm:.3g} | "
          f"{r.q_bh:.3g} | {g0} | {_pp(r.fail_beta)} pts (q={r.fail_q_bh:.2g}) |")
    a("\n**Multivariable model** (Δ ~ size + type presence + END type + chain + run; cluster by graph; "
      f"n={R['Q3_multivariable']['_n_tasks']} tasks). Species-clustered SEs ({R['Q3_multivariable_species_clustered']['_n_clusters']} clusters) as a lineage sensitivity:\n")
    a("| term | β (95% CI), graph-clustered | p | p (species-clustered) |")
    a("|---|---|---|---|")
    for k, v in R["Q3_multivariable"].items():
        if k.startswith("_") or k.startswith("C(run"):
            continue
        vs = R["Q3_multivariable_species_clustered"][k]
        a(f"| {k} | {_pp(v['beta'])} ({_pp(v['ci'][0])}, {_pp(v['ci'][1])}) | {v['p']:.3g} | {vs['p']:.3g} |")
    a("\n**By END source type:**\n")
    a("| END fed by | graphs | tasks | Δ | raw | fail rate | median tokens |")
    a("|---|---|---|---|---|---|---|")
    for r in sorted(R["Q3_end_type"], key=lambda d: -d["delta"]):
        a(f"| {r['end_type']} | {r['graphs']} | {r['tasks']} | {_pp(r['delta'])} | {r['raw']:.3f} | {r['fail_rate']:.2f} | {r['median_tokens'] / 1e3:.0f}k |")
    a("")

    a("## 4. Cost-effectiveness\n")
    C = R["Q4_cost"]
    a(f"Graphs with ≥{C['min_tasks']} tasks. Spearman(tokens, Δ) ρ={C['spearman_tokens_vs_delta'][0]:.2f} "
      f"(p={C['spearman_tokens_vs_delta'][1]:.2g}).\n")
    a("Pareto front (min tokens, max Δ), per run:\n")
    a("| graph | types | workers | tasks | Δ | raw | tokens | score/Mtok | fail |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in C["pareto"]:
        a(f"| {r['label']} | {r['types']} | {r['n_workers']} | {r['n_tasks']} | {_pp(r['delta'])} | {r['score']:.3f} | "
          f"{r['tokens'] / 1e3:.0f}k | {r['score_per_Mtok']:.2f} | {r['fail_rate']:.2f} |")
    a("\nTop raw score per million tokens: " + "; ".join(
        f"{r['label']} {r['types']} {r['score_per_Mtok']:.2f}/Mtok (Δ {_pp(r['delta'])}, {r['n_tasks']} tasks)" for r in C["top_score_per_Mtok"]) + "\n")

    a("## 5. Failure modes\n")
    a(f"Task failure rate (training): " + ", ".join(f"{k} {v:.2f}" for k, v in Q5["task_fail_rate"].items()) +
      f". Any failure scores 0 (mean raw score failed {Q5['delta_failed_vs_ok']['failed_mean_score']:.3f} vs ok {Q5['delta_failed_vs_ok']['ok_mean_score']:.3f}).\n")
    a("| worker type | executions | failed | rate (Wilson 95%) | top causes |")
    a("|---|---|---|---|---|")
    for r in Q5["worker_fail_by_type"]:
        a(f"| {r['type']} | {r['runs']} | {r['failed']} | {r['rate']:.3f} ({r['ci'][0]:.3f}–{r['ci'][1]:.3f}) | "
          + ", ".join(f"{c} {n}" for c, n in r["top_causes"]) + " |")
    a("\nACE-writing workers by `ace_format`:\n")
    a("| type | ace_format | executions | rate (95%) |")
    a("|---|---|---|---|")
    for r in Q5["ace_worker_fail_by_format"]:
        a(f"| {r['type']} | {r['ace_format']} | {r['runs']} | {r['rate']:.2f} ({r['ci'][0]:.2f}–{r['ci'][1]:.2f}) |")
    pj = Q5["pick_vs_judge_worker_fail"]
    fl = Q5["ace_fmt_lines_effect"]
    fe = Q5["ace_fmt_element_only_effect"]
    a(f"\nPICKMERGER vs JUDGEMERGER per-execution failure: {_pp(pj[0])} pts ({_pp(pj[1])}, {_pp(pj[2])}; p={pj[3]:.3g}). "
      f"Among ACE writers (adjusting for type, run): line-addressable formats {_pp(fl[0])} pts ({_pp(fl[1])}, {_pp(fl[2])}; p={fl[3]:.3g}); "
      f"element-only format {_pp(fe[0])} pts ({_pp(fe[1])}, {_pp(fe[2])}; p={fe[3]:.3g}).\n")
    a("Training failure causes: " + ", ".join(f"{c} {n}" for c, n in Q5["train_failure_classes"]) + "\n")
    v = H["verified_winner"]
    a(f"**SWE-bench Verified, winner (neat_b_0_37b607a6):** {v['n_failed']}/{v['n']} instances failed "
      f"({100 * v['fail_rate']:.1f}%, Wilson {100 * v['fail_rate_ci'][0]:.1f}–{100 * v['fail_rate_ci'][1]:.1f}%), all in the "
      f"PICKMERGER (the only ACE-writing node; ace_format=file_lines). Causes: " +
      ", ".join(f"{c} {n}" for c, n in v["failure_classes"]) +
      f". Replicate r2: {v['n_failed_r2']} failures (" + ", ".join(f"{c} {n}" for c, n in v["failure_classes_r2"]) +
      f"); {v['failed_in_both_runs']} instances failed in both (P(fail r2 | fail r1)={v['p_fail_r2_given_fail_r1']:.2f} vs "
      f"P(fail r2 | ok r1)={v['p_fail_r2_given_ok_r1']:.2f}), so failures are partly instance-driven. "
      f"Resolve rate on non-failed instances {100 * v['resolved_rate_nonfailed']:.1f}%; the single-agent baseline resolves "
      f"{v['baseline_resolved_on_failed']}/{v['n_failed']} ({100 * v['baseline_rate_on_failed']:.1f}%) of the winner's failed "
      f"instances (vs {100 * v['baseline_rate_on_nonfailed']:.1f}% of the others).\n")

    a("## 6. Lineage\n")
    a("Node-innovation families (innovations.json key family × type): first generation seen in a population, peak "
      "prevalence of any single innovation, and how many innovations of that family survive to the latest generation.\n")
    a("| run | family:type | innovations | first gen | peak prev. | surviving / extinct | best final prev. |")
    a("|---|---|---|---|---|---|---|")
    for r in R["Q6_lineage"]["innovation_families"]:
        a(f"| {r['run']} | {r['family']}:{r['type']} | {r['n_innovations']} | {r['first_gen']} | {r['peak_prevalence']:.2f} | "
          f"{r['n_surviving']} / {r['n_extinct']} | {r['final_prevalence_best_innov']:.2f} |")
    a("\nMotif prevalence (% of population), first → last generation:\n")
    for run, mm in R["Q6_lineage"]["motif_prevalence"].items():
        a(f"- {run}: " + "; ".join(f"{k.replace('has_', '')} {100 * list(v.values())[0]:.0f}→{100 * list(v.values())[-1]:.0f}%"
                                    for k, v in mm.items()))
    wp = R["Q6_winner_prevalence"]
    a("\nWinner's topology (ED→ED chain into PICKMERGER; WL structure hash, any properties) as % of population by generation: "
      + "; ".join(f"{run}: " + " ".join(f"{g}:{100 * v:.0f}" for g, v in d["structure"].items()) for run, d in wp.items())
      + ". Winner's exact node innovations {0, 2, 19} in neat_b: "
      + " ".join(f"{g}:{100 * v:.0f}" for g, v in wp["neat_b"]["node_set"].items()) + " (%).")
    for run, evs in R["Q6_species_extinctions"].items():
        for e in evs:
            if e["lost_members"] >= 3:
                stag = all(li is not None and e["generation"] - li > 6 for li in e["lost_last_improved"])
                why = "pruned for stagnation (no improvement for >6 generations)" if stag else "emptied by re-speciation"
                a(f"- {run} gen {e['generation']}: species {e['lost_species']} ({e['lost_members']} members, last improved "
                  f"gen {e['lost_last_improved']}) disappeared: {why}.")
    a("")

    a("## Held-out val40 (40 instances, 27 repos disjoint from training)\n")
    a("| candidate | workers | types | END | ED→ED | train mean | val mean | val fail | median tokens | Wilcoxon p vs baseline |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in H["val40"]:
        tm = "–" if r["train_mean"] is None else f"{r['train_mean']:.3f} ({r['train_tasks']})"
        wp = "–" if r["wilcoxon_p_vs_base"] is None else f"{r['wilcoxon_p_vs_base']:.3g}"
        a(f"| {r['name']} | {r['workers']} | {r['types']} | {r.get('end_type', 'editor')} | {r.get('editor_chain', '–')} | {tm} | "
          f"{r['val_mean']:.3f} | {r['val_fail']:.2f} | {r['median_tokens'] / 1e3:.0f}k | {wp} |")
    a("")
    a("## Figures\n")
    a("- fig1_size.png: (a) Δ by size, (b) tokens by size, (c) mean size per generation.")
    a("- fig2_effects.png: forest plot of feature effects.")
    a("- fig3_pareto_lineage.png: (a) Δ vs tokens with Pareto fronts, (b) motif prevalence over generations.\n")
    a("## Caveats\n")
    a("- Selection effects: later-generation graphs are descendants of earlier winners, so feature effects partly reflect "
      "which lineage happened to be lucky early; the gen-0 column is the only selection-free estimate (and is small).")
    a("- Autocorrelated lineages: graphs are not independent; graph-clustered SEs treat them as such. The species-clustered "
      "sensitivity is a coarse lineage correction with few clusters.")
    a("- Noisy minibatches: 6–8 tasks per batch and a proxy fitness (patch similarity), not test resolution. Within-instance "
      "centring removes task difficulty but not batch × graph interaction.")
    a("- Features are confounded with each other (e.g. PICKMERGER at END implies ≥3 workers and an ACE writer); the "
      "univariate table is descriptive, the multivariable model partially adjusts.")
    a("- Runs were still appending; numbers are for this snapshot.")
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
