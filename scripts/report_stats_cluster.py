#!/usr/bin/env python3
"""Per-instance statistics for the report that cannot be computed from the committed summaries.

Not run on real data (per-instance data lives only on the cluster); smoke-tested end-to-end on synthetic
data in the same format. Run on the cluster, e.g.:

    uv run --with scipy --with statsmodels --with pandas scripts/report_stats_cluster.py \\
        --preds baseline_nolimit=$ACE/runs/verified/baseline_unl \\
        --preds neat=$ACE/runs/verified/neat_best \\
        --preds neat_r2=$ACE/runs/verified/neat_best_r2 \\
        --preds baseline_30step=$ACE/runs/verified/baseline_r1 \\
        --preds baseline_30step_r2=$ACE/runs/verified/baseline_r2 \\
        --verified $ACE/data/verified.jsonl --out results/report_stats/cluster \\
        [--ids-file results/gold_validation/gold_resolvable_ids.txt]

Each ``--preds NAME=DIR`` is a system directory in the format read by scripts/analyze.py: ``preds*.jsonl``
(instance_id, model_patch, usage{prompt_tokens, completion_tokens, cached_tokens, calls}, failure,
proxy_score) and ``eval*.jsonl`` (instance_id, resolved, status). Instances missing from eval count as
unresolved, as in analyze.py. The first --preds is the reference; ``--primary`` names the system of the
pre-registered comparison (default ``neat``); ``--replicate A:B`` pairs replicate runs.

Computes:
  1. exact McNemar + paired Newcombe RD CI within each difficulty stratum, and a stratified
     (Mantel-Haenszel for matched pairs) test, plus GEE with a system x difficulty interaction;
  2. replicate discordance (McNemar, Cohen's kappa, Jaccard of resolved sets);
  3. paired instance-bootstrap CIs for resolved per Mtok (all and uncached) and the system/reference ratio;
  4. proxy validity with per-group SDs: Cohen's d (pooled), Glass's delta, and instance-clustered bootstrap
     CIs for AUC, point-biserial r and d;
  5. token Wilcoxon effect sizes (matched-pairs rank-biserial r, Hodges-Lehmann log-ratio with bootstrap CI),
     with Holm across comparisons;
  6. failure / empty-patch cross-tabulations by instance.
Writes ``<out>/cluster_stats.json`` and ``<out>/cluster_stats.md``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

# Reuse the committed summary helpers (pure functions; importing does not run anything).
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_stats import (  # noqa: E402
    DIFF_ORDER, Z, cohen_h, mcnemar_exact, newcombe_paired, p_adjust, paired_or_exact, wilson,
)


# ----------------------------------------------------------------------------- loaders (mirror analyze.py)

def jsonl(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_system(directory: Path):
    preds = {r["instance_id"]: r for r in jsonl(sorted(glob.glob(str(directory / "preds*.jsonl"))))}
    evals = {r["instance_id"]: r for r in jsonl(sorted(glob.glob(str(directory / "eval*.jsonl"))))}
    return preds, evals


def usage_tokens(pred, uncached=False):
    u = (pred or {}).get("usage") or {}
    total = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    if uncached:
        total -= u.get("cached_tokens", 0)
    return total


def proxy_of(pred, gold):
    if "proxy_score" in pred and pred["proxy_score"] is not None:
        return float(pred["proxy_score"])
    from ace.swebench.proxy import patch_similarity  # same fallback as analyze.py

    return float(patch_similarity(pred.get("model_patch", ""), gold))


# ----------------------------------------------------------------------------- helpers

def table2x2(a: np.ndarray, b: np.ndarray):
    """a = reference resolved, b = system resolved (bool arrays). Returns n11, n10 (ref only), n01 (sys only), n00."""
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def kappa(a: np.ndarray, b: np.ndarray):
    po = float((a == b).mean())
    p1, p2 = a.mean(), b.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def paired_block(a, b):
    n11, n10, n01, n00 = table2x2(a, b)
    two, greater, _ = mcnemar_exact(n10, n01)
    d, lo, hi = newcombe_paired(n11, n10, n01, n00)
    por, plo, phi = paired_or_exact(n10, n01) if n10 + n01 else (float("nan"),) * 3
    return {
        "n": int(len(a)), "both": n11, "ref_only": n10, "system_only": n01, "neither": n00,
        "rate_ref": float(a.mean()), "rate_system": float(b.mean()),
        "rd": d, "rd_newcombe_paired95": [lo, hi],
        "mcnemar_p_two_sided": two, "mcnemar_p_one_sided_greater": greater,
        "paired_or": por, "paired_or_exact95": [plo, phi],
        "cohen_h": cohen_h(float(b.mean()), float(a.mean())),
    }


def mh_matched_pairs(strata):
    """Stratified McNemar (Mantel-Haenszel for matched pairs): sum of discordant counts across strata.

    Equivalent to McNemar on the pooled pairs because the MH estimator for 1:1 matched pairs is
    sum(n01)/sum(n10); returned with the exact conditional p and the pooled paired OR."""
    b = sum(s["ref_only"] for s in strata)
    c = sum(s["system_only"] for s in strata)
    two, greater, _ = mcnemar_exact(b, c)
    por, lo, hi = paired_or_exact(b, c)
    return {"ref_only": b, "system_only": c, "p_two_sided": two, "p_one_sided_greater": greater,
            "mh_paired_or": por, "exact95": [lo, hi]}


def gee_interaction(ids, a, b, diff):
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = []
    for i, x, y, d in zip(ids, a, b, diff):
        # merge >4 h (n=3) into 1-4 h so every cell has data
        d2 = "1-4 hours" if d == ">4 hours" else d
        rows.append({"iid": i, "system": 0, "resolved": int(x), "difficulty": d2})
        rows.append({"iid": i, "system": 1, "resolved": int(y), "difficulty": d2})
    df = pd.DataFrame(rows)
    out = {}
    for label, formula in (("main", "resolved ~ system + C(difficulty)"),
                           ("interaction", "resolved ~ system * C(difficulty)")):
        try:
            fit = smf.gee(formula, "iid", df, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            terms = {}
            ci = fit.conf_int()
            for t in fit.params.index:
                if "system" in t:
                    terms[t] = {"or": float(np.exp(fit.params[t])),
                                "ci95": [float(np.exp(ci.loc[t, 0])), float(np.exp(ci.loc[t, 1]))],
                                "p": float(fit.pvalues[t])}
            out[label] = terms
            if label == "interaction":
                inter = [t for t in fit.params.index if t.startswith("system:")]
                if inter:
                    R = np.zeros((len(inter), len(fit.params)))
                    for k, t in enumerate(inter):
                        R[k, list(fit.params.index).index(t)] = 1
                    w = fit.wald_test(R, scalar=True)
                    out["interaction_wald"] = {"stat": float(w.statistic), "df": len(inter), "p": float(w.pvalue)}
        except Exception as exc:  # noqa: BLE001
            out[label] = {"error": str(exc)}
    return out


def boot_ratio_ci(num_a, den_a, num_b, den_b, rng, B):
    """Paired instance bootstrap of resolved-per-Mtok for two systems and their ratio (b / a)."""
    n = len(num_a)
    ra, rb, rr = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        da, db = den_a[idx].sum(), den_b[idx].sum()
        xa = num_a[idx].sum() / (da / 1e6) if da else np.nan
        xb = num_b[idx].sum() / (db / 1e6) if db else np.nan
        ra.append(xa)
        rb.append(xb)
        rr.append(xb / xa if xa else np.nan)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    return q(ra), q(rb), q(rr)


def auc_of(x, y):
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC (Mann-Whitney), ties count 1/2
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def d_stats(x, y):
    pos, neg = x[y == 1], x[y == 0]
    n1, n0 = len(pos), len(neg)
    s1, s0 = pos.std(ddof=1), neg.std(ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    return {"n1": n1, "n0": n0, "mean1": float(pos.mean()), "mean0": float(neg.mean()),
            "sd1": float(s1), "sd0": float(s0), "sd_pooled": sp,
            "cohen_d": float((pos.mean() - neg.mean()) / sp),
            "hedges_g": float((pos.mean() - neg.mean()) / sp * (1 - 3 / (4 * (n1 + n0) - 9))),
            "glass_delta_unresolved_sd": float((pos.mean() - neg.mean()) / s0)}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True, help="NAME=DIR (first is the reference)")
    ap.add_argument("--verified", type=Path, default=Path("/usr/project/xtmp/kg349/ace/data/verified.jsonl"))
    ap.add_argument("--ids-file", type=Path, default=None)
    ap.add_argument("--primary", default="neat")
    ap.add_argument("--replicate", action="append", default=None,
                    help="A:B replicate pair (default: neat:neat_r2 and baseline_30step:baseline_30step_r2)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = {r["instance_id"]: r for r in jsonl([args.verified])}
    ids = sorted(meta)
    if args.ids_file:
        keep = {l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()}
        ids = [i for i in ids if i in keep]
    diff = np.array([meta[i].get("difficulty", "?") for i in ids])

    systems, resolved, tok, unc, fail, empty = {}, {}, {}, {}, {}, {}
    for spec in args.preds:
        name, _, d = spec.partition("=")
        preds, evals = load_system(Path(d))
        systems[name] = (preds, evals)
        resolved[name] = np.array([bool(evals.get(i, {}).get("resolved", False)) for i in ids])
        tok[name] = np.array([usage_tokens(preds.get(i)) for i in ids], dtype=float)
        unc[name] = np.array([usage_tokens(preds.get(i), uncached=True) for i in ids], dtype=float)
        fail[name] = np.array([bool((preds.get(i) or {}).get("failure")) for i in ids])
        empty[name] = np.array([i in preds and not preds[i].get("model_patch") for i in ids])
    names = list(systems)
    ref = names[0]
    report = {"n_instances": len(ids), "reference": ref, "primary": args.primary, "systems": names}

    # sanity check against the committed summary numbers
    report["resolved_counts"] = {n: int(resolved[n].sum()) for n in names}

    # 1. difficulty strata (paired)
    strata = {}
    for other in names[1:]:
        a, b = resolved[ref], resolved[other]
        per = []
        for d in DIFF_ORDER + sorted(set(diff) - set(DIFF_ORDER)):
            m = diff == d
            if m.sum() == 0:
                continue
            blk = paired_block(a[m], b[m])
            blk["bin"] = d
            per.append(blk)
        merged = diff.copy()
        merged[np.isin(merged, ["1-4 hours", ">4 hours"])] = ">1 hour"
        per_merged = []
        for d in ["<15 min fix", "15 min - 1 hour", ">1 hour"]:
            m = merged == d
            if m.sum():
                blk = paired_block(a[m], b[m])
                blk["bin"] = d
                per_merged.append(blk)
        ps = [p["mcnemar_p_two_sided"] for p in per_merged]
        for p, h in zip(per_merged, p_adjust(ps, "holm")):
            p["mcnemar_p_two_sided_holm"] = float(h)
        strata[other] = {"per_bin": per, "per_bin_merged": per_merged, "stratified": mh_matched_pairs(per),
                         "gee": gee_interaction(ids, a, b, diff)}
    report["difficulty_strata"] = strata

    # 2. replicate discordance
    pairs = args.replicate or ["neat:neat_r2", "baseline_30step:baseline_30step_r2"]
    reps = {}
    for pr in pairs:
        x, _, y = pr.partition(":")
        if x in resolved and y in resolved:
            blk = paired_block(resolved[x], resolved[y])
            blk["cohen_kappa"] = kappa(resolved[x], resolved[y])
            u = (resolved[x] | resolved[y]).sum()
            blk["jaccard_resolved"] = float((resolved[x] & resolved[y]).sum() / u) if u else float("nan")
            blk["failure_both"] = int((fail[x] & fail[y]).sum())
            blk["failure_kappa"] = kappa(fail[x], fail[y]) if fail[x].any() or fail[y].any() else None
            reps[pr] = blk
    report["replicates"] = reps

    # 3. bootstrap CIs on resolved per Mtok (paired instance bootstrap vs reference)
    eff = {}
    for other in names[1:]:
        out = {}
        for label, T in (("all_tokens", tok), ("uncached", unc)):
            a_ci, b_ci, r_ci = boot_ratio_ci(resolved[ref].astype(float), T[ref], resolved[other].astype(float),
                                             T[other], rng, args.bootstrap)
            pa = resolved[ref].sum() / (T[ref].sum() / 1e6)
            pb = resolved[other].sum() / (T[other].sum() / 1e6)
            out[label] = {"ref": float(pa), "ref95": a_ci, "system": float(pb), "system95": b_ci,
                          "ratio": float(pb / pa) if pa else None, "ratio95": r_ci}
        eff[other] = out
    report["resolved_per_Mtok_bootstrap"] = eff

    # 5. token Wilcoxon effect sizes
    tw = {}
    for other in names[1:]:
        ta, tb = tok[ref], tok[other]
        m = (ta > 0) & (tb > 0)
        lr = np.log(tb[m] / ta[m])
        w = stats.wilcoxon(tb[m], ta[m])
        dpos = lr[lr != 0]
        ranks = stats.rankdata(np.abs(dpos))
        rpos, rneg = ranks[dpos > 0].sum(), ranks[dpos < 0].sum()
        rbc = float((rpos - rneg) / (rpos + rneg)) if len(dpos) else float("nan")
        # Hodges-Lehmann pseudo-median of the log ratio (Walsh averages), bootstrap CI
        def hl(v):
            if len(v) > 3000:
                v = rng.choice(v, 3000, replace=False)
            i, j = np.triu_indices(len(v))
            return float(np.median((v[i] + v[j]) / 2))
        boots = [hl(lr[rng.integers(0, len(lr), len(lr))]) for _ in range(min(args.bootstrap, 2000))]
        tw[other] = {"n": int(m.sum()), "wilcoxon_p": float(w.pvalue), "rank_biserial_r": rbc,
                     "hl_ratio": float(np.exp(hl(lr))),
                     "hl_ratio95": [float(np.exp(np.percentile(boots, 2.5))), float(np.exp(np.percentile(boots, 97.5)))]}
    ps = [v["wilcoxon_p"] for v in tw.values()]
    for v, h in zip(tw.values(), p_adjust(ps, "holm")):
        v["wilcoxon_p_holm"] = float(h)
    report["token_wilcoxon"] = tw

    # 4. proxy validity with per-group SDs and instance-clustered bootstrap
    xs, ys, cl = [], [], []
    for name, (preds, evals) in systems.items():
        for k, i in enumerate(ids):
            pr = preds.get(i)
            gold = meta[i].get("patch")
            if pr is None or i not in evals or not gold:
                continue
            xs.append(proxy_of(pr, gold))
            ys.append(int(bool(evals[i].get("resolved"))))
            cl.append(k)
    x, y, cl = np.array(xs), np.array(ys), np.array(cl)
    if len(x) == 0 or not (0 < y.sum() < len(y)):
        raise SystemExit("proxy validity: no usable (proxy, resolved) pairs; check --preds/--verified paths")
    pv = {"n": int(len(x)), "auc": auc_of(x, y), "r_pb": float(stats.pointbiserialr(y, x).statistic)}
    pv.update(d_stats(x, y))
    by_inst = {}
    for k, c in enumerate(cl):
        by_inst.setdefault(c, []).append(k)
    groups = list(by_inst.values())
    bs = {"auc": [], "r_pb": [], "cohen_d": []}
    for _ in range(min(args.bootstrap, 2000)):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        xb, yb = x[idx], y[idx]
        if 0 < yb.sum() < len(yb):
            bs["auc"].append(auc_of(xb, yb))
            bs["r_pb"].append(float(stats.pointbiserialr(yb, xb).statistic))
            bs["cohen_d"].append(d_stats(xb, yb)["cohen_d"])
    for k, v in bs.items():
        pv[f"{k}_cluster_boot95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    report["proxy_validity"] = pv

    # 6. failures and empty patches, paired by instance
    fx = {}
    for other in names[1:]:
        f = fail[other]
        fx[other] = {
            "system_failures": int(f.sum()), "system_empty": int(empty[other].sum()),
            "ref_rate_on_system_failed": float(resolved[ref][f].mean()) if f.any() else None,
            "ref_rate_on_system_ok": float(resolved[ref][~f].mean()) if (~f).any() else None,
            "system_rate_on_ok": float(resolved[other][~f].mean()) if (~f).any() else None,
            "ref_empty_and_system_resolved": int((empty[ref] & resolved[other]).sum()),
            "fisher_ref_resolved_vs_system_failed_p": float(stats.fisher_exact(
                [[int((resolved[ref] & f).sum()), int((~resolved[ref] & f).sum())],
                 [int((resolved[ref] & ~f).sum()), int((~resolved[ref] & ~f).sum())]]).pvalue),
        }
        fx[other]["empty_patch_resolved"] = int((empty[other] & resolved[other]).sum())
    fx[ref] = {"empty_patch_resolved": int((empty[ref] & resolved[ref]).sum())}
    report["failures"] = fx

    (args.out / "cluster_stats.json").write_text(json.dumps(report, indent=2, default=float))

    # short markdown
    L = [f"# Per-instance statistics (n = {len(ids)}, reference = {ref})", ""]
    L.append("## Paired difficulty strata (exact McNemar, Newcombe paired RD)")
    for other, s in strata.items():
        L.append(f"\n**{other} vs {ref}**\n")
        L.append("| bin | n | both/ref only/sys only/neither | RD pts (95%) | McNemar p (two-sided) | Holm |")
        L.append("|---|---|---|---|---|---|")
        for p in s["per_bin_merged"]:
            L.append(f"| {p['bin']} | {p['n']} | {p['both']}/{p['ref_only']}/{p['system_only']}/{p['neither']} | "
                     f"{100 * p['rd']:+.1f} ({100 * p['rd_newcombe_paired95'][0]:+.1f} to {100 * p['rd_newcombe_paired95'][1]:+.1f}) | "
                     f"{p['mcnemar_p_two_sided']:.2e} | {p['mcnemar_p_two_sided_holm']:.2e} |")
        st = s["stratified"]
        L.append(f"\nStratified paired OR {st['mh_paired_or']:.2f} ({st['exact95'][0]:.2f}–{st['exact95'][1]:.2f}); "
                 f"GEE interaction Wald p = {s['gee'].get('interaction_wald', {}).get('p', 'n/a')}")
    L.append("\n## Replicates")
    for pr, b in reps.items():
        L.append(f"- {pr}: both {b['both']}, only first {b['ref_only']}, only second {b['system_only']}; McNemar p = "
                 f"{b['mcnemar_p_two_sided']:.3f}; kappa = {b['cohen_kappa']:.2f}; Jaccard = {b['jaccard_resolved']:.2f}")
    L.append("\n## Resolved per Mtok, paired bootstrap 95% CI")
    for other, e in eff.items():
        for lab, v in e.items():
            L.append(f"- {other} vs {ref} ({lab}): {v['system']:.3f} ({v['system95'][0]:.3f}–{v['system95'][1]:.3f}) vs "
                     f"{v['ref']:.3f} ({v['ref95'][0]:.3f}–{v['ref95'][1]:.3f}); ratio {v['ratio']:.2f} "
                     f"({v['ratio95'][0]:.2f}–{v['ratio95'][1]:.2f})")
    L.append("\n## Token Wilcoxon effect sizes")
    for other, v in tw.items():
        L.append(f"- {other}: HL ratio {v['hl_ratio']:.2f} ({v['hl_ratio95'][0]:.2f}–{v['hl_ratio95'][1]:.2f}), "
                 f"rank-biserial r {v['rank_biserial_r']:+.2f}, p {v['wilcoxon_p']:.2e} (Holm {v['wilcoxon_p_holm']:.2e})")
    L.append("\n## Proxy validity (per-group SDs, instance-clustered bootstrap)")
    L.append(f"- AUC {pv['auc']:.3f} ({pv['auc_cluster_boot95'][0]:.3f}–{pv['auc_cluster_boot95'][1]:.3f}); r_pb {pv['r_pb']:.3f} "
             f"({pv['r_pb_cluster_boot95'][0]:.3f}–{pv['r_pb_cluster_boot95'][1]:.3f})")
    L.append(f"- resolved {pv['mean1']:.3f} (SD {pv['sd1']:.3f}, n {pv['n1']}) vs unresolved {pv['mean0']:.3f} (SD {pv['sd0']:.3f}, n {pv['n0']}); "
             f"Cohen's d {pv['cohen_d']:.2f} ({pv['cohen_d_cluster_boot95'][0]:.2f}–{pv['cohen_d_cluster_boot95'][1]:.2f}), "
             f"Hedges' g {pv['hedges_g']:.2f}, Glass's Δ {pv['glass_delta_unresolved_sd']:.2f}")
    L.append("\n## Failures / empty patches")
    for k, v in fx.items():
        L.append(f"- {k}: {json.dumps(v)}")
    (args.out / "cluster_stats.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
