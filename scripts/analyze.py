#!/usr/bin/env python3
"""Statistical comparison of agent pipelines on SWE-bench Verified (and NEAT run summaries).

    python scripts/analyze.py \\
        --system baseline=runs/verified/baseline_r1 --system neat=runs/verified/neat_best \\
        --verified data/verified.jsonl --neat-runs runs/neat_a runs/neat_b --out runs/analysis

Each ``--system NAME=DIR`` directory holds ``preds*.jsonl`` (scripts/run_swe.py output:
patch, usage, proxy) and ``eval*.jsonl`` (scripts/swebench_eval.py output: ``resolved``).
Instances missing from a system's eval results count as unresolved (and are reported).

Tests (H1: optimized pipeline resolves more issues than the naive one):

* resolve rates with Wilson 95% CIs;
* exact McNemar test on the paired discordant counts (two-sided and one-sided p);
* paired bootstrap 95% CI of the resolve-rate difference (10,000 resamples);
* conditional logistic check: logistic regression ``resolved ~ system + difficulty``
  (GEE with exchangeable correlation, clustered by instance);
* tokens per instance (total and uncached prompt + completion): medians and the Wilcoxon
  signed-rank test, and resolved-per-million-tokens;
* proxy validity: AUC and point-biserial correlation of the training proxy score
  (patch similarity to the gold patch) against test-based resolution;
* NEAT runs: per-generation fitness and a Spearman trend test of generation vs. mean
  fitness of newly evaluated genomes.

Writes ``<out>/report.json``, ``<out>/report.md`` and PNG figures.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


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


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mcnemar_exact(b: int, c: int):
    """b: A resolved & B not; c: B resolved & A not. Returns (two-sided p, one-sided p for B>A)."""
    from scipy.stats import binomtest

    n = b + c
    if n == 0:
        return 1.0, 1.0
    two = binomtest(c, n, 0.5, alternative="two-sided").pvalue
    one = binomtest(c, n, 0.5, alternative="greater").pvalue
    return float(two), float(one)


def load_system(directory: Path):
    preds = {r["instance_id"]: r for r in jsonl(sorted(glob.glob(str(directory / "preds*.jsonl"))))}
    evals = {}
    for r in jsonl(sorted(glob.glob(str(directory / "eval*.jsonl")))):
        evals[r["instance_id"]] = r
    return preds, evals


def usage_tokens(pred, uncached=False):
    u = (pred or {}).get("usage") or {}
    total = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    if uncached:
        total -= u.get("cached_tokens", 0)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", required=True, help="NAME=DIR (first is the reference)")
    ap.add_argument("--verified", type=Path, default=Path("/usr/project/xtmp/kg349/ace/data/verified.jsonl"))
    ap.add_argument("--neat-runs", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="restrict to these instance ids (e.g. the gold-resolvable subset)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = {r["instance_id"]: r for r in jsonl([args.verified])}
    ids = sorted(meta)
    if args.ids_file:
        keep = {l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()}
        ids = [i for i in ids if i in keep]
    systems = {}
    for spec in args.system:
        name, _, d = spec.partition("=")
        systems[name] = load_system(Path(d))
    report: dict = {"n_instances": len(ids), "systems": {}, "comparisons": [], "notes": []}
    resolved = {}
    for name, (preds, evals) in systems.items():
        r = np.array([bool(evals.get(i, {}).get("resolved", False)) for i in ids])
        resolved[name] = r
        k = int(r.sum())
        p, lo, hi = wilson(k, len(ids))
        tok = np.array([usage_tokens(preds.get(i)) for i in ids], dtype=float)
        unc = np.array([usage_tokens(preds.get(i), uncached=True) for i in ids], dtype=float)
        calls = np.array([((preds.get(i) or {}).get("usage") or {}).get("calls", 0) for i in ids], dtype=float)
        statuses: dict[str, int] = {}
        for i in ids:
            s = evals.get(i, {}).get("status", "missing")
            statuses[s] = statuses.get(s, 0) + 1
        report["systems"][name] = {
            "resolved": k, "rate": p, "wilson95": [lo, hi],
            "preds": len(preds), "evaluated": len(evals),
            "empty_patches": sum(1 for i in ids if i in preds and not preds[i].get("model_patch")),
            "agent_failures": sum(1 for i in ids if i in preds and preds[i].get("failure")),
            "eval_status": statuses,
            "tokens_total_sum": float(tok.sum()), "tokens_median": float(np.median(tok)),
            "tokens_uncached_median": float(np.median(unc)), "calls_median": float(np.median(calls)),
            "resolved_per_Mtok": k / (tok.sum() / 1e6) if tok.sum() else None,
            "resolved_per_Mtok_uncached": k / (unc.sum() / 1e6) if unc.sum() else None,
        }
        # by difficulty
        diff = {}
        for i, ok in zip(ids, r):
            d = meta[i].get("difficulty", "?")
            a = diff.setdefault(d, [0, 0])
            a[0] += int(ok)
            a[1] += 1
        report["systems"][name]["by_difficulty"] = {d: {"resolved": a[0], "n": a[1]} for d, a in diff.items()}

    names = list(systems)
    ref = names[0]
    from scipy.stats import wilcoxon

    for other in names[1:]:
        a, b = resolved[ref], resolved[other]
        n01 = int((a & ~b).sum())
        n10 = int((~a & b).sum())
        two, one = mcnemar_exact(n01, n10)
        diffs = []
        n = len(ids)
        for _ in range(10000):
            idx = rng.integers(0, n, n)
            diffs.append(b[idx].mean() - a[idx].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pa, pb = systems[ref][0], systems[other][0]
        ta = np.array([usage_tokens(pa.get(i)) for i in ids], dtype=float)
        tb = np.array([usage_tokens(pb.get(i)) for i in ids], dtype=float)
        mask = (ta > 0) & (tb > 0)
        try:
            w = wilcoxon(tb[mask], ta[mask])
            wstat, wp = float(w.statistic), float(w.pvalue)
        except ValueError:
            wstat, wp = float("nan"), float("nan")
        comp = {
            "reference": ref, "system": other,
            "rate_diff": float(b.mean() - a.mean()), "bootstrap95": [float(lo), float(hi)],
            "discordant_ref_only": n01, "discordant_system_only": n10,
            "mcnemar_exact_p_two_sided": two, "mcnemar_exact_p_one_sided": one,
            "tokens_median_ratio": float(np.median(tb[mask] / ta[mask])) if mask.any() else None,
            "tokens_wilcoxon": {"statistic": wstat, "p": wp, "n": int(mask.sum())},
        }
        # GEE logistic regression with difficulty
        try:
            import pandas as pd
            import statsmodels.api as sm
            import statsmodels.formula.api as smf

            rows = []
            for i, x, y in zip(ids, a, b):
                d = meta[i].get("difficulty", "?")
                rows.append({"iid": i, "system": 0, "resolved": int(x), "difficulty": d})
                rows.append({"iid": i, "system": 1, "resolved": int(y), "difficulty": d})
            df = pd.DataFrame(rows)
            model = smf.gee("resolved ~ system + C(difficulty)", "iid", df,
                            family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
            fit = model.fit()
            comp["gee"] = {"odds_ratio_system": float(np.exp(fit.params["system"])),
                           "ci95": [float(np.exp(v)) for v in fit.conf_int().loc["system"]],
                           "p": float(fit.pvalues["system"])}
        except Exception as exc:  # noqa: BLE001
            comp["gee"] = {"error": str(exc)}
        report["comparisons"].append(comp)

    # proxy validity
    from scipy.stats import pointbiserialr

    xs, ys = [], []
    for name, (preds, evals) in systems.items():
        for i in ids:
            pr = preds.get(i)
            if pr is None or i not in evals:
                continue
            gold = meta[i].get("patch")
            if not gold:
                continue
            from ace.swebench.proxy import patch_similarity

            xs.append(pr.get("proxy_score", patch_similarity(pr.get("model_patch", ""), gold)))
            ys.append(int(bool(evals[i].get("resolved"))))
    if xs and 0 < sum(ys) < len(ys):
        xs_a, ys_a = np.array(xs), np.array(ys)
        pos, neg = xs_a[ys_a == 1], xs_a[ys_a == 0]
        auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
        r, p = pointbiserialr(ys_a, xs_a)
        report["proxy_validity"] = {"n": len(xs), "auc": auc, "pointbiserial_r": float(r), "p": float(p),
                                    "mean_proxy_resolved": float(pos.mean()), "mean_proxy_unresolved": float(neg.mean())}

    # NEAT runs
    from scipy.stats import spearmanr

    report["neat"] = {}
    for run in args.neat_runs:
        gens = jsonl([run / "generations.jsonl"]) if (run / "generations.jsonl").is_file() else []
        evals = jsonl([run / "evaluations.jsonl"]) if (run / "evaluations.jsonl").is_file() else []
        per_gen = {}
        for e in evals:
            per_gen.setdefault(e.get("batch", e.get("generation")), []).append(float(e.get("score", 0)))
        series = [(g, float(np.mean(v)), len(v)) for g, v in sorted(per_gen.items()) if g is not None]
        trend = spearmanr([s[0] for s in series], [s[1] for s in series]) if len(series) > 2 else None
        fails = sum(1 for e in evals if e.get("failure"))
        report["neat"][str(run)] = {
            "generations": len(gens),
            "task_evaluations": len(evals),
            "failure_rate": fails / len(evals) if evals else None,
            "mean_task_score_by_batch": series,
            "best_fitness_by_generation": [g.get("best_fitness") for g in gens],
            "mean_fitness_by_generation": [g.get("mean_fitness") for g in gens],
            "spearman_generation_vs_mean": None if trend is None else
                {"rho": float(trend.statistic), "p": float(trend.pvalue)},
        }

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["# SWE-bench Verified comparison", ""]
    for name, s in report["systems"].items():
        lines.append(f"- **{name}**: {s['resolved']}/{len(ids)} = {s['rate']:.3f} "
                     f"(Wilson 95% {s['wilson95'][0]:.3f}-{s['wilson95'][1]:.3f}); median tokens "
                     f"{s['tokens_median']:.0f}; resolved/Mtok {s['resolved_per_Mtok']}")
    for c in report["comparisons"]:
        lines.append(f"- {c['system']} - {c['reference']}: diff {c['rate_diff']:+.3f} "
                     f"(bootstrap 95% {c['bootstrap95'][0]:+.3f}..{c['bootstrap95'][1]:+.3f}); McNemar exact "
                     f"p one-sided {c['mcnemar_exact_p_one_sided']:.4f}, two-sided {c['mcnemar_exact_p_two_sided']:.4f} "
                     f"(discordant {c['discordant_ref_only']} vs {c['discordant_system_only']}); token ratio "
                     f"{c['tokens_median_ratio']}, Wilcoxon p {c['tokens_wilcoxon']['p']}")
    if "proxy_validity" in report:
        pv = report["proxy_validity"]
        lines.append(f"- proxy validity: AUC {pv['auc']:.3f}, r_pb {pv['pointbiserial_r']:.3f} (p {pv['p']:.2g}, n {pv['n']})")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if report["neat"]:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for run, d in report["neat"].items():
                ax.plot(d["best_fitness_by_generation"], label=f"{Path(run).name} best")
                ax.plot(d["mean_fitness_by_generation"], "--", label=f"{Path(run).name} mean")
            ax.set_xlabel("generation")
            ax.set_ylabel("proxy fitness (minibatch)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(args.out / "neat_fitness.png", dpi=150)
    except Exception as exc:  # noqa: BLE001
        print("plotting failed:", exc)


if __name__ == "__main__":
    main()
