#!/usr/bin/env python3
"""Report statistics computed from the committed summaries in results/ (no per-instance data).

    uv run scripts/report_stats.py [--results results] [--out results/report_stats]

Inputs (all committed):
  results/analysis_v2/report.json, results/analysis_v2_gold/report.json  (main comparison)
  results/analysis_v1/report.json                                          (baseline r1 vs r2 discordance)
  results/selection/selection.json, results/graph_analysis/{report.json,features.csv}

Everything here is derived from aggregate counts. A key observation: each pairwise comparison
in report.json gives both marginals and both discordant counts, so the full paired 2x2 table
(both / ref only / system only / neither) is recoverable and paired effect sizes can be computed
exactly. What cannot be recovered without per-instance data is listed in summary.md under
"Needs cluster data" and computed by scripts/report_stats_cluster.py.

Writes summary.md, summary.json, fig_main.{png,pdf}, fig_difficulty.{png,pdf},
fig_neat_curves.{png,pdf} and stats_methods_paragraph.md into --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

Z = 1.959963984540054

LABELS = {
    "baseline_nolimit": "Baseline (no step limit)",
    "neat": "NEAT best r1",
    "neat_r2": "NEAT best r2",
    "baseline_30step": "Baseline 30-step r1",
    "baseline_30step_r2": "Baseline 30-step r2",
}
SHORT = {
    "baseline_nolimit": "Naive no-limit",
    "neat": "Pipeline r1",
    "neat_r2": "Pipeline r2",
    "baseline_30step": "Naive 30-step r1",
    "baseline_30step_r2": "Naive 30-step r2",
}
PRIMARY = ("baseline_nolimit", "neat")
DIFF_ORDER = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
DIFF_SHORT = {"<15 min fix": "<15 min", "15 min - 1 hour": "15 min–1 h", "1-4 hours": "1–4 h", ">4 hours": ">4 h"}

# Colourblind-safe reference palette (dataviz skill, validated): blue / orange; others neutral grey.
C_NEAT = "#2a78d6"
C_BASE = "#eb6834"
C_GREY = "#8c8b87"
C_GREY_L = "#bdbcb6"
INK = "#0b0b0b"
INK2 = "#52514e"


# ----------------------------------------------------------------------------- basic estimators

def wilson(k: int, n: int, z: float = Z):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe_unpaired(k1, n1, k0, n0):
    """Newcombe (1998) hybrid score CI for p1 - p0, independent samples."""
    p1, l1, u1 = wilson(k1, n1)
    p0, l0, u0 = wilson(k0, n0)
    d = p1 - p0
    return d, d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def wald_unpaired(k1, n1, k0, n0):
    p1, p0 = k1 / n1, k0 / n0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    d = p1 - p0
    return d, d - Z * se, d + Z * se


def newcombe_paired(n11, n10, n01, n00):
    """Newcombe (1998b, method 10) score CI for pS - pR from a paired 2x2.

    n11 both resolved, n10 reference only, n01 system only, n00 neither. Returns (d, lo, hi)."""
    n = n11 + n10 + n01 + n00
    kS, kR = n11 + n01, n11 + n10
    pS, lS, uS = wilson(kS, n)
    pR, lR, uR = wilson(kR, n)
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    num = n11 * n00 - n10 * n01
    phi = 0.0 if denom == 0 else (max(num - n / 2, 0) if num > 0 else num) / denom
    d = pS - pR
    dl = math.sqrt(max((pS - lS) ** 2 - 2 * phi * (pS - lS) * (uR - pR) + (uR - pR) ** 2, 0))
    du = math.sqrt(max((uS - pS) ** 2 - 2 * phi * (uS - pS) * (pR - lR) + (pR - lR) ** 2, 0))
    return d, d - dl, d + du


def mcnemar_exact(b: int, c: int):
    """b = reference only, c = system only. Exact conditional binomial test."""
    n = b + c
    if n == 0:
        return 1.0, 1.0, 1.0
    two = stats.binomtest(c, n, 0.5, alternative="two-sided").pvalue
    greater = stats.binomtest(c, n, 0.5, alternative="greater").pvalue
    less = stats.binomtest(c, n, 0.5, alternative="less").pvalue
    return float(two), float(greater), float(less)


def mcnemar_midp_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * stats.binom.cdf(k, n, 0.5) - stats.binom.pmf(k, n, 0.5)))


def paired_or_exact(b, c, alpha=0.05):
    """Conditional (paired) OR = c/b with exact Clopper-Pearson CI via pi = c/(b+c)."""
    n = b + c
    lo_pi = stats.beta.ppf(alpha / 2, c, n - c + 1) if c > 0 else 0.0
    hi_pi = stats.beta.ppf(1 - alpha / 2, c + 1, n - c) if c < n else 1.0
    f = (lambda p: p / (1 - p) if p < 1 else float("inf"))
    return (c / b if b else float("inf")), f(lo_pi), f(hi_pi)


def cohen_h(p1, p0):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))


def paired_rr(n11, n10, n01, n00):
    """RR = pS/pR with the paired-sample log-scale Wald CI."""
    kS, kR = n11 + n01, n11 + n10
    rr = kS / kR
    se = math.sqrt((n10 + n01) / (kS * kR)) if kS and kR else float("nan")
    return rr, rr * math.exp(-Z * se), rr * math.exp(Z * se)


def p_adjust(p, method):
    p = np.asarray(p, dtype=float)
    m = len(p)
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p)
    if method == "holm":
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p[i]))
            adj[i] = running
        return adj
    if method == "bh":
        adj = np.empty(m)
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, p[i] * m / (rank + 1))
            adj[i] = min(running, 1.0)
        return adj
    raise ValueError(method)


def cochran_armitage(ks, ns, scores):
    ks, ns, t = (np.asarray(x, dtype=float) for x in (ks, ns, scores))
    N, K = ns.sum(), ks.sum()
    pbar = K / N
    T = float(np.sum(t * (ks - ns * pbar)))
    var = pbar * (1 - pbar) * (np.sum(ns * t ** 2) - np.sum(ns * t) ** 2 / N)
    z = T / math.sqrt(var) if var > 0 else float("nan")
    return z, float(2 * stats.norm.sf(abs(z)))


def fisher_z_ci(r, n):
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(z - Z * se), math.tanh(z + Z * se)


def hanley_mcneil_se(auc, n1, n0):
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))


def sci(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) >= 1e-3 and abs(x) < 1e4:
        return f"{x:.{digits + 1}g}" if abs(x) < 1 else f"{x:.{digits}f}"
    return f"{x:.{digits}e}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


# ----------------------------------------------------------------------------- analyses

def paired_table(rep, comp):
    sysA = rep["systems"][comp["reference"]]
    sysB = rep["systems"][comp["system"]]
    n = rep["n_instances"]
    b, c = comp["discordant_ref_only"], comp["discordant_system_only"]
    n11 = sysA["resolved"] - b
    assert n11 == sysB["resolved"] - c, "inconsistent marginals/discordant counts"
    n00 = n - n11 - b - c
    return n11, b, c, n00


def comparisons_block(rep, label):
    rows = []
    for comp in rep["comparisons"]:
        n11, b, c, n00 = paired_table(rep, comp)
        n = rep["n_instances"]
        two, greater, less = mcnemar_exact(b, c)
        pR = (n11 + b) / n
        pS = (n11 + c) / n
        d_np, lo_np, hi_np = newcombe_paired(n11, b, c, n00)
        rr, rr_lo, rr_hi = paired_rr(n11, b, c, n00)
        por, por_lo, por_hi = paired_or_exact(b, c)
        rd = pS - pR
        rows.append({
            "subset": label,
            "reference": comp["reference"], "system": comp["system"],
            "primary": (comp["reference"], comp["system"]) == PRIMARY,
            "n": n, "table": {"both": n11, "ref_only": b, "system_only": c, "neither": n00},
            "rate_ref": pR, "rate_system": pS,
            "mcnemar_exact_p_one_sided_greater": greater,
            "mcnemar_exact_p_two_sided": two,
            "mcnemar_exact_p_one_sided_less": less,
            "mcnemar_midp_two_sided": mcnemar_midp_two_sided(b, c),
            "json_p_one_sided": comp["mcnemar_exact_p_one_sided"],
            "json_p_two_sided": comp["mcnemar_exact_p_two_sided"],
            "risk_difference": rd,
            "rd_bootstrap95_json": comp["bootstrap95"],
            "rd_newcombe_paired95": [lo_np, hi_np],
            "extra_resolved_per_100": 100 * rd,
            "nnt": (1 / rd) if rd > 0 else None,
            "relative_risk": rr, "rr_paired95": [rr_lo, rr_hi],
            "cohen_h": cohen_h(pS, pR),
            "gee_or": comp["gee"].get("odds_ratio_system"), "gee_or95": comp["gee"].get("ci95"),
            "gee_p": comp["gee"].get("p"),
            "paired_or_c_over_b": por, "paired_or_exact95": [por_lo, por_hi],
            "tokens_median_ratio": comp["tokens_median_ratio"],
            "tokens_wilcoxon_p": comp["tokens_wilcoxon"]["p"],
        })
    for key in ("mcnemar_exact_p_one_sided_greater", "mcnemar_exact_p_two_sided"):
        ps = [r[key] for r in rows]
        for meth in ("bonferroni", "holm", "bh"):
            for r, a in zip(rows, p_adjust(ps, meth)):
                r[f"{key}_{meth}"] = float(a)
    return rows


def difficulty_block(rep):
    sysd = rep["systems"]
    out = {"per_system": {}, "rd_neat_vs_nolimit": [], "trend": {}, "mh": None}
    scores = list(range(len(DIFF_ORDER)))
    for name, s in sysd.items():
        bins = []
        for d in DIFF_ORDER:
            k, n = s["by_difficulty"][d]["resolved"], s["by_difficulty"][d]["n"]
            p, lo, hi = wilson(k, n)
            bins.append({"bin": d, "resolved": k, "n": n, "rate": p, "wilson95": [lo, hi]})
        out["per_system"][name] = bins
        ks, ns = [b["resolved"] for b in bins], [b["n"] for b in bins]
        z, p = cochran_armitage(ks, ns, scores)
        # chi-square of independence over the bins (and with the sparse >4 h bin merged into 1-4 h)
        tab = np.array([ks, [n - k for k, n in zip(ks, ns)]])
        chi4 = stats.chi2_contingency(tab, correction=False)
        tab3 = np.array([ks[:2] + [ks[2] + ks[3]], [ns[0] - ks[0], ns[1] - ks[1], ns[2] + ns[3] - ks[2] - ks[3]]])
        chi3 = stats.chi2_contingency(tab3, correction=False)
        z3, p3 = cochran_armitage(tab3[0], tab3.sum(0), [0, 1, 2])
        out["trend"][name] = {
            "cochran_armitage_z_4bins": z, "cochran_armitage_p_4bins": p,
            "cochran_armitage_z_3bins": z3, "cochran_armitage_p_3bins": p3,
            "chi2_4bins": float(chi4.statistic), "chi2_df_4bins": int(chi4.dof), "chi2_p_4bins": float(chi4.pvalue),
            "chi2_min_expected_4bins": float(chi4.expected_freq.min()),
            "chi2_3bins": float(chi3.statistic), "chi2_df_3bins": int(chi3.dof), "chi2_p_3bins": float(chi3.pvalue),
        }
    ref, sysn = PRIMARY
    tables = []
    for bR, bS in zip(out["per_system"][ref], out["per_system"][sysn]):
        d, lo, hi = newcombe_unpaired(bS["resolved"], bS["n"], bR["resolved"], bR["n"])
        _, wlo, whi = wald_unpaired(bS["resolved"], bS["n"], bR["resolved"], bR["n"])
        rr = bS["rate"] / bR["rate"] if bR["rate"] > 0 else None
        out["rd_neat_vs_nolimit"].append({
            "bin": bR["bin"], "n": bR["n"], "rate_ref": bR["rate"], "rate_system": bS["rate"],
            "rd": d, "newcombe95": [lo, hi], "wald95": [wlo, whi], "rr": rr,
            "cohen_h": cohen_h(bS["rate"], bR["rate"]),
        })
        if bR["resolved"] + bS["resolved"] > 0:
            tables.append([[bS["resolved"], bS["n"] - bS["resolved"]], [bR["resolved"], bR["n"] - bR["resolved"]]])
    try:
        from statsmodels.stats.contingency_tables import StratifiedTable

        st = StratifiedTable([np.array(t, dtype=float) + 0.0 for t in tables])
        tst = st.test_null_odds(correction=False)
        lo, hi = st.oddsratio_pooled_confint()
        res = {"mh_or": float(st.oddsratio_pooled), "mh_or95": [float(lo), float(hi)],
               "cmh_chi2": float(tst.statistic), "cmh_p": float(tst.pvalue),
               "strata_used": len(tables), "note": "strata with zero resolved in both arms dropped"}
        try:
            bd = st.test_equal_odds(adjust=False)
            res["breslow_day_chi2"], res["breslow_day_p"] = float(bd.statistic), float(bd.pvalue)
        except Exception as exc:  # noqa: BLE001
            res["breslow_day_error"] = str(exc)
        # homogeneity on the two informative strata only (the 1-4 h stratum has a zero cell)
        st2 = StratifiedTable([np.array(t, dtype=float) for t in tables[:2]])
        bd2 = st2.test_equal_odds(adjust=False)
        res["breslow_day_2strata_chi2"], res["breslow_day_2strata_p"] = float(bd2.statistic), float(bd2.pvalue)
        out["mh"] = res
    except Exception as exc:  # noqa: BLE001
        out["mh"] = {"error": str(exc)}
    return out


def token_block(rep):
    rows = {}
    for name, s in rep["systems"].items():
        tot = s["tokens_total_sum"]
        unc_tot = s["resolved"] / s["resolved_per_Mtok_uncached"] * 1e6 if s["resolved_per_Mtok_uncached"] else None
        rows[name] = {
            "resolved": s["resolved"], "rate": s["rate"],
            "tokens_median": s["tokens_median"], "tokens_uncached_median": s["tokens_uncached_median"],
            "tokens_total": tot, "tokens_mean": tot / rep["n_instances"],
            "tokens_uncached_total_derived": unc_tot,
            "cached_share_derived": None if unc_tot is None else 1 - unc_tot / tot,
            "resolved_per_Mtok": s["resolved_per_Mtok"], "resolved_per_Mtok_uncached": s["resolved_per_Mtok_uncached"],
            "Mtok_per_resolved": tot / 1e6 / s["resolved"],
            "calls_median": s["calls_median"], "empty_patches": s["empty_patches"],
            "agent_failures": s["agent_failures"],
        }
    ratios = {}
    for ref in ("baseline_nolimit", "baseline_30step"):
        a, b = rows["neat"], rows[ref]
        ratios[f"neat/{ref}"] = {
            "rate": a["rate"] / b["rate"],
            "tokens_median": a["tokens_median"] / b["tokens_median"],
            "tokens_total": a["tokens_total"] / b["tokens_total"],
            "resolved_per_Mtok": a["resolved_per_Mtok"] / b["resolved_per_Mtok"],
            "resolved_per_Mtok_uncached": a["resolved_per_Mtok_uncached"] / b["resolved_per_Mtok_uncached"],
            "median_of_paired_ratios_json": next(c["tokens_median_ratio"] for c in rep["comparisons"]
                                                 if c["system"] == "neat" and c["reference"] == ref)
            if ref == rep["comparisons"][0]["reference"] else None,
        }
    return rows, ratios


def replicate_block(rep, rep_v1):
    s = rep["systems"]
    out = {}
    # baseline 30-step r1 vs r2: discordant counts exist in analysis_v1
    c = next(c for c in rep_v1["comparisons"] if c["reference"] == "baseline" and c["system"] == "baseline_r2")
    n11 = rep_v1["systems"]["baseline"]["resolved"] - c["discordant_ref_only"]
    n00 = rep_v1["n_instances"] - n11 - c["discordant_ref_only"] - c["discordant_system_only"]
    d, lo, hi = newcombe_paired(n11, c["discordant_ref_only"], c["discordant_system_only"], n00)
    po = (n11 + n00) / rep_v1["n_instances"]
    p1 = (n11 + c["discordant_ref_only"]) / rep_v1["n_instances"]
    p2 = (n11 + c["discordant_system_only"]) / rep_v1["n_instances"]
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    out["baseline_30step_r1_vs_r2"] = {
        "source": "results/analysis_v1/report.json (baseline vs baseline_r2)",
        "table": {"both": n11, "r1_only": c["discordant_ref_only"], "r2_only": c["discordant_system_only"], "neither": n00},
        "rate_diff_r2_minus_r1": c["rate_diff"], "bootstrap95_json": c["bootstrap95"],
        "newcombe_paired95": [lo, hi], "mcnemar_exact_p_two_sided": mcnemar_exact(c["discordant_ref_only"], c["discordant_system_only"])[0],
        "percent_agreement": po, "cohen_kappa": (po - pe) / (1 - pe),
        "jaccard_resolved": n11 / (n11 + c["discordant_ref_only"] + c["discordant_system_only"]),
        "gee_or": c["gee"]["odds_ratio_system"], "gee_or95": c["gee"]["ci95"],
    }
    # NEAT r1 vs r2: only marginals; discordance not identifiable
    k1, k2, n = s["neat"]["resolved"], s["neat_r2"]["resolved"], rep["n_instances"]
    d, lo, hi = newcombe_unpaired(k2, n, k1, n)
    lo_both, hi_both = max(0, k1 + k2 - n), min(k1, k2)
    # range of exact McNemar p over all feasible discordance patterns (b - c fixed at k1 - k2)
    ps = []
    for both in range(lo_both, hi_both + 1):
        b, cc = k1 - both, k2 - both
        if b + cc + both <= n:
            ps.append(mcnemar_exact(b, cc)[0])
    out["neat_r1_vs_r2"] = {
        "rate_diff_r2_minus_r1": (k2 - k1) / n, "newcombe_unpaired95": [lo, hi],
        "unpaired_two_proportion_p": float(stats.chi2_contingency(np.array([[k1, n - k1], [k2, n - k2]]), correction=False).pvalue),
        "feasible_both_resolved_range": [lo_both, hi_both],
        "mcnemar_p_range_over_feasible_tables": [min(ps), max(ps)],
        "needs": "per-instance resolved flags of neat_best and neat_best_r2 (eval*.jsonl) to get the r1/r2 discordant counts",
    }
    # both NEAT replicates vs no-limit baseline, from the v2 JSON
    for sysn in ("neat", "neat_r2"):
        comp = next(c for c in rep["comparisons"] if c["system"] == sysn)
        out[f"{sysn}_vs_nolimit_json"] = {"rate_diff": comp["rate_diff"], "bootstrap95": comp["bootstrap95"]}
    return out


def proxy_block(pv, n_resolved):
    r, n = pv["pointbiserial_r"], pv["n"]
    n1 = n_resolved
    n0 = n - n1
    p = n1 / n
    m1, m0 = pv["mean_proxy_resolved"], pv["mean_proxy_unresolved"]
    # r_pb = (m1 - m0) * sqrt(p q) / s_total (population SD)  ->  s_total, within-group pooled SD, d
    s_total = (m1 - m0) * math.sqrt(p * (1 - p)) / r
    s_within = math.sqrt(s_total ** 2 - p * (1 - p) * (m1 - m0) ** 2)
    d = (m1 - m0) / s_within
    auc_se = hanley_mcneil_se(pv["auc"], n1, n0)
    out = {
        "n_pairs_pooled": n, "n_resolved": n1, "n_unresolved": n0,
        "auc": pv["auc"], "auc_hanley_mcneil95": [pv["auc"] - Z * auc_se, pv["auc"] + Z * auc_se],
        "r_pb": r, "r_pb_fisher95_n_pooled": list(fisher_z_ci(r, n)),
        "r_pb_fisher95_n_instances500": list(fisher_z_ci(r, 500)),
        "p_json": pv["p"],
        "mean_proxy_resolved": m1, "mean_proxy_unresolved": m0, "mean_diff": m1 - m0,
        "sd_total_derived": s_total, "sd_within_pooled_derived": s_within,
        "cohen_d_derived": d,
        "cohen_d_from_r_formula": r / math.sqrt(p * (1 - p) * (1 - r * r)),
        "auc_implied_d_binormal": math.sqrt(2) * stats.norm.ppf(pv["auc"]),
    }
    return out


def neat_block(rep):
    out = {}
    for run, d in rep["neat"].items():
        name = Path(run).name
        best = np.array(d["best_fitness_by_generation"], dtype=float)
        mean = np.array(d["mean_fitness_by_generation"], dtype=float)
        g = np.arange(len(best))
        batch = np.array([x[1] for x in d["mean_task_score_by_batch"]][: len(g)], dtype=float)

        def acf1(x):
            x = x - x.mean()
            return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))

        def dw(y):
            res = y - np.polyval(np.polyfit(g, y, 1), g)
            return float(np.sum(np.diff(res) ** 2) / np.sum(res ** 2))

        sp_best = stats.spearmanr(g, best)
        sp_mean = stats.spearmanr(g, mean)
        # Spearman of generation vs mean fitness after removing the minibatch's mean task score
        X = np.column_stack([np.ones_like(batch), batch])
        resid = mean - X @ np.linalg.lstsq(X, mean, rcond=None)[0]
        sp_resid = stats.spearmanr(g, resid)
        out[name] = {
            "generations": int(len(g)), "task_evaluations": d["task_evaluations"], "failure_rate": d["failure_rate"],
            "best_fitness": best.tolist(), "mean_fitness": mean.tolist(), "batch_mean_task_score": batch.tolist(),
            "spearman_json_batch_mean_vs_gen": d["spearman_generation_vs_mean"],
            "spearman_best_vs_gen": {"rho": float(sp_best.statistic), "p": float(sp_best.pvalue)},
            "spearman_mean_vs_gen": {"rho": float(sp_mean.statistic), "p": float(sp_mean.pvalue)},
            "spearman_mean_resid_batch_vs_gen": {"rho": float(sp_resid.statistic), "p": float(sp_resid.pvalue)},
            "pearson_mean_vs_batch": float(stats.pearsonr(mean, batch).statistic),
            "lag1_acf_best": acf1(best), "lag1_acf_mean": acf1(mean),
            "durbin_watson_mean_linear": dw(mean), "durbin_watson_best_linear": dw(best),
            "best_first_last_max": [float(best[0]), float(best[-1]), float(best.max()), int(best.argmax())],
            "mean_first_last_max": [float(mean[0]), float(mean[-1]), float(mean.max()), int(mean.argmax())],
        }
    return out


def selection_block(sel, graph_rep):
    val = {v["name"]: v for v in graph_rep["held_out"]["val40"]}
    rows = []
    for s in sel:
        v = val.get(s["name"], {})
        # implied run-level prior mean of the shrinkage (k = 6): shrunk = (n m + k mu) / (n + k)
        k = 6
        mu = (s["shrunk"] * (s["train_tasks"] + k) - s["train_tasks"] * s["train_mean"]) / k
        rows.append({
            "name": s["name"], "run": Path(s["run"]).name, "train_tasks": s["train_tasks"],
            "train_mean": s["train_mean"], "shrunk_mean": s["shrunk"], "implied_prior_mean_k6": mu,
            "workers": s["workers"], "types": s["types"],
            "val40_mean": v.get("val_mean"), "val40_fail": v.get("val_fail"),
            "val40_median_tokens": v.get("median_tokens"), "val40_wilcoxon_p_vs_baseline": v.get("wilcoxon_p_vs_base"),
        })
    ps = [r["val40_wilcoxon_p_vs_baseline"] for r in rows]
    for meth in ("bonferroni", "holm", "bh"):
        for r, a in zip(rows, p_adjust(ps, meth)):
            r[f"val40_p_{meth}"] = float(a)
    base = val["baseline (single editor agent)"]
    return {
        "candidates": rows,
        "baseline_val40": {"mean": base["val_mean"], "median_tokens": base["median_tokens"], "fail": base["val_fail"]},
        "winner": "neat_b_0_37b607a6",
        "nested_tree_val40": {"mean": 0.326, "winner_mean": 0.316, "wilcoxon_p": 0.46, "wins": 12, "ties": 20,
                              "losses": 8, "failures": 11, "winner_failures": 10,
                              "median_tokens": 918_000, "winner_median_tokens": 403_000,
                              "sign_test_p_two_sided": float(stats.binomtest(12, 20, 0.5).pvalue),
                              "source": "INTERVENTIONS.md and earlier experiment notes (no JSON)"},
    }


def graph_block(feat_csv, graph_rep):
    rows = list(csv.DictReader(open(feat_csv)))
    sig = [r for r in rows if r["q_bh"] and float(r["q_bh"]) < 0.05]
    sig.sort(key=lambda r: -abs(float(r["beta"])))
    top = [{
        "feature": r["feature"], "label": r["label"], "graphs_with": int(float(r["graphs_with"])) if r["graphs_with"] else None,
        "graphs_without": int(float(r["graphs_without"])) if r["graphs_without"] else None,
        "beta_pts": 100 * float(r["beta"]), "ci95_pts": [100 * float(r["ci_lo"]), 100 * float(r["ci_hi"])],
        "p_perm": float(r["p_perm"]), "q_bh": float(r["q_bh"]),
    } for r in sig]
    q1 = graph_rep["Q1_size"]
    size = [{"n_workers": s["n_workers"], "graphs": s["graphs"], "tasks": s["tasks"], "delta_pts": 100 * s["delta"],
             "ci95_pts": [100 * s["ci"][0], 100 * s["ci"][1]], "median_tokens": s["median_tokens"],
             "fail_rate": s["fail_rate"]} for s in q1["by_size"]]
    return {
        "n_features_tested": len(rows), "n_bh_significant_q05": len(sig), "top5_by_abs_effect": top[:5],
        "top5_with_ge10_graphs": [t for t in top if (t["graphs_with"] or 0) >= 10][:5],
        "size": size, "quadratic": q1["quadratic"], "linear_slope": q1["linear_slope"],
        "perm_p_linear": q1["perm_p_linear"], "tokens_pct_per_worker": q1["tokens_pct_per_worker"],
        "multivariable_n_workers": graph_rep["Q3_multivariable"]["n_workers"],
    }


# ----------------------------------------------------------------------------- figures

def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
        "font.family": "DejaVu Sans", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def savefig(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=150)
    fig.savefig(out / f"{stem}.pdf")


def colour(name):
    return {"neat": C_NEAT, "baseline_nolimit": C_BASE}.get(name, C_GREY)


def fig_main(rep, out):
    plt = setup_mpl()
    order = ["neat", "neat_r2", "baseline_nolimit", "baseline_30step", "baseline_30step_r2"]
    s = rep["systems"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.35, 3.3), gridspec_kw={"hspace": 0.55})
    y = np.arange(len(order))[::-1]
    for yi, name in zip(y, order):
        p, (lo, hi) = s[name]["rate"], s[name]["wilson95"]
        col = colour(name)
        prim = name in PRIMARY
        a.errorbar(100 * p, yi, xerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="o", color=col,
                   ms=5 if prim else 4, mfc=col if prim else "white", mew=1.2, elinewidth=1.4 if prim else 1.0,
                   capsize=2)
        a.text(100 * hi + 0.8, yi, f"{100 * p:.1f}%  ({s[name]['resolved']}/{rep['n_instances']})",
               va="center", fontsize=6.3, color=INK)
    a.set_yticks(y, [SHORT[n] for n in order])
    for t, name in zip(a.get_yticklabels(), order):
        if name in PRIMARY:
            t.set_fontweight("bold")
    a.set_xlim(0, 40)
    a.set_xlabel("Resolve rate on SWE-bench Verified (%), Wilson 95% CI")
    a.set_title("(a) Resolve rate (filled = primary comparison)", loc="left")
    a.grid(axis="x", color="#e4e3df", lw=0.5)
    a.set_axisbelow(True)

    vals = [s[n]["resolved_per_Mtok"] for n in order]
    for yi, name, v in zip(y, order, vals):
        prim = name in PRIMARY
        b.barh(yi, v, height=0.62, color=colour(name) if prim else C_GREY_L, edgecolor="none")
        b.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6.3, color=INK)
    b.set_yticks(y, [SHORT[n] for n in order])
    for t, name in zip(b.get_yticklabels(), order):
        if name in PRIMARY:
            t.set_fontweight("bold")
    b.set_xlim(0, 0.6)
    b.set_xlabel("Resolved instances per million tokens (all tokens)")
    b.set_title("(b) Token efficiency", loc="left")
    b.grid(axis="x", color="#e4e3df", lw=0.5)
    b.set_axisbelow(True)
    savefig(fig, out, "fig_main")
    plt.close(fig)


def fig_difficulty(diff, out):
    """Grouped bars; the 1-4 h (n=42) and >4 h (n=3) bins are merged into >1 h (n=45) for display."""
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    labels = None
    w = 0.36
    for off, name, col in ((-w / 2, "baseline_nolimit", C_BASE), (w / 2, "neat", C_NEAT)):
        bins = diff["per_system"][name]
        ks = [bins[0]["resolved"], bins[1]["resolved"], bins[2]["resolved"] + bins[3]["resolved"]]
        ns = [bins[0]["n"], bins[1]["n"], bins[2]["n"] + bins[3]["n"]]
        ci = [wilson(k, n) for k, n in zip(ks, ns)]
        r = np.array([c[0] for c in ci]) * 100
        lo = np.array([c[1] for c in ci]) * 100
        hi = np.array([c[2] for c in ci]) * 100
        x = np.arange(3)
        ax.bar(x + off, r, width=w - 0.03, color=col, label=SHORT[name], edgecolor="none")
        ax.errorbar(x + off, r, yerr=[np.clip(r - lo, 0, None), np.clip(hi - r, 0, None)], fmt="none",
                    ecolor=INK, elinewidth=0.8, capsize=1.8)
        for xi, k, n, ri, h in zip(x + off, ks, ns, r, hi):
            ax.text(xi, h + 1.0, f"{k}", ha="center", va="bottom", fontsize=5.8, color=INK2)
        labels = [f"<15 min\nn={ns[0]}", f"15 min–1 h\nn={ns[1]}", f">1 h\nn={ns[2]}"]
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Resolve rate (%), Wilson 95% CI")
    ax.set_xlabel("Task difficulty (annotated fix time); numbers above bars = resolved")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 45)
    savefig(fig, out, "fig_difficulty")
    plt.close(fig)


def fig_neat(neat, out):
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    cols = {"neat_a": C_NEAT, "neat_b": C_BASE}
    names = {"neat_a": "Run A", "neat_b": "Run B"}
    for run, d in neat.items():
        g = np.arange(d["generations"])
        col = cols.get(run, C_GREY)
        ax.plot(g, d["best_fitness"], "-", color=col, lw=1.5, marker="o", ms=2.6, label=f"{names.get(run, run)}: best")
        ax.plot(g, d["mean_fitness"], "--", color=col, lw=1.2, label=f"{names.get(run, run)}: mean")
    ax.axvline(5, color=INK2, lw=0.7, ls=":")
    ax.text(4.8, 0.03, "candidates selected\n(after gen 5)", fontsize=5.8, color=INK2, ha="right", va="bottom")
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Proxy fitness (patch similarity)")
    ax.set_ylim(0, 0.6)
    ax.set_xlim(-0.5, max(d["generations"] for d in neat.values()) - 0.5)
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
              handlelength=2.2)
    savefig(fig, out, "fig_neat_curves")
    plt.close(fig)


# ----------------------------------------------------------------------------- markdown

def fmt_ci(ci, scale=1.0, d=1, sign=False):
    f = "{:+.%df}" % d if sign else "{:.%df}" % d
    return f"{f.format(scale * ci[0])} to {f.format(scale * ci[1])}"


def write_markdown(S, out):
    L = []
    A = L.append
    A("# Report statistics (computed from committed summaries)\n")
    A("Generated by `scripts/report_stats.py` from `results/analysis_v2{,_gold}/report.json`, "
      "`results/analysis_v1/report.json`, `results/selection/selection.json` and `results/graph_analysis/`. "
      "No per-instance data was available locally; every number below is derived from aggregate counts. "
      "Each pairwise comparison gives both marginals and both discordant counts, so the **full paired 2x2 "
      "table is recoverable** and the paired effect sizes below are exact. Statistics that need per-instance "
      "data are listed in the last section.\n")
    A("**Primary analysis (pre-registered):** NEAT best r1 vs the no-step-limit baseline, one-sided exact McNemar "
      "(H1: NEAT resolves more). Everything else is secondary/descriptive.\n")

    # 1. multiple comparisons
    A("## 1. Multiple-comparison correction (four comparisons vs the no-limit baseline)\n")
    A("Exact McNemar p recomputed as a binomial test on the discordant pairs (c of b+c, p=0.5). The "
      "`0.0000` in report.md is rounding; the recomputed values match report.json to machine precision. "
      "Adjustment is over the family of four comparisons in each subset.\n")
    for subset in ("all500", "gold464"):
        rows = S["comparisons"][subset]
        A(f"**{ 'All 500 instances' if subset == 'all500' else 'Gold-resolvable 464'}**\n")
        A("| comparison | discordant (ref only / sys only) | p one-sided (H1: sys > ref) | Bonferroni | Holm | BH q | p two-sided | Holm (two-sided) |")
        A("|---|---|---|---|---|---|---|---|")
        for r in rows:
            nm = f"{SHORT[r['system']]} − {SHORT[r['reference']]}" + (" **(primary)**" if r["primary"] else "")
            A(f"| {nm} | {r['table']['ref_only']} / {r['table']['system_only']} | "
              f"{sci(r['mcnemar_exact_p_one_sided_greater'])} | {sci(r['mcnemar_exact_p_one_sided_greater_bonferroni'])} | "
              f"{sci(r['mcnemar_exact_p_one_sided_greater_holm'])} | {sci(r['mcnemar_exact_p_one_sided_greater_bh'])} | "
              f"{sci(r['mcnemar_exact_p_two_sided'])} | {sci(r['mcnemar_exact_p_two_sided_holm'])} |")
        A("")
    A("The one-sided alternative is only meaningful for the NEAT comparisons; for the two 30-step-baseline "
      "comparisons the question is two-sided (does the step limit change anything?), so their two-sided "
      "Holm-adjusted p is the one to quote (both ≈0.76, n.s.). The primary result survives any correction "
      "by ~12 orders of magnitude.\n")

    # 2. effect sizes
    A("## 2. Effect sizes (all 500; paired 2x2 reconstructed from marginals + discordant counts)\n")
    A("| comparison | both / ref only / sys only / neither | RD pts (bootstrap 95%) | RD pts (Newcombe paired 95%) | extra resolved per 100 | NNT | RR (paired 95%) | Cohen's h | GEE OR adj. difficulty (95%) | paired OR c/b (exact 95%) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in S["comparisons"]["all500"]:
        t = r["table"]
        nm = f"{SHORT[r['system']]} − {SHORT[r['reference']]}"
        A(f"| {nm} | {t['both']} / {t['ref_only']} / {t['system_only']} / {t['neither']} | "
          f"{100 * r['risk_difference']:+.1f} ({fmt_ci(r['rd_bootstrap95_json'], 100, 1, True)}) | "
          f"{fmt_ci(r['rd_newcombe_paired95'], 100, 1, True)} | {r['extra_resolved_per_100']:+.1f} | "
          f"{'%.1f' % r['nnt'] if r['nnt'] else '–'} | {r['relative_risk']:.2f} ({fmt_ci(r['rr_paired95'], 1, 2)}) | "
          f"{r['cohen_h']:+.2f} | {r['gee_or']:.2f} ({fmt_ci(r['gee_or95'], 1, 2)}) | "
          f"{r['paired_or_c_over_b']:.2f} ({fmt_ci(r['paired_or_exact95'], 1, 2)}) |")
    A("")
    p = S["comparisons"]["all500"][0]
    A(f"Primary: NEAT resolves {p['extra_resolved_per_100']:.1f} more instances per 100 (one extra resolution "
      f"for every {p['nnt']:.1f} instances), {p['relative_risk']:.2f}× the baseline rate. Cohen's h = "
      f"{p['cohen_h']:.2f} is a small-to-medium effect by Cohen's labels, which understate it here because both "
      "rates are low. The GEE OR is a population-averaged (marginal) OR; the paired OR c/b is the "
      "conditional (within-instance) OR, which is larger as expected.\n")
    A("Gold-resolvable subset (464): primary RD " +
      f"{100 * S['comparisons']['gold464'][0]['risk_difference']:+.1f} pts "
      f"(bootstrap {fmt_ci(S['comparisons']['gold464'][0]['rd_bootstrap95_json'], 100, 1, True)}), RR "
      f"{S['comparisons']['gold464'][0]['relative_risk']:.2f}, h {S['comparisons']['gold464'][0]['cohen_h']:.2f}, "
      f"GEE OR {S['comparisons']['gold464'][0]['gee_or']:.2f}. The discordant counts are identical to the full set "
      "(no discordant pair lies among the 36 gold-failing instances), so the McNemar p-values are unchanged.\n")

    # 3. difficulty
    D = S["difficulty"]
    A("## 3. Difficulty-stratified analysis\n")
    A("Resolve rate (%) with Wilson 95% CI per difficulty bin (bins are SWE-bench Verified's annotated fix time).\n")
    A("| system | " + " | ".join(f"{DIFF_SHORT[d]} (n={D['per_system']['neat'][i]['n']})" for i, d in enumerate(DIFF_ORDER)) + " | Cochran–Armitage trend z, p (4 bins) | χ² (3 bins, >1 h merged) |")
    A("|---|" + "---|" * (len(DIFF_ORDER) + 2))
    for name in ("neat", "neat_r2", "baseline_nolimit", "baseline_30step", "baseline_30step_r2"):
        bins = D["per_system"][name]
        tr = D["trend"][name]
        cells = [f"{b['resolved']}: {pct(b['rate'])} ({pct(b['wilson95'][0])}–{pct(b['wilson95'][1])})" for b in bins]
        A(f"| {SHORT[name]} | " + " | ".join(cells) +
          f" | z={tr['cochran_armitage_z_4bins']:.2f}, p={sci(tr['cochran_armitage_p_4bins'])} | "
          f"χ²({tr['chi2_df_3bins']})={tr['chi2_3bins']:.1f}, p={sci(tr['chi2_p_3bins'])} |")
    A("")
    A("Trend scores 0,1,2,3 for <15 min, 15 min–1 h, 1–4 h, >4 h. Every system shows a strong decreasing trend. "
      "The 4-bin χ² has expected counts <5 in the >4 h bin, so the 3-bin χ² (1–4 h and >4 h merged) is reported.\n")
    A("**NEAT r1 − no-limit baseline by bin** (unpaired approximation: within-bin pairing is not recoverable "
      "from the JSON, so these CIs ignore the positive within-instance correlation and are conservative):\n")
    A("| bin | n | baseline % | NEAT % | RD pts (Newcombe 95%) | RD pts (Wald 95%) | RR | Cohen's h |")
    A("|---|---|---|---|---|---|---|---|")
    for r in D["rd_neat_vs_nolimit"]:
        A(f"| {DIFF_SHORT[r['bin']]} | {r['n']} | {pct(r['rate_ref'])} | {pct(r['rate_system'])} | "
          f"{100 * r['rd']:+.1f} ({fmt_ci(r['newcombe95'], 100, 1, True)}) | {fmt_ci(r['wald95'], 100, 1, True)} | "
          f"{'%.2f' % r['rr'] if r['rr'] else '∞ (0 baseline)' if r['rate_system'] > 0 else '–'} | {r['cohen_h']:+.2f} |")
    A("")
    mh = D["mh"]
    if "mh_or" in mh:
        A(f"Mantel–Haenszel common OR across bins (unpaired) = {mh['mh_or']:.2f} (95% {fmt_ci(mh['mh_or95'], 1, 2)}), "
          f"CMH χ² = {mh['cmh_chi2']:.1f}, p = {sci(mh['cmh_p'])}. Breslow–Day homogeneity on the two informative "
          f"bins (<15 min, 15 min–1 h): χ²(1) = {mh['breslow_day_2strata_chi2']:.2f}, p = {sci(mh['breslow_day_2strata_p'])}"
          " — no evidence that the NEAT advantage (on the OR scale) differs between easy and medium tasks. "
          "In absolute terms the gain is similar (≈+20 pts vs ≈+16 pts); in relative terms it is larger on medium "
          "tasks (RR ≈3.7 vs ≈2.4). On the 45 tasks rated >1 h both systems are near floor (NEAT 2/45, baseline 0/45).\n")

    # 4. tokens
    T, R = S["tokens"], S["token_ratios"]
    A("## 4. Cost–performance (token efficiency)\n")
    A("“All tokens” = prompt + completion including prefix-cache hits; “uncached” subtracts cached prompt "
      "tokens (closer to marginal compute). Uncached totals are derived as resolved ÷ (resolved per uncached Mtok).\n")
    A("| system | resolved | median tok/instance | mean tok/instance | total tokens | uncached total (derived) | resolved / Mtok (all) | resolved / Mtok (uncached) | Mtok per resolved |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name in ("neat", "neat_r2", "baseline_nolimit", "baseline_30step", "baseline_30step_r2"):
        t = T[name]
        A(f"| {SHORT[name]} | {t['resolved']} | {t['tokens_median'] / 1e3:.0f}K | {t['tokens_mean'] / 1e3:.0f}K | "
          f"{t['tokens_total'] / 1e6:.0f}M | {t['tokens_uncached_total_derived'] / 1e6:.0f}M | "
          f"{t['resolved_per_Mtok']:.3f} | {t['resolved_per_Mtok_uncached']:.3f} | {t['Mtok_per_resolved']:.1f} |")
    A("")
    A("| ratio NEAT r1 / reference | resolve rate | median tokens | total tokens | resolved/Mtok (all) | resolved/Mtok (uncached) |")
    A("|---|---|---|---|---|---|")
    for k, r in R.items():
        A(f"| {k.replace('neat/', 'NEAT r1 / ').replace('baseline_nolimit', 'no-limit baseline').replace('baseline_30step', '30-step baseline r1')} | "
          f"{r['rate']:.2f}× | {r['tokens_median']:.2f}× | {r['tokens_total']:.2f}× | {r['resolved_per_Mtok']:.1f}× | {r['resolved_per_Mtok_uncached']:.1f}× |")
    A("")
    A("Against the primary reference NEAT dominates on both axes: 2.9× the resolve rate at 0.63× the median "
      "(0.17× the total) tokens, i.e. ~17× more resolutions per million tokens. Against the 30-step baseline NEAT "
      "costs more per instance (1.2× median, 1.3× total) but still resolves 2.5× more per Mtok. The no-limit "
      "baseline's mean (≈3.0M) is 4× its median (701K): a heavy tail of runs that exhaust the 64K context for many "
      "steps without producing an edit drives its total. The median of per-instance token ratios (Wilcoxon, in "
      "report.json) is 0.61 for NEAT vs no-limit (p = 2.4e-22).\n")

    # 5. replicates
    P = S["replicates"]
    b = P["baseline_30step_r1_vs_r2"]
    n = P["neat_r1_vs_r2"]
    A("## 5. Replicate agreement\n")
    A("| pair | r2 − r1 (pts) | 95% CI | test | agreement |")
    A("|---|---|---|---|---|")
    A(f"| Baseline 30-step r2 vs r1 | {100 * b['rate_diff_r2_minus_r1']:+.1f} | bootstrap {fmt_ci(b['bootstrap95_json'], 100, 1, True)}; "
      f"Newcombe paired {fmt_ci(b['newcombe_paired95'], 100, 1, True)} | McNemar exact p = {b['mcnemar_exact_p_two_sided']:.2f} "
      f"(discordant {b['table']['r1_only']} vs {b['table']['r2_only']}) | {b['table']['both']} resolved by both; "
      f"Cohen's κ = {b['cohen_kappa']:.2f}; Jaccard of resolved sets = {b['jaccard_resolved']:.2f} |")
    A(f"| NEAT r2 vs r1 | {100 * n['rate_diff_r2_minus_r1']:+.1f} | Newcombe **unpaired** {fmt_ci(n['newcombe_unpaired95'], 100, 1, True)} "
      f"| unpaired χ² p = {n['unpaired_two_proportion_p']:.2f} | discordance unknown |")
    A("")
    A("Baseline r1-vs-r2 discordance comes from `results/analysis_v1/report.json`, where baseline r1 was the "
      "reference. For NEAT r1 vs r2 the JSON only has marginals; the number resolved by both can be anywhere in "
      f"[{n['feasible_both_resolved_range'][0]}, {n['feasible_both_resolved_range'][1]}], and the exact McNemar p "
      f"ranges over [{n['mcnemar_p_range_over_feasible_tables'][0]:.3f}, {n['mcnemar_p_range_over_feasible_tables'][1]:.2f}] "
      "across those tables, so it cannot be bounded usefully. Needed: per-instance resolved flags for `neat_best` "
      "and `neat_best_r2`. Both NEAT replicates beat the no-limit baseline by similar margins "
      f"(+{100 * P['neat_vs_nolimit_json']['rate_diff']:.1f}, bootstrap {fmt_ci(P['neat_vs_nolimit_json']['bootstrap95'], 100, 1, True)}; "
      f"+{100 * P['neat_r2_vs_nolimit_json']['rate_diff']:.1f}, {fmt_ci(P['neat_r2_vs_nolimit_json']['bootstrap95'], 100, 1, True)}). "
      "Note that baseline κ is only moderate: the *same* system resolves a substantially different set of "
      "instances on a re-run, even though the totals agree closely.\n")

    # 6. proxy
    X = S["proxy"]
    A("## 6. Proxy validity (training proxy vs execution resolution)\n")
    A(f"- Pooled over the five systems: n = {X['n_pairs_pooled']} (instance, system) pairs, {X['n_resolved']} resolved.\n"
      f"- AUC = {X['auc']:.3f} (Hanley–McNeil 95% {fmt_ci(X['auc_hanley_mcneil95'], 1, 3)}).\n"
      f"- Point-biserial r = {X['r_pb']:.3f}; Fisher-z 95% CI {fmt_ci(X['r_pb_fisher95_n_pooled'], 1, 3)} with n = {X['n_pairs_pooled']}, "
      f"or {fmt_ci(X['r_pb_fisher95_n_instances500'], 1, 3)} with the conservative n = 500 instances (pairs share instances, "
      "so the pooled n overstates precision).\n"
      f"- Mean proxy: resolved {X['mean_proxy_resolved']:.3f} vs unresolved {X['mean_proxy_unresolved']:.3f} (difference {X['mean_diff']:.3f}).\n"
      f"- Cohen's d = {X['cohen_d_derived']:.2f}. Per-group SDs are **not** in the JSON; d is derived exactly from r_pb, the group "
      f"means and group sizes (pooled within-group SD {X['sd_within_pooled_derived']:.3f}, population-SD convention). "
      f"It assumes a common within-group SD; Glass's Δ or a heteroscedastic d needs per-group SDs (cluster). "
      f"Cross-check: the binormal d implied by the AUC is {X['auc_implied_d_binormal']:.2f}.\n")

    # 7. NEAT curves
    A("## 7. NEAT learning curves\n")
    A("| run | gens | task evals | failure rate | best: gen0 → last (max @gen) | mean: gen0 → last | Spearman ρ best~gen (p) | Spearman ρ mean~gen (p) | report.json ρ batch-mean~gen (p) | lag-1 ACF mean | r(mean, batch mean score) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for run, d in S["neat"].items():
        bf, mf = d["best_first_last_max"], d["mean_first_last_max"]
        sj = d["spearman_json_batch_mean_vs_gen"]
        A(f"| {run} | {d['generations']} | {d['task_evaluations']} | {d['failure_rate']:.2f} | {bf[0]:.3f} → {bf[1]:.3f} ({bf[2]:.3f} @{bf[3]}) | "
          f"{mf[0]:.3f} → {mf[1]:.3f} | {d['spearman_best_vs_gen']['rho']:+.2f} ({d['spearman_best_vs_gen']['p']:.2f}) | "
          f"{d['spearman_mean_vs_gen']['rho']:+.2f} ({d['spearman_mean_vs_gen']['p']:.2f}) | {sj['rho']:+.2f} ({sj['p']:.2f}) | "
          f"{d['lag1_acf_mean']:+.2f} | {d['pearson_mean_vs_batch']:.2f} |")
    A("")
    A("No run shows a monotone improvement after generation 1. The curves are dominated by **minibatch "
      "difficulty**: each generation is scored on a different 6–8-task minibatch, and population mean fitness "
      "tracks the minibatch's mean task score almost perfectly (last column), so generation-to-generation "
      "movement is mostly batch luck (e.g. run A's spike at gen 8, batch mean 0.43, and collapse at gen 9, 0.03). "
      "After regressing out the batch mean, the generation trend is still null "
      f"(A ρ = {S['neat']['neat_a']['spearman_mean_resid_batch_vs_gen']['rho']:+.2f}, p = {S['neat']['neat_a']['spearman_mean_resid_batch_vs_gen']['p']:.2f}; "
      f"B ρ = {S['neat']['neat_b']['spearman_mean_resid_batch_vs_gen']['rho']:+.2f}, p = {S['neat']['neat_b']['spearman_mean_resid_batch_vs_gen']['p']:.2f}). "
      "**Autocorrelation:** consecutive generations share elites (whose running-mean fitness is carried over "
      "and re-scored) and descend from each other, so the series are not independent and the Spearman p-values "
      "(which assume exchangeable observations) are only descriptive; lag-1 ACF and Durbin–Watson statistics are "
      "in summary.json. Run A's speciation threshold was reset after gen 2, and candidates were selected at gen 5; "
      "later generations (B ran to gen 18) were not used for selection.\n")

    # 8. selection
    Sel = S["selection"]
    A("## 8. Selection and validation (val40)\n")
    A("| candidate | run | workers | types | train tasks | train mean | shrunk mean (k=6) | implied prior mean | val40 mean | val40 fail | val40 median tok | Wilcoxon p vs baseline | Holm (4) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in Sel["candidates"]:
        star = " **(winner)**" if r["name"] == Sel["winner"] else ""
        A(f"| {r['name']}{star} | {r['run']} | {r['workers']} | {'+'.join(r['types'])} | {r['train_tasks']} | {r['train_mean']:.3f} | "
          f"{r['shrunk_mean']:.3f} | {r['implied_prior_mean_k6']:.3f} | {r['val40_mean']:.3f} | {r['val40_fail']:.2f} | "
          f"{r['val40_median_tokens'] / 1e3:.0f}K | {r['val40_wilcoxon_p_vs_baseline']:.3f} | {r['val40_p_holm']:.3f} |")
    bl = Sel["baseline_val40"]
    A(f"| baseline (single agent) | – | 1 | editor | – | – | – | – | {bl['mean']:.3f} | {bl['fail']:.2f} | {bl['median_tokens'] / 1e3:.0f}K | – | – |")
    A("")
    nt = Sel["nested_tree_val40"]
    A(f"The pre-registered rule (highest val40 mean proxy) picked neat_b_0_37b607a6: 0.316 vs baseline 0.211, paired "
      f"Wilcoxon p = 0.025. That p is uncorrected and is computed on the same 40 instances used to choose the "
      f"winner among four candidates; Holm over the four candidate-vs-baseline tests gives p = "
      f"{next(r['val40_p_holm'] for r in Sel['candidates'] if r['name'] == Sel['winner']):.3f}, so val40 alone does not "
      "establish superiority — the confirmatory evidence is the independent 500-instance test. The shrinkage prior "
      "mean implied by the reported numbers is run-specific (≈0.171 for A, ≈0.281 for B), i.e. candidates were "
      "shrunk toward their own run's mean. "
      f"Hand-built nested tree vs winner on val40: {nt['mean']:.3f} vs {nt['winner_mean']:.3f}, Wilcoxon p = {nt['wilcoxon_p']}, "
      f"{nt['wins']} wins / {nt['ties']} ties / {nt['losses']} losses (sign test on non-ties p = {nt['sign_test_p_two_sided']:.2f}), "
      f"{nt['median_tokens'] / 1e3:.0f}K vs {nt['winner_median_tokens'] / 1e3:.0f}K median tokens (≈2.3×).\n")

    # 9. graph analysis
    G = S["graph"]
    A("## 9. Graph-analysis highlights (for the Discussion)\n")
    A(f"{G['n_bh_significant_q05']} of {G['n_features_tested']} features are BH-significant at q < 0.05. "
      "Effects are Δ proxy points (score ×100) relative to other genomes on the same instance and minibatch, "
      "graph-clustered 95% CIs.\n")
    A("Five largest |effect|, all q < 0.05:\n")
    A("| feature | graphs with / without | Δ pts (95% CI) | p_perm | q (BH) |")
    A("|---|---|---|---|---|")
    for t in G["top5_by_abs_effect"]:
        A(f"| {t['label']} | {t['graphs_with']} / {t['graphs_without']} | {t['beta_pts']:+.1f} ({fmt_ci(t['ci95_pts'], 1, 1, True)}) | {t['p_perm']:.4f} | {t['q_bh']:.3f} |")
    A("")
    A("Several of these rest on very few graphs (5–9 with the feature), so they are fragile. The five largest among "
      "features carried by ≥10 graphs are more robust candidates for the Discussion:\n")
    A("| feature | graphs with / without | Δ pts (95% CI) | q (BH) |")
    A("|---|---|---|---|")
    for t in G["top5_with_ge10_graphs"]:
        A(f"| {t['label']} | {t['graphs_with']} / {t['graphs_without']} | {t['beta_pts']:+.1f} ({fmt_ci(t['ci95_pts'], 1, 1, True)}) | {t['q_bh']:.3f} |")
    A("")
    A("**Size sweet spot.** Δ by number of workers:\n")
    A("| workers | graphs | tasks | Δ pts (95% CI) | median tokens | task fail rate |")
    A("|---|---|---|---|---|---|")
    for s in G["size"]:
        A(f"| {s['n_workers']} | {s['graphs']} | {s['tasks']} | {s['delta_pts']:+.1f} ({fmt_ci(s['ci95_pts'], 1, 1, True)}) | {s['median_tokens'] / 1e3:.0f}K | {s['fail_rate']:.2f} |")
    A("")
    q = G["quadratic"]
    mv = G["multivariable_n_workers"]
    A(f"Three workers is the only size with a CI entirely above zero (+3.9 pts). No linear trend (slope "
      f"{100 * G['linear_slope'][0]:+.1f} pts/worker, permutation p = {G['perm_p_linear']:.2f}) but a clear inverted U "
      f"(quadratic p = {q['p_b2']:.1e}, vertex {q['vertex']:.1f} workers); tokens rise {G['tokens_pct_per_worker']:.0f}% per worker. "
      f"Holding composition fixed, each extra worker costs {100 * mv['beta']:+.1f} pts ({fmt_ci(mv['ci'], 100, 1, True)}). "
      "The winner (3 workers: two editors into a pick-merger) sits exactly at the sweet spot.\n")
    return L


def needs_cluster_md():
    return [
        "## Needs cluster data\n",
        "These require per-instance predictions/evaluations (`$ACE/runs/verified/*/preds*.jsonl`, `eval*.jsonl`) "
        "and `data/verified.jsonl`; `scripts/report_stats_cluster.py` computes all of them (written to match "
        "`scripts/analyze.py`'s loaders; smoke-tested on synthetic data only, not yet run on the real runs):\n",
        "1. **McNemar within difficulty strata** (exact, per bin) and the paired RD with Newcombe paired CI per bin; "
        "stratified (Mantel–Haenszel-type) paired test and a GEE with a system × difficulty interaction. Section 3 uses an "
        "unpaired approximation.",
        "2. **Replicate discordance** NEAT r1 vs r2 (McNemar, κ, Jaccard); also NEAT r2 vs baseline r1/no-limit tables are "
        "available but r1-vs-r2 is not.",
        "3. **Bootstrap CI on resolved/Mtok** per system and on the NEAT / baseline ratio (paired instance bootstrap), all "
        "tokens and uncached. Section 4 has point estimates only.",
        "4. **Cohen's d on the proxy with per-group SDs** (and Glass's Δ), plus an instance-clustered bootstrap CI for AUC "
        "and r_pb that respects the 5 systems sharing each instance. Section 6 derives d assuming a pooled SD.",
        "5. **Holm/BH on the token Wilcoxon tests** is computable here, but effect sizes for them (matched-pairs rank-biserial "
        "r, Hodges–Lehmann median ratio with CI) need the per-instance token vectors.",
        "6. Conditional (within-instance) analyses of **empty patches and agent failures** (e.g. resolve rate of the baseline on "
        "instances where NEAT's pick-merger failed, paired by instance).",
        "7. NEAT curves corrected for minibatch difficulty at the task level (instance fixed effects) need "
        "`$ACE/runs/neat_{a,b}/evaluations.jsonl`; this is partly done in `results/graph_analysis`.",
        "",
    ]


def inconsistencies_md(S):
    return [
        "## Inconsistencies and caveats noticed in existing results\n",
        "- `report.md` prints McNemar p as `0.0000`; the exact values are 9.1e-15 (NEAT r1) and 5.8e-13 (NEAT r2). "
        "the earlier experiment notes' “p < 1e-4” is true but understates them by 10 orders of magnitude.",
        "- the earlier experiment notes' proxy AUC “0.88” is the v1 value (2000 pairs); v2 (2500 pairs incl. the no-limit baseline) gives 0.891.",
        "- the earlier experiment notes stated both “NEAT uses 0.61× the tokens” (vs no-limit, median of per-instance ratios) and "
        "“about 1.4–1.5× the baseline's tokens” (vs the 30-step baseline, from v1). Both are right but the reference must "
        "be named each time. Ratio of medians (0.63×) and ratio of totals (0.17×) also differ a lot from 0.61× because "
        "of the no-limit baseline's heavy token tail.",
        "- Uncached token medians differ 2.5× between the two identical 30-step baseline runs (44K vs 108K) while all-token "
        "medians agree (362K vs 368K): the cache-hit rate depended on cluster load (see INTERVENTIONS.md thrash entries). "
        "Uncached efficiency figures therefore reflect serving conditions as much as the pipeline.",
        "- baseline_30step r1 has 2 eval `error` statuses counted as unresolved; NEAT r1/r2 have 95/93 agent failures "
        "(pick-merger) counted as unresolved. Both conventions are conservative for the respective system but should be stated.",
        "- The NEAT block in analysis_v2 (A 13, B 19 generations) and the graph-analysis snapshot (A 11, B 13 generations) "
        "are different snapshots of still-running jobs; numbers quoted from the two sources will not match exactly.",
        "- `spearman_generation_vs_mean` in report.json is computed on the per-batch mean task score (all genomes evaluated on "
        "that batch, including one more batch than logged generations), not on `mean_fitness_by_generation`; both are "
        "reported in Section 7.",
        "- Proxy validity pools 5 systems × 500 instances as if independent (n = 2500); the p = 3.7e-181 is overstated.",
        "- The val40 p = 0.025 is uncorrected for four candidates and is computed on the selection set (winner's curse); "
        "Holm-adjusted it is ≈0.10.",
        "- The 30-step baseline comparisons were tested one-sided in the direction “30-step > no-limit”, which was not a "
        "hypothesis anyone held; quote the two-sided p.",
        "- django__django-10097 is “resolved” by an empty patch (gold_validation); if any system's run resolved it with an "
        "empty patch it is a free point — check on the cluster.",
        "",
    ]


METHODS = """# Statistical methods (draft)

The primary, pre-registered analysis compared the NEAT-selected graph (first run) with the no-step-limit baseline on all 500 SWE-bench Verified instances using a one-sided exact McNemar test on the paired discordant outcomes. Secondary comparisons (the second NEAT run and both 30-step baseline runs, each against the same reference) were adjusted with the Holm procedure, and Benjamini–Hochberg q-values were also reported. Resolve rates were summarised with Wilson 95% confidence intervals; effect sizes were the risk difference with a paired bootstrap 95% CI (10,000 resamples), the relative risk, Cohen's h, the conditional odds ratio from discordant pairs with an exact CI, and a difficulty-adjusted odds ratio from a GEE logistic model clustered by instance. Token use was compared with the Wilcoxon signed-rank test and summarised as resolved instances per million tokens. Trends across difficulty bins were tested with the Cochran–Armitage test. Proxy validity was assessed with the AUC and the point-biserial correlation (Fisher-z CI). NEAT learning curves were tested with Spearman correlation and treated as descriptive. Analyses were repeated on the 464 gold-resolvable instances.
"""


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/report_stats"))
    args = ap.parse_args()
    R = args.results
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((R / "analysis_v2/report.json").read_text())
    gold = json.loads((R / "analysis_v2_gold/report.json").read_text())
    v1 = json.loads((R / "analysis_v1/report.json").read_text())
    sel = json.loads((R / "selection/selection.json").read_text())
    grep_ = json.loads((R / "graph_analysis/report.json").read_text())

    S = {
        "sources": {"main": str(R / "analysis_v2/report.json"), "gold": str(R / "analysis_v2_gold/report.json"),
                    "replicate_baseline": str(R / "analysis_v1/report.json")},
        "primary": {"reference": PRIMARY[0], "system": PRIMARY[1], "test": "exact McNemar, one-sided (system > reference)"},
        "comparisons": {"all500": comparisons_block(rep, "all500"), "gold464": comparisons_block(gold, "gold464")},
        "difficulty": difficulty_block(rep),
        "replicates": replicate_block(rep, v1),
        "proxy": proxy_block(rep["proxy_validity"], sum(s["resolved"] for s in rep["systems"].values())),
        "neat": neat_block(rep),
        "selection": selection_block(sel, grep_),
        "graph": graph_block(R / "graph_analysis/features.csv", grep_),
    }
    S["tokens"], S["token_ratios"] = token_block(rep)
    S["difficulty_gold"] = difficulty_block(gold)["trend"]

    # sanity: recomputed exact p equals report.json
    for subset in S["comparisons"].values():
        for r in subset:
            assert math.isclose(r["mcnemar_exact_p_one_sided_greater"], r["json_p_one_sided"], rel_tol=1e-9)
            assert math.isclose(r["mcnemar_exact_p_two_sided"], r["json_p_two_sided"], rel_tol=1e-9)

    fig_main(rep, out)
    fig_difficulty(S["difficulty"], out)
    fig_neat(S["neat"], out)
    S["figures"] = [str(out / f"{s}.{e}") for s in ("fig_main", "fig_difficulty", "fig_neat_curves") for e in ("png", "pdf")]

    md = write_markdown(S, out)
    md += ["## 10. Figures\n",
           "- `fig_main.png/pdf`: (a) resolve rate per system with Wilson 95% CIs, primary pair filled; (b) resolved per million tokens.",
           "- `fig_difficulty.png/pdf`: resolve rate by difficulty bin, NEAT r1 vs no-limit baseline, Wilson 95% CIs (1–4 h and >4 h merged into >1 h, n = 45, for display; the >4 h bin has n = 3).",
           "- `fig_neat_curves.png/pdf`: best (solid) and mean (dashed) proxy fitness per generation, run A (pop 16, batch 6) and run B (pop 12, batch 8); dotted line = selection point.",
           ""]
    md += needs_cluster_md()
    md += inconsistencies_md(S)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    (out / "stats_methods_paragraph.md").write_text(METHODS)
    words = len(METHODS.split("\n", 2)[2].split())
    print(f"wrote {out}/summary.md, summary.json, figures; methods paragraph {words} words")


if __name__ == "__main__":
    main()
