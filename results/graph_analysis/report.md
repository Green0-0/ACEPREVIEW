# Which evolved worker graphs were effective? (secondary analysis of NEAT runs A and B)

Snapshot: neat_a 1313 / neat_b 1432 task records (11 / 13 generations logged); records of genomes not yet logged skipped (94 / 88); mock records skipped (0 / 0). 2563 records used (centred cells need ≥2 genomes); distinct graphs: neat_a 103, neat_b 127 (identical graph JSON pooled).

**Outcome.** Δ = proxy score minus the mean score of all genomes that attempted the same instance in the same minibatch (instance fixed effect). Minibatch means range 0.03–0.43; instance cells explain 53% of score variance. Effects are in score points (proxy ×100). **Inference.** Task-level OLS Δ ~ x + run with cluster-robust SEs by graph (CIs); stratified permutation test across graphs within run (p_perm); Benjamini–Hochberg over all 127 features (q). 'gen0' = estimate from the generation-0 minibatch only (random initial genomes + bootstrap mutants; no selection).

## Key findings

- **Size sweet spot at 3 workers.** Δ = +3.9 pts at 3 workers vs -7.4 at 1 and -1.1 at 4 (no linear trend, p=0.74; quadratic p=9e-07). Holding composition fixed, each extra worker shifts Δ by -2.0 pts, and tokens rise 25% per worker. Run A drifted to larger graphs (3.5→5.2 workers); run B stayed near 3.
- **What helped:** a PICKMERGER feeding END +4.9 pts (95% CI +2.8 to +7.0, q=0.011), an EDITOR→EDITOR chain +5.0 pts (95% CI +3.0 to +7.0, q=0.0063), more editors +2.7 pts (95% CI +1.1 to +4.3, q=0.02). **What hurt:** a VIEWEDITOR -8.4 pts (95% CI -12.0 to -4.8, q=0.0063), a lone EDITOR at END -6.0 pts (95% CI -8.5 to -3.5, q=0.012). CRITIC, VIEWER and LISTVIEWER presence had no reliable univariate effect.
- **Settings:** PICKMERGER ace_format=element -19.2 pts (95% CI -27.2 to -11.1, q=0.012) and context_budget=12000 -8.8 pts (95% CI -12.5 to -5.1, q=0.0063); EDITOR max_steps=60 -4.5 pts (95% CI -7.5 to -1.5, q=0.019), context_budget=6000 +6.4 pts (95% CI +2.0 to +10.7, q=0.039). Most prompt-text options are not distinguishable after BH.
- **Failures drive fitness.** Tool-using workers never fail; ACE writers fail per execution vieweditor 57%, critic 50%, pickmerger 31% vs JUDGEMERGER 8%. Line-addressable ACE formats fail 13.5 pts less often. PICKMERGER graphs still win on Δ despite failing more, so its successful merges must outscore picking a single candidate.
- **Winner on Verified:** 95/500 (19%) graph failures, all in the PICKMERGER; 55 are overlapping ACE blocks (43 of them the same block emitted twice). Resolve rate on non-failed instances is 30.4%. Note its topology: editor 0 (not a VIEWER) passes VIEW+REPORT to editor 2, and both edits go to the PICKMERGER.
- **Lineage:** the winner's lineage (nodes 0, 2, 19) held 25–58% of run B's population through generation 8, then went extinct at generation 9 when stagnation pruning removed its 3 species (11 of 12 genomes). A record set on a lucky minibatch at gen 2 could not be beaten on later batches.

## 1. Size

| workers | graphs | tasks | Δ (95% CI) | raw score | median tokens | task fail rate |
|---|---|---|---|---|---|---|
| 1 | 15 | 140 | -7.4 (-11.2, -4.2) | 0.156 | 241k | 0.00 |
| 2 | 43 | 465 | -2.0 (-3.3, -0.9) | 0.196 | 305k | 0.22 |
| 3 | 79 | 841 | +3.9 (+2.1, +5.5) | 0.302 | 522k | 0.18 |
| 4 | 53 | 637 | -1.1 (-3.5, +1.0) | 0.191 | 578k | 0.55 |
| 5 | 21 | 231 | -0.8 (-5.4, +3.3) | 0.179 | 454k | 0.55 |
| 6 | 15 | 225 | -1.2 (-3.8, +1.0) | 0.163 | 991k | 0.57 |
| 7 | 3 | 18 | -5.8 (-12.4, -2.5) | 0.030 | 544k | 0.83 |
| 8 | 1 | 6 | -8.2 (-8.2, -8.2) | 0.000 | 279k | 1.00 |

Linear slope +0.1 points/worker (95% CI -0.6, +0.9; cluster p=0.718; perm p=0.736). Quadratic term p=8.65e-07 (vertex 3.8 workers); joint test of size categories p=1.36e-46. Gen-0 only: +0.6 (-1.7, +2.9). Tokens rise 25% per worker (log-slope 0.221, CI 0.160–0.281).

- neat_a: mean workers 3.50 → 5.19, slope +0.131/gen (CI +0.087, +0.174; Spearman ρ=0.93; generations are autocorrelated, so treat as descriptive).
- neat_b: mean workers 2.25 → 3.00, slope +0.034/gen (CI -0.011, +0.078; Spearman ρ=0.30; generations are autocorrelated, so treat as descriptive).

## 2–3. Composition, settings and topology

All features with q<0.10 plus the pre-specified topology features (full list: features.csv). Settings rows (e.g. 'ED max_steps=30') are estimated only among graphs containing that worker type, with x = share of that type's workers using the option, so β contrasts 'all such workers use it' vs 'none do'; with/without counts are graphs with share>0 vs share=0.

| feature | graphs with/without | Δ effect (95% CI) | p_cluster | p_perm | q (BH) | gen0 Δ | effect on fail rate |
|---|---|---|---|---|---|---|---|
| has VIEWEDITOR | 21/209 | -8.4 (-12.0, -4.8) | 4.78e-06 | 0.00025 | 0.00635 | -11.6 | +41.9 pts (q=1.2e-09) |
| has PICKMERGER | 94/136 | +4.5 (+2.3, +6.6) | 5.11e-05 | 0.00025 | 0.00635 | +3.1 | +9.4 pts (q=0.023) |
| EDITOR -> EDITOR chain | 139/91 | +5.0 (+3.0, +7.0) | 1.3e-06 | 0.00025 | 0.00635 | +3.3 | +1.9 pts (q=0.79) |
| END source writes ACE | 129/101 | +6.5 (+4.0, +9.0) | 3.85e-07 | 0.00025 | 0.00635 | +2.1 | +14.9 pts (q=0.013) |
| PM context_budget=12000 | 13/81 | -8.8 (-12.5, -5.1) | 3.59e-06 | 0.00025 | 0.00635 | – | +25.2 pts (q=1.8e-05) |
| END fed by PICKMERGER | 91/139 | +4.9 (+2.8, +7.0) | 6.53e-06 | 0.0005 | 0.0106 | +3.1 | +7.9 pts (q=0.079) |
| PM ace_format=element | 5/89 | -19.2 (-27.2, -11.1) | 3.43e-06 | 0.00075 | 0.0119 | – | +63.8 pts (q=3.9e-18) |
| END fed by EDITOR | 42/188 | -6.0 (-8.5, -3.5) | 2.13e-06 | 0.00075 | 0.0119 | -0.2 | -27.0 pts (q=9.1e-19) |
| ED max_steps=60 | 111/119 | -4.5 (-7.5, -1.5) | 0.00307 | 0.0015 | 0.019 | +3.2 | -3.5 pts (q=0.65) |
| ED role=#1 'careful senior' | 9/221 | -13.0 (-22.7, -3.3) | 0.00858 | 0.0015 | 0.019 | -6.8 | +19.1 pts (q=0.65) |
| # EDITORs (per +1) | (numeric) | +2.7 (+1.1, +4.3) | 0.00097 | 0.00175 | 0.0202 | +4.0 | +3.6 pts (q=0.36) |
| ED context_budget=6000 | 70/160 | +6.4 (+2.0, +10.7) | 0.0041 | 0.00375 | 0.0391 | +2.0 | -2.5 pts (q=0.81) |
| CR context_budget=24000 | 45/14 | +7.0 (+2.8, +11.2) | 0.00106 | 0.004 | 0.0391 | +6.4 | -26.9 pts (q=1.9e-05) |
| has any merger | 152/78 | +3.3 (+1.2, +5.3) | 0.00173 | 0.0045 | 0.0408 | +1.0 | +12.9 pts (q=7.2e-05) |
| VE ace_format=element_file | 8/13 | +13.4 (+6.7, +20.1) | 8.91e-05 | 0.005 | 0.0423 | – | -22.2 pts (q=0.015) |
| JM view_full_weight=2 | 8/52 | -11.3 (-20.8, -1.8) | 0.0192 | 0.00575 | 0.0456 | – | +26.3 pts (q=0.32) |
| CR context_budget=12000 | 7/52 | -9.0 (-16.1, -2.0) | 0.0119 | 0.00775 | 0.0579 | -2.5 | +26.9 pts (q=4.1e-08) |
| PM context_budget=48000 | 48/46 | +3.9 (+0.8, +7.0) | 0.014 | 0.0127 | 0.0899 | +3.9 | -12.9 pts (q=0.01) |
| JM strategy=consistency with the codebase | 41/19 | +6.8 (+0.8, +12.9) | 0.0272 | 0.0142 | 0.0952 | – | -27.7 pts (q=0.01) |
| depth (per +1) | (numeric) | +0.8 (+0.1, +1.6) | 0.0337 | 0.083 | 0.302 | +2.0 | +6.1 pts (q=3.4e-05) |
| END merger fan-in (per +1) | (numeric) | +1.6 (-0.2, +3.4) | 0.0868 | 0.1 | 0.317 | -0.7 | +13.8 pts (q=1.3e-06) |
| has VIEWER | 65/165 | -1.8 (-4.1, +0.5) | 0.131 | 0.129 | 0.373 | +3.8 | -5.3 pts (q=0.36) |
| has CRITIC | 59/171 | -1.9 (-4.4, +0.7) | 0.151 | 0.153 | 0.429 | -4.0 | +17.5 pts (q=2.7e-05) |
| END fed by JUDGEMERGER | 59/171 | -2.0 (-4.8, +0.8) | 0.169 | 0.162 | 0.429 | -4.9 | +5.1 pts (q=0.45) |
| has LISTVIEWER | 55/175 | +1.8 (-0.7, +4.3) | 0.154 | 0.191 | 0.468 | +4.2 | -3.1 pts (q=0.65) |
| # edit generators (per +1) | (numeric) | +0.5 (-1.0, +1.9) | 0.524 | 0.561 | 0.725 | -0.0 | +12.7 pts (q=2.5e-07) |

**Multivariable model** (Δ ~ size + type presence + END type + chain + run; cluster by graph; n=2563 tasks). Species-clustered SEs (24 clusters) as a lineage sensitivity:

| term | β (95% CI), graph-clustered | p | p (species-clustered) |
|---|---|---|---|
| n_workers | -2.0 (-3.5, -0.6) | 0.00677 | 0.00212 |
| has_viewer | +1.8 (-2.0, +5.6) | 0.346 | 0.498 |
| has_listviewer | +4.0 (+1.0, +7.0) | 0.00817 | 0.00895 |
| has_vieweditor | -7.4 (-11.0, -3.8) | 6.51e-05 | 0.000419 |
| has_critic | +4.7 (+0.2, +9.2) | 0.0389 | 0.0677 |
| end_pickmerger | +8.2 (+2.5, +13.9) | 0.0046 | 0.0131 |
| end_judgemerger | +3.7 (-2.2, +9.6) | 0.216 | 0.16 |
| editor_chain | +3.7 (-1.8, +9.2) | 0.187 | 0.163 |

**By END source type:**

| END fed by | graphs | tasks | Δ | raw | fail rate | median tokens |
|---|---|---|---|---|---|---|
| pickmerger | 91 | 1091 | +2.6 | 0.239 | 0.45 | 620k |
| critic | 38 | 398 | -0.2 | 0.183 | 0.54 | 298k |
| judgemerger | 59 | 644 | -1.3 | 0.232 | 0.27 | 867k |
| editor | 42 | 430 | -4.3 | 0.197 | 0.00 | 310k |

## 4. Cost-effectiveness

Graphs with ≥6 tasks. Spearman(tokens, Δ) ρ=0.01 (p=0.89).

Pareto front (min tokens, max Δ), per run:

| graph | types | workers | tasks | Δ | raw | tokens | score/Mtok | fail |
|---|---|---|---|---|---|---|---|---|
| A:6dbf1a67 | LV1+ED1+CR1 | 3 | 6 | -9.0 | 0.000 | 111k | 0.00 | 1.00 |
| A:3f71ae17 | ED1+CR1 | 2 | 6 | +0.7 | 0.096 | 208k | 0.46 | 0.83 |
| A:f43a73cb | LV1+ED1+CR1 | 3 | 6 | +3.2 | 0.228 | 250k | 0.91 | 0.50 |
| A:9b36bbdd | LV1+ED1+CR1 | 3 | 6 | +7.1 | 0.497 | 251k | 1.98 | 0.17 |
| A:96cadcf6 | LV1+ED1+CR1 | 3 | 33 | +8.5 | 0.306 | 286k | 1.07 | 0.24 |
| A:b1a65581 | VW1+LV1+ED2+PM1 | 5 | 33 | +10.8 | 0.300 | 745k | 0.40 | 0.30 |
| A:54ce4881 | LV1+ED2+PM1 | 4 | 13 | +15.1 | 0.351 | 852k | 0.41 | 0.15 |
| B:811ad129 | ED1 | 1 | 8 | -17.0 | 0.125 | 94k | 1.33 | 0.00 |
| B:3360bbf9 | ED1+CR1 | 2 | 8 | -3.7 | 0.132 | 101k | 1.31 | 0.62 |
| B:8c88a133 | ED1+CR1 | 2 | 8 | +3.2 | 0.201 | 114k | 1.77 | 0.50 |
| B:151b0208 | ED2+PM1 | 3 | 8 | +4.4 | 0.347 | 249k | 1.39 | 0.38 |
| B:c5fe07fc | ED2+PM1 | 3 | 8 | +6.5 | 0.412 | 260k | 1.59 | 0.12 |
| B:0a218a97 | ED2+PM1 | 3 | 8 | +10.6 | 0.410 | 271k | 1.51 | 0.12 |
| B:03ea564d | ED2+PM1 | 3 | 16 | +12.2 | 0.381 | 274k | 1.39 | 0.19 |
| B:634f75d2 | ED2+PM1 | 3 | 8 | +15.5 | 0.450 | 382k | 1.18 | 0.00 |
| B:97bf1858 | ED2+PM1 | 3 | 8 | +15.6 | 0.454 | 484k | 0.94 | 0.25 |
| B:cca99cde | ED2+PM1 | 3 | 8 | +16.0 | 0.448 | 975k | 0.46 | 0.12 |

Top raw score per million tokens: A:9b36bbdd LV1+ED1+CR1 1.98/Mtok (Δ +7.1, 6 tasks); B:8c88a133 ED1+CR1 1.77/Mtok (Δ +3.2, 8 tasks); B:c5fe07fc ED2+PM1 1.59/Mtok (Δ +6.5, 8 tasks); B:0a218a97 ED2+PM1 1.51/Mtok (Δ +10.6, 8 tasks); B:45328a02 ED2+PM1 1.43/Mtok (Δ +8.5, 16 tasks)

## 5. Failure modes

Task failure rate (training): neat_a 0.54, neat_b 0.17. Any failure scores 0 (mean raw score failed 0.000 vs ok 0.337).

| worker type | executions | failed | rate (Wilson 95%) | top causes |
|---|---|---|---|---|
| vieweditor | 225 | 128 | 0.569 (0.504–0.632) | unknown element 32, overlap (duplicate block) 25, no ACE block 19 |
| critic | 668 | 332 | 0.497 (0.459–0.535) | unknown element 115, malformed element path 50, overlap (duplicate block) 48 |
| pickmerger | 1043 | 327 | 0.314 (0.286–0.342) | unknown element 93, overlap (duplicate block) 75, no ACE block 61 |
| listviewer | 614 | 53 | 0.086 (0.067–0.111) | listviewer selection unresolved 53 |
| judgemerger | 520 | 40 | 0.077 (0.057–0.103) | judge reply names no candidate 39, empty input view 1 |
| editor | 4190 | 0 | 0.000 (0.000–0.001) |  |
| viewer | 821 | 0 | 0.000 (0.000–0.005) |  |

ACE-writing workers by `ace_format`:

| type | ace_format | executions | rate (95%) |
|---|---|---|---|
| critic | element | 303 | 0.58 (0.53–0.64) |
| critic | element_file | 86 | 0.52 (0.42–0.63) |
| critic | element_file_lines | 227 | 0.40 (0.34–0.46) |
| critic | file_lines | 52 | 0.38 (0.26–0.52) |
| pickmerger | element | 7 | 0.71 (0.36–0.92) |
| pickmerger | element_file | 34 | 0.56 (0.39–0.71) |
| pickmerger | element_file_lines | 414 | 0.38 (0.33–0.43) |
| pickmerger | file_lines | 588 | 0.25 (0.22–0.28) |
| vieweditor | element | 89 | 0.61 (0.50–0.70) |
| vieweditor | element_file | 94 | 0.49 (0.39–0.59) |
| vieweditor | element_file_lines | 30 | 0.77 (0.59–0.88) |
| vieweditor | file_lines | 12 | 0.42 (0.19–0.68) |

PICKMERGER vs JUDGEMERGER per-execution failure: +14.9 pts (+9.8, +20.1; p=1.06e-08). Among ACE writers (adjusting for type, run): line-addressable formats -13.5 pts (-20.7, -6.4; p=0.000214); element-only format +12.1 pts (+5.1, +19.2; p=0.000702).

Training failure causes: unknown element 240, overlap (duplicate block) 148, no ACE block 118, unclosed fence 92, malformed element path 86, listviewer selection unresolved 53, judge reply names no candidate 39, overlap (distinct blocks) 25, conflicting blocks 21, header not allowed by format 18, element header on non-code file 13, unknown path 8, malformed header 7, empty input view 4, line span out of file 3, other 3, NEW on existing file 2

**SWE-bench Verified, winner (neat_b_0_37b607a6):** 95/500 instances failed (19.0%, Wilson 15.8–22.7%), all in the PICKMERGER (the only ACE-writing node; ace_format=file_lines). Causes: overlap (duplicate block) 43, no ACE block 23, unclosed fence 13, overlap (distinct blocks) 12, header not allowed by format 2, malformed element path 1, line span out of file 1. Replicate r2: 93 failures (overlap (duplicate block) 40, unclosed fence 22, no ACE block 15, overlap (distinct blocks) 9, malformed element path 4, line span out of file 2, header not allowed by format 1); 33 instances failed in both (P(fail r2 | fail r1)=0.35 vs P(fail r2 | ok r1)=0.15), so failures are partly instance-driven. Resolve rate on non-failed instances 30.4%; the single-agent baseline resolves 4/95 (4.2%) of the winner's failed instances (vs 8.1% of the others).

## 6. Lineage

Node-innovation families (innovations.json key family × type): first generation seen in a population, peak prevalence of any single innovation, and how many innovations of that family survive to the latest generation.

| run | family:type | innovations | first gen | peak prev. | surviving / extinct | best final prev. |
|---|---|---|---|---|---|---|
| neat_a | branch:editor | 5 | 0 | 0.25 | 4 / 1 | 0.12 |
| neat_a | init:editor | 3 | 0 | 1.00 | 3 / 0 | 1.00 |
| neat_a | init:viewer | 1 | 0 | 0.44 | 1 / 0 | 0.31 |
| neat_a | init:pickmerger | 2 | 0 | 0.44 | 2 / 0 | 0.31 |
| neat_a | init:listviewer | 1 | 0 | 0.25 | 1 / 0 | 0.12 |
| neat_a | init:critic | 1 | 0 | 0.56 | 1 / 0 | 0.56 |
| neat_a | merge:pickmerger | 6 | 0 | 0.25 | 3 / 3 | 0.12 |
| neat_a | merge:judgemerger | 3 | 0 | 0.19 | 3 / 0 | 0.19 |
| neat_a | mirror:vieweditor | 6 | 0 | 0.19 | 3 / 3 | 0.19 |
| neat_a | source:viewer | 2 | 0 | 0.06 | 0 / 2 | 0.00 |
| neat_a | split:listviewer | 6 | 0 | 0.44 | 5 / 1 | 0.44 |
| neat_a | critic:critic | 3 | 5 | 0.25 | 1 / 2 | 0.25 |
| neat_b | branch:editor | 3 | 0 | 1.00 | 2 / 1 | 1.00 |
| neat_b | init:editor | 2 | 0 | 1.00 | 0 / 2 | 0.00 |
| neat_b | init:pickmerger | 1 | 0 | 0.58 | 0 / 1 | 0.00 |
| neat_b | init:viewer | 1 | 0 | 0.33 | 0 / 1 | 0.00 |
| neat_b | merge:judgemerger | 5 | 0 | 0.92 | 1 / 4 | 0.42 |
| neat_b | mirror:vieweditor | 6 | 0 | 0.25 | 2 / 4 | 0.25 |
| neat_b | critic:critic | 3 | 3 | 0.25 | 0 / 3 | 0.00 |
| neat_b | source:viewer | 3 | 3 | 0.58 | 1 / 2 | 0.58 |
| neat_b | split:listviewer | 4 | 3 | 0.08 | 0 / 4 | 0.00 |
| neat_b | merge:pickmerger | 2 | 9 | 0.25 | 1 / 1 | 0.25 |

Motif prevalence (% of population), first → last generation:

- neat_a: viewer 31→31%; listviewer 31→69%; vieweditor 6→19%; critic 50→56%; judgemerger 6→25%; pickmerger 50→56%; editor_chain 50→56%
- neat_b: viewer 33→58%; listviewer 0→0%; vieweditor 8→33%; critic 0→0%; judgemerger 8→42%; pickmerger 33→25%; editor_chain 42→42%

Winner's topology (ED→ED chain into PICKMERGER; WL structure hash, any properties) as % of population by generation: neat_a: 0:0 1:0 2:0 3:0 4:0 5:0 6:0 7:0 8:0 9:0 10:0; neat_b: 0:0 1:0 2:0 3:8 4:25 5:8 6:0 7:0 8:0 9:0 10:0 11:0 12:0. Winner's exact node innovations {0, 2, 19} in neat_b: 0:33 1:33 2:50 3:42 4:58 5:25 6:42 7:25 8:25 9:0 10:0 11:0 12:0 (%).
- neat_a gen 3: species [3, 7, 9, 12] (5 members, last improved gen [2, 2, 2, 2]) disappeared: emptied by re-speciation.
- neat_a gen 5: species [10, 16, 17] (3 members, last improved gen [1, 4, 4]) disappeared: emptied by re-speciation.
- neat_a gen 6: species [14, 15, 18] (5 members, last improved gen [4, 4, 5]) disappeared: emptied by re-speciation.
- neat_a gen 8: species [21, 22] (3 members, last improved gen [6, 6]) disappeared: emptied by re-speciation.
- neat_b gen 2: species [3] (3 members, last improved gen [1]) disappeared: emptied by re-speciation.
- neat_b gen 9: species [0, 1, 2] (11 members, last improved gen [2, 2, 2]) disappeared: pruned for stagnation (no improvement for >6 generations).

## Held-out val40 (40 instances, 27 repos disjoint from training)

| candidate | workers | types | END | ED→ED | train mean | val mean | val fail | median tokens | Wilcoxon p vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| baseline (single editor agent) | 1 | - | editor | – | – | 0.210 | 0.00 | 347k | – |
| neat_a_0_54ce4881 | 4 | LV1+ED2+PM1 | pickmerger | 1 | 0.351 (13) | 0.246 | 0.40 | 729k | 0.273 |
| neat_a_1_b1a65581 | 5 | VW1+LV1+ED2+PM1 | pickmerger | 1 | 0.320 (21) | 0.243 | 0.40 | 859k | 0.52 |
| neat_b_0_37b607a6 | 3 | ED2+PM1 | pickmerger | 1 | 0.439 (16) | 0.316 | 0.25 | 403k | 0.0248 |
| neat_b_1_abe07840 | 3 | ED2+PM1 | pickmerger | 1 | 0.433 (16) | 0.302 | 0.30 | 301k | 0.0268 |

## Figures

- fig1_size.png: (a) Δ by size, (b) tokens by size, (c) mean size per generation.
- fig2_effects.png: forest plot of feature effects.
- fig3_pareto_lineage.png: (a) Δ vs tokens with Pareto fronts, (b) motif prevalence over generations.

## Caveats

- Selection effects: later-generation graphs are descendants of earlier winners, so feature effects partly reflect which lineage happened to be lucky early; the gen-0 column is the only selection-free estimate (and is small).
- Autocorrelated lineages: graphs are not independent; graph-clustered SEs treat them as such. The species-clustered sensitivity is a coarse lineage correction with few clusters.
- Noisy minibatches: 6–8 tasks per batch and a proxy fitness (patch similarity), not test resolution. Within-instance centring removes task difficulty but not batch × graph interaction.
- Features are confounded with each other (e.g. PICKMERGER at END implies ≥3 workers and an ACE writer); the univariate table is descriptive, the multivariable model partially adjusts.
- Runs were still appending; numbers are for this snapshot.
