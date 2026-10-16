# SWE-bench Verified comparison

- **baseline**: 37/500 = 0.074 (Wilson 95% 0.054-0.100); median tokens 361757; resolved/Mtok 0.19246093582673585
- **neat**: 124/500 = 0.248 (Wilson 95% 0.212-0.288); median tokens 440417; resolved/Mtok 0.4826407079140368
- **baseline_r2**: 36/500 = 0.072 (Wilson 95% 0.052-0.098); median tokens 367590; resolved/Mtok 0.17765438912810408
- **neat_r2**: 117/500 = 0.234 (Wilson 95% 0.199-0.273); median tokens 349278; resolved/Mtok 0.4915154085060388
- neat - baseline: diff +0.174 (bootstrap 95% +0.134..+0.214); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 16 vs 103); token ratio 1.478966301970232, Wilcoxon p 4.750821600621576e-11
- baseline_r2 - baseline: diff -0.002 (bootstrap 95% -0.024..+0.020); McNemar exact p one-sided 0.6445, two-sided 1.0000 (discordant 15 vs 14); token ratio 1.0125727933204873, Wilcoxon p 0.25271440875710915
- neat_r2 - baseline: diff +0.160 (bootstrap 95% +0.122..+0.200); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 15 vs 95); token ratio 1.394700839466143, Wilcoxon p 1.7196037086859833e-06
- proxy validity: AUC 0.880, r_pb 0.523 (p 3.8e-141, n 2000)
