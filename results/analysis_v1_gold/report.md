# SWE-bench Verified comparison

- **baseline**: 37/464 = 0.080 (Wilson 95% 0.058-0.108); median tokens 356462; resolved/Mtok 0.2090462245102099
- **neat**: 124/464 = 0.267 (Wilson 95% 0.229-0.309); median tokens 418914; resolved/Mtok 0.5289761510888984
- **baseline_r2**: 36/464 = 0.078 (Wilson 95% 0.057-0.106); median tokens 361258; resolved/Mtok 0.19385007617877215
- **neat_r2**: 117/464 = 0.252 (Wilson 95% 0.215-0.294); median tokens 346250; resolved/Mtok 0.5359501995074618
- neat - baseline: diff +0.187 (bootstrap 95% +0.147..+0.231); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 16 vs 103); token ratio 1.486466398036235, Wilcoxon p 1.0082346209175506e-09
- baseline_r2 - baseline: diff -0.002 (bootstrap 95% -0.024..+0.022); McNemar exact p one-sided 0.6445, two-sided 1.0000 (discordant 15 vs 14); token ratio 1.0155948141774136, Wilcoxon p 0.2154830574493185
- neat_r2 - baseline: diff +0.172 (bootstrap 95% +0.131..+0.213); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 15 vs 95); token ratio 1.394700839466143, Wilcoxon p 3.846936569631444e-06
- proxy validity: AUC 0.880, r_pb 0.537 (p 5.9e-139, n 1856)
