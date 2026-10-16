# SWE-bench Verified comparison

- **baseline_nolimit**: 43/464 = 0.093 (Wilson 95% 0.070-0.123); median tokens 658142; resolved/Mtok 0.03129903968111758
- **neat**: 124/464 = 0.267 (Wilson 95% 0.229-0.309); median tokens 418914; resolved/Mtok 0.5289761510888984
- **neat_r2**: 117/464 = 0.252 (Wilson 95% 0.215-0.294); median tokens 346250; resolved/Mtok 0.5359501995074618
- **baseline_30step**: 37/464 = 0.080 (Wilson 95% 0.058-0.108); median tokens 356462; resolved/Mtok 0.2090462245102099
- **baseline_30step_r2**: 36/464 = 0.078 (Wilson 95% 0.057-0.106); median tokens 361258; resolved/Mtok 0.19385007617877215
- neat - baseline_nolimit: diff +0.175 (bootstrap 95% +0.131..+0.218); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 19 vs 100); token ratio 0.6390557872400096, Wilcoxon p 4.037031785662217e-20
- neat_r2 - baseline_nolimit: diff +0.159 (bootstrap 95% +0.116..+0.203); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 20 vs 94); token ratio 0.6345547688523844, Wilcoxon p 2.396186124420632e-22
- baseline_30step - baseline_nolimit: diff -0.013 (bootstrap 95% -0.039..+0.013); McNemar exact p one-sided 0.8721, two-sided 0.4177 (discordant 22 vs 16); token ratio 0.43372422579448383, Wilcoxon p 7.41565974406189e-30
- baseline_30step_r2 - baseline_nolimit: diff -0.015 (bootstrap 95% -0.045..+0.013); McNemar exact p one-sided 0.8785, two-sided 0.3817 (discordant 27 vs 20); token ratio 0.4136147001190269, Wilcoxon p 5.185197978444128e-32
- proxy validity: AUC 0.891, r_pb 0.542 (p 5.3e-177, n 2320)
