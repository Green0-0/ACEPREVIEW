# SWE-bench Verified comparison

- **baseline_nolimit**: 43/500 = 0.086 (Wilson 95% 0.064-0.114); median tokens 701071; resolved/Mtok 0.029013466958296544
- **neat**: 124/500 = 0.248 (Wilson 95% 0.212-0.288); median tokens 440417; resolved/Mtok 0.4826407079140368
- **neat_r2**: 117/500 = 0.234 (Wilson 95% 0.199-0.273); median tokens 349278; resolved/Mtok 0.4915154085060388
- **baseline_30step**: 37/500 = 0.074 (Wilson 95% 0.054-0.100); median tokens 361757; resolved/Mtok 0.19246093582673585
- **baseline_30step_r2**: 36/500 = 0.072 (Wilson 95% 0.052-0.098); median tokens 367590; resolved/Mtok 0.17765438912810408
- neat - baseline_nolimit: diff +0.162 (bootstrap 95% +0.122..+0.204); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 19 vs 100); token ratio 0.6113074599673937, Wilcoxon p 2.383413012887618e-22
- neat_r2 - baseline_nolimit: diff +0.148 (bootstrap 95% +0.108..+0.186); McNemar exact p one-sided 0.0000, two-sided 0.0000 (discordant 20 vs 94); token ratio 0.6120651003055047, Wilcoxon p 8.115606457947457e-25
- baseline_30step - baseline_nolimit: diff -0.012 (bootstrap 95% -0.036..+0.012); McNemar exact p one-sided 0.8721, two-sided 0.4177 (discordant 22 vs 16); token ratio 0.422988025895472, Wilcoxon p 2.0652900748563647e-33
- baseline_30step_r2 - baseline_nolimit: diff -0.014 (bootstrap 95% -0.040..+0.012); McNemar exact p one-sided 0.8785, two-sided 0.3817 (discordant 27 vs 20); token ratio 0.4085700980908834, Wilcoxon p 1.2343206370987378e-35
- proxy validity: AUC 0.891, r_pb 0.530 (p 3.7e-181, n 2500)
