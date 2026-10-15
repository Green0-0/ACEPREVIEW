**Slide 1 — Topics in Information Science**

Overview of statistical hypothesis testing  
(Week 2/9)

5/W,6/W  
@H-260

Hiroki Tanaka  
International Christian University

2026@Hiroki Tanaka

---

**Slide 2 — Contents**

1. Overview of multimodal information processing
2. Overview of statistical hypothesis testing
3. Overview of machine learning
4. Examples of Statistical and machine learning-based data analysis (data search)
5. Natural language processing: analysis and machine learning (Programming implementation 1)
6. Speech and image processing: analysis and machine learning
7. EEG information processing (group work, data search)
8. Multimodal information integration: integration methods and synchronous timing (group work, Programming implementation 2)
9. Presentation and Discussion

---

**Slide 3 — Data Modality**

- Audio (speech / music)
- Image (facial expression etc)
- Text (natural language)
- EEG
- Heart rate
- Others (Respiration, Smell, sensation)

How should information scientists use statistical methods in their research?

---

**Slide 4 — What is statistical hypothesis test?**

- A statistical hypothesis test is a method of statistical inference used to decide whether the data sufficiently supports a particular hypothesis.
- Does it occur more than a chance?

[Bar chart comparing Method A and Method B.]

---

**Slide 5 — Application of statistical hypothesis testing**

- Useful to compare more than two conditions
  - UX system evaluation
  - Questionnaire analysis
  - Demographic group comparison
  - Machine learning algorithm comparison
  - A/B/C testing
  - Teaching method comparison
  - Healthcare intervention comparison
  - Speech recognition system comparison
  - NLP model comparison
  - Usability testing
  - Psychological experiments
  - EEG experiment comparison

---

**Slide 6 — Linguistic and speech analysis**

Table 3: Difference of mean values between ASD and TD based on language and speech features from children’s utterances. Each table cell notes which of the two classes has the greater mean on the corresponding feature (*: p < 0.01, **: p < 0.005).

| Feature | Result | Feature | Result | Feature | Result |
|---|---|---|---|---|---|
| WPS | – | affect | TD** | nonfluent | – |
| 6 let. | ASD* | cognitive | TD* | fillers | ASD* |
| laughter | – | perceptual | – | fsd | TD** |
| adverb | – | biological | – | fcov | TD* |
| pronoun | – | relativity | – | psd | – |
| conjunctions | – | work | – | pcov | – |
| negations | – | achievement | – | speech rate | – |
| quantifiers | – | leisure | – | a3 | – |
| numbers | – | home | – | h1h2 | – |
| social | TD** | assent | ASD** | h1a3 | ASD** |

Tanaka, H., Sakti, S., Neubig, G., Toda, T., & Nakamura, S. (2014, June). Linguistic and acoustic features for automatic identification of autism spectrum disorders in children’s narrative. In *Proceedings of the workshop on computational linguistics and clinical psychology: From linguistic signal to clinical reality* (pp. 88–96).

[Mother–infant interaction diagram, with the following labels:]

- SYNCHRONY
- RECIPROCITY
- 1: mothers remain still and stop interaction
- 2: ↗ infants increase vocalization and pause during SF
- 3: after SF: infants maintain SF level of vocalization but decrease pauses
- 4: after SF: mothers do not increase vocalization but show less pause and use more non e-IDS
- after SF: ↗ overlap; ↗ infant response to maternal vocalization ratio (both e-IDS, non e-IDS); ↘ joint silence
- 3-month infant ≈ 6-month infant
- SF with affectionate touch (➜ cortisol) ≠ classic SF (↗ cortisol)

Bourvis, N., Singer, M., Saint Georges, C., Bodeau, N., Chetouani, M., Cohen, D., & Feldman, R. (2018). Pre-linguistic infants employ complex communicative loops to engage mothers in social exchanges and repair interaction ruptures. *Royal Society open science, 5*(1), 170274.

---

**Slide 7 — Scenario**

Describe the scenario, including what you would like to investigate and what data you would collect. Write your scenario in the spreadsheet.

[Spreadsheet](https://docs.google.com/spreadsheets/d/1pzJm0uYZQI2JW-Is_ZE00o7tTan8Is7bn4zy8xuZy7E/edit?gid=0#gid=0)

---

**Slide 8 — Data type**

- First, check your data type

[Diagram:]

- Data
  - Quantitative
    - Discrete
    - Continuous
  - Qualitative

---

**Slide 9**

Descriptive statistics

---

**Slide 10 — Data visualization**

- e.g. x={13,14,7,12,10,6,8,15,4,14,9,6,10,12,5,12,8,8,12,15}
- Histogram created by R

```r
x<-c(13,14,7,12,10,6,8,15,4,14,9,6,10,12,5,12,8,8,12,15)
hist(x)
```

[Histogram of x. Axes: Frequency; x.]

---

**Slide 11 — Descriptive statistics**

- Mean
  - n: number of samples

\[
\bar{x}=\frac{x_1+x_2+\cdots+x_n}{n}=10
\]

- Ordered data: Minimum - 25% - Median - 75% - Maximum

```text
4 5 6 6 7 8 8 8 9 10 10 12 12 12 12 13 14 14 15 15
```

```r
sort(x)
```

- Mode
  - Most frequent samples

```r
names(which.max(table(x)))
[1] "12"
```

- Others: e.g., kurtosis (tailedness), skewness (asymmetry) …

---

**Slide 12 — Boxplot**

- Good to visualize descriptive statistics.

[Boxplot with labels:]

- maximum
- 75th percentile
- median

---

**Slide 13 — Descriptive statistics (Cont’d)**

- (unbiased) Variance

\[
\sigma^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
\]

- Standard deviation (SD)

\[
\sigma=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}
\]

- Range (e.g., min to max)

---

**Slide 14 — Standardization and Normalization**

- Transforming data
- Standardization: e.g., Z-score (the scale() function in R)
  - Mean = 0, SD = 1

\[
Z_{\text{score}}=\frac{x-\bar{x}}{\sigma}
\]

- Normalization: Rescale values to a specified range, such as [0, 1] or [−1, 1]

[Feature scaling diagram:]

Normalization:

\[
X_{\text{new}}=\frac{X-X_{\min}}{X_{\max}-X_{\min}}
\]

Standardization:

\[
X'=\frac{X-\text{Mean}}{\text{Standard deviation}}
\]

[Source](https://python.plainenglish.io/)

---

**Slide 15 — Scatter plots**

- Relationship between 2 variables

[Two scatter plots labeled “Positive correlation” and “No correlation.”]

---

**Slide 16 — Covariance**

- Relationship between two variables:

\[
s_{xy}=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})
\]

- Positive relationship → Positive correlation
- Negative relationship → Negative correlation
- Normalized covariance → Correlation coefficient

---

**Slide 17 — Correlation**

- Pearson’s correlation coefficient:

\[
r_{xy}=\frac{s_{xy}}{s_xs_y}
\]

| Value | Description |
|---|---|
| .00–.19 | “very weak” |
| .20–.39 | “weak” |
| .40–.59 | “moderate” |
| .60–.79 | “strong” |
| .80–1.0 | “very strong” |

[Source](http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf)

- For example, a correlation value of r = .42 would be a “moderate positive correlation”
- Ordered data: Spearman’s correlation (effect size: ρ)

---

**Slide 18 — Cross-tabulation**

- Relationships among categorical variables
- e.g., whether someone likes or dislikes a movie

| Sport | Movie: like | Movie: dislike |
|---|---:|---:|
| like | 10 | 4 |
| dislike | 2 | 4 |

---

**Slide 19**

In what situations are descriptive statistics sufficient to summarize the data?

Are descriptive statistics sufficient to fully represent a dataset?

---

**Slide 20**

Inferential statistics

---

**Slide 21 — Population and sample**

- Population: the complete set of all target data
  - Population mean: the true mean
  - Population variance: the true variance
- Sample: a subset of the population collected for analysis
  - Sample mean: the sample mean
  - Sample variance: the sample variance

[Diagram: sampling from Population to sample; infer from Known to Unknown.]

---

**Slide 22 — Inferential statistics**

[Diagram:]

- Inferential statistics
  - estimation
    - point estimation
    - Interval estimation (confidence interval)
  - statistical hypothesis test

---

**Slide 23 — Statistical hypothesis test**

- Infer population characteristics from sample data
  - Is the observed effect greater than what could be expected by chance?
  - Do Methods A and B differ significantly in accuracy?
  - Is Method A significantly more accurate than Method B?

[Bar chart comparing Method A and Method B.]

---

**Slide 24 — General procedure**

| Proc. | |
|---|---|
| 1 | Set null hypothesis (H₀) and alternative hypothesis (H₁) (two-sided or one-sided test) on the population |
| 2 | Choose a statistical distribution |
| 3 | Determine the value of the significance level α (generally, 0.05, 0.01, 0.001) |
| 4 | (Collect data) Find value of statistical distribution from data |
| 5 | If the value of the statistical distribution falls within the rejection range, the null hypothesis (H₀) is rejected and the alternative hypothesis is accepted. |

---

**Slide 25 — Example) Paired T-test**

- Example: Paired t-test (one-tailed)
- Hypothesis:
  - The right hand is longer than the left hand.
- Set α = 0.05
- Data: all right-handed participants (N=3)

R command:

```r
right<-c(212,194,160)
left<-c(208,188,158)
Difference<-right-left
```

[Hand illustrations labeled “Left hand” and “Right hand.”]

| | Right hand | Left hand | difference |
|---|---:|---:|---:|
| Sub 1 | 212mm | 208mm | 4mm |
| Sub 2 | 194mm | 188mm | 6mm |
| Sub 3 | 160mm | 158mm | 2mm |

---

**Slide 26 — Paired T-test (Cont’d)**

- Null hypothesis (H₀): The mean of the paired differences is 0.
- Alternative hypothesis (H₁): The mean of the paired differences is greater than 0.
- Form of T-Value:

\[
t=\frac{\bar{x}}{\sigma/\sqrt{n}}
\]

What factors affect the t-value?

- Calculate followings:

\[
\bar{x}=?
\]

\[
\sigma=?
\]

\[
t=?
\]

Ans. t = 3.47

---

**Slide 27 — Student’s t distribution**

- d.f. (degree of freedom) = n − 1

[Distribution curves labeled:]

- Standard Normal (with df = ∞)
- t (df = 13)
- t (df = 5)
- Areas of tail

[Source](https://financetrain.com/students-t-distribution/)

---

**Slide 28 — Refer to the T table**

[Background image: a t-distribution reference table, partially covered by an enlarged table.]

Areas of one tail = P-value

| n | df | 0.10 | 0.05 | 0.01 |
|---:|---:|---:|---:|---:|
| 3 | 2 | 1.88 | 2.92 | 4.30 |

Right-handed people have significantly longer right hands than left hands (p < 0.05).

Ans. t = 3.47

---

**Slide 29 — Paired T-test (Cont’d)**

- The null hypothesis was rejected, providing evidence in support of the alternative hypothesis (p < 0.05).

- R command:

```r
t.test(right,left,alternative="greater",paired=TRUE)
```

```text
Paired t-test

data:  right and left
t = 3.4641, df = 2, p-value = 0.03709
alternative hypothesis: true difference in means is greater than 0
95 percent confidence interval:
 0.6282911       Inf
sample estimates:
mean of the differences
                      4
```

---

**Slide 30 — Common Statistical Hypothesis Tests**

- Single-variable tests:
  - Z-test: Tests whether a population mean differs from a specified value when the population variance is known.
  - One-sample t-test: Tests whether the mean of a sample differs from a specified value.
  - Binomial test: Tests whether an observed proportion differs from a specified probability.
  - Pearson’s correlation test: Tests whether there is a statistically significant linear correlation between two continuous variables.
- Two-variable comparisons:
  - Paired t-test: Compares the means of two related or paired groups.
  - Student’s t-test: Compares the means of two independent groups, assuming equal variances.
  - Welch’s t-test: Compares the means of two independent groups without assuming equal variances.
  - Wilcoxon signed-rank test: A nonparametric test for comparing two related or paired groups.
  - Chi-square test of independence: Tests whether two categorical variables are statistically associated.

---

**Slide 31**

When is statistical hypothesis testing enough to summarize the data?

What are the limitations of statistical hypothesis testing?

---

**Slide 32**

Multiple comparisons

---

**Slide 33 — ANOVA (Analysis of Variance)**

- Comparing averages across 3 or more groups.
- Uses the F-distribution (a continuous distribution that arises when testing whether two or more variances are equal).
- If the result is significant, proceed with post-hoc analysis.

[Bar chart comparing A, B, and C.]

---

**Slide 34 — When the null hypothesis is not rejected**

- We cannot conclude that either the null hypothesis or the alternative hypothesis is definitively true or false.
- We can only say that there is not enough evidence to reject H₀.
- The probability of making a Type I error (rejecting H₀ when it is actually true) is the significance level, α.

| (Fact) | Reject | Fail to reject |
|---|---|---|
| Null hypothesis (H₀) is true | Type I error (Probability = α) | Correct inference |
| Null hypothesis (H₀) is false | Correct inference | Type II error (Probability = β) |

---

**Slide 35 — Family-wise error rate (FWER)**

- Problem of multiple testing (repeated hypothesis tests)
- Family-Wise Error Rate (FWER): The probability of making at least one Type I error across a family (set) of hypothesis tests.
- How can we address the problem of multiple testing?

\[
FWER\leq 1-(1-\alpha_{IT})^c
\]

αᵢₜ = alpha level for an individual test (e.g. .05),  
c = Number of comparisons.

With 10 comparisons, the risk of at least one false positive exceeds 40%.

[Bar chart showing three groups and pairwise comparisons.]

---

**Slide 36 — Bonferroni correction**

- Reducing the risk of Type I error in multiple comparisons.
- Adjust the significance level (α) for each comparison.
- For example, using the Bonferroni correction, divide the overall α by the number of comparisons.

[Diagram showing comparisons among A, B, and C.]

Suppose we are comparing 3 groups control the overall α=0.05.  
A difference is considered statistically significant only if p < 0.017.

---

**Slide 37 — Bootstrapping and permutation test**

- When the number of comparisons is large, this method becomes very conservative, making it difficult to detect true differences (increased Type II error).

[Diagram labels:]

- Original data X
  - A₁, A₂, A₃, A₄, A₅
  - B₁, B₂, B₃, B₄, B₅
- Permutated data
  - X′₁, X′₂, X′₃, …, X′₁₀₀₀
- e.g. \(\binom{10}{5}=252\) times
- T-value (or e.g. mean) as original
- T-value
- Compare original t-values and simulated t-values
- The frequency of the t-value over the original t-value is regarded as the p value

Original image:  
[Source](https://www.dynacom.co.jp/product_service/packages/snpalyze/sa_t2_permutation.html)

---

**Slide 38 — False discovery rate (FDR) correction**

- Among all rejected null hypotheses, the FDR is the expected proportion of Type I errors.

Procedure:

- Select desired limit q (e.g. 0.05) on FDR
- Order p-values,

\[
p_1\leq p_2\leq \cdots\leq p_N
\]

- Let r be largest i such that

\[
p_i\leq(i/N)\times q
\]

- Reject all null hypotheses corresponding to p₁, …, pᵣ

Ding, Z., Ma, W., Feng, L., Zhang, M., & Li, X. (2025). Quantifying Task-locked Information Transmission between Cortical Areas with TMS-EEG. *NeuroImage*, 121323.

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300.

---

**Slide 39**

Alternative approaches

---

**Slide 40 — Effect size and general linear model**

- Differences between two groups: Use Cohen’s d to quantify the standardized mean difference.

\[
d=\frac{|\bar{x}_1-\bar{x}_2|}{s_c}
\]

\[
s_c=\sqrt{\frac{n_1s_1^2+n_2s_2^2}{n_1+n_2}}
\]

- Multiple factors: Use a general linear model (GLM) to examine the effects of multiple factors.

McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall.

---

**Slide 41 — Choosing an appropriate test**

- Data type and measurement scale
- Parametric or nonparametric test?
- Test assumptions: Normality? Equal variances?
- Study design: Paired or unpaired? One-tailed or two-tailed?
- Multiple comparisons: Is a correction needed?

---

**Slide 42 — References**

- Permutation test:
  - [https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/](https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/)
  - [http://www.fieldtriptoolbox.org/tutorial/cluster_permutation_timelock](http://www.fieldtriptoolbox.org/tutorial/cluster_permutation_timelock)
- Multiple comparison:
  - Coursera Module 27&28: FWER and FDR Correction
  - [https://www.coursera.org/lecture/functional-mri/module-27-fwer-correction-WhrBO](https://www.coursera.org/lecture/functional-mri/module-27-fwer-correction-WhrBO)

---

**Slide 43 — Tools for the statistical hypothesis testing**

- R: A programming language designed for statistical analysis and data visualization.
- Python: A versatile programming language with a wide range of statistical libraries.
  - Statsmodels: Provides tools for estimating statistical models, performing statistical tests, and analyzing data.
  - SciPy: A Python library for mathematics, science, and engineering, including a wide range of statistical functions.

---

**Slide 44 — Demonstration**

- Colab
  - [https://colab.research.google.com/drive/12LllKx0-hw4P5pF8KzJ1M8T2Uba53FKJ?usp=sharing](https://colab.research.google.com/drive/12LllKx0-hw4P5pF8KzJ1M8T2Uba53FKJ?usp=sharing)
- IRIS data
  - [https://www.kaggle.com/datasets/webirlab/iris-data?resource=download](https://www.kaggle.com/datasets/webirlab/iris-data?resource=download)

---

**Slide 45 — Scenario and discussion**

Please consider a realistic and concrete scenario in which statistical hypothesis testing could be applied. Then, choose an appropriate statistical hypothesis test and formulate a null hypothesis for your scenario. Explain why you chose the test. Write your answers in the spreadsheet.

[Spreadsheet](https://docs.google.com/spreadsheets/d/1pzJm0uYZQI2JW-Is_ZE00o7tTan8Is7bn4zy8xuZy7E/edit?usp=sharing)

---

**Slide 46 — Exercise**

- Q1. Calculate by hand whether A differs from B using a paired t-test (two-tailed, α = 0.05).
- Q2. Perform the same analysis using a computer and calculate the effect size for the difference between A and B.

| | A | B |
|---|---:|---:|
| Sub 1 | 3 | 0 |
| Sub 2 | 4 | 1 |
| Sub 3 | 1 | 2 |
| Sub 4 | 8 | 3 |
| Sub 5 | 10 | 0 |

---

**Slide 47 — Summary**

- Overview of Statistical Tests
- Descriptive Statistics
- Inferential Statistics
- Multiple Comparisons
- Large-Scale Data Analysis
- Effect Size