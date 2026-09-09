# Kelly Portfolio Optimization with Covariance Denoising, HRP and Detoning

Testing whether three machine learning techniques improve portfolio construction on a
ten-stock US equity universe, and being honest about when they don't.

**Headline result:** covariance denoising reduced the condition number of the correlation
matrix by 58.9% and changed the optimal portfolio weights by exactly zero. The reason is
worth more than the technique.

---

## The question

Denoising, hierarchical clustering and detoning are standard tools in the quantitative
finance literature (López de Prado, 2020). Each addresses estimation error in the
covariance matrix. The question here is not whether they work in principle, but whether
they help on a specific, realistic problem: ten large-cap US stocks, two years of daily
data, a long-only mandate with a 20% position limit.

## Data and setup

| | |
|---|---|
| Universe | TSLA, WMT, BAC, GS, LLY, MRK, GOOG, META, AAPL, XOM |
| Training window | 3 Jan 2024 – 30 Sep 2025 (437 daily observations) |
| Test window | 1 Oct – 31 Dec 2025 (64 observations, never seen in estimation) |
| Cross-validation | 5 purged folds with a 5-day embargo, over all 501 observations |
| Risk-free rate | 4.50% annual |
| Constraint | Long-only, fully invested, no position above 20% |

`q = T/N = 43.7`. That number turns out to drive the main finding.

## What was built

**Kelly optimizer.** Maximizes empirical log growth, `E[ln(1 + wᵀr)]`, on the realized
return distribution rather than a Gaussian approximation. Every asset in the universe shows
excess kurtosis between 2.4 and 16.3, and log utility is sensitive to the left tail, so the
empirical form matters. SLSQP with 20 random restarts on the simplex; feasibility asserted
rather than clipped.

**Fractional Kelly comparison.** Half-, full- and double-Kelly, with leverage financed at
the risk-free rate.

**Purged K-fold cross-validation.** Contiguous folds, no shuffling, with purging and an
embargo on both sides of each test block (López de Prado, 2018). Plain K-fold on a return
series leaks future information into training and is invalid.

**Marchenko-Pastur denoising.** Fits the noise variance σ² by minimizing the distance
between the empirical eigenvalue density and the theoretical MP density, then clips every
eigenvalue below λ₊ to their common mean. Trace preserved.

**Detoning.** Removes the leading (market) eigenvector after denoising.

**Hierarchical Risk Parity.** Correlation-distance clustering, quasi-diagonalization,
recursive bisection. No matrix inversion at any point.

## Results

### 1. The Kelly criterion concentrates

Unconstrained, the optimizer put the entire portfolio in three of ten stocks: META 45.8%,
GS 39.1%, TSLA 15.1%. Under a 20% cap it held exactly five, every one sitting at the
ceiling. Annualized growth fell from 57.98% to 53.67%, a cost of 4.31 percentage points.

The estimated returns behind that concentration were not distinguishable from one another.
GS returned 49.4% and META 50.7% over 437 observations. The optimizer gave one of them
more than twice the weight of the other.

### 2. A Sharpe ratio cannot evaluate leverage

| | Half-Kelly | Kelly | Double-Kelly |
|---|---|---|---|
| Cumulative return | 4.66% | 8.07% | 14.42% |
| **Sharpe ratio** | **1.4562** | **1.4562** | **1.4562** |
| Max drawdown | −3.16% | −6.52% | −13.07% |
| 95% cVaR (daily) | −1.23% | −2.47% | −4.96% |

This is not a bug. Excess return scales linearly with the Kelly fraction, so mean and
standard deviation both scale by `f` and the ratio is invariant. Any leverage decision
judged on Sharpe alone is being judged on a statistic that is blind to it.

In the weakest cross-validation fold, double-Kelly grew at 6.12% against full Kelly's 7.38%
while taking twice the drawdown. Growth is not monotonic in leverage, exactly as Kelly's
theory predicts.

### 3. Denoising worked, and changed nothing

| | Value |
|---|---|
| Condition number, raw | 16.96 |
| Condition number, denoised | 6.97 (−58.9%) |
| Eigenvalues retained as signal | 3 of 10 |
| Change in Kelly portfolio weights | **0.00** |

The 20% cap forces at least five positions (1 ÷ 0.20 = 5). The capped Kelly portfolio held
exactly five, all at the ceiling. When a constraint pins every active weight to the
boundary, the covariance matrix has no influence on the answer. No amount of work on the
estimate can move a solution the feasible set has already determined.

### 4. Cross-validation overturned the single-period ranking

On the test quarter, HRP + denoised won five of six metrics. Across five purged folds:

| Portfolio | Growth (mean) | Sharpe (mean) | Sharpe (sd) | Max DD |
|---|---|---|---|---|
| Equal weight | **39.00%** | 1.99 | 1.73 | −9.70% |
| HRP (raw) | 34.15% | **2.32** | 2.31 | **−7.49%** |
| HRP + denoised | 33.24% | 2.21 | 2.20 | −7.66% |
| HRP + den. + detoned | 35.58% | 2.23 | 2.01 | −8.31% |
| Kelly | 33.25% | 1.34 | 0.71 | −10.59% |

Equal weight had the highest mean growth rate of anything tested, with no estimation and no
code (DeMiguel, Garlappi and Uppal, 2009). HRP's Sharpe standard deviation of 2.31 equals
its mean of 2.32.

Only one claim survives both tests: **HRP consistently reduces risk.** Mean drawdown from
−10.59% to −7.49%, mean cVaR from −2.79% to −1.92%. The growth advantage does not.

## What I'd take into practice

**Check whether a constraint is binding before improving an estimate.** This is the
cheapest diagnostic in the project and the one that mattered most. If a mandate limit pins
every active position, better inputs cannot change the output.

**Match the tool to q.** Denoising is built for the regime where N approaches T. At
q = 43.7 there was little noise available to remove. The technique is not weak; the problem
was not the one it solves.

**Judge methods on cross-fold dispersion, not on a winning quarter.** A single 64-day test
window cannot separate a real edge from a favorable period.

## Repository layout

```
├── notebooks/
│   └── kelly_hrp_denoising.ipynb     # full analysis, all outputs saved
├── reports/
│   ├── technical_report.pdf          # methodology, results, interpretation
│   └── estimation_error_talk.pdf     # 6-slide technical talk
├── figures/                          # exported charts
├── src/
│   └── portfolio.py                  # optimizer, denoising, HRP, CV utilities
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/kelly_hrp_denoising.ipynb
```

Prices are pulled from Yahoo Finance via `yfinance`. The data cell raises rather than
falling back to anything synthetic, so every number in the outputs traces to real market
data.

## References

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification: How
Inefficient Is the 1/N Portfolio Strategy? *Review of Financial Studies*, 22(5), 1915–1953.

Jagannathan, R., & Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the
Wrong Constraints Helps. *Journal of Finance*, 58(4), 1651–1683.

Kelly, J. L. (1956). A New Interpretation of Information Rate. *Bell System Technical
Journal*, 35(4), 917–926.

López de Prado, M. (2016). Building Diversified Portfolios That Outperform Out of Sample.
*Journal of Portfolio Management*, 42(4), 59–69.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University
Press.

Marchenko, V. A., & Pastur, L. A. (1967). Distribution of Eigenvalues for Some Sets of
Random Matrices. *Matematicheskii Sbornik*, 114(4), 507–536.

---

*Completed as part of an MSc in Financial Engineering. Results are from a specific universe
and period and are not investment advice.*
