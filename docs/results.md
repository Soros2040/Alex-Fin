# Manuscript-reported results and reproduction record

English | [简体中文](results.zh-CN.md) · [Home](../README.md)

The author-supplied Alex-Fin portfolio manuscript reports 18.72% annualized return, a 1.43 annualized Sharpe ratio, and 12.65% maximum drawdown in its principal out-of-sample experiment. The author confirms these refer to completed research experiments. This page records that evidence level explicitly: the figures have been checked against manuscript Table 6-1, while the full experiment has not been independently rerun from this public edition.


## Complete table register

The [full Chinese manuscript and PDF](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper) preserve the historical record. All rows of Tables 6-1 through 6-7 and 7-1 are transcribed in the two bilingual cases below. The cases preserve negative drawdown signs; the summary on this page displays positive loss magnitudes.

| Source table | Subject | Complete reading |
| --- | --- | --- |
| 6-1 | Ten-model comparison | [Case 2](../cases/02-risk-rewards-and-evaluation.md) |
| 6-2 | Five frequency settings | [Case 1](../cases/01-causal-multifrequency-graphs.md) |
| 6-3 | MoE across market regimes | [Case 1](../cases/01-causal-multifrequency-graphs.md) |
| 6-4 | GRPO/PPO convergence and generalization | [Case 2](../cases/02-risk-rewards-and-evaluation.md) |
| 6-5 | Four models across market regimes | [Case 2](../cases/02-risk-rewards-and-evaluation.md) |
| 6-6 | Three models across volatility regimes | [Case 2](../cases/02-risk-rewards-and-evaluation.md) |
| 6-7 | Five sectors | [Case 2](../cases/02-risk-rewards-and-evaluation.md) |
| 7-1 | Full model and five ablations | [Case 2](../cases/02-risk-rewards-and-evaluation.md) |

The detailed reading records two interpretation conflicts: PPO also has positive bear-market Sharpe in Table 6-5, and the DSR replacement has a larger Sharpe decline than MoE removal in Table 7-1. It also identifies unassigned market-regime dates and explains why epoch counts do not establish wall-clock efficiency. These findings preserve the original data while narrowing the conclusions.


## Reported comparison

| Model | Annualized return | Annualized volatility | Maximum drawdown magnitude | Annualized Sharpe | Calmar | Annualized turnover, as labeled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Alex-Fin** | **18.72%** | **11.34%** | **12.65%** | **1.43** | **1.48** | **1.27** |
| MVO | 7.35% | 14.62% | 33.03% | 0.33 | 0.22 | 2.84 |
| Equal weight | 9.12% | 16.78% | 38.47% | 0.39 | 0.24 | 0.32 |
| CSI 300 index | 4.26% | 18.23% | 40.12% | 0.09 | 0.11 | — |
| LSTM | 10.57% | 15.89% | 29.74% | 0.51 | 0.36 | 2.16 |
| GAT | 12.33% | 14.25% | 26.31% | 0.69 | 0.47 | 1.98 |
| FinCast baseline | 13.68% | 13.17% | 22.45% | 0.85 | 0.61 | 1.83 |
| PPO | 14.21% | 13.02% | 20.18% | 0.90 | 0.70 | 1.75 |
| DQN | 11.05% | 15.36% | 28.64% | 0.55 | 0.39 | 2.31 |
| SAC | 13.84% | 13.56% | 21.73% | 0.84 | 0.64 | 1.89 |

Source: portfolio manuscript, Table 6-1, “全样本外测试区间核心投资绩效对比.” The original table expresses drawdowns as negative returns; this table shows positive loss magnitudes. The turnover label is retained and needs a precise calculation convention in the run manifest.

## Protocol described in the manuscript

- Universe: CSI 300 constituent stocks. Data coverage: January 4, 2010–February 28, 2026; final out-of-sample coverage: January 2017–February 2026.
- Data sources: Wind and Tushare; the manuscript also names China iVIX as a market feature. Availability and redistribution rights must be checked for each series before sharing data.
- Rolling evaluation: ten overlapping calendar windows shifted by one year, with a stated five-year training period, one-year validation period, and one-year test period. Exact trading-date boundaries and the final partial period are still needed.
- Execution: daily decisions after the close, trades at the next opening, initial cash of CNY 1,000,000 per backtest cycle, whole-lot trading, and residual cash.
- Costs: 0.03% commission and 0.1% sell-side stamp tax in the manuscript specification. A turnover penalty represents a further trading-friction term in the learning objective.
- Risk-free rate: 2.5% annualized in the stated backtest settings. The original architecture document contains other cost and rate descriptions; the released experiment configuration must identify the actual settings used.
- Constraints: long-only, no leverage, cash allowed, and sector exposure capped at 25%.

The study's date range, historical tax assumptions, market-data coverage, and claimed source availability should be evaluated as recorded experimental choices. A future rerun should record historically applicable trading rules separately from any deliberately fixed counterfactual assumptions.

## What the public evidence establishes

| Item | Current evidence | Reproduction requirement |
| --- | --- | --- |
| Headline metrics | Present in the supplied manuscript | Recompute from dated portfolio values and the exact metric code. |
| Architecture | Two research documents and an authorized source snapshot | Resolve the [source review findings](source-status.md) and link the exact experiment configuration. |
| Baseline comparison | A manuscript table with ten models | Match datasets, time splits, budgets, constraints, and evaluation rules. |
| Statistical significance | The manuscript describes bootstrap analysis | Publish sampling method, seed, dependence treatment, and confidence intervals. |
| Data handling | Described at a procedural level | Provide point-in-time manifests, adjustment rules, missingness masks, and checksums where licensing permits. |
| Runtime reproducibility | Public rerun pending | Supply dependency lock, hardware details, run commands, logs, and checkpoint provenance. |

The manuscript mentions exclusions of delisted and long-suspended stocks. Their effect on survivorship bias needs to be measured in a point-in-time universe audit. It also describes quarterly rebalancing for equal weight while other policies use daily decisions; this difference should remain visible when interpreting comparisons.

## Metric conventions for a reproducible release

For net portfolio value $V_t$, define simple returns $r_t=V_t/V_{t-1}-1$. With $T$ observed trading returns and annualization constant $A$:

$$
\operatorname{CAGR}=(V_T/V_0)^{A/T}-1,\qquad
\operatorname{Sharpe}=\sqrt A\,\frac{\overline{r-r_f}}{s(r-r_f)},
$$

$$
\operatorname{MDD}=\max_t\left(1-\frac{V_t}{\max_{u\le t}V_u}\right).
$$

Record the return convention, annualization constant, standard-deviation degrees of freedom, daily risk-free conversion, cash-flow treatment, and turnover definition. A ratio calculated from rounded annual summary columns can be an arithmetic check, but it is not a substitute for recomputing daily-series metrics.

The released [source snapshot](source-status.md) contains concrete model modules, but its implementation review identifies differences affecting graph use, action selection, accounting, and metrics. No claim is made that this particular revision generated Table 6-1; establish that link through the original run record.

## Minimum run package

A public rerun should include the exact code revision; all model and baseline configurations; train/validation/test boundaries; data hashes or a licensed retrieval manifest; random seeds; per-day portfolio values, positions, trades, and costs; per-window and per-seed metrics; and runtime/resource measurements. Outputs should identify whether each return is gross or net.

First reproduce an equal-weight or MVO baseline with the same execution ledger, then add Alex-Fin modules and ablations. Freeze preprocessing and model selection before opening the final test results. Keep full-run status as **pending** until the published package can regenerate the reported table. The [second learning case](../cases/02-risk-rewards-and-evaluation.md) provides a small accounting example before any training run.
