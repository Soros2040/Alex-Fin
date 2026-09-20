# Case 2 — Read the reward, rolling evaluation, and ablations as an evidence chain

English | [简体中文](02-risk-rewards-and-evaluation.zh-CN.md) · [Home](../README.md) · [Case 1](01-causal-multifrequency-graphs.md)

**Research question:** what does a portfolio result establish when its action rule, costs, model selection, and comparison conditions are made explicit?

Prerequisites are simple returns, mean and variance, and train/validation/test separation. Read the [full manuscript](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper) alongside this case. Its author confirms the reported numerical results describe completed experiments. This case transcribes and interprets that record; it performs no experiment or rerun and does not establish that the public source revision generated the reported numbers.

## 1. Connect the chosen weights to the return they earn

A long-only portfolio satisfies $w_i\ge0$ and $\sum_iw_i=1$ if all wealth is invested. [`GRPOPolicy.sample_weights`](../src/agents/grpo.py) samples Gaussian latent actions $z$ and sets $w=\operatorname{softmax}(z)$. Its log probability is calculated in latent Gaussian space. Treat the latent action and its transformed portfolio as distinct objects when deriving a policy objective.

The paper describes decisions after the close, trading at the next opening, whole lots, residual cash, and a sector cap of $25\%$. The current [environment](../src/envs/trading_env.py) instead holds a vector of fractional asset weights; [its cap helper](../src/envs/constraints.py) has no sector membership input and clips then renormalizes. Renormalization can increase a clipped weight again. A sector constraint needs a sector map and a feasible projection or execution procedure.

The source computes

$$
r_t^{net}=w_{t-1}^{\top}r_t-c\sum_i|w_{t,i}^{target}-w_{t-1,i}|,
$$

then installs the target weights. With old weights $[0.6,0.4]$, target $[0.5,0.5]$, and returns $[0.01,-0.005]$, gross return is $0.004$. Turnover is $0.2$; at the source default $c=0.0005$, fee is $0.0001$ and net return is $0.0039$. This hand calculation explains **the source's ordering**, not a reconstructed manuscript trade.

Immediate candidate rewards thus share the same old-weight gross return and differ through trading penalties. They do not yet score each candidate's subsequent holding-period return. The environment also keeps target weights fixed between steps instead of drifting holdings with prices. A next-open ledger must explicitly account for the close-to-open interval, fills, cash, quantity rounding, price drift, and the period during which new weights earn returns.

## 2. Distinguish a DSR learning signal from a reported Sharpe ratio

The released [reward module](../src/agents/reward.py) computes differential Sharpe using previous exponentially weighted moments:

$$
D_t=\frac{B_{t-1}(r_t-A_{t-1})-\frac12A_{t-1}(r_t^2-B_{t-1})}
{(B_{t-1}-A_{t-1}^2+\epsilon)^{3/2}},
$$

$$
A_t=A_{t-1}+\eta(r_t-A_{t-1}),\qquad
B_t=B_{t-1}+\eta(r_t^2-B_{t-1}).
$$

It subtracts a turnover penalty and clips the reward to $[-10,10]$. The default $\eta=2/253$ controls adaptation. Fees already reduce the input return, while the additional turnover penalty changes the optimization preference. Record both, rather than treating them as one fee. Cold-start moment initialization and clipping can strongly affect early rewards.

A reporting convention for daily net returns is

$$
\operatorname{Sharpe}=\sqrt{252}\frac{\overline{r-r_f}}{s(r-r_f)},\qquad
\operatorname{MDD}=\max_t\left(1-\frac{V_t}{\max_{0\le u\le t}V_u}\right).
$$

Specify the risk-free conversion, variance convention, and initial wealth. The paper states $2.5\%$ annual risk-free rate; current `_calc_metrics` uses zero risk-free rate and omits initial wealth from its running-peak array. A matching metric name therefore does not establish a matching calculation. DSR is a sequential learning signal, not the full-period Sharpe ratio.

## 3. Separate group-relative learning from evaluation-time choice

For candidate returns or rewards $R_g$ sampled from one state, group-relative advantages and a clipped objective can be expressed as

$$
\widehat A_g=\frac{R_g-\overline R}{s_R+\epsilon},\qquad
L=\frac1G\sum_g\min(\rho_g\widehat A_g,\operatorname{clip}(\rho_g,1-\varepsilon,1+\varepsilon)\widehat A_g).
$$

[`_run_period_grpo`](../src/training/trainer.py) samples a group, calls `env.simulate` for each candidate, standardizes rewards, forms Gaussian-density ratios, and adds entropy and expert auxiliary terms. These are concrete training operations to inspect.

The same routine chooses `argmax(rewards_t)` even when `train_mode=False`. Evaluation must state an action rule available before outcomes are known, such as a fixed deterministic policy or a predeclared sampling rule. Selecting the best realized candidate after observing that period's reward is not the same evaluation. Here the environment's old-weight return ordering further means the candidates mostly compete on transaction penalties. Both issues need resolution before connecting this runner to the manuscript's GRPO comparison.

## 4. Make rolling evaluation an auditable calendar

The manuscript specifies data from January 4, 2010 to February 28, 2026, CSI 300 constituents, ten windows shifted by one year, and five years of training, one of validation, one of testing. It states January 2017–February 2026 as final out-of-sample coverage. Exact exchange dates and the final partial year are still needed.

| Protocol item | Manuscript record | Source/readout question |
| --- | --- | --- |
| Window lengths | Five/one/one years | `TrainConfig` defaults to seven/one/one; supply actual historical configuration. |
| Model selection | Validation DSR, rolling transfer of checkpoints | `state_dict()` snapshots lack deep copies; selection state persists across windows. |
| Test aggregation | Ten rolling windows | Current training function performs the final test on `rolling_slices[-1]`; show how all historical test years were assembled. |
| Trading costs | $0.03\%$ commission, $0.1\%$ sell-side tax | Source has one symmetric fee default of $0.05\%$. |
| Baselines | Ten models including equal weight | Paper states quarterly equal-weight rebalance; other policies decide daily. |
| Universe | CSI 300, with exclusions | Audit delistings, long suspensions, changing constituents, and survivorship. |

Overlapping training windows are normal. Out-of-sample returns require a unique decision record per date; duplicated tests cannot be counted as independent evidence. Freeze model selection before opening the test period. A run manifest should attach code revision, preprocessing fit ranges, data availability, seeds, hyperparameters, baseline budgets, and daily holdings/costs.

## 5. Inspect every reported result table

Drawdowns below retain the paper's negative-return sign. Each value is transcribed from the named table; no missing series is reconstructed. Tables 6-2 and 6-3 appear in [Case 1](01-causal-multifrequency-graphs.md). The following six tables complete the record.

### Table 6-1 — Full out-of-sample comparison

| Model | Annual return | Annual volatility | Maximum drawdown | Sharpe | Calmar | Annual turnover, as labeled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alex-Fin | 18.72% | 11.34% | -12.65% | 1.43 | 1.48 | 1.27 |
| MVO | 7.35% | 14.62% | -33.03% | 0.33 | 0.22 | 2.84 |
| Equal weight | 9.12% | 16.78% | -38.47% | 0.39 | 0.24 | 0.32 |
| CSI 300 | 4.26% | 18.23% | -40.12% | 0.09 | 0.11 | — |
| LSTM | 10.57% | 15.89% | -29.74% | 0.51 | 0.36 | 2.16 |
| GAT | 12.33% | 14.25% | -26.31% | 0.69 | 0.47 | 1.98 |
| FinCast baseline | 13.68% | 13.17% | -22.45% | 0.85 | 0.61 | 1.83 |
| PPO | 14.21% | 13.02% | -20.18% | 0.90 | 0.70 | 1.75 |
| DQN | 11.05% | 15.36% | -28.64% | 0.55 | 0.39 | 2.31 |
| SAC | 13.84% | 13.56% | -21.73% | 0.84 | 0.64 | 1.89 |

The headline record is $18.72\%$, $1.43$, and a $12.65\%$ drawdown magnitude. The Calmar arithmetic $18.72/12.65\approx1.48$ is consistent at the reported precision. Sharpe must be recomputed from excess-return observations; dividing rounded annual return by annual volatility is not that computation. The turnover label lacks a complete unit convention, and the paper's bootstrap significance claim lacks resamples and dependence handling in the public package.

### Table 6-4 — GRPO and PPO training/generalization

| Algorithm | Convergence epochs | Training Sharpe | Validation Sharpe | Out-of-sample Sharpe | Sharpe decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO | 42 | 1.68 | 1.51 | 1.43 | 14.88% |
| PPO | 76 | 1.72 | 1.24 | 0.90 | 47.67% |

Decay matches $(S_{train}-S_{test})/S_{train}$ after rounding. Epoch counts are not wall-clock or sample-efficiency measurements: group size, environment calls, and convergence criterion matter. The embedded manuscript image labeled as a convergence plot was identified as a DeepSeek-V3 reference figure; it is not used as an Alex-Fin result plot. The numerical table remains the available Alex-Fin record.

### Table 6-5 — Sharpe by market regime

| Model | Bull | Bear | Sideways |
| --- | ---: | ---: | ---: |
| Alex-Fin | 1.57 | 0.98 | 1.36 |
| MVO | 0.41 | -0.37 | 0.28 |
| CSI 300 | 0.27 | -0.52 | 0.05 |
| PPO | 0.97 | 0.24 | 0.82 |

The stated bull interval is January 2019–February 2021; bear is January–October 2022; sideways combines January 2017–December 2018 and January 2023–February 2026. These labels leave March–December 2021 and November–December 2022 unassigned. They are descriptive subsets until a complete classification rule is supplied. The paper's nearby assertion that Alex-Fin is the only positive bear-market model conflicts with PPO's $0.24$; the table supports a higher value, not uniqueness.

### Table 6-6 — Sharpe by volatility regime

| Model | Low | Medium | High |
| --- | ---: | ---: | ---: |
| Alex-Fin | 1.52 | 1.47 | 1.31 |
| MVO | 0.45 | 0.36 | 0.12 |
| PPO | 0.98 | 0.92 | 0.74 |

The paper uses CSI 300's 20-day rolling volatility, with boundaries at the $25\%$ and $75\%$ quantiles. High-volatility performance is lower even for Alex-Fin. Record whether thresholds were estimated on training history, expanding history, or the complete test sample; post-hoc subgroup description and an implementable regime detector have different information requirements.

### Table 6-7 — Sharpe by sector

| Sector | Alex-Fin | MVO | CSI 300 sector index |
| --- | ---: | ---: | ---: |
| Financials | 1.24 | 0.28 | 0.11 |
| Cyclicals | 1.37 | 0.31 | 0.15 |
| Consumer | 1.48 | 0.42 | 0.23 |
| Technology | 1.51 | 0.35 | 0.18 |
| Healthcare | 1.39 | 0.33 | 0.16 |

Within this table technology has the highest Alex-Fin value. Establish sector membership dates and whether policies were retrained in each universe or attribution was computed from one portfolio before comparing mechanisms across sectors.

### Table 7-1 — Module ablations

| Configuration | Sharpe | Maximum drawdown | Annual return |
| --- | ---: | ---: | ---: |
| Full Alex-Fin | 1.43 | -12.65% | 18.72% |
| Daily data only | 0.92 | -20.37% | 13.15% |
| Serial coupled temporal/spatial architecture | 1.05 | -18.42% | 14.37% |
| Single FFN replacing MoE | 0.87 | -22.19% | 12.64% |
| PPO replacing GRPO | 0.90 | -20.18% | 14.21% |
| Daily-return reward replacing DSR | 0.68 | -28.54% | 11.83% |

Each row removes or replaces a component relative to the full system. These are interacting counterfactual configurations; their Sharpe gaps cannot be added into a contribution waterfall. The largest listed drop is the reward replacement, $1.43-0.68=0.75$, versus $0.56$ for MoE removal. The nearby manuscript text calls MoE the largest decrease; that interpretation does not cover every row.

## 6. Relate conclusions to their available evidence

The tables support a coherent reported ranking and motivate the multi-frequency, expert, and reward designs. They do not by themselves establish causal attribution, statistical uncertainty, execution realism, or current-source reproducibility.

The manuscript also reports placebo and robustness checks: shuffled-return Sharpe mean $0.12$ and $95\%$ quantile $0.37$ across $1000$ repetitions; randomized-graph Sharpe $0.58$ with $25.74\%$ drawdown; a changed-period Sharpe of $1.37$; and a higher-cost Sharpe of $1.28$ with $16.34\%$ return. These remain prose-level records until run outputs, sampling procedures, and configurations are available. The alternate period and universes require their own coverage and licensing manifests. See [Results](../docs/results.md) and the complete manuscript for the full scope.

## 7. Complete a useful review without running an experiment

| Task | Deliverable under `contributions/` | Acceptance |
| --- | --- | --- |
| Execution contract | Observation → decision → fill → holding-return timeline | Explain old/new weights, costs, cash, and available prices. |
| Table reconciliation | One table's original row, interpretation, source location | Preserve numbers; identify one claim supported and one unresolved. |
| Rolling protocol | Calendar template and required run fields | Identify unique test dates, validation-only selection, partial years, and per-window seeds. |
| Ablation fairness | Comparison checklist for one replacement | Specify data, budget, capacity, reward, tuning, and uncertainty. |

**Worked review:** for Table 6-5, record the claim “only positive in bear markets,” the Alex-Fin and PPO cells $0.98$ and $0.24$, and the proposed wording “highest reported bear-market Sharpe among these models.” Add the missing regime dates as an independent protocol question. This corrects an interpretation while preserving the historical table.

Use the [copyable record](../contributions/first-review.md), open an issue, and submit a focused PR through the [contribution guide](../CONTRIBUTING.md). An accepted record links the manuscript, exact source functions, and the reason for each conclusion. Algorithm changes and future experiments belong in separately scoped follow-up work.
