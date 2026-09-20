# Alex-Fin research architecture

English | [简体中文](architecture.zh-CN.md) · [Home](../README.md)

Alex-Fin models allocation as a sequential decision problem. At a decision cutoff, the policy observes only information already available, combines asset-level and market-level features, and chooses target weights. An execution model turns those targets into holdings and returns; the learning procedure updates the policy from risk-adjusted feedback.

This page summarizes the author-supplied portfolio manuscript and architecture document. It describes the intended research system. The [source map and implementation review](source-status.md) identify the released modules and their current differences from this design. Run artifacts needed for a public reproduction are listed in [Results](results.md).

## Modules and interfaces

| Module | Input | Output | Main responsibility |
| --- | --- | --- | --- |
| Availability and preprocessing | Prices, volume, fundamentals, availability timestamps | Per-frequency, per-asset histories | Enforce the decision cutoff and fit transformations on permitted history. |
| Frequency representation | Native-frequency patches | Tokens with a common hidden width and frequency identifiers | Preserve the distinction between sampling resolutions. |
| Graph estimation | Trailing observations for each frequency | A separate asset adjacency matrix per frequency | Provide a statistical relationship structure. |
| Parallel temporal and spatial attention | Tokens, time masks, graph masks | Asset-level temporal and relational features | Learn serial behavior and cross-asset relationships. |
| Cross-frequency attention | Frequency-specific representations | Fused asset representations | Exchange information under availability masks. |
| Sparse experts | Fused tokens | Adaptive features | Combine shared structure with routed specialization. |
| Policy and feasible allocation | Adaptive features and current holdings | Target risky-asset and cash weights | Produce a portfolio satisfying the declared constraints. |
| Execution and learning | Targets, fill rules, returns, costs | Holdings, reward, policy update | Keep economic accounting separate from the training objective. |

For $N$ assets and frequency $f$, native observations have shape $T_f\times N\times d_f$. Patch projections can map these to $L_f\times N\times D$ without requiring equal sequence lengths. Masks must distinguish unavailable observations from a measured value of zero. Quarterly fundamentals also require publication timestamps; a quarter-end date alone does not establish availability.

## Recorded design settings

The source design uses seven frequencies: second, minute, hour, day, week, month, and quarter. It describes a hidden width of 256, eight attention heads, dropout 0.1, a single spatial graph-attention layer, and one shared plus eight routed experts. Each token activates two routed experts alongside the shared expert. Expert specialization is learned; the design does not assign an expert permanently to a named market regime.

The portfolio is long-only and unlevered with a cash component. Target weights sum to one. The manuscript also specifies a 25% sector exposure cap. Cash makes this feasible even when the available risky assets span fewer than four sectors.

## Graphs and the meaning of direction

A Graphical Lasso estimate solves a sparse precision-matrix problem:

$$
\widehat\Omega_f=\arg\min_{\Omega\succ0}\{\operatorname{tr}(S_f\Omega)-\log\det\Omega+\lambda_f\|\Omega\|_{1,\mathrm{off}}\}.
$$

Here $S_f$ is a trailing sample covariance and $\lambda_f$ controls sparsity. For Gaussian observations, a zero off-diagonal precision entry corresponds to conditional independence. The precision matrix is symmetric. This statistical association is a useful graph prior, but it does not by itself identify shock transmission over a future horizon.

The manuscript proposes normalizing squared precision entries and calls the resulting construction GLASSO-DY. Its equivalence to Diebold–Yilmaz forecast-error variance decomposition remains a derivation to verify. Row normalization may create asymmetric weights without adding a temporal mechanism. A faithful implementation should preserve the manuscript formula as a named variant and compare it with an explicitly specified dynamic spillover estimator. [Case 1](../cases/01-causal-multifrequency-graphs.md) explains the distinction with numbers.

## Policy learning and portfolio accounting

The design uses a differential Sharpe ratio (DSR) reward, with exponentially updated first and second return moments, and a turnover penalty. It adapts group-relative policy optimization to compare sampled allocation actions from the same state. A group-relative advantage has the form

$$
\widehat A_g=\frac{r_g-\bar r}{s_r+\epsilon}.
$$

All actions in a group need the same initial holdings, market state, reward-moment state, and future evaluation interval. Each candidate must be evaluated in an independent environment copy. Otherwise the comparison depends on candidate evaluation order. Group normalization does not establish unbiased gradients or guarantee superior generalization.

Softmax ensures positive weights that sum to one. A sector cap requires an additional explicit construction, such as a constrained optimization layer or a hierarchical capped allocation. The density used in the policy ratio must correspond to the sampled action representation; transforming a Gaussian action into constrained weights needs a carefully specified likelihood convention.

The reward is a training signal. Portfolio performance must come from a self-financing value ledger with actual fill prices, holdings drift, cash, fees, and rejected orders. A turnover penalty alone is not a cash deduction. [Case 2](../cases/02-risk-rewards-and-evaluation.md) works through these differences.

## Reproduction decisions to record

1. Specify every rolling train, validation, and test boundary. The manuscript describes 5/1/1-year windows, while the architecture document also contains a 7/1/1-year reference; the public run manifest must resolve the difference.
2. Use point-in-time constituents and publication times; document treatment of delistings, suspensions, corporate actions, and missing frequencies.
3. Define graph diagonals, normalization order, empty-neighbor handling, edge orientation, estimation window, and how hyperparameters are selected without test data.
4. Define the feasible action mapping and its effect on policy likelihoods and sector exposure.
5. Separate reward penalties from executed costs, and record execution timing, lot sizes, cash interest, and risk-free-rate conversion.
6. Freeze all choices before the held-out run and publish per-seed and per-window outputs alongside aggregate metrics.

These items make the research testable. They are not completed-run claims. Source details and foundational references are in [Sources](sources.md).
