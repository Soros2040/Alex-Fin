# Case 1 — From multi-frequency observations to graphs, experts, and a policy

English | [简体中文](01-causal-multifrequency-graphs.zh-CN.md) · [Home](../README.md) · [Case 2](02-risk-rewards-and-evaluation.md)

**Research question:** can a portfolio model preserve information arriving at different speeds, represent relationships between assets, and adapt its representation before choosing weights?

This case follows the [complete research manuscript](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper) and [architecture manuscript](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-architecture), then checks their interfaces against the released source. Prerequisites are matrix algebra, covariance, attention, and portfolio weights. Numerical examples are hand calculations; experiment tables are historical manuscript records. Reading and contributing require no training run.

## 1. Follow the original design

![Original Alex-Fin multi-frequency and graph architecture](../docs/assets/manuscript/multifrequency-architecture.png)

**Original manuscript illustration:** seven frequency branches feed GLASSO-DY and MB-PSTSA-GNN. Chinese labels describe native-frequency patches and time-based masks. The image is unchanged; the source implements a narrower route below. [Figure provenance](../docs/assets/manuscript/README.md).

The design poses four questions: which observations are available, which asset relations guide aggregation, which features are shared or specialized, and how the representation becomes a portfolio. A module name is insufficient: its tensor axes, estimation window, and accounting contract determine the answer.

## 2. Establish availability before aligning frequencies

The paper uses second, minute, hour, day, week, month, and quarter observations. Let $`X_f\in\mathbb R^{T_f\times N\times d_f}`$ denote time positions, assets, and input features. Projecting a patch of length $`P_f`$ to width $`D`$ gives:

```math
z_{f,k,i}=W_f\mathrm{vec}(\mathrm{Norm}(X_{f,k,i}))+e_f+e_i.
```

Frequency embeddings identify a branch; they cannot recover missing observations. Equal embedding width does not imply equal sequence length. The manuscript uses $`D=256`$ and eight attention heads; `TrainConfig` supplies these values, while standalone encoder defaults are $`D=128`$ and four heads.

A close available at 15:01 can enter a 15:05 decision. A first-quarter statement published on April 30 cannot enter an April 29 decision because its reporting period ended on March 31. Preserve event time, availability time, and revision time. A valid attention mask satisfies

```math
M_{qk}=\begin{cases}0,&a_k\le t_q,\\-\infty,&a_k>t_q,\end{cases}
\qquad \mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt D+M)V.
```

Here $`a_k`$ is availability, not row order. A quarterly token can attend to an earlier minute token if it satisfies the cutoff. A triangle over assets cannot enforce this temporal rule. The paper pairs decisions using the completed close with execution at the next opening.

[`_build_multi_frequency_inputs`](../src/training/trainer.py) currently resamples daily closes into week, month, and quarter endpoints, calculates percentage changes, zero-pads history, and returns `[batch, asset, lookback]`. These are four views of daily data, not seven independently observed streams. A source review should record endpoint labels, incomplete periods, missingness, and whether each value existed at the cutoff.

## 3. Define what a graph edge means

For covariance estimate $`S`$, Graphical Lasso estimates a sparse precision matrix:

```math
\widehat\Omega=\arg\min_{\Omega\succ0}\lbrace \mathrm{tr}(S\Omega)-\log\det\Omega+\lambda\sum_{i\ne j}|\Omega_{ij}|\rbrace .
```

Under a Gaussian interpretation, $`\rho_{ij\mid-ij}=-\Omega_{ij}/\sqrt{\Omega_{ii}\Omega_{jj}}`$ describes conditional association. Diagonal entries $`4,9`$ and off-diagonal entry $`-3`$ give $`0.5`$. This symmetric association does not establish intervention-based causation.

The paper and [`_dy_decomposition`](../src/agents/glasso_dy.py) use a variance-weighted squared-precision expression:

```math
d_{ij}=\frac{\sigma_{jj}^{-1}\Omega_{ij}^{2}}{\sum_k\sigma_{kk}^{-1}\Omega_{ik}^{2}}.
```

For row $`[2,1]`$ and variances $`[1,4]`$, the components are $`4/4.25\approx0.9412`$ and $`0.25/4.25\approx0.0588`$. Squaring discards the sign. Different denominators can produce asymmetry, but neither operation introduces a forecast horizon.

A conventional generalized forecast-error variance decomposition additionally needs a fitted dynamic model, impulse-response matrices, innovation covariance, and horizon summation. The function accepts `H=5` but does not use it. Treat its output as the implemented graph proxy; interpreting it as a verified five-step Diebold–Yilmaz measure requires a separate derivation. [Primary references](../docs/sources.md).

![Original manuscript graph illustration](../docs/assets/manuscript/risk-graph.png)

**Original graph illustration:** placed beside graph construction in the manuscript. No dated edge matrix or run manifest accompanies it. It illustrates the graph presentation rather than validating a transmission estimate. Missing glyphs in the original heading remain unchanged.

`_row_normalize_spillover` zeroes the diagonal and normalizes **columns**, defining `A[i,j]` as `j → i`. Zero columns receive uniform off-diagonal weights. When every column is normalized, `sum(A)/N` is mechanically one. Consequently that exported statistic cannot measure changing total system risk. This is a consequence of the normalization, ascertainable without running code.

## 4. Track the encoder's axes

The manuscript proposes parallel temporal and spatial branches: mix valid times for one asset, and mix asset neighbors at a usable time. A graph-bias interface can be written as

```math
S_{ij}=q_i^\top k_j/\sqrt D+\beta A_{ij},\qquad h_i^{space}=\sum_j\mathrm{softmax}_j(S_{ij})v_j.
```

A bias changes relative scores; a hard mask forbids edges. A specification must select its rule, including self-loops and zero edges, before claiming that a graph constrains attention. The manuscript then fuses branches and exchanges frequency tokens using timestamps.

In [`MBPSTSAGNNEncoder.forward`](../src/agents/mbpstsa_gnn.py), the actual path is:

| Operation | Tensor/interface | Interpretation |
| --- | --- | --- |
| Input projection | `[B,N,L] → [B,N,D]` | Lookback is absorbed into a projection. |
| Frequency addition | One frequency vector per asset token | Branch identity is explicit. |
| `time_attn` | Attention over `N`, with an `N × N` triangular mask | Its sequence axis is assets, not retained times. |
| `_space_attention` | `adjacency` is passed; its addition is commented out | Current output is independent of the graph. |
| Cross-frequency attention | `[B,N,F,D] → [B*N,F,D]` | Frequency summaries mix separately for each asset. |
| Frequency mean | `[B,N,D]` | Token timestamps have already been compressed. |

Spatial attention also receives `h_time`, making this part sequential. This concrete source snapshot needs alignment before it represents the paper's parallel native-time graph architecture. A manuscript graph ablation cannot automatically be attributed to this revision.

## 5. Read experts as a computation

The design uses one shared expert and eight routed experts, selecting two routed outputs per token:

```math
y=x+E_{shared}(\mathrm{RMSNorm}(x))+
\sum_{k\in\mathrm{Top2}(g(x)+b)}\widetilde p_k E_k(\mathrm{RMSNorm}(x)).
```

Selected probabilities $`0.4`$ and $`0.2`$ renormalize to $`2/3`$ and $`1/3`$. The shared path is always active. Specialization must be assessed with routing and representations; Top-2 alone cannot identify bull-market and bear-market experts.

[`AdaptiveMoE.forward`](../src/agents/moe.py) implements RMSNorm, SwiGLU, Top-2 weights, bias updates, and residual LayerNorm. Three distinctions matter:

- It evaluates **all eight** expert networks before weighting selected outputs. Sparse selection is not yet sparse computation.
- The paper specifies auxiliary-loss-free balancing; the source adds squared load-balance and orthogonality losses to the policy objective.
- Its orthogonality term compares shared output with the combined routed output. The paper describes pairwise constraints among the shared and selected experts. Cancellation in a mixture is not equivalent to pairwise orthogonality.

[`_compose_state_embed`](../src/training/trainer.py) then averages asset features and adds padded risk factors. [`GRPOPolicy`](../src/agents/grpo.py) maps the state to Gaussian latent actions and Softmax weights. Case 2 examines reward and evaluation.

## 6. Read the existing mechanism evidence

All rows of manuscript Tables 6-2 and 6-3 appear below. They belong to the stated January 2017–February 2026 out-of-sample study. Run configurations, seeds, and daily outputs are needed to associate them with an executable revision. Drawdowns retain the original negative-return sign.

### Table 6-2 — Frequency input settings

| Input | Annualized return | Annualized Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| All seven frequencies | 18.72% | 1.43 | -12.65% |
| Daily only | 13.15% | 0.92 | -20.37% |
| High frequency only: second/minute/hour | 10.62% | 0.68 | -24.51% |
| Low frequency only: week/month/quarter | 9.84% | 0.57 | -26.83% |
| Conventional up/downsampled alignment | 12.47% | 0.83 | -21.69% |

Daily-only Sharpe declines by $`1.43-0.92=0.51`$, or $`35.66\%`$. This motivates checking extra information. It does not isolate coverage, capacity, alignment, or tuning-budget effects. A fair comparison needs those controls. The current daily-data builder cannot alone recreate the seven-frequency comparison.

### Table 6-3 — Experts across regimes

| Setting | Bull Sharpe | Bear Sharpe | Sideways Sharpe | Full-period Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Full Alex-Fin | 1.57 | 0.98 | 1.36 | 1.43 |
| MoE removed | 1.12 | -0.15 | 0.87 | 0.87 |

The bear-market sign change is a reported contrast. To interpret it as specialization, request regime rules, routing distributions, a matched-capacity replacement, and seed uncertainty. Performance differences do not reveal which expert learned which economic mechanism. Case 2 records the regime dates and coverage gaps.

## 7. Source reading and contribution tasks

| Source | Inspect | Reading output |
| --- | --- | --- |
| [Frequency embedding](../src/agents/frequency_embedding.py), [input builder](../src/training/trainer.py) | Identifiers, padding, endpoints | Availability table for one decision. |
| [Graph estimator](../src/agents/glasso_dy.py) | Fallback, horizon, direction, normalization | Formula-to-function correspondence. |
| [Encoder](../src/agents/mbpstsa_gnn.py) | Axes and actual graph use | Annotated shape trace. |
| [Experts](../src/agents/moe.py) | Routing computation and losses | Design-to-source comparison. |
| [Policy](../src/agents/grpo.py) | Latent density and portfolio weights | Action-space explanation. |

**Task A — axis review.** Claim an issue and copy the [contribution record](../contributions/first-review.md). Trace `[B,N,L]` to the policy. Acceptance: name every axis and the mask axis; distinguish source facts from proposed changes.

**Task B — graph semantics.** Derive why column normalization fixes the matrix sum, and identify where a forecast horizon must enter a dynamic decomposition. Acceptance: equations, exact functions, and a primary methodological source.

**Task C — mechanism evidence.** Select a row of Table 6-2 or 6-3 and write its minimum comparison manifest. Acceptance: original numbers plus availability, capacity/tuning controls, and uncertainty requirements.

Submit a focused record under `contributions/` through the [workflow](../CONTRIBUTING.md). These tasks require reading and derivation, not experiment execution.
