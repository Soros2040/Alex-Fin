# Case 1 — Build a multi-frequency graph without future information

English | [简体中文](01-causal-multifrequency-graphs.zh-CN.md) · [Architecture](../docs/architecture.md) · [Next case](02-risk-rewards-and-evaluation.md)

**Question:** At today's decision time, which observations can a portfolio policy use, and what exactly does an edge between two assets tell it?

This case connects two parts of Alex-Fin: representing data at different frequencies and giving spatial attention a graph. All numbers below are teaching examples, independent of the manuscript experiment. A calculator and basic matrix knowledge are sufficient; no dataset download or model training is required. Allow about 60–90 minutes.

## Learning objectives

By the end, you should be able to distinguish an event date from an availability date, construct a causal attention mask, calculate a conditional-dependence edge, and explain why an asymmetric matrix does not automatically measure directional risk transmission.

## 1. Put the decision on a timeline

Suppose a policy forms its target portfolio at **15:05 on April 29, 2024**, using information available by that instant, and trades at the next opening. The following observations belong to its candidate feature store:

| Observation | Period or event time | Available time | Usable at the cutoff? |
| --- | --- | --- | --- |
| One-minute bar | April 29, 14:59–15:00 | April 29, 15:00:02 | Yes |
| Final daily close | April 29, 15:00 | April 29, 15:01 | Yes |
| First-quarter earnings | Quarter ended March 31 | April 30, 18:00 | No |
| Corrected historical volume | April 26 | April 30, 09:00 | No; use the previously available version |
| Completed weekly bar | Week ended April 26 | April 26, 15:02 | Yes |

The relevant predicate is `available_at <= decision_cutoff`. A database may attach a financial statement to March 31 because that is the reporting period. Joining it to March trading dates would leak a later announcement. Likewise, historical revisions need a version timestamp if the experiment claims to reproduce what was knowable at the time.

At the cutoff, the policy may know the closing price, but it cannot also assume that a trade triggered by that completed close was filled at the same close without a specified executable mechanism. This case uses the next opening to make the ordering explicit.

## 2. Keep sequence length separate from embedding width

Let $X_f\in\mathbb R^{T_f\times N\times d_f}$ hold observations at frequency $f$. Each frequency can have its own history length $T_f$, feature width $d_f$, and patch size $P_f$. A projection maps a valid patch into a common width $D$:

$$
z_{f,k,i}=W_f\,\operatorname{vec}(\operatorname{Norm}(X_{f,k,i}))+e_f+e_i.
$$

$e_f$ identifies frequency and $e_i$ identifies the asset. A common hidden width makes attention compatible; it does not make the sequences equal in length or equal in informational content. Do not manufacture minute observations by repeating a quarterly report.

Patch normalization uses only the observations in an available patch. A partial patch needs a declared rule: exclude it or encode it with a validity mask. Padding with zeros without a mask confuses “not observed” with a real zero return.

For query token $q$ at decision time $t_q$ and key token $k$ with availability $a_k$, a simple causal mask is

$$
M_{qk}=\begin{cases}0,&a_k\le t_q\\-\infty,&a_k>t_q.\end{cases}
$$

Then attention uses $\operatorname{softmax}(QK^\top/\sqrt D+M)V$. Availability masks also need to exclude invalid or padded observations. Fitting a scaler, selecting features, or estimating a graph on the entire dataset can still leak the future even if this attention mask is correct.

## 3. Calculate an interpretable graph edge

A covariance matrix measures marginal co-movement. Its inverse, the precision matrix $\Omega$, captures conditional relationships under a multivariate Gaussian model. Suppose three assets have the following positive-definite precision matrix:

$$
\Omega=\begin{pmatrix}1&-0.5&0\\-0.5&2&-0.5\\0&-0.5&1\end{pmatrix}.
$$

Partial correlation is

$$
\rho_{ij\mid\mathrm{others}}=-\frac{\Omega_{ij}}{\sqrt{\Omega_{ii}\Omega_{jj}}}.
$$

Therefore $\rho_{12}=\rho_{23}=0.5/\sqrt2\approx0.3536$ and $\rho_{13}=0$. The graph contains edges 1–2 and 2–3. Under the stated Gaussian model, assets 1 and 3 are conditionally independent given asset 2. This does not imply that their unconditional correlation is zero, nor that asset 2 causes either asset's returns.

Graphical Lasso estimates a sparse positive-definite $\Omega$ by balancing data fit and an off-diagonal sparsity penalty. Sparsity is a modeling choice, not a significance test. Choose its strength using training or validation periods and document the convention for penalizing diagonal entries.

## 4. Do not mistake normalization for transmission

Square the off-diagonal entries of the example matrix, set the diagonal to zero, and normalize each nonempty row. The result is

$$
A=\begin{pmatrix}0&1&0\\0.5&0&0.5\\0&1&0\end{pmatrix}.
$$

Now $A_{12}=1$ while $A_{21}=0.5$. The asymmetry came from different row totals, although the original relationship was symmetric. We have learned how each node divides attention among its neighbors. We have not identified a time direction or forecast horizon for a shock.

A standard generalized forecast-error variance decomposition requires temporal dynamics. For a fitted VAR with moving-average coefficients $\Phi_h$ and innovation covariance $\Sigma$, a common form is

$$
\theta_{ij}(H)=\frac{\sigma_{jj}^{-1}\sum_{h=0}^{H-1}(e_i^\top\Phi_h\Sigma e_j)^2}{\sum_{h=0}^{H-1}e_i^\top\Phi_h\Sigma\Phi_h^\top e_i}.
$$

The forecast horizon $H$ and lag propagation $\Phi_h$ are part of the definition. Generalized shares may require row normalization. Even this predictive connectedness should not be relabeled as structural causality without identification assumptions.

Alex-Fin's manuscript proposes a precision-based GLASSO-DY variant. To study it fairly, implement the stated formula under its own name, document its edge interpretation, and compare it with a temporal estimator rather than assuming equivalence. If diagonals are removed, decide whether to renormalize afterward; the operations are not interchangeable. For an isolated node, retain its own representation through a residual or self-loop rule rather than dividing by zero.

## 5. Send the graph into attention

For asset $i$, graph attention aggregates only permitted neighbors:

$$
h'_i=\sum_{j\in\mathcal N(i)}\alpha_{ij}Wh_j,\qquad
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k\in\mathcal N(i)}\exp(e_{ik})}.
$$

Choose an edge convention once. In this case, row $i$, column $j$ means “asset $i$ receives information from asset $j$.” A graph prior controls which interactions are allowed; learned attention weights determine their relative contribution. An attention weight is not itself a causal effect or a portfolio position.

Estimate each graph using the corresponding frequency's history available at the cutoff. A monthly graph can update less often than a minute graph. Record its last estimation time so a learner can inspect whether a relation was stale or unavailable.

## Inspect the released implementation

Read [frequency_embedding.py](../src/agents/frequency_embedding.py), [glasso_dy.py](../src/agents/glasso_dy.py), and [mbpstsa_gnn.py](../src/agents/mbpstsa_gnn.py). In the source snapshot, `_dy_decomposition` exposes a horizon parameter without using it; the subsequent normalization is by column. The encoder accepts adjacency but currently leaves its use commented out. These are concrete audit tasks, so the equations above describe the learning concepts rather than silently certifying that this implementation satisfies them.

Trace the tensor shape immediately after `input_proj`: the lookback is compressed into the embedding, leaving the asset axis as the sequence dimension. Check whether the following `time_attn` and triangular mask therefore have the intended temporal meaning. A useful exercise is to permute asset order and compare the correspondingly unpermuted outputs. Read the [implementation findings](../docs/source-status.md) before interpreting a successful shape test as a methodological validation.

## Exercises

1. Move the decision cutoff to April 29 at 14:59:30. Which rows in the observation table become unavailable?
2. Compute the three partial correlations and all nonzero entries of $A$ by hand. Explain the source of its asymmetry in one sentence.
3. Remove asset 2's edges. Define what its spatial layer should do and how you would test that behavior.
4. Design a comparison between a partial-correlation graph and a forecast-error spillover graph. Keep the policy, time split, random seeds, and graph sparsity budget comparable.
5. Write a short contribution containing a timestamp table, one numerical assertion, a source reference, and a proposed test. State what observation would falsify your claim.

## Answer checks

At 14:59:30, the final minute bar and daily close are unavailable; the completed previous week's bar remains available. The quarter report and revision remain unavailable. The three correlations are approximately 0.3536, 0.3536, and zero. Row normalization produces the 1 versus 0.5 edge weights because the middle asset has two equal neighbors while each outer asset has one.

For an isolated node, an explicit residual-only output or a documented self-loop gives finite behavior; the test should verify finiteness and absence of messages from excluded nodes. A graph comparison should report both predictive results and graph stability. Stronger predictive performance alone does not establish a causal interpretation.

## Sources and next step

Read [Graphical Lasso](https://doi.org/10.1093/biostatistics/kxm045), [Diebold–Yilmaz connectedness](https://doi.org/10.1016/j.ijforecast.2011.02.006), and [Graph Attention Networks](https://arxiv.org/abs/1710.10903). [FinCast](https://arxiv.org/abs/2508.19609) provides background on financial forecasting across temporal resolutions. [Project source notes](../docs/sources.md) connect these references to the manuscript.

Continue with [Case 2](02-risk-rewards-and-evaluation.md) to turn target weights into an economically meaningful result.
