# Case 2 — Turn portfolio actions into defensible results

English | [简体中文](02-risk-rewards-and-evaluation.zh-CN.md) · [Previous case](01-causal-multifrequency-graphs.md) · [Reported results](../docs/results.md)

**Question:** When a policy receives a better training reward, what evidence is needed to conclude that investors obtained a better net portfolio outcome?

This case follows one rebalance through weights, costs, portfolio returns, differential Sharpe rewards, and group-relative policy comparison. The numbers are invented teaching examples. They do not reproduce or extend the manuscript's 18.72% result. A calculator or a basic Python interpreter is enough; allow about 60–90 minutes.

## Learning objectives

You will distinguish target weights from executed holdings, calculate a self-financing one-step net return, evaluate a DSR reward, normalize a group of candidate rewards, and specify a fair out-of-sample comparison.

## 1. Start with holdings, not the last target

Suppose the portfolio immediately before trading is worth 100 currency units. Its **current** risky-asset and cash weights are

$$
\widetilde w=(0.50,0.30,0.20),
$$

and the new target is $w=(0.60,0.20,0.20)$. The current weights already reflect price movement since the previous trade. Comparing against an old target can understate or overstate the required trading.

For this simplified example, define total traded risky-asset notional as a fraction of pretrade value:

$$
u=|0.60-0.50|+|0.20-0.30|=0.20.
$$

We buy 10% and sell 10%. A convention that defines one-way turnover as half of the absolute changes would report 10%; this case uses 20% total traded notional. State the convention whenever reporting turnover.

Assume the all-in charge is 10 basis points, or $c=0.001$, per unit traded. The approximate fee fraction is $cu=0.0002$, costing 0.02 currency units. To keep this hand calculation explicit, the model charges that fraction first, then allocates the remaining wealth exactly to the target. It permits fractional holdings and ignores market impact. A production ledger must solve actual share quantities, buy and sell fees, lot sizes, and cash feasibility together.

## 2. Calculate gross and net returns separately

Suppose the two assets return $+1\%$ and $-2\%$ over the holding interval, and cash earns zero. The target's gross return is

$$
r_{\mathrm{gross}}=0.60(0.01)+0.20(-0.02)=0.002=0.20\%.
$$

Under the simplified fee-first convention:

$$
V_1=100(1-0.0002)(1+0.002)=100.17996,
$$

so the net return is **0.17996%**, approximately 18 basis points. The additive approximation $r_{\mathrm{gross}}-cu=0.0018$ differs by a small cross-product term. State which convention your implementation uses.

The next pretrade weights drift with realized prices. With zero cash interest and no further flows, they are

$$
\widetilde w_{i,1}=\frac{w_i(1+r_i)}{1+r_{\mathrm{gross}}}.
$$

They are approximately $(0.6047904,0.1956088,0.1996008)$. The next rebalance must start from those holdings. A reward penalty for turnover would influence learning, but only an actual deduction in the value ledger changes the net return just calculated.

## 3. Calculate the differential Sharpe reward

DSR provides a local risk-adjusted learning signal using exponentially weighted moments. Let $A$ estimate mean return and $B$ estimate the second raw moment. For realized return $R_t$:

$$
\Delta A_t=R_t-A_{t-1},\quad \Delta B_t=R_t^2-B_{t-1},
$$

$$
D_t=\frac{B_{t-1}\Delta A_t-\tfrac12A_{t-1}\Delta B_t}{(B_{t-1}-A_{t-1}^2+\epsilon)^{3/2}}.
$$

Only after computing the reward from the previous moments, update

$$
A_t=A_{t-1}+\eta\Delta A_t,\qquad B_t=B_{t-1}+\eta\Delta B_t.
$$

Use an independent numerical example: $A_{t-1}=0.001$, $B_{t-1}=0.000101$, and $R_t=0.003$. Ignoring a negligible numerical floor for the hand calculation:

- Estimated variance: $0.000101-0.001^2=0.0001$.
- Moment changes: $\Delta A=0.002$ and $\Delta B=-0.000092$.
- Numerator: $0.000101(0.002)-0.5(0.001)(-0.000092)=0.000000248$.
- Denominator: $0.0001^{3/2}=0.000001$.
- Therefore **$D_t=0.248$**.

With $\eta=0.01$, the updated moments are $A_t=0.00102$ and $B_t=0.00010008$. The initialization, variance floor, return definition, and reward clipping all affect learning and must be recorded. Near-zero variance can create unstable rewards. DSR is a local sensitivity-based objective; maximizing its realized sum does not automatically prove that a finite-sample final Sharpe ratio is globally optimal.

If net returns already deduct trading costs, an additional turnover penalty is a deliberate preference for lower trading, not the mechanism by which fees enter portfolio value. Tune it on training/validation data and name it separately from costs.

## 4. Compare actions using a common environment state

Alex-Fin adapts group-relative policy optimization to candidate portfolio actions. Suppose three candidates receive rewards

$$
r=(0.3,0.1,-0.1).
$$

Their mean is 0.1 and population standard deviation is $\sqrt{0.08/3}\approx0.163299$. Normalizing gives

$$
\widehat A\approx(1.224745,0,-1.224745).
$$

The first candidate is better relative to this group. This says nothing by itself about its absolute profitability. If every candidate loses money, the least bad candidate may still have positive relative advantage.

To make the comparison meaningful, copy the same initial holdings, cash, observations, and DSR moments for every candidate. Evaluate them over the same historical interval. Compute each reward without modifying a shared moment accumulator. Otherwise later candidates receive a different problem merely because they were evaluated later.

A clipped policy objective uses the importance ratio $\rho_g=\pi_\theta(a_g\mid s)/\pi_{\mathrm{old}}(a_g\mid s)$:

$$
J(\theta)=\frac1G\sum_g\min\left(\rho_g\widehat A_g,\operatorname{clip}(\rho_g,1-\varepsilon,1+\varepsilon)\widehat A_g\right).
$$

This learning rule still needs a specified stochastic policy and action density. For simplex-constrained weights, choose and document the distribution and transformation; a Gaussian density on an unrelated unconstrained variable cannot be silently substituted for a weight-space density. Equal rewards yield zero advantages under a safe zero-variance convention. The exact reward horizon and use of discounting must also be fixed before comparisons.

## 5. Check constraints where they matter

Softmax can turn scores into nonnegative weights that sum to one. It does not guarantee a 25% sector cap. A weight vector allocating 60% to one bank still violates that cap, even when all weights sum correctly.

The policy therefore needs an explicit feasible mapping. After execution, test sector weights again using actual holdings and fill prices. Lot rounding and partial fills can make realized holdings differ from targets. If prices subsequently drift beyond a cap, the protocol must say whether this triggers a rebalance, permits a temporary deviation, or records a violation.

## 6. Evaluate on a portfolio path

For a toy path $V=(100,110,99,108)$, total return is **8%**. The running peaks are $(100,110,110,110)$; the largest decline from a peak is $(110-99)/110=10\%$. Maximum drawdown is therefore **10%**, even though the final value is above the starting value.

Do not annualize the three-period example without knowing its sampling frequency. For a real daily run, publish daily net returns and specify the annualization constant, risk-free rate, and standard-deviation convention. Compute metrics from the series instead of reconstructing them from a few rounded summary values.

For a fair Alex-Fin comparison, use identical point-in-time universes, observation cutoffs, train/validation/test windows, execution accounting, risk constraints, and data access. Match or report tuning budgets and compute resources. Repeat stochastic methods across seeds. Report per-window results and confidence intervals; financial serial dependence makes an unexamined independent-day bootstrap inappropriate. State the chosen dependence-aware resampling method and its parameters.

## Connect the calculation to source

[reward.py](../src/agents/reward.py) implements the DSR moment update used above. The included [arithmetic checks](../tests/test_case_arithmetic.py) load this module and verify both hand-calculated rewards without training a neural network:

```bash
python -m unittest discover -s tests -p "test_case_arithmetic.py" -v
```

Next inspect [trading_env.py](../src/envs/trading_env.py), [constraints.py](../src/envs/constraints.py), and [trainer.py](../src/training/trainer.py). The environment snapshot uses old weights for the current return and then stores targets, which differs from the fee-first target-holding example here. Its cap helper clips and renormalizes individual weights. The runner also selects candidates using simulated historical rewards during evaluation. Use these differences to propose explicit timing, constraint, and inference tests; the [source review](../docs/source-status.md) records them as issues to resolve before a faithful rerun.

## Exercises

1. Repeat the one-step portfolio calculation with 30 basis points charged per unit of total traded notional.
2. Compute the trade needed to return the drifted weights to $(0.60,0.20,0.20)$. Explain why the old target is an insufficient starting point.
3. Recalculate DSR if the realized return in its separate example is $-0.003$.
4. Give three negative rewards whose normalized group still has a positive leading advantage. What claim can and cannot be made about that leading action?
5. Draft a result record containing the run revision, dates, data definition, seed, fee rules, reward convention, metric definitions, and daily-output locations. Mark which entries are currently unavailable for the manuscript experiment.

## Answer checks

At 30 basis points, the fee fraction is $0.003\times0.20=0.0006$, so net value is $100(0.9994)(1.002)=100.13988$ and net return is **0.13988%**. Returning to the target requires risky-asset changes of about $-0.0047904$ and $+0.0043912$ of current wealth; the cash balance absorbs the difference before any new fees.

For $R_t=-0.003$, $\Delta A=-0.004$ and $\Delta B=-0.000092$; the numerator is $-0.000000358$, giving **DSR = -0.358** under the same negligible-floor approximation. Rewards $(-0.1,-0.3,-0.5)$ produce a positive relative advantage for the first candidate while every reward remains negative. This identifies relative preference, not positive net investment return.

## Sources and a first contribution

See [Moody and Saffell](https://doi.org/10.1109/72.935097) for DSR, [PPO](https://arxiv.org/abs/1707.06347) for the clipped policy objective, and [DeepSeekMath](https://arxiv.org/abs/2402.03300) for GRPO's original setting. [Qlib](https://github.com/microsoft/qlib) provides examples of quantitative research infrastructure. Portfolio-specific adaptation and accounting remain the responsibility of this project.

A useful first contribution is a small deterministic example that checks fee deduction, weight drift, or identical-state group evaluation. Submit its assumptions, expected numerical output, and source through the [contribution workflow](../CONTRIBUTING.md).
