# Alex-Fin

**Multi-frequency graphs, adaptive experts, and reinforcement learning for portfolio allocation.**

English | [简体中文](README.zh-CN.md)

[Read the complete work](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper) · [Start with Case 1](cases/01-causal-multifrequency-graphs.md) · [Inspect all results](cases/02-risk-rewards-and-evaluation.md) · [Contribute](CONTRIBUTING.md)

Alex-Fin asks how a portfolio policy can combine observations arriving at different speeds, represent relationships between assets, and adapt as market conditions change. The research links native-frequency representations, graph-based temporal/spatial attention, a shared-and-routed expert module, and group-relative policy learning.

The author's research manuscript reports **18.72% annualized return, 1.43 annualized Sharpe, and 12.65% maximum drawdown magnitude** for January 2017–February 2026 out-of-sample coverage. The author confirms these are completed experiment results. This edition provides the manuscript record, authorized source, and detailed evidence review; a run manifest connecting the exact source revision to those historical results remains to be supplied.

![Original Alex-Fin research architecture](docs/assets/manuscript/multifrequency-architecture.png)

*Original manuscript illustration: seven frequency branches, risk-graph construction, and the temporal/spatial encoder. This depicts the research design. [Case 1](cases/01-causal-multifrequency-graphs.md) follows the actual source and identifies where its tensor axes and graph use differ. The image is unchanged; [provenance](docs/assets/manuscript/README.md) records its source and hash.*

## What you can learn and inspect

Follow a complete reasoning chain: available observations → graph meaning → expert routing → feasible weights → return accounting → out-of-sample evidence. You can inspect a precise methodological claim, trace it to code, and submit a review without downloading market data or training a model.

The intended reader knows basic Python, matrix algebra, probability, and simple financial returns. Each case introduces the formulas it uses and ends with a deliverable that another reader can assess.

| Route | Read | Leave with |
| --- | --- | --- |
| Understand the research question | [Architecture](docs/architecture.md) and [full architecture work](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-architecture) | A map of observations, representations, actions, and execution. |
| Trace the model | [Case 1: frequencies, graphs, experts, policy](cases/01-causal-multifrequency-graphs.md) | A tensor-axis trace, graph derivation, and interpretation of Tables 6-2 and 6-3. |
| Assess the evidence | [Case 2: rewards, rolling evaluation, ablations](cases/02-risk-rewards-and-evaluation.md) | Every row of Tables 6-1, 6-4–6-7, and 7-1, with comparison conditions. |
| Inspect implementation | [Source map and review](docs/source-status.md) | Concrete modules and the fidelity gaps to resolve. |
| Make a first contribution | [Copyable review record](contributions/first-review.md) | An issue and focused PR connecting a claim, source, and proposed improvement. |

## What the existing experiment record says

| Model | Annualized return | Annualized Sharpe | Maximum drawdown magnitude |
| --- | ---: | ---: | ---: |
| Alex-Fin | 18.72% | 1.43 | 12.65% |
| MVO | 7.35% | 0.33 | 33.03% |
| PPO | 14.21% | 0.90 | 20.18% |
| CSI 300 | 4.26% | 0.09 | 40.12% |

Source: manuscript Table 6-1. The [complete table register](docs/results.md) covers ten models, frequency settings, experts, optimization, market and volatility regimes, sectors, and ablations. The manuscript specifies five/one/one-year rolling training/validation/test windows, trading costs, and a 2.5% annual risk-free rate. Exact run configurations, daily outputs, and statistical resampling records remain necessary for independent reproduction.

The deeper cases preserve uncertain and adverse findings: high-volatility Sharpe falls, market-regime dates have gaps, two prose interpretations conflict with their tables, and component ablation gaps are not additive contributions. Original reference images are distinguished from Alex-Fin illustrations; no borrowed convergence or ablation image is presented as an Alex-Fin result.

## Materials, status, and next deliverables

| Area | Available | Actual status | Next deliverable |
| --- | --- | --- | --- |
| Complete works | [Research paper](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper), [architecture work](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-architecture) | Chinese full text, PDF, original-figure record, bilingual guides | Resolve documented manuscript ambiguities. |
| Learning cases | Two detailed cases in English and Chinese | Formulas, hand calculations, source navigation, all eight result tables, review tasks | Reviewed contributions for individual claims. |
| Source | Authorized model and selected tests | Research snapshot with recorded hashes; implementation review published | Align graph, attention, execution, and evaluation contracts. |
| Experiments | Historical manuscript tables | Author-reported completed experiments; no rerun in this edition | Exact run manifest, daily ledgers, configurations, and uncertainty records. |

Key implementation findings include daily-derived frequency views, asset-axis attention labeled temporal, currently unused adjacency, computation of all experts despite sparse routing, and candidate selection using simulated outcomes during evaluation. These are specific source findings, not a claim about which revision produced the historical experiments. They guide future implementation work.

## Contributions and project history

Julius presents the research problem, model architecture, and experiment narrative as part of a broader portfolio of financial and quantum research. The source package credits **Benjamin Team**; per-file revision and hash records remain in [the source manifest](docs/source-manifest.json). This edition distinguishes research design, team implementation, and upstream methods. Microsoft Qlib is an independently authored infrastructure reference.

[BenjaminAgent](https://github.com/Soros2040/Fintelligence) develops financial research workflows; [Q-Fintelligence](https://github.com/Soros2040/Q-Fintelligence) focuses on quantum computing and quantum machine learning. [Julius' future](https://github.com/Soros2040/julius-future) connects complete works, project reflection, and future questions.

## First contributions

| Open task | Suggested location | Acceptance criterion |
| --- | --- | --- |
| Review a tensor axis or graph formula | `contributions/`, linked to Case 1 | Exact function, equation, and shape trace; source facts separated from proposals. |
| Reconcile one result interpretation | `contributions/`, linked to Case 2 | Original table row, comparison conditions, and a supported wording change. |
| Improve bilingual explanations | Matching English/Chinese case | Preserve equations, numbers, source locations, and evidence status. |

Claim an issue → create a branch → submit a focused PR → respond to review → register the accepted contribution. The [contribution guide](CONTRIBUTING.md) and [worked example](contributions/example-review.md) explain this process. Full research runs are future work with an agreed protocol.

## Sources and license

[Primary sources](docs/sources.md) explain portfolio theory, Graphical Lasso, attention, experts, and policy optimization. [Original figure provenance](docs/assets/manuscript/README.md) identifies unchanged manuscript assets. [NOTICE](NOTICE) records source attribution.

Original code uses [MIT](LICENSE); original documentation uses [CC BY-NC-SA 4.0](LICENSE-DOCS). Third-party code, data, and referenced material retain their respective rights.
