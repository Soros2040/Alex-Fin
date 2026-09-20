# Sources and attribution / 来源与归属

[Home](../README.md) · [中文首页](../README.zh-CN.md)

## Project evidence / 项目证据

The architecture and result summaries are original public-facing explanations based on the author's supplied documents:

架构与结果页面是依据作者提供的下列文档重新撰写的公开说明：

- **Alex-Fin portfolio manuscript:** “Alex-Fin：多频率时空图网络与深度强化学习的自适应动态投资组合优化模型.” Research design in Chapters 4–5; principal results in Table 6-1. / 第 4–5 章描述方法与实验设计，表 6-1 记录主要实验结果。
- **Alex-Fin architecture document:** “alexfin架构（markdown）.” A detailed specification of frequency representation, graphs, experts, policy learning, and evaluation. / 描述频率表征、图、专家、策略学习与评估的详细规格。

These source documents are retained privately. This release publishes a summary and selected result values, rather than the full manuscripts. Manuscript-reported experiment results and newly constructed teaching examples are labeled separately.

源文档保留在私人资料中。本次发布的是重新组织的说明和选定结果数值，未公开论文全文。论文实验结果与本次编写的教学例子分别标明来源。

## Foundational references / 基础参考资料

| Source / 来源 | Role in the project / 在本项目中的作用 |
| --- | --- |
| [Markowitz, Portfolio Selection, 1952](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x) | Mean–variance portfolio baseline. / 均值方差投资组合基准。 |
| [Friedman, Hastie, Tibshirani, Sparse inverse covariance estimation with the graphical lasso, 2008](https://doi.org/10.1093/biostatistics/kxm045) | Sparse precision estimation and conditional-dependence graphs. / 稀疏精度估计与条件依赖图。 |
| [Diebold and Yilmaz, Better to give than to receive, 2012](https://doi.org/10.1016/j.ijforecast.2011.02.006) | Directional spillover measurement through forecast-error variance decomposition. / 基于预测误差方差分解的方向性溢出度量。 |
| [Veličković et al., Graph Attention Networks](https://arxiv.org/abs/1710.10903) | Attention over graph neighborhoods. / 图邻域上的注意力聚合。 |
| [FinCast: A Foundation Model for Financial Time-Series Forecasting](https://arxiv.org/abs/2508.19609) | Multi-resolution financial representation background. / 多时间分辨率金融表征背景。 |
| [Moody and Saffell, Learning to trade via direct reinforcement, 2001](https://doi.org/10.1109/72.935097) | Differential Sharpe ratio and direct reinforcement learning for trading. / 微分夏普比率及交易中的直接强化学习。 |
| [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | Clipped policy-surrogate background and a policy baseline. / 裁剪策略目标与策略基准。 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Original GRPO introduction in mathematical reasoning; portfolio use requires a separate adaptation. / GRPO 在数学推理任务中的原始提出，投资组合应用需要另行适配。 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | Shared/routed experts and routing-balance background; Alex-Fin's expert counts are project settings. / 共享与路由专家、路由均衡背景；Alex-Fin 的专家数量属于本项目设定。 |
| [Microsoft Qlib](https://github.com/microsoft/qlib) and [documentation](https://qlib.readthedocs.io/en/latest/) | Quantitative research infrastructure and point-in-time data concepts. / 量化研究基础设施与时点可知数据概念。 |

The teaching examples are newly constructed and use invented numbers. They explain mechanics and are not additional empirical evidence for Alex-Fin. References establish the provenance of the underlying methods; they do not validate every adaptation or performance claim in the project.

教学例子为本次编写，使用假设数值解释机制，不构成 Alex-Fin 新增的实验证据。参考资料说明基础方法的出处，不意味着它们已经验证本项目的每项改造或性能主张。

## Code provenance / 代码来源

The public edition includes the owner-authorized model source described in the [source map](source-status.md), with per-file provenance in the [source manifest](source-manifest.json). An earlier archive also contained a Qlib copy; this edition links to [upstream Qlib](https://github.com/microsoft/qlib), under its own [MIT license](https://github.com/microsoft/qlib/blob/main/LICENSE), as a separate infrastructure reference.

本公开版本包含[源码导航](source-status.md)所列的所有者授权模型代码，[源码清单](source-manifest.json)记录逐文件来源。较早的归档也包含 Qlib 副本，本版本将采用自身 [MIT 许可](https://github.com/microsoft/qlib/blob/main/LICENSE)的 [Qlib 上游](https://github.com/microsoft/qlib)单独列为基础设施参考。
