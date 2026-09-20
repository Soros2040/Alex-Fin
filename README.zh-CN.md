# Alex-Fin

**结合多频率图网络、自适应专家与强化学习的投资组合研究。**

[English](README.md) | 简体中文

[阅读完整作品](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper) · [从案例一开始](cases/01-causal-multifrequency-graphs.zh-CN.md) · [查看完整结果](cases/02-risk-rewards-and-evaluation.zh-CN.md) · [参与贡献](CONTRIBUTING.zh-CN.md)

Alex-Fin 研究组合策略如何结合不同速度到达的观测、表示资产之间的关系，并适应市场变化。研究设计连接原生频率表征、图约束时空注意力、共享与路由专家，以及组相对策略学习。

作者研究稿报告，2017 年 1 月至 2026 年 2 月样本外覆盖区间的**年化收益率为 18.72%、年化夏普为 1.43、最大回撤幅度为 12.65%**；作者确认它们来自已完成实验。本版本提供稿件记录、授权源码和深入证据核查，具体源码版本与历史结果之间的运行清单仍需补齐。

![Alex-Fin 原稿研究架构](docs/assets/manuscript/multifrequency-architecture.png)

*原稿插图：七个频率分支、风险图构建与时空编码器，表达研究设计。[案例一](cases/01-causal-multifrequency-graphs.zh-CN.md)追踪真实源码，说明张量轴和图使用方式的差异。图像保持原样，[来源清单](docs/assets/manuscript/README.md)记录出处与校验值。*

## 可以学到和核查什么

沿完整推理链阅读：可得观测→图的含义→专家路由→可行权重→收益核算→样本外证据。你可以核查具体方法主张，定位实现，提交一份有依据的复核，无需下载行情数据或训练模型。

适合具有基础 Python、矩阵代数、概率和简单金融收益知识的读者。每个案例介绍使用的公式，并以他人能够验收的产出结束。

| 路线 | 阅读材料 | 阅读产出 |
| --- | --- | --- |
| 理解研究问题 | [架构说明](docs/architecture.zh-CN.md)、[完整架构作品](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-architecture) | 观测、表征、动作与执行之间的对应。 |
| 追踪模型 | [案例一：频率、图、专家与策略](cases/01-causal-multifrequency-graphs.zh-CN.md) | 张量轴追踪、图推导、表 6-2 和 6-3 解读。 |
| 评估证据 | [案例二：奖励、滚动评估与消融](cases/02-risk-rewards-and-evaluation.zh-CN.md) | 表 6-1、6-4 至 6-7、7-1 全部行及其比较条件。 |
| 检查实现 | [源码导航与核查](docs/source-status.md) | 具体模块及其需要对齐的实现差异。 |
| 首次贡献 | [可复制复核记录](contributions/first-review.md) | 连接主张、来源和改进建议的 Issue 与聚焦 PR。 |

## 现有实验记录说明了什么

| 模型 | 年化收益率 | 年化夏普 | 最大回撤幅度 |
| --- | ---: | ---: | ---: |
| Alex-Fin | 18.72% | 1.43 | 12.65% |
| MVO | 7.35% | 0.33 | 33.03% |
| PPO | 14.21% | 0.90 | 20.18% |
| 沪深 300 | 4.26% | 0.09 | 40.12% |

来源为稿件表 6-1。[完整表格索引](docs/results.zh-CN.md)覆盖十模型比较、频率、专家、优化、市场与波动率区间、行业和消融。稿件声明五／一／一年的滚动训练／验证／测试窗口、交易成本和 2.5% 年无风险利率。独立复现仍需要准确运行配置、逐日输出与统计重采样记录。

深入案例保留不确定及不利发现：高波动夏普下降、行情日期存在缺口、两处正文解释与表格不符、组件消融差不能相加为贡献。参考文献图片与 Alex-Fin 图示分别记录，没有把借用的收敛或消融图展示为 Alex-Fin 结果。

## 材料、状态与下一项交付

| 模块 | 已有材料 | 实际状态 | 下一项交付 |
| --- | --- | --- | --- |
| 完整作品 | [研究稿](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper)、[架构稿](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-architecture) | 中文全文、PDF、原图记录、双语导读 | 解决已经登记的稿件疑点。 |
| 学习案例 | 两篇完整中英文案例 | 公式、手算、源码定位、八张结果表、复核任务 | 逐项主张的经过审阅的贡献。 |
| 源码 | 授权模型与部分测试 | 含校验值的研究快照，已有实现核查 | 对齐图、注意力、执行与评估约定。 |
| 实验 | 历史稿件表格 | 作者报告的已完成实验，本版未复跑 | 具体运行清单、逐日账本、配置和不确定性记录。 |

关键实现发现包括从日度派生的频率视图、名为时间注意力的资产轴处理、当前未使用的邻接矩阵、稀疏路由下仍计算全部专家，以及评估时按模拟结果选择候选。这些是针对具体源码的发现，不用于推断历史实验由哪个版本生成，而是为未来实现工作提供方向。

## 贡献与项目经历

Julius 将本项目的研究问题、模型架构和实验叙述作为金融及量子研究作品集的一部分。源码包署名为 **Benjamin Team**，逐文件版本与校验值记录见[源码清单](docs/source-manifest.json)。本版分别记录研究设计、团队实现和上游方法；Microsoft Qlib 是独立作者维护的基础设施参考。

[BenjaminAgent](https://github.com/Soros2040/Fintelligence)发展金融研究工作流；[量融智枢](https://github.com/Soros2040/Q-Fintelligence)聚焦量子计算与量子机器学习；[Julius' future](https://github.com/Soros2040/julius-future)连接完整作品、项目回顾和未来问题。

## 首次贡献任务

| 开放任务 | 建议交付位置 | 验收标准 |
| --- | --- | --- |
| 复核一个张量轴或图公式 | `contributions/`，回链案例一 | 具体函数、公式、形状追踪，区分事实与建议。 |
| 核对一项结果解释 | `contributions/`，回链案例二 | 原表行、比较条件、有依据的措辞修改。 |
| 改善双语解释 | 对应的中英文案例 | 公式、数字、来源位置、证据状态一致。 |

Issue 认领→建立分支→提交聚焦 PR→回应审阅→登记已接受贡献。[贡献指南](CONTRIBUTING.zh-CN.md)和[完整示例](contributions/example-review.zh-CN.md)说明具体流程。完整研究运行属于后续按约定协议推进的工作。

## 来源与许可

[一手来源](docs/sources.md)解释组合理论、Graphical Lasso、注意力、专家和策略优化。[原图来源](docs/assets/manuscript/README.md)标识保持原样的稿件图片，[NOTICE](NOTICE)记录源码归属。

原创代码采用 [MIT](LICENSE)，原创文档采用 [CC BY-NC-SA 4.0](LICENSE-DOCS)，第三方代码、数据与引用材料保留各自权利。
