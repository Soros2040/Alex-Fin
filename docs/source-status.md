# Source map and implementation review / 源码导航与实现核查

[Home](../README.md) · [中文首页](../README.zh-CN.md)

## Provenance / 来源

This edition includes the owner's authorized model source from revision `333dd4f5434ba43335ff781f91298291f1532d52`, compared with revision `24d7f4abd29eb396864a6805130de58958320f45` of the related model project. The first revision includes explicit frequency embeddings, GLASSO-DY, the graph encoder, experts, and training/evaluation orchestration and is the primary source for this release. The original projects and their Git histories remain private.

本版本纳入所有者授权的模型源码，主版本为 `333dd4f5434ba43335ff781f91298291f1532d52`，并与相关模型项目版本 `24d7f4abd29eb396864a6805130de58958320f45` 比较。主版本包含明确的频率嵌入、GLASSO-DY、图编码器、专家与训练评估流程。原项目及其 Git 历史继续保持私有。

The [source manifest](source-manifest.json) records the original location and SHA-256 of each copied source/test file. Core algorithm files are preserved as a research snapshot. The public package adds a focused dependency manifest and places the existing local feature reader in an importable package with a small initializer. It excludes private data, model weights, access settings, server deployment code, and source manuscripts.

[源码清单](source-manifest.json)记录每个复制源码与测试文件的原位置和 SHA-256。核心算法以研究快照保留；公开包新增精简依赖清单，将已有本地特征读取器放入可导入包，并提供小型初始化文件。私人数据、模型权重、访问配置、服务器部署代码与源论文均不属于本次发布内容。

## Read the code / 阅读顺序

| Responsibility / 职责 | Source / 源码 | What to inspect / 阅读重点 |
| --- | --- | --- |
| Frequency identifiers / 频率标识 | [frequency_embedding.py](../src/agents/frequency_embedding.py) | Embedding dimensions and identifiers. / 嵌入维度与频率索引。 |
| Graph construction / 图构建 | [glasso_dy.py](../src/agents/glasso_dy.py) | Precision estimation, fallback, and normalization. / 精度估计、回退与归一化。 |
| Encoder / 编码器 | [mbpstsa_gnn.py](../src/agents/mbpstsa_gnn.py) | Tensor axes and adjacency use. / 张量轴与邻接矩阵是否实际生效。 |
| Experts / 专家 | [moe.py](../src/agents/moe.py) | Shared path, routing, auxiliary losses. / 共享路径、路由与辅助损失。 |
| Policy / 策略 | [grpo.py](../src/agents/grpo.py) | Gaussian latent actions and softmax weights. / 高斯潜在动作与 Softmax 权重。 |
| Reward / 奖励 | [reward.py](../src/agents/reward.py) | Previous-moment DSR and update order. / 使用旧矩的 DSR 与更新顺序。 |
| Environment / 环境 | [trading_env.py](../src/envs/trading_env.py) | Action timing, fees, simulation state. / 动作时间、费用与模拟状态。 |
| Training and evaluation / 训练评估 | [trainer.py](../src/training/trainer.py) | Data slices, candidate selection, checkpoints, metrics. / 数据窗口、候选选择、模型快照与指标。 |

## Material findings before a faithful manuscript rerun / 忠实复跑前的关键发现

These are findings from the released source, not conclusions about which exact code generated the author's historical experiment. The exact correspondence requires the original run manifest.

以下结论来自本次发布源码，不用于断言作者历史实验究竟使用了哪个版本；二者对应关系需要原始运行清单确认。

1. **Graph use:** the encoder accepts `adjacency`, but its spatial attention currently does not use it. Its so-called temporal attention operates on the asset axis after the lookback dimension is projected away. A change-in-adjacency test and an asset-permutation test are needed before claiming the stated graph/temporal architecture. **图的使用：**编码器接收 `adjacency`，空间注意力当前未实际使用它；所谓时间注意力在回看维度投影后作用于资产轴。需要邻接变化测试和资产排列测试，才能确认对应的图与时间架构。
2. **Frequencies:** the training builder derives day/week/month/quarter inputs from daily closes; it does not supply native second/minute/hour observations. **频率：**训练构造器从日收盘价生成日、周、月、季输入，没有提供原生秒、分钟和小时数据。
3. **Graph estimator:** `_dy_decomposition` accepts `H` but does not use it. `_row_normalize_spillover` performs column normalization. The off-diagonal sparsity count subtracts the number of off-diagonal positions from a total nonzero count, which can trigger unintended fallback; the fallback is a ridge-regularized inverse covariance, despite a warning naming Ledoit–Wolf. **图估计器：**`H` 参数未参与计算，名称包含 row 的归一化函数实际按列归一化；非对角非零计数扣除了非对角位置总数，可能触发不符合预期的回退，回退实际是岭正则逆协方差，而提示文字称为 Ledoit–Wolf。
4. **Allocation constraints:** the cap helper clips each asset then renormalizes; it has no sector membership input and can exceed the cap after renormalization. The environment has no explicit cash asset or share-lot ledger. **配置约束：**上限函数逐资产截断后归一化，没有行业映射输入，归一化后仍可能突破上限；环境没有显式现金资产或整手股数账本。
5. **Evaluation choices:** the runner samples candidates, evaluates their historical rewards, and selects the best candidate even when `train_mode=False`. A deployable evaluation must choose actions without consulting outcomes for that decision. **评估选择：**即使 `train_mode=False`，流程仍在计算历史奖励后选择最佳候选；可部署评估必须在不查看该次决策后果的情况下选择动作。
6. **Return timing:** the environment computes the period's gross return from previous weights, charges target turnover, and then sets the target weights. Immediate candidate rewards therefore do not capture the candidate's subsequent holding-period returns. Holdings are not drifted with prices. This timing requires reconciliation with the learning objective and manuscript execution rule. **收益时序：**环境按旧权重计算当期毛收益、扣目标换手费，然后设置目标权重；候选即时奖励未包含其后续持有期收益，持仓也未按价格漂移，需要与学习目标和稿件执行规则统一。
7. **Checkpoints and metrics:** selected `state_dict()` objects are retained without a deep copy, so later updates can change the supposed best snapshot. Current Sharpe uses zero risk-free rate; drawdown omits initial wealth from the running-peak series. **模型快照与指标：**最佳 `state_dict()` 未深复制，后续训练可能改变所谓最佳快照；当前夏普口径未减无风险利率，回撤峰值序列没有纳入初始净值。
8. **Feature reader:** the optional lightweight training branch calls `load_feature`, while the supplied reader exposes `load_bin_feature`. Use the daily Parquet input route for inspection until this interface is reconciled. **特征读取：**可选轻量输入调用 `load_feature`，而读取器提供 `load_bin_feature`；接口统一前，阅读运行流程应使用日度 Parquet 输入。

The code is useful for studying concrete modules and designing corrections. Until these issues are resolved and the run provenance is established, it should not be presented as a verified reproduction of Table 6-1.

The expanded [model case](../cases/01-causal-multifrequency-graphs.md) also checks expert computation and graph statistics: all routed experts are evaluated before selection; auxiliary load balancing and the implemented orthogonality term differ from the manuscript; column-normalized adjacency makes `sum(A)/N` constant when every column sums to one. The [evaluation case](../cases/02-risk-rewards-and-evaluation.md) traces the five/one/one-year manuscript protocol against seven/one/one-year runner defaults, and notes that the final training report evaluates the final rolling test slice. These are source-reading findings; this documentation update did not execute the model, tests, notebooks, or experiments.

深化后的[模型案例](../cases/01-causal-multifrequency-graphs.zh-CN.md)继续核对专家计算与图统计：源码先计算所有路由专家再选择，辅助负载均衡与正交项和稿件有差异，各列和为一时邻接矩阵的 `sum(A)/N` 恒定。[评估案例](../cases/02-risk-rewards-and-evaluation.zh-CN.md)对照稿件五／一／一年窗口和运行器七／一／一年默认值，并指出训练最终报告只评估最后一个滚动测试切片。这些是源码阅读发现，本次文档深化没有执行模型、测试、Notebook 或实验。

这些源码可用于研究具体模块和设计改进。在解决问题并建立运行来源对应关系之前，不能将它称为表 6-1 的已验证复现。

## Bounded checks / 小范围检查

Python 3.12 or newer is the declared baseline. From the repository root, create an environment and install the development dependencies:

声明的 Python 基线为 3.12 或更新版本，在仓库根目录创建环境并安装开发依赖：

```bash
python -m venv .venv
# Activate .venv using your operating system's normal activation command.
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -p "test_case_arithmetic.py" -v
python -m pytest tests/agents tests/test_replay_audit.py
python -m src.scripts.run_training --help
```

The unit checks cover selected module properties, not manuscript performance or financial correctness of the entire runner. The inherited suite contains weak and version-sensitive assertions, so a green suite must be interpreted narrowly. Full training is not required for the learning cases. No historical performance numbers are generated by these checks.

单元检查验证部分模块性质，不能证明论文性能或完整运行流程的金融正确性。继承的测试含有较弱断言和版本敏感条件，因此全部通过也只代表有限结论。学习案例不需要完整训练，这些检查不会生成历史绩效数值。

The publication preparation checks passed: 28 Python files parsed, all 26 copied files matched their recorded source hashes, and all three teaching arithmetic checks passed. A text scan found no credential assignments, private connection hosts, or absolute local user paths in the selected public files. The current preparation runtime lacks PyTorch, SciPy, scikit-learn, PyArrow, and pytest, so the inherited neural-network/module suite was not executed here. No full training or performance rerun was performed.

发布准备检查已通过：28 个 Python 文件语法解析通过，26 个复制文件与记录的源校验值一致，三个教学算术检查全部通过。文本扫描未发现凭据赋值、私人连接主机或绝对本地用户路径。当前准备环境缺少 PyTorch、SciPy、scikit-learn、PyArrow 和 pytest，因此未在此环境运行继承的神经网络及模块测试，也没有执行完整训练或绩效复跑。
