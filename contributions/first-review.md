# First review record / 首次复核记录

[English workflow](../CONTRIBUTING.md) · [中文流程](../CONTRIBUTING.zh-CN.md) · [Accepted-record index / 记录索引](README.md)

Copy the fields below into a new Markdown file under `contributions/` after claiming an issue. A reading contribution can be complete without running an experiment. Maintainers register a record as accepted only after its PR is reviewed and merged.

通过 Issue 认领后，将下列字段复制到 `contributions/` 下的新 Markdown 文件。阅读贡献可以在不运行实验的情况下完成。维护者在 PR 审阅合并后，将记录登记为已接受。

## Template / 模板

- **Issue and scope / Issue 与范围:** link and the single claim being reviewed / 链接与本次复核的一项主张。
- **Status / 状态:** claimed → submitted → changes requested → accepted / 已认领→已提交→待修改→已接受。
- **Original wording or result / 原始表述或结果:** quotation or exact table row / 引文或准确表格行。
- **Primary evidence / 一手证据:** work, chapter/table, source function, revision where relevant / 作品、章节或表号、函数及适用版本。
- **Reasoning / 推理:** formula, shape trace, arithmetic, or comparison condition / 公式、形状追踪、算术或比较条件。
- **Supported conclusion / 支持的结论:** what follows from that evidence / 证据能够支持什么。
- **Open question / 未决问题:** what additional record is required / 还需要哪些记录。
- **Proposed edit / 建议修改:** corresponding English and Chinese wording / 对应中英文表述。
- **Validation / 检查:** links, numbers, counterpart language, and exact execution scope / 链接、数字、另一语言与实际运行范围。
- **Review and registration / 审阅与登记:** PR, reviewer, merge date, final record link / PR、审阅者、合并日期和记录链接。

## Worked example / 完整例子

**Scope / 范围:** manuscript Table 6-5 and the adjoining bear-market interpretation / 稿件表 6-5 与相邻熊市解释。

**Original claim / 原主张:** Alex-Fin is the only model with positive bear-market Sharpe / Alex-Fin 是熊市唯一正夏普模型。

**Evidence / 证据:** [full work](https://github.com/Soros2040/julius-future/tree/main/works/alex-fin-paper), Table 6-5: Alex-Fin `0.98`, MVO `-0.37`, CSI 300 `-0.52`, PPO `0.24`. The [evaluation case](../cases/02-risk-rewards-and-evaluation.md) preserves the complete row and conditions.

**Reasoning / 推理:** both `0.98` and `0.24` are positive. The ordering supports the highest reported value among the listed models; it does not support uniqueness. / 两个数值都大于零，排序支持这些模型中的最高报告数值，不支持唯一性。

**Proposed wording / 建议措辞:** “Alex-Fin has the highest reported bear-market Sharpe among the models in Table 6-5.” / “Alex-Fin 在表 6-5 所列模型中具有最高的报告熊市夏普。”

**Open question / 未决问题:** the stated regime dates leave parts of the test period unassigned; request the complete calendar classification. / 已列行情日期没有覆盖全部测试期，需要完整分类日历。

**Validation / 检查:** compare the original table cells, check both language versions, and preserve the historical record. No experiment is required. / 核对原表格单元、两种语言，保留历史记录，无需实验。

**Status / 状态:** illustrative completed review, not a submitted issue or merged contribution / 本页是已写完整的复核示例，尚未提交 Issue 或合并贡献。
