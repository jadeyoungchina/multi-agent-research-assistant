# 三分钟项目讲解与演示

当前展示交付采用已完成的 AI 复核，不安排后续人工复核。真人验证为 0 是实验方法限制，
不是继续演示、交付或开发的阻塞项。以下内容对应现有代码和已保存结果。

## 演示顺序

| 时间 | 操作 | 说明重点 |
| --- | --- | --- |
| 0:00–0:30 | 运行 `python examples/run_demo.py`，打开打印的本机地址 | 当前是 fake 离线流程，真实模型评测单独保存 |
| 0:30–1:10 | 选中两份预载入资料，输入建议问题，点击“开始研究” | 观察 Planner、Retriever、Researcher、Critic、Writer 和引用验证阶段 |
| 1:10–1:40 | 打开“查看引用”，展示文件名、片段位置和原文 | 引用可追踪不等于语义支持已得到证明 |
| 1:40–2:30 | 打开正式结果和对比图 | 普通 RAG 本组最高；多代理相对单代理的提升伴随更长耗时 |
| 2:30–3:00 | 展示失败案例、notebook 与贡献边界 | 讨论复现、未知账单、AI 复核限制和可验证的后续问题 |

## 英文讲解稿

This project explores a practical question: when does multi-agent orchestration
help document research enough to justify its overhead?

The application accepts local PDF, Markdown and text files. It preserves source
locations during chunking and retrieval, then produces a report whose citations
can be inspected in the browser. The workflow separates planning, retrieval,
synthesis, critique and writing. Revision is bounded, and SQLite stores stage
checkpoints and ordered events so progress can be recovered and replayed.

This live demonstration uses deterministic fake providers. It demonstrates the
upload, workflow and citation interface without model API calls. The real-model
experiment is a separate saved artifact.

The experiment compares four workflows: direct generation, plain RAG, planned
single-agent RAG and the multi-agent workflow. It uses forty held-out questions
across eight documentation families, with three fresh repeats. The matrix retains
all four hundred and eighty executions, including seven failures. Code, data,
configuration, execution order and the primary comparison were frozen before
generation. Repeats are averaged within each question, and uncertainty is explored
by resampling document families.

The result is a tradeoff rather than a universal win for more agents. Plain RAG
has the highest observed primary quality, at about zero point nine two.
Multi-agent RAG scores about zero point eight eight, compared with zero point
eight two for single-agent RAG, but its median successful latency is roughly
five point eight times greater. Citation structure checks also do not establish
that the cited passage supports every claim.

The engineering contribution is the integration of traceable retrieval, bounded
execution, persistent progress, request accounting and reproducible comparison.
Selected upstream notebook material is attributed, and coding assistants were
used in implementation and evaluation. References and scoring are AI assisted;
there is no independent human validation. Original judgments, explicit source
corrections, unknown charges and failed executions remain visible.

A useful next investigation would target retrieval misses and unsupported
generalizations using development material and newly designed evaluation cases,
while keeping this held-out result as a fixed baseline.

## 常见追问

| 问题 | 可核查的回答要点 |
| --- | --- |
| 为什么有多智能体，普通 RAG 却最高？ | 问题本身是比较架构取舍；该小型文档集上，额外规划和写作步骤增加耗时，也可能引入遗漏和泛化错误，不能由流程复杂度预设赢家。 |
| 480 条是不是 480 道独立题？ | 不是。40 题 × 4 流程 × 3 次；先在题内平均重复，再配对，按 8 个文档家族 bootstrap。 |
| 有引用为什么还会错？ | ID 存在只证明引用结构合法。支持率要判断所引段落是否支持断言；失败案例包含格式合法但内容只获部分支持的回答。 |
| 如何避免答案泄漏？ | 三条 RAG 搜索同一完整 split 候选池，工作流只接收问题和候选资料；参考关键点与 gold 定位仅供评价端使用。 |
| 成本是否精确？ | 保留 3960 次物理请求；17 次用量未知，DeepSeek 账号价未核实，总金额未知。不能把条件成本差值当完整账单。 |
| 哪些属于上游适配？ | 按 notices 和源文件头逐项说明 Planner/图、文档 intake、trace/metric 的适配；其余项目工程模块和概念影响分别列出。 |
| 谁复核了结果？ | DeepSeek 新上下文和 Codex 原文裁定；同模型别名可能产生相关错误。后续人工复核已取消，不将 AI 结果写成人工验证。 |

[运行说明](demo-guide.md) · [英文项目摘要](project-brief.md) · [技术报告](../technical-report.md)
