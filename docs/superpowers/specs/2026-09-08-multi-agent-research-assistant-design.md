# Multi-Agent Research Assistant 设计规格

> 日期：2026-09-08  
> 状态：已确认
> 用途：个人学习成果与非商业作品展示  
> Upstream：[`NirDiamant/GenAI_Agents`](https://github.com/NirDiamant/GenAI_Agents)，审计基准 `4c95ae14cc2462c442b5c064cccd74430d02bc46`

## 1. 背景

原项目因误删需要从现有环境仓库重新构建。目标是交付一个可运行、可测试、可追溯、可展示的多智能体研究助手，而不是仅恢复若干 notebook 示例。

系统接收用户上传的 PDF、Markdown 和 TXT 文档以及研究问题，通过规划、检索、分析、批评和写作阶段，生成带可验证引用的结构化研究报告。

本项目以 `NirDiamant/GenAI_Agents` 为技术底座。适合的 Agent、workflow 和 utility 实现应优先抽取与适配；仅在 upstream 缺失、存在已知缺陷，或无法满足生产化边界时自行实现。

## 2. 目标与非目标

### 2.1 目标

- 提供 Planner、Retriever、Researcher、Critic、Writer 五类职责明确的 Agent。
- 使用 LangGraph 实现显式状态、条件路由、有限返工循环和阶段恢复。
- 支持 PDF、Markdown、TXT 的本地摄入、切块、向量检索和来源追踪。
- 默认通过阿里云百炼 DashScope 的 OpenAI 兼容接口调用 `qwen3.7-flash`。
- 保留 OpenAI Provider 扩展点，并提供确定性 Fake Provider。
- 提供 FastAPI、简洁同源网页、运行进度事件和持久化报告。
- 提供 30 条可复现 benchmark，对比四种工作流配置。
- 完整保留 upstream 许可证、作者归属和逐文件修改记录。
- 形成透明标注的 2026 年 7 月至 9 月重建提交历史，并发布 `v1.0.0`。

### 2.2 非目标

- v1.0 不提供互联网搜索、爬虫或在线论文搜索。
- v1.0 不提供用户注册、登录、权限与多租户隔离。
- v1.0 不采用 React/Vite 等独立前端工具链。
- v1.0 不面向大规模并发或分布式任务队列。
- v1.0 不承诺逐字复现外部模型输出；只保证配置、输入、过程和指标可追踪。
- 本项目不得用于商业用途，除非另行取得 upstream 许可人的书面授权。

## 3. Upstream 复用策略

### 3.1 复用原则

1. 优先抽取已有流程、状态模型、评分器和工具逻辑。
2. notebook 代码进入应用前必须去除全局状态、交互式输入、输出副作用和硬编码凭据。
3. 对存在错误、安全风险或来源丢失的问题进行修复，不机械复制。
4. 每个适配文件记录来源 notebook、cell、固定 commit、修改内容和适用许可证。
5. 仅借鉴概念而未复制具体表达的文件在 `docs/upstream-analysis.md` 标记为 `concept-only`。

### 3.2 主要映射

| 本项目能力 | Upstream 来源 | 处理方式 |
|---|---|---|
| LangGraph 状态与条件路由 | `scientific_paper_agent_langgraph.ipynb` cells 15、19、21 | 适配并扩展状态、节点和有限返工环 |
| Planner 与 Critic 结构化输出 | 同上 cells 13、19 | 适配 Pydantic schema 与节点职责 |
| 顺序多 Agent baseline | `multi_agent_collaboration_system.ipynb` cells 6、11–21 | 提取为实验基线，不作为主工作流 |
| 文件格式检查与摄入路由 | `document_intake_agent_langgraph.ipynb` cells 13、15、19、23 | 复用路由思想，改为安全本地读取 |
| 切块、候选检索、查询改写与重排 | `EU_Green_Compliance_FAQ_Bot.ipynb` cells 21、25、36 | 保留算法流程，修复距离解释、来源丢失和接口错误 |
| Trace 评估与质量门禁 | `trace_based_agent_evaluation.ipynb` cells 5、9、13、15、19 | 直接演化为框架无关 evaluator 和 pytest |
| 网页正文抓取 | `search_the_internet_and_summarize.ipynb` cell 8 | v1.0 不启用，仅登记为未来可选能力 |

### 3.3 许可证与归属

- 根目录保留 upstream 自定义非商业许可证。
- 增加 `THIRD_PARTY_NOTICES.md`，记录来源、固定 commit、贡献者、文件映射和修改。
- 增加 `THIRD_PARTY_LICENSES/GenAI_Agents-LICENSE.txt` 原文副本。
- README 明确本项目是独立学习项目，不是官方 fork、关联项目或获背书项目。
- 不将含 upstream 派生代码的项目宣称为 MIT、Apache 或商业可用项目。

## 4. 总体架构

系统是一个模块化 Python 单体：

```text
Browser
  │ upload / start / SSE / report
  ▼
FastAPI API + Static UI
  │
  ├── Document Service ── Loader → Cleaning → Chunking → Embedding → Local Index
  │
  └── Research Service ── LangGraph Workflow
                         Planner
                            ↓
                         Retriever
                            ↓
                         Researcher
                            ↓
                          Critic
                         ↙      ↘
                    revise      sufficient
                       │             ↓
                       └────────── Writer
                                      ↓
                              Citation Validator
                                      ↓
                              Research Report

SQLite: documents, chunks, runs, events, reports
```

### 4.1 模块边界

```text
app/
├── agents/          # 五类 Agent 节点与 prompt
├── api/             # FastAPI 路由、schema、依赖注入
├── domain/          # 领域模型、枚举、错误类型
├── evaluation/      # benchmark、trace、metrics、报告器
├── providers/       # Chat/Embedding Provider 抽象与实现
├── retrieval/       # loader、chunking、index、query fusion
├── services/        # 文档与研究应用服务
├── storage/         # SQLite repository 与迁移初始化
├── workflow/        # LangGraph state、graph、routing、events
├── static/          # HTML、CSS、JavaScript
├── config.py        # 类型化 settings
└── main.py          # 应用工厂与启动入口
```

各模块通过明确接口协作，Agent 不直接访问 FastAPI、环境变量或数据库。

## 5. 数据模型与持久化

### 5.1 核心模型

- `Document`：ID、原文件名、媒体类型、内容哈希、状态、创建时间。
- `EvidenceChunk`：ID、document ID、页码、chunk 序号、文本、检索分数、embedding 版本。
- `ResearchPlan`：研究目标、子问题、检索查询、完成条件。
- `Finding`：论点、支持 Evidence ID、冲突 Evidence ID、置信说明。
- `Critique`：是否充分、证据缺口、修订查询、反馈。
- `Citation`：Evidence ID、文件名、页码、chunk、原文摘录。
- `ResearchReport`：摘要、主要发现、证据冲突、局限、参考来源。
- `WorkflowState`：请求、计划、证据、发现、批评、迭代次数、事件、错误和最终报告。
- `ResearchRun`：运行状态、模型配置、时间、token、重试、阶段和结果。

### 5.2 SQLite

SQLite 保存文档元数据、chunk 文本与向量、研究任务、阶段事件和报告。Embedding 以带模型版本的二进制数组保存，检索时使用 NumPy 进行归一化余弦计算。

本地上传文件、数据库、运行输出和索引目录全部加入 `.gitignore`。

## 6. 文档摄入与检索

### 6.1 摄入

```text
inspect → pdf / native text / unsupported → normalize → chunk → embed → persist
```

- PDF 使用本地解析器，保留页码。
- Markdown 和 TXT 使用 UTF-8，并提供明确的编码错误信息。
- 上传只接受允许的扩展名和媒体类型，限制单文件及总请求大小。
- 文件名经过净化，服务只访问配置的数据根目录。
- chunk ID 由 document ID、页码、序号和内容哈希确定。

### 6.2 Embedding 与检索

- Chat 与 Embedding 使用独立 Provider 配置。
- 真实运行默认使用 DashScope Embedding 模型，名称由 `DASHSCOPE_EMBEDDING_MODEL` 配置。
- 测试与离线演示使用确定性 Fake Embedding。
- Retriever 同时执行原查询和 Planner 生成的扩展查询。
- 候选按 Evidence ID 去重并融合排序，再交给 Researcher。
- 检索失败或无足够证据时返回可解释状态，不允许 Writer 补造事实。

## 7. Agent 工作流

### 7.1 Planner

将研究问题转换为目标、子问题、检索查询和完成条件。简单问题仍进入可追踪流程，但可以生成单步骤计划。

### 7.2 Retriever

从本地索引获取候选 EvidenceChunk，保留所有来源字段和检索分数。

### 7.3 Researcher

基于证据提取发现、比较来源并显式记录冲突。任何结论必须绑定 Evidence ID。

### 7.4 Critic

同时执行两类检查：

- 确定性检查：Evidence ID 是否存在、覆盖率、空引用和重复引用。
- LLM 检查：答案是否覆盖计划、证据是否充分、是否需要补检索。

Critic 最多触发两轮补检索，工作流另有总步数上限。

### 7.5 Writer

将已验证发现写成 Markdown 报告。Writer 只能引用当前状态中的 Evidence ID。生成后由 Citation Validator 再次验证并展开为可点击引用。

## 8. Provider 层

### 8.1 接口

- `ChatProvider.generate()`：文本生成。
- `ChatProvider.generate_structured()`：Pydantic 结构化输出。
- `EmbeddingProvider.embed_documents()` 与 `embed_query()`。
- 统一返回模型、token、耗时和重试元数据。

### 8.2 实现

- `DashScopeProvider`：默认，OpenAI 兼容 base URL，聊天模型默认 `qwen3.7-flash`。
- `OpenAIProvider`：可选，不在 Agent 内写厂商分支。
- `FakeProvider`：确定性测试、离线演示和 CI。

结构化输出失败时按“解析 → 校验 → 修复提示 → 有限重试”处理。网络调用设置超时、指数退避和错误归一化。API Key 只从环境变量读取，不记录、不返回。

## 9. API 与网页

### 9.1 API

- `GET /health`：应用、数据库和配置状态，不暴露秘密。
- `POST /api/documents`：上传并摄入一个或多个文档。
- `GET /api/documents`：列出已摄入文档。
- `DELETE /api/documents/{id}`：删除文档及关联索引。
- `POST /api/research`：创建异步研究任务并返回 run ID。
- `GET /api/research/{id}`：返回状态、指标和报告。
- `GET /api/research/{id}/events`：使用 SSE 返回阶段事件。

统一错误响应包含稳定错误码、用户可读消息和 request ID。

### 9.2 网页

原生 HTML/CSS/JavaScript，由 FastAPI 同源提供：

- 文档上传与摄入状态。
- 已摄入文档选择。
- 研究问题输入。
- Planner 至 Writer 的阶段进度时间线。
- Markdown 报告显示。
- 点击引用时展开源文件、页码、chunk 和原文。
- 失败、证据不足和重试状态清晰可见。

页面以桌面作品展示为主，同时支持基础窄屏布局。

## 10. 可靠性、安全与恢复

- 模型调用采用可配置超时、指数退避和最大重试次数。
- 所有 Agent 输出经过 Pydantic 校验。
- 每个阶段开始与完成时写入事件和状态快照。
- 服务重启后可从最后完成阶段重新调度未完成任务。
- 单个失败任务不影响其他任务和文档索引。
- 文件读取限制在数据根目录，拒绝路径穿越和不支持的类型。
- 不抓取任意 URL，避免 SSRF 和远程 prompt injection 面。
- 日志包含 request ID、run ID、阶段、模型、耗时、token、重试和错误类型。
- 日志、API 响应和测试 fixture 均不得包含密钥。

## 11. 评估

### 11.1 Benchmark

仓库包含 30 条研究型问题，每条记录：

- 问题与关联文档。
- 预期证据 ID 或证据条件。
- 参考答案要点。
- 允许的最大循环次数。
- 可选时延门槛。

比较四种配置：

1. Baseline LLM。
2. LLM + RAG。
3. Single Agent + RAG。
4. Multi-Agent + RAG。

### 11.2 指标

- retrieval recall@k。
- citation precision 与 evidence coverage。
- 答案要点 F1/覆盖率。
- 成功率和失败率。
- p50/p95 时延。
- token 用量和模型调用次数。
- Critic 循环次数。

评估结果输出 JSON、CSV 和 Markdown。真实模型的波动必须保留原始配置和运行记录，不把单次结果写成普遍结论。

## 12. 测试与完成标准

测试包括：

- loader、chunking、配置和安全路径单元测试。
- Provider 超时、重试、错误归一化和结构化输出测试。
- 检索排序、查询融合和来源保留测试。
- 正常、返工、证据不足、最大循环和恢复路径图测试。
- 引用完整性测试。
- API、SSE、数据库和静态资源集成测试。
- 从 upstream trace evaluator 演化的 benchmark 与质量门禁测试。

完成标准：

- 全部默认测试通过。
- Fake Provider 可执行完整离线演示。
- 本地配置 Key 后，`qwen3.7-flash` 真实 smoke test 通过。
- 自动化测试中的引用完整性为 100%。
- README、架构、upstream 分析、实验说明、技术报告、CHANGELOG 和许可证齐全。
- 仓库不包含 `.env`、API Key、上传文档、数据库或本地运行结果。

## 13. Git 重建与发布

现有 `main` 包含已推送的环境初始化和设计规格提交。用户已明确授权在完整备份与远端租约保护下重写该历史。

### 13.1 安全措施

1. 重写前创建本地 Git bundle 备份。
2. 重新构造 `main` 后核对提交树、文件状态和测试结果。
3. 使用 `git push --force-with-lease origin main`，避免覆盖意外出现的远端更新。
4. 推送 `v1.0.0` 标签。

### 13.2 透明性

README 和 CHANGELOG 必须说明：项目因误删而重建，提交日期是用于恢复开发里程碑的重建时间线，不是原 Git 对象的恢复。不得声称这些提交是未经重建的原始历史。

### 13.3 里程碑时间

所有提交使用 `Asia/Shanghai` 时区且安排在 20:00 之后：

| 日期时间 | 里程碑 |
|---|---|
| 2026-07-06 20:18 | 环境、许可证与 upstream 分析 |
| 2026-07-13 21:07 | 项目骨架、配置和领域模型 |
| 2026-07-22 22:16 | 基础 Agent 与 LangGraph 工作流 |
| 2026-07-30 20:43 | 文档摄入与切块 |
| 2026-08-08 21:26 | 向量检索与引用追踪 |
| 2026-08-17 22:09 | Critic 返工循环和恢复 |
| 2026-08-25 20:51 | Qwen/OpenAI Provider 与 API |
| 2026-09-01 21:34 | 评估框架和 benchmark |
| 2026-09-05 22:12 | 网页界面与完整集成 |
| 2026-09-08 21:40 | 测试、文档和 v1.0.0 |

提交按功能里程碑组织，不制造无意义的微提交。对较晚加入 upstream 的参考实现，只在其可用日期之后的重建里程碑中登记和适配。

## 14. 交付物

- 可运行的 FastAPI 应用和原生网页。
- 完整多 Agent RAG 工作流。
- DashScope、OpenAI 和 Fake Provider。
- SQLite 本地持久化与恢复。
- 30 条 benchmark、评估运行器和报告生成器。
- 自动化测试套件。
- 示例文档与离线演示。
- README、架构文档、upstream 分析、实验文档、技术报告、CHANGELOG。
- 许可证、第三方通知和逐文件来源记录。
- 重建后的 Git 历史、GitHub 推送和 `v1.0.0` 标签。
