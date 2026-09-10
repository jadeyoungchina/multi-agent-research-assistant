# Multi-Agent Research Assistant

[English](README.md)

面向本地文档的研究助手，提供 FastAPI API、同源网页和 LangGraph 工作流：
Planner → Retriever → Researcher → Critic → Writer → Citation Validator。
上传 PDF、Markdown 或 UTF-8 TXT，输入研究问题，查看进度和带来源引用的报告。

本仓库于 2026-09-08 在原本地项目意外删除后重建。提交日期用于恢复已记录的开发里程碑；这些提交不是原始 Git 对象，也不是未经重建的历史记录。

本项目是独立的非商业学习项目，选择性适配
[NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)，固定参考提交为
`4c95ae14cc2462c442b5c064cccd74430d02bc46`，与 Nir Diamant 无关联且未获其背书。
上游自定义 [LICENSE](LICENSE) 限制商业用途，商业使用须事先获得许可人的书面授权。
作者归属、来源 cell 和逐文件映射见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 已实现能力与边界

- 本地解析、保留来源的切块、embedding、余弦检索和倒数排名融合。
- 结构化 Agent 输出、有限 Critic 返工、引用 ID 校验和证据不足标记。
- SQLite 保存文档、向量、报告、阶段快照和有序事件；启动时重新调度未完成任务。
- 网页上传、文档选择、报告与原文摘录、SSE 进度和轮询回退。
- DashScope、OpenAI 兼容接口，以及用于离线演示和 CI 的确定性 fake provider。
- [30 条合成 benchmark](benchmarks/cases.jsonl)、四种工作流和 JSON/CSV/Markdown 输出。

本版本无互联网搜索、OCR、身份认证、多租户隔离或分布式任务队列。
引用校验验证来源 ID 和格式，不能证明论点真实。Fake 输出用于验证软件流程，
不能用作真实模型研究质量的证据。

## 离线 fake 快速开始

推荐 Python 3.11，支持范围为 3.11–3.12。安装依赖需要下载软件包；完成安装后，
fake 应用和离线测试不调用模型服务。在仓库根目录执行：

```bash
python -m venv .venv
```

Linux/macOS 用 `source .venv/bin/activate` 激活；PowerShell 用
`.venv\Scripts\Activate.ps1` 激活，然后安装：

```bash
python -m pip install -r requirements/upstream-requirements.txt
python -m pip install -r requirements/project-requirements.txt
```

Bash 启动命令：

```bash
CHAT_PROVIDER=fake EMBEDDING_PROVIDER=fake RETRIEVAL_MIN_SIMILARITY=-1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

PowerShell 启动命令：

```powershell
$env:CHAT_PROVIDER = 'fake'
$env:EMBEDDING_PROVIDER = 'fake'
$env:RETRIEVAL_MIN_SIMILARITY = '-1'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开[本地网页](http://127.0.0.1:8000)，上传并选择 `examples/demo-corpus` 中的文档，
提问“项目目标和评估局限是什么？”。API 交互文档见
[本地 Swagger UI](http://127.0.0.1:8000/docs)。Fake 会确定性地组合提供的证据，不是真实 LLM。
演示将相似度阈值设为 -1，使简化哈希向量能够展示来源证据；真实 provider 默认阈值是 0.15。
该演示阈值不能作为检索质量的依据。

## DashScope / Qwen 配置

复制 `.env.example` 为 `.env`，设置 `CHAT_PROVIDER=dashscope`、
`EMBEDDING_PROVIDER=dashscope`，在本地文件的空白 `DASHSCOPE_API_KEY` 字段填入自己的密钥。
默认模型和地址如下；如账号区域或模型支持情况不同，调整相应配置：

```dotenv
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen3.7-flash
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BATCH_SIZE=10
```

DashScope 兼容配置默认将 embedding 批次控制在当前 `text-embedding-v4`
的限制内；结构化 JSON 请求会关闭 Qwen thinking 输出，使工作流阶段返回
规模可控、便于校验的机器可读结果。

删除终端中先前设置的 provider 和相似度环境变量，或打开新终端并激活环境，再启动真实配置：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

系统环境变量优先于 `.env`。配置模型名不代表账号可用性或已经通过真实调用验证。
真实调用可能收费，并向服务商发送文档文本。也可将两个 provider 设为 `openai` 并在
本地设置 `OPENAI_API_KEY`，模型和地址字段见 `.env.example`。
切换 embedding 模型时使用独立数据库/上传路径并重新摄入文档，避免混用不同模型的向量。

## API 与 SSE 示例

以下采用 Bash curl 语法；Windows 请使用 `curl.exe` 和当前终端的引号规则。
上传必须显式提供与扩展名匹配的 MIME 类型：

```bash
curl -F 'files=@examples/demo-corpus/project-overview.md;type=text/markdown' http://127.0.0.1:8000/api/documents
curl http://127.0.0.1:8000/api/documents
```

上传返回 HTTP 201 和 `{"documents":[...]}`。将其中的 `id` 替换下方 `DOCUMENT_ID`：

```bash
curl -i -H 'Content-Type: application/json' -d '{"question":"What are the project goals?","document_ids":["DOCUMENT_ID"]}' http://127.0.0.1:8000/api/research
```

HTTP 202 返回 `run_id`、`status` 和 `Location` 响应头。将真实 `run_id` 替换 `RUN_ID`：

```bash
curl -N http://127.0.0.1:8000/api/research/RUN_ID/events
curl http://127.0.0.1:8000/api/research/RUN_ID
curl -N -H 'Last-Event-ID: 5' 'http://127.0.0.1:8000/api/research/RUN_ID/events?after=3'
```

状态依次为 `queued`、`running`，终态为 `completed` 或 `failed`。查询返回 `run` 和
`report`（报告尚未保存时为 null）。SSE 使用 `event: workflow`、整数 `id`，JSON 包含
`sequence`、`run_id`、`stage`、`event_type`、`payload`、`created_at`。
断线续传取 `after` 与 `Last-Event-ID` 的较大值，上例从序号 5 之后继续。
空闲时发送 keep-alive 注释，终态事件发送完毕后关闭流。
删除文档使用 `DELETE /api/documents/DOCUMENT_ID`。`/health` 的 provider 就绪状态只反映
配置和对象初始化，不会在线探测模型服务。

## 测试与实验

完整测试套件还需要 Node.js 22 来运行浏览器行为测试。应用运行不需要 Node 构建步骤，
CI 会配置 Python 和 Node 两种运行环境。

```bash
python -m pytest -m "not live" -q
python scripts/verify_repository.py
python -m app.evaluation.cli --provider fake --variants all --output-dir artifacts/evaluation/fake --enforce-gate
```

输出目录生成 `results.json`、`results.csv`、`report.md`。发布门禁只约束
`multi_agent_rag`，另外三种工作流用于对照。生成结果不提交 Git，CI 以 artifact 保存。
指标定义、门槛、真实模型命令和限制见[实验说明](docs/experiments.md)。
Fake benchmark 不支持真实模型质量、速度提升或成本节省的结论。

## 隐私与开发

默认上传目录为 `data/uploads`，数据库为 `data/research_assistant.db`。
Fake 模式下文档、文本、向量、问题、状态快照和报告保存在本机；真实 embedding/chat
模式会把文档文本、问题和检索证据发送给配置的服务商。本应用不加密本地文件，
也不提供自动保留期限。删除文档不保证清除旧研究报告或快照中的文本。
自定义语料的评估结果同样可能包含敏感信息。合成治理文档中的保留规则是虚构素材，
不是应用的数据治理实现。

个人使用请绑定 `127.0.0.1`。密钥只保存在本地环境配置，不应进入 Git、截图或实验报告。
仓库校验器检查 Git 跟踪的工作目录文件、常见密钥格式、运行数据、许可证、benchmark
证据与适配文件归属，不检查完整 Git 历史；提交前仍需审查暂存差异。
更多内容见[架构](docs/architecture.md)、[技术报告](docs/technical-report.md)、
[重建历史](docs/reconstruction-history.md)和[发布清单](docs/release-checklist.md)。
