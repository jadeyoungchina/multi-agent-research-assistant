**第一步执行记录：环境、DeepSeek CLI 与实验设计**

记录日期：2026-10-18（Asia/Shanghai）。本次完成阶段 A，尚未执行阶段 B 的模型接入与预算控制，也没有调用付费模型。

**环境与离线检查**

项目现有 `.venv` 可以正常启动，Python 为 3.11.9；`pip check` 未发现依赖冲突。上一轮出现的基础解释器访问失败本次未复现，因此保留当前环境，无须重装或升级依赖。

| 检查 | 本次结果 |
| --- | --- |
| Python | 3.11.9 |
| pytest / OpenAI SDK / Pydantic | 8.3.2 / 1.43.0 / 2.8.2 |
| FastAPI / LangGraph / LangChain / NumPy | 0.112.2 / 0.2.18 / 0.2.16 / 1.26.4 |
| Node / npm | 24.15.0 / 11.12.1 |
| `python -m pip check` | No broken requirements found |
| `python -m pytest -m "not live" -q` | **492 passed, 1 deselected，64.80 秒** |
| `python scripts/verify_repository.py` | **PASS：167 个已跟踪文件，许可证、归属及 30 道回归题验证通过** |
| fake benchmark，四流程全部执行 | **120 条 trace，quality gate PASS** |

仓库验证器只覆盖 Git 已跟踪文件。新增计划、协议和本记录在本阶段尚未提交，另外进行了 JSON、计数、链接和内容检查。上述离线测试不证明真实模型质量或账号可用性。

fake 结果保存在 `artifacts/evaluation/step-a-20261017/fake/`。目录名沿用本轮开始时的日期；本记录使用完成时的客户端日期。环境版本快照及摘要另存于同级 `readiness.json`，该目录被 Git 忽略。CI 当前使用 Node 22，本机使用 Node 24；本次已通过现有离线检查，后续不把 CLI 对本机 Node 的兼容性等同于在 CI 安装 CLI 的兼容性。

**DeepSeek CLI 的实际状态**

- 已安装官方包 `@deepseek-ai/dsh@0.2.1-rc.2`，命令为 `dsh`。
- `dsh --version`、`dsh --help` 和 `dsh headless --help` 均正常退出。
- 支持 `headless` 单次任务、从 stdin 接收输入及 `--json` 事件输出，适合后续受控辅助任务。
- 当前本地 `headless` 配置选择 `deepseek-account`、`deepseek-flash`、`reasoningEffort: high`。
- 检测到本地 DeepSeek 账号凭据记录；未检测到默认 `DEEPSEEK_API_KEY` 配置。本次仅检查本地存在性，没有验证登录是否过期、推理接口是否可用或账户余额。
- 没有打印、复制或修改凭据，没有修改用户 CLI 的全局/个人配置。

CLI 账号登录模式和 API Key 模式是独立路由，不能仅凭“CLI 已登录”推断项目 API 已配置；也不能直接把 API Key 路由价格当作账号模式的实际账单。第二步先确认账号模式的计费、usage 和限额；若需要 API Key 路由，仅在本机配置，不在聊天里传递凭据。

官方 CLI 的单次任务入口参考 [命令帮助](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cli-help.md)。本次以本机安装包的帮助和配置为实际版本依据。配置与凭据路由依据本机安装包中的 `dsh-llm-deepseek`、`dsh-credentials-local` 和 `dsh-headless` 说明核对。

**固定设计，保留实现期校准**

机器可读草案见 [实验协议草案](../../benchmarks/real-v1/protocol.draft.json)。

设计确定为：

- 60 道自建真实文档问题，20 dev / 40 test；五类任务各 12 道，中英文各 30 道。
- 文档族、版本族、翻译/改写题按组划分；候选文档池与 gold 来源分离。
- Qwen 同一固定模型驱动四流程，测试集各运行 3 次，共 480 条 trace。
- DeepSeek CLI 优先承担受控辅助任务；其默认高思考配置不直接用于简单任务，第二步通过独立任务配置关闭思考并设置输出和循环上限。
- 语义裁判与被测生成分开；DeepSeek 四流程复验最多 160 条 trace，预算允许再执行。
- 总 API 预算上限 ¥100；共享预算入口实施前不启动付费 CLI 或 API 工作。
- 真人审核 60 道参考答案；20 份开发答案校准；正式抽查 80 份回答。
- 先按问题汇总重复运行，再做配对差值与分组区间；不预设多智能体一定提高质量。

草案明确保留待定项：真实文档清单及许可证、固定 Qwen 模型可用性、CLI 账号模式实际计费与 usage、开发集校准后的输出上限/检索参数、裁判规则和最终样本选择种子。冻结正式测试配置必须等这些项目完成，不能把本草案称为已经完成的实验预注册或实测结果。

**下一步的具体入口**

进入阶段 B：先实现共享费用账本和调用预算，再接入受控 `dsh headless --json` 辅助任务入口，确认其账号模式是否满足可核算费用的条件。后续 CLI 工作在独立任务目录和显式文件范围中执行，限制输出、重试、工具与任务轮数，避免默认代理自行扩展任务影响成本与可追溯性。

正式四流程评测仍通过相同的工作流/provider 接口运行。若将完整 Agent CLI 作为其中一条正式流程，必须作为新的实验变体单独定义，不能无记录地替代 DeepSeek 模型调用。CLI 内部任何模型、搜索或补充请求均需纳入账本；无法可靠计量或限制的模式先保留为待解决项。
