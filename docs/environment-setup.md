# Environment Setup

> 生成日期：2026-09-07
> 目标项目：Multi-Agent Research Assistant（基于 upstream: NirDiamant/GenAI_Agents）
> 本轮范围：开发环境配置 + upstream 环境审计 + baseline 验证（不含业务代码开发）

## OS

- Windows 11 专业版，Build 26100（24H2）
- PowerShell 5.1.26100.6899（系统内置）

## Python

- **已安装：Python 3.11.9**（python.org 官方安装器，当前用户安装，无需管理员权限）
  - 路径：`C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe`
  - 选择原因：upstream README 要求 Python 3.9+，但其 requirements.txt 锁定 `numpy==1.26.4`（无 Python 3.13 预编译包），实际兼容范围 3.9–3.12；按约定优先选择 3.11
  - 系统 PATH 已通过官方安装器 PrependPath 配置
- pip 24.0 / setuptools 65.5.0 / wheel（venv 内置版本，见 Known Issues #2）

## Git

- Git 2.55.0.windows.3（已预装，未重复安装）
- git-lfs 3.7.1（已预装；upstream 仓库未使用 LFS，无需额外配置）
- `user.name` / `user.email`：**未配置，待用户输入**

## VS Code

- 已安装 1.119.0（用户级安装，`code` 命令已在 PATH），未重复安装

## Node.js

- 已安装 Node v22.22.2 / npm 10.9.7
- upstream 为纯 Python 项目，**不需要 Node.js**；保留现有安装即可

## Docker

- **未安装（刻意跳过）**
- 审计结论：upstream 仓库无 Dockerfile / docker-compose.yml / 任何 Docker 配置文件；README 仅提及"部分 agent 需要 Docker"。核心 baseline 与目标项目（云端 LLM API）均不需要
- 建议：仅当后续选用依赖容器化组件的教程时再安装 Docker Desktop

## Virtual Environment

- 项目目录：`D:\AIProjects\multi-agent-research-assistant\`
- 虚拟环境：`D:\AIProjects\multi-agent-research-assistant\.venv\`（Python 3.11.9）
- 激活方式：
  - PowerShell：`D:\AIProjects\multi-agent-research-assistant\.venv\Scripts\Activate.ps1`
  - Git Bash：`source D:/AIProjects/multi-agent-research-assistant/.venv/Scripts/activate`
- upstream 参考克隆：`D:\AIProjects\GenAI_Agents\`（浅克隆，commit 4c95ae1，未做任何修改）

## Project Dependencies

依赖分两层（均在 `.venv` 中，未污染系统 Python）：

1. **upstream 层**（`requirements/upstream-requirements.txt`，88 个锁定包）
   - 来源：`D:\AIProjects\GenAI_Agents\requirements.txt` 的副本，含 3 处最小修复（见 Known Issues #1）
   - 核心：langchain 0.2.16 / langchain-openai 0.1.23 / langchain-community 0.2.16 / langgraph 0.2.18 / openai 1.43.0 / pydantic 2.8.2 / tiktoken 0.7.0 / duckduckgo-search 6.2.13 / nltk 3.9.1 / autogen 0.3.0
2. **项目层**（`requirements/project-requirements.txt`）
   - fastapi 0.112.2 / uvicorn[standard] 0.30.6 / pytest 8.3.2
   - upstream 不要求这些包；它们是目标项目技术栈（FastAPI 服务 + pytest 测试）所需，版本与锁定栈同期兼容

复现命令：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements/upstream-requirements.txt
.venv/Scripts/python.exe -m pip install -r requirements/project-requirements.txt
```

## Environment Variables

- 模板：`.env.example`（已生成，仅含变量名，无任何真实 Key）
- 必需：`OPENAI_API_KEY`（upstream 29 处引用，云端 LLM 主力）
- 可选：`TAVILY_API_KEY` / `LANGCHAIN_API_KEY` / `GROQ_API_KEY` / `GOOGLE_API_KEY` / `NEWSAPI_KEY`（仅部分 upstream 教程使用）
- 填写方式：`cp .env.example .env` 后在 `.env` 中填入真实值
- 防护：`.gitignore` 已排除 `.env`、`.venv/`、`__pycache__/` 等

## Verification

| 验证项 | 结果 |
|---|---|
| `git --version` | ✅ 2.55.0.windows.3 |
| `python --version`（venv） | ✅ Python 3.11.9 |
| `python -m pip --version`（venv） | ✅ pip 24.0 |
| venv 创建与激活 | ✅ |
| upstream 依赖安装 | ✅ 88 包全部安装成功 |
| `import langchain` | ✅ 0.2.16 |
| `import langgraph`（含 StateGraph 编译运行） | ✅ 0.2.18 |
| `import pydantic` | ✅ 2.8.2 |
| `import fastapi` | ✅ 0.112.2 |
| `pytest` 运行 | ✅ 8.3.2 |
| upstream 测试套件（tests/） | ✅ 37 passed / 2 skipped / 1 failed（失败项为沙箱伪影，见 Known Issues #3） |
| 最小 LangGraph baseline（无需 API Key） | ✅ 图编译 + invoke 成功 |
| Docker | ⏭️ 跳过（项目不需要） |
| API Key 泄露检查 | ✅ 无真实 Key 写入任何文件 |

## Known Issues

1. **upstream requirements.txt 自相矛盾（upstream 自身缺陷，已绕开）**
   - 上游锁定 `aiohttp==3.14.3` 但未同步升级其子依赖，导致 pip 解析必然失败
   - 项目副本中做了 3 处最小修复：`aiohappyeyeballs 2.4.0→2.5.0`、`aiosignal 1.3.1→1.4.0`、`yarl 1.9.11→1.18.3`
   - upstream 克隆目录中的原始文件**未做任何修改**
2. **venv 内 pip/setuptools 保持创建时自带版本（pip 24.0 / setuptools 65.5.0）**
   - 原因：当前沙箱环境拦截 venv 内的文件删除操作（`SHFileOperationW 失败: 0x2`），pip 自升级需要删除旧文件，两次尝试均因此失败并曾导致 venv 损坏（已通过重建 venv 修复）
   - 影响：pip 24.0 对本项目依赖解析与安装完全够用；如后续需要升级，请在沙箱外终端执行 `python -m pip install --upgrade pip`
3. **upstream 测试 1 项失败为环境伪影**
   - `test_local_image_cannot_escape_repository_root`：测试断言本身通过，失败发生在 tearDown 阶段删除临时目录时被沙箱拦截
   - 在正常 Windows 终端中运行应全部通过；非代码问题、非依赖问题
4. **Git 身份未配置**：`user.name` / `user.email` 待用户提供后执行：
   ```bash
   git config --global user.name "<你的名字>"
   git config --global user.email "<你的邮箱>"
   ```
5. **LLM 相关 baseline 卡点**：所有调用云端 LLM 的教程均需在 `.env` 填入真实 `OPENAI_API_KEY` 后才能运行；无需 Key 的部分已全部验证通过

## 下一阶段建议（待确认后执行）

1. 配置 Git 身份（需用户提供 name/email）
2. 在 `.env` 填入 `OPENAI_API_KEY`，运行 1 个 upstream 教程 notebook 做端到端 LLM 验证
3. 在 `multi-agent-research-assistant` 目录初始化项目骨架（git init + 目录结构），再开始 Multi-Agent Research Assistant 设计
