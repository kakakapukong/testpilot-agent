# TestPilot Agent（B 方案）

TestPilot 是一个为小型 Python 项目做“测试驱动修复”的单 Agent。你给它一个仓库、一句任务描述和**固定的受限 pytest 验证命令**；它可以读文件、搜索、修改、运行受限的 pytest 命令。只有最后由宿主重新运行那条固定命令并且通过，它才会报告成功。

它不是“把一个大框架包起来”的玩具：工具协议、上下文、循环、错误回填、终止条件、路径约束、验证门和 JSONL 审计轨迹都在本仓库自己实现。它故意保持小：便于读懂、演示和答辩，也能真正完成“失败测试 -> 精确修复 -> 独立验证”的完整闭环。

## 架构和成功条件

循环是：`模型 -> 工具调用 -> 本地执行 -> 结果回填 -> 模型`。模型调用 `finish` 不是成功；宿主会运行模型无法改写的 `--verify` 命令，并把失败结果回填给模型。默认测试目录、pytest 配置、显式选择的测试目标和审计文件都属于只读资产。

七个工具是：`list_files`、`read_file`、`search_text`、`edit_file`、`write_file`、`run_command`、`finish`。

成功同时需要四件事：模型显式 finish、至少一次成功的 `.py`/`.pyi` 源码修改、验证发生在最后一次修改之后、固定验证命令退出码为 0。只改 README 之类的文件不能冒充修复成功。

JSONL 轨迹记录轮次、工具名、参数的类型/长度摘要、模型/工具/验证耗时、退出码和停止原因；它不保存工具参数原文，避免把待修改源码抄入审计文件，并且不能被 Agent 的文件工具覆盖。

## 先看离线演示（不需要 API Key）

安装开发依赖后运行：

```powershell
pip install -e ".[dev]"
python -m testpilot.demo
```

它会在临时目录创建一个错误的 `subtract`，真实运行失败的 pytest，再让内置 FakeModel 读取、精确修改并申请 finish。预期输出为：

```text
BEFORE=FAIL
AGENT=SUCCESS
AFTER=PASS
```

也可以留下演示仓库检查轨迹：`python -m testpilot.demo --keep .\demo-workspace`。

## 使用真实模型

TestPilot 使用普通的 Chat Completions function tools（`tool_choice="auto"`），并不绑定某一个“最新模型”。请自行填写一个支持 function calling 的模型名；API Key 只从环境变量读取。接口形式见 [OpenAI Chat Completions API 文档](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions)。

真实模型还需要安装可选的 API 依赖：

```powershell
pip install -e ".[api,dev]"
```

PowerShell 中可先复制模板，再在当前窗口设置变量（不要把真实 key 写进仓库）：

```powershell
$env:OPENAI_API_KEY = ""
$env:OPENAI_MODEL = ""
# 可选：兼容服务端点
$env:OPENAI_BASE_URL = ""
```

然后运行，例如：

```powershell
python -m testpilot --workspace . --verify "python -m pytest -q" "修复失败的计算逻辑，但不要修改 tests"
```

常用参数：`--max-iterations 12` 限制循环轮数；`--trace .testpilot\traces\my-run.jsonl` 指定审计文件。trace 必须是 workspace 内尚不存在的新 `.jsonl` 文件，CLI 会先独占创建它，避免向已有用户文件追加内容；默认也会创建唯一的 `workspace/.testpilot/traces/run-*.jsonl`。最终终端只输出状态、停止原因、改动文件、验证退出码和 trace 路径，不输出模型正文或工具输出。

`--verify` 只接受受限的 pytest 形式，例如 `python -m pytest -q` 或指定 workspace 内测试目标；允许常用的筛选、简洁度、回溯和耗时选项。它会拒绝工作区外目标、参数文件、插件加载、输出覆盖选项，以及 `python verify.py` 这类自定义脚本。pytest 自动加载的第三方插件也会关闭，以免环境悄悄改变验证含义。

## 没有使用哪些现成 Agent

本项目没有使用 LangChain、OpenAI Agents SDK、AutoGen、CrewAI、Claude Code、Codex 或 OpenCode，也没有使用 MCP、托管文件工具或托管代码执行。模型 API 只是“给出下一步工具调用”；Agent 的状态图式阶段、循环和验证门都在本地代码中。

当前只做单 Agent。它没有 LangGraph 式持久化/断点恢复或人工审批，也没有多 Agent、插件、长期记忆、MCP 和 Web UI；这些是有意识地留在后续扩展，而不是把功能藏在框架里。

## 安全边界（诚实说明）

已做的边界包括：路径限制在 workspace、原子写入、受保护的 tests/pytest 配置/显式验证目标/trace、按实际目录项计数的扫描上限、读取/搜索/命令输出上限、纯文本搜索、命令超时、固定 pytest verifier，以及从子进程和 trace 中过滤 `API_KEY`、`*_KEY`、token、secret、password、credential 等常见敏感环境变量。pytest 参数和环境注入也会检查；模型不能把测试指到工作区外，也不能自行改验证资产。当读取因字符预算提前停止时，结果会明确标记截断，并把无法确定的总行数设为未知，而不会为了统计行数继续扫描整个大文件。

但这**不是 OS 沙箱**。pytest 与仓库代码本身仍会执行本地代码；它不抵抗同机恶意进程或路径竞态。对于不可信仓库，应在容器或虚拟机中运行，而不是直接在你的电脑上运行。
