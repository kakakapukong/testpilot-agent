# TestPilot Agent（B 方案，三 Agent）

TestPilot 是一个为小型 Python 项目做“测试驱动修复”的三 Agent 系统。你给它一个仓库、一句任务描述和**固定的受限 pytest 验证命令**：Repair Agent 负责浏览和修改代码，独立的 Reviewer Agent 只读检查修复，Memory Agent 只在成功审批后把可复用经验整理成结构化记忆。真实 CLI 只有在宿主 pytest 通过、Reviewer 通过、且用户查看安全摘要后明确批准，才会报告成功。每个完整操作都会保存本地安全检查点，模型请求失败或程序正常中断后可用同一 `run_id` 继续。

它不是“把一个大框架包起来”的玩具：三个 Agent 的提示词、独立上下文、不同工具权限、调度顺序、结构化审查反馈、一次返工上限、终止条件、验证门、持久化恢复、仓库级长期记忆和 JSONL 审计轨迹都在本仓库自己实现。它故意保持小：便于读懂、演示和答辩，也能真正完成“历史经验检索 -> 失败测试 -> 精确修复 -> 中断恢复 -> 独立验证 -> 只读审查 -> 人工决定 -> 经验沉淀”的完整闭环。

## 架构和成功条件

Repair Agent 的循环是：`模型 -> 工具调用 -> 本地执行 -> 结果回填 -> 模型`。它调用 `finish` 不是成功；宿主会运行模型无法改写的 `--verify` 命令，并把失败结果回填给 Repair Agent。默认测试目录、pytest 配置、显式选择的测试目标和审计文件都属于只读资产。

宿主 pytest 第一次通过后，Reviewer Agent 用全新的上下文检查当前代码。它只有 `list_files`、`read_file`、`search_text` 和 `submit_review` 四个工具，没有写文件、执行命令、联网或批准权限。它把仓库内容当作不可信数据，而且必须先成功完成至少一次列文件、读文件或搜索操作，宿主才会接受 `pass` 或 `request_changes`：

```text
Repair Agent 修改 -> 固定 pytest -> Reviewer Agent
Reviewer pass -> 人工审批 -> 成功 -> Memory Agent 总结并由宿主保存
Reviewer request_changes -> Repair Agent 最多返工一次 -> 固定 pytest -> 最终 Review
最终仍 request_changes -> 失败，不再循环，也不进入人工审批
```

第一次退回的有界反馈会作为 `finish` 工具结果交给 Repair Agent；在新的源码修改发生前，重复 `finish` 会被拒绝。这样既有真实的 Agent 分工，也不会让 Repair 与 Reviewer 无限争论或同时写文件。

真实 CLI 的完整成功路径严格按这个顺序执行：**写入 workspace（写前记录原始状态） -> 宿主运行固定 pytest -> Reviewer 通过 -> 只显示改动摘要 -> 一次人工决定**。摘要只有验证退出码，以及按路径排序的 `M/A "path" (+新增行/-删除行)`；路径用 JSON 字符串形式转义，换行、终端控制符和双向文字控制符不能伪造审批界面。行数采用线性的保守统计，不显示源码或 diff 正文。输入 `y` 或 `yes` 批准后保留改动并把它作为下次运行的新基线；拒绝或无法取得输入时，恢复本次运行前记录的文件字节和权限。批准或成功回滚后旧快照都会清空，因此同一个运行器可以安全开始下一次任务。若回滚操作自身失败，也不会报告成功，而是返回 `rollback_failed`。

Repair Agent 的七个工具是：`list_files`、`read_file`、`search_text`、`edit_file`、`write_file`、`run_command`、`finish`。Reviewer Agent 的四个工具是：`list_files`、`read_file`、`search_text`、`submit_review`。Memory Agent 只有一个终止工具 `submit_memory`。三个角色可以使用同一个配置的模型名，但真实 CLI 会创建三个彼此独立的模型客户端、三套上下文和三套不同权限的工具注册表。

真实 CLI 成功同时需要六件事：Repair Agent 显式 finish、至少一次成功的 `.py`/`.pyi` 源码修改、验证发生在最后一次修改之后、固定验证命令退出码为 0、Reviewer 最终通过、用户在一次性提示中明确批准。只改 README 之类的文件不能冒充修复成功。直接构造 `AgentRunner` 时 Reviewer、审批 workflow 和记忆依赖仍是可选边界；真实 CLI 始终启用三 Agent，离线 demo 使用确定性的 FakeModel 和模拟批准，不等待终端输入。

JSONL 轨迹记录 Repair 轮次、工具名、参数的类型/长度摘要、模型/工具/验证耗时、Reviewer 和 Memory 阶段、审批结果、检索命中数量、记忆 ID 与停止原因。Review 事件不保存任务、路径或反馈正文；Memory 事件只保存字段长度、数量、ID 和错误码等元数据。轨迹不保存工具参数原文、提示输入、源码、diff、API Key、Reviewer 反馈或记忆正文，并且不能被 Agent 的文件工具覆盖。

## 仓库级长期记忆

每个仓库拥有自己的本地记忆库：`workspace/.testpilot/memories/entries.jsonl`。新任务开始时，宿主用本地关键词相关度做确定性排序，只把正分的前 3 条结构化经验放进 Repair Agent 的初始上下文；Reviewer Agent 不接收这些历史信息，仍用全新上下文独立判断当前修复。

记忆不会在“模型声称完成”时写入。只有固定 pytest 退出码为 0、Reviewer 最终通过、用户明确批准三项证据同时成立，Memory Agent 才会收到有界的任务、完成说明、改动文件名和 Reviewer 反馈，并提交 `problem`、`root_cause`、`solution`、`verification`、`keywords` 五个字段。宿主随后再次校验、清理常见凭据、去重并尝试原子保存，同时附上运行 ID、测试退出码和审批状态等不可由模型伪造的证据；重复记录或辅助阶段失败会分别显示 `duplicate` 或稳定警告。源码文件、完整 diff 和完整对话不会作为记忆条目保存。

记忆生成或存储失败不会推翻已经通过的修复，而会在结果中显示稳定的 `memory_warning`。恢复任务不会重新检索，以免同一个运行中途改变提示；它继续使用检查点里原本保存的上下文。记忆目录和检查点一样对所有 Agent 文件工具不可见，并由 `.gitignore` 排除。真实 API 运行会把检索出的结构化记忆放入 Repair Agent 上下文并发送给所配置的模型端点，因此记忆库应只记录允许交给该端点处理的信息；Git 忽略规则只能降低误提交风险，提交前仍需主动检查。

## 安全节点与断点恢复

真实 CLI 首次调用模型前会生成 16 位十六进制 `run_id`，并把最近一个完整安全节点原子写入 `workspace/.testpilot/checkpoints/<run_id>.json`。新路径第一次写入前，宿主会先持久化 ChangeJournal 原始快照和上一完整事务的文件指纹；完整的 assistant/tool 事务结束后，才推进对话上下文、累计状态和当前文件指纹。因此，进程在多工具写入中途退出时，旧事务不会被半截改动“冒充”成可恢复状态，也不会出现只有工具调用、没有工具结果的上下文。检查点保存有界 Repair 上下文、累计轮数、重复调用保护、Reviewer 轮数与返工计数，以及 ChangeJournal 的写前快照；不保存 API Key、模型客户端或环境变量。

中断结果会显示 `resume_available=yes`。继续时只需要 workspace 和 `run_id`：

```powershell
python -m testpilot --workspace . --resume 0123456789abcdef
```

恢复入口先严格解析检查点，确认它属于当前 workspace，并重新计算所有已修改路径的存在状态、普通文件类型、权限和 SHA-256 指纹。任一文件被外部改动、删除、替换为链接或超过快照上限时，都会在创建模型客户端之前拒绝继续。校验通过后会重建同一个回滚基线、上下文和累计状态；旧的 pytest 通过、Reviewer 通过和人工批准证据会失效，所以恢复后必须重新 pytest、重新 Reviewer，再进入人工审批。Reviewer 已提出的返工要求及“一次返工”计数不会因重启清零。

恢复沿用原任务、固定 verifier、原 JSONL trace 和原本已经注入的历史经验；`--max-iterations` 若显式提供，只覆盖这一次调用的循环预算，累计轮数继续增长。用户批准后，宿主先把检查点标为 terminal 并清理敏感状态，再提交本次 ChangeJournal；拒绝或审批不可用时则先成功回滚，再完成终态清理。这样，进程不会在清空回滚基线后留下一个仍可恢复的活动检查点。检查点目录对三个 Agent 的所有文件工具均不可见，并已由 `.gitignore` 排除；它只适合留在本机，不应上传到公开仓库。

## 先看离线演示（不需要 API Key）

安装开发依赖后运行：

```powershell
pip install -e ".[dev]"
python -m testpilot.demo
```

它会完成两个逻辑任务。第一个任务创建错误的 `subtract` 并跑出真实失败的 pytest；Repair FakeModel 修改后模拟中断，程序从磁盘重建运行器，再重新 pytest、只读 Review 和模拟批准，随后由实际的 `MemoryAgent` 调度与存储代码配合脚本化 FakeModel，把固定结构化经验保存到本地。第二个全新任务创建相似的 `difference` 错误，先检索到刚保存的 1 条经验，再完成相同的验证闭环。依赖安装完成后，演示运行本身不需要 API Key、网络或终端输入，也不会打印记忆正文。预期输出严格为：

```text
BEFORE=FAIL
INTERRUPTED=CHECKPOINTED
RESUMED=SUCCESS
VERIFIED=PASS
REVIEWED=PASS
APPROVED=SIMULATED
AFTER=PASS
MEMORY_FIRST_SAVED=yes
MEMORY_SECOND_RETRIEVED=1
MEMORY_REUSED=yes
```

这里的 `MEMORY_REUSED=yes` 精确表示第二个 fresh run 检索并注入了 1 条经验，而且脚本化任务成功完成；它用于证明数据流和隔离边界，不把它解释成真实模型效果提升。

也可以留下演示仓库检查元数据：`python -m testpilot.demo --keep .\demo-workspace`。保留目录中可看到第一次中断与恢复共用一个 `run_id`、第二个任务使用独立 trace、成功终态的检查点均已删除，以及记忆文件恰好保留一条去重后的记录。录屏时只展示文件存在性、条目数量和终端摘要，不打开记忆正文。

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

首次安全保存后，CLI 会在模型调用前输出可恢复标识：

```text
run_id=0123456789abcdef
checkpoint=.testpilot/checkpoints/0123456789abcdef.json
```

如果本次因 API 暂时失败等原因停止且最终摘要显示 `resume_available=yes`，使用上面的 `run_id` 继续：

```powershell
python -m testpilot --workspace . --resume 0123456789abcdef
# 可选：只调整这一次恢复调用的模型循环预算
python -m testpilot --workspace . --resume 0123456789abcdef --max-iterations 6
```

固定验证通过后，CLI 会显示类似下面的安全摘要并等待一次决定：

```text
APPROVAL_REQUIRED
verification_exit=0
M "calculator.py" (+1/-1)
Accept verified changes? [y/N]:
```

只有 Reviewer 已通过时才会出现这个审批。只有 `y`/`yes`（忽略大小写和首尾空白）会批准；直接回车、其他文本、非字符串输入、EOF 或 `Ctrl+C` 都按拒绝处理并触发回滚。批准后文件保留；拒绝或无法批准时恢复原始状态。

最终紧凑结果还会显示：

```text
review=passed|changes_requested|unavailable|-
review_rounds=0|1|2
review_reworks=0|1
approval=approved|rejected|unavailable|-
run_id=0123456789abcdef|-
resume_available=yes|no
checkpoint_warning=checkpoint_cleanup_failed|-
memories_retrieved=0|1|2|3
memory_saved=yes|no|duplicate
memory_warning=memory_invalid|memory_load_failed|memory_save_failed|其他稳定错误码|-
```

其中 `-` 表示该运行没有相应警告或没有进入相应阶段。`duplicate` 表示等价经验已存在，因此没有重复追加。终端不会打印 Reviewer 反馈、Memory Agent 输出或记忆正文；只有 Repair Agent 的有界上下文能够收到第一次退回意见和本次新任务检索出的历史经验。

常用参数：新任务未指定时使用 12 轮 Repair Agent 预算；`--max-iterations N` 可设置本次新建或恢复调用的预算。Reviewer 有独立的 6 轮只读预算、一次返工机会，Memory Agent 最多 3 轮且只有结构化提交工具。新任务可用 `--trace .testpilot\traces\my-run.jsonl` 指定审计文件；恢复模式固定沿用原 trace，不接受新的 task、verify 或 trace。新 trace 必须是 workspace 内尚不存在的 `.jsonl` 文件，CLI 会先独占创建它，避免向已有用户文件追加内容；默认也会创建唯一的 `workspace/.testpilot/traces/run-*.jsonl`。最终终端只输出状态、停止原因、改动文件、验证退出码、Review 状态/轮数/返工数、审批状态、检查点与记忆元数据和 trace 路径，不输出模型正文或工具输出。`changed_files=` 使用 ASCII 安全的 JSON 数组，例如 `changed_files=["calculator.py"]`，文件名中的控制字符不会变成新的终端行。

`--verify` 只接受受限的 pytest 形式，例如 `python -m pytest -q` 或指定 workspace 内测试目标；允许常用的筛选、简洁度、回溯和耗时选项。它会拒绝工作区外目标、参数文件、插件加载、输出覆盖选项，以及 `python verify.py` 这类自定义脚本。pytest 自动加载的第三方插件也会关闭，以免环境悄悄改变验证含义。

## 没有使用哪些现成 Agent

本项目没有使用 LangChain、OpenAI Agents SDK、AutoGen、CrewAI、Claude Code、Codex 或 OpenCode，也没有使用 MCP、托管文件工具或托管代码执行。模型 API 只是“给出下一步工具调用”；Agent 的状态图式阶段、循环和验证门都在本地代码中。

当前代码已经独立实现**顺序式三 Agent 协作、一次性的本地终端审批、持久化断点恢复和仓库级长期记忆**：Repair Agent 能写，Reviewer Agent 只能读，Memory Agent 只能提交结构化经验，宿主负责确定性调度、成功证据和持久化；安全检查点恢复后仍会重新经过 pytest、Reviewer 与人工审批门。

## 安全边界（诚实说明）

已做的边界包括：路径限制在 workspace、原子写入、写前变更记录、单文件 1,000,000 字节的回滚快照上限、受保护的 tests/pytest 配置/显式验证目标/trace、Agent 不可见的检查点与记忆目录、按实际目录项计数的扫描上限、读取/搜索/命令输出上限、纯文本搜索、命令超时、固定 pytest verifier，以及从子进程、trace 和记忆中清理 `API_KEY`、`*_KEY`、token、secret、password、credential 等常见敏感值。检查点采用严格版本化 JSON、16 MB 总大小上限；记忆采用严格 JSONL、单条与总量上限、最多 200 条和重复指纹；两者都使用临时文件 + `fsync` + 原子替换。恢复前会校验 workspace 身份、生命周期和 journal 路径对应文件的完整指纹。pytest 参数和环境注入也会检查；模型不能把测试指到工作区外，也不能自行改验证资产。超过快照上限的文件会在写入前被拒绝，避免为了审批把任意大的旧文件读进内存。当读取因字符预算提前停止时，结果会明确标记截断，并把无法确定的总行数设为未知，而不会为了统计行数继续扫描整个大文件。

Reviewer 的“通过”不会扩大任何权限：它的注册表从结构上没有修改或执行工具，也不能跳过固定 pytest。人工批准同样不能扩大权限；危险操作和受保护路径仍会在写入前直接拒绝。审批界面只显示内容无关的行数摘要；拒绝和输入不可用都会精确恢复 journal 记录的运行前状态。若 Reviewer 无法运行、返回无效结果，或第二轮仍发现阻塞问题，系统会失败关闭且不进入审批。若外部进程同时改写同一路径、路径结构发生变化或底层文件操作失败，系统也会失败关闭，并可能报告 `rollback_failed`，不会假装已经安全恢复。

但这**不是 OS 沙箱**。pytest 与仓库代码本身仍会执行本地代码；它不抵抗同机恶意进程或路径竞态。对于不可信仓库，应在容器或虚拟机中运行，而不是直接在你的电脑上运行。
