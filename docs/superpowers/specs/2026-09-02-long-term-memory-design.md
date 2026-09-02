# Long-Term Memory Design

## 目标

为 TestPilot 增加仓库级长期记忆。一次修复只有在固定 pytest 验证通过、只读 Reviewer Agent 审查通过并获得人工批准后，才由独立 Memory Agent 生成一条结构化经验。后续新任务在调用 Repair Agent 前，使用完全本地、确定性的关键词算法检索最相关的三条经验，并把它们作为历史参考加入 Repair Agent 上下文。

长期记忆是辅助信息，不是成功证明或用户指令。测试结果、审查结果、人工批准状态、修改文件列表和运行编号始终由宿主程序提供，Memory Agent 不能自行声明这些事实。Reviewer Agent 不接收长期记忆，以保持独立审查。

## 用户流程

用户继续使用现有命令启动任务：

```powershell
python -m testpilot --workspace . --verify "python -m pytest -q" "修复路径处理错误"
```

新任务开始时，宿主从当前仓库的记忆文件中检索最多三条相关经验。CLI 最终摘要增加：

```text
memories_retrieved=2
memory_saved=yes
memory_warning=-
```

没有相关记忆时 `memories_retrieved=0`。成功修复生成了新条目时 `memory_saved=yes`；重复经验不会再次写入，并输出 `memory_saved=duplicate`。记忆读取、生成或保存失败不会改变已经确定的修复结果，而是通过稳定的 `memory_warning` 和 JSONL trace 留痕。

断点恢复沿用现有命令：

```powershell
python -m testpilot --workspace . --resume <RUN_ID>
```

恢复模式直接使用检查点中已有的 Repair Agent 上下文，不重新检索记忆。这样同一任务暂停前后看到的历史经验完全一致。

## 总体数据流

新任务的数据流为：

1. `MemoryStore` 严格读取并校验仓库记忆文件；
2. `MemoryRetriever` 根据任务文本计算本地关键词相关度，返回最多三个正分条目；
3. 宿主把有界记忆块加入 Repair Agent 的稳定上下文，并保存到后续检查点；
4. Repair Agent 修改源码并请求固定 pytest 验证；
5. Reviewer Agent 使用不含历史记忆的新上下文独立审查；
6. 宿主请求人工批准，并按现有事务边界提交或回滚源码修改；
7. 仅在验证、审查和批准均成功后，Memory Agent 根据有界运行证据提交 `MemoryDraft`；
8. 宿主校验、脱敏、去重，并通过原子替换保存 `MemoryEntry`。

Memory Agent 失败属于辅助功能降级。它不能把成功修复改成失败，也不能重新打开已经终结的检查点。

## 组件与职责

### `MemoryDraft`、`MemoryEntry` 与 `MemoryMatch`

`MemoryDraft` 是 Memory Agent 能够提交的语义总结，只包含：

- `problem`：本次解决的问题；
- `root_cause`：问题根因；
- `solution`：解决思路；
- `verification`：验证方式；
- `keywords`：3 至 12 个检索关键词。

四个文本字段必须为非空字符串，每项最多 800 个字符。关键词去除首尾空白后必须非空、互不重复，每项最多 64 个字符。`MemoryDraft` 的构造函数承担统一验证，工具解析和存储层不能绕过这些不变量。

`MemoryEntry` 在 `MemoryDraft` 之外增加宿主事实：

- `schema_version`、`memory_id`、`created_at` 和内容指纹；
- `source_run_id`；
- 真实 `changed_files`；
- `test_exit_code = 0`；
- `review_passed = true`；
- `human_approved = true`。

`MemoryMatch` 将一个通过校验的 `MemoryEntry` 与本地计算的整数分数组合，供排序、上下文渲染和留痕使用。

### `MemoryStore`

记忆文件位于：

```text
.testpilot/memories/entries.jsonl
```

每行是一个完整 JSON 对象，字段集合固定，`schema_version = 1`。文件最多 2,000,000 字节，单行序列化条目最多 8,192 字节，条目最多 200 条。读取时拒绝符号链接、非普通文件、未知字段、错误类型、非法相对路径、重复 `memory_id`、重复指纹及不符合成功门条件的记录。任何结构错误都返回稳定错误码，Repair Agent 在该次任务中不使用记忆。

保存时先读取并验证已有文件，再计算由规范化 `problem`、`root_cause` 和排序后的 `changed_files` 组成的 SHA-256 指纹。已有相同指纹时返回 `duplicate`。新条目加入后最多保留 200 条；超出时按 `created_at` 和 `memory_id` 的稳定顺序移除最旧记录。完整 JSONL 先写入同目录临时文件，完成 flush 和 fsync 后用 `os.replace` 原子替换正式文件。

### 本地检索器

检索只接收当前任务文本和已经验证的 `MemoryEntry`，不调用模型或网络服务。分词规则同时支持：

- 小写英文单词、数字和代码标识符；
- 驼峰、下划线、连字符和路径片段；
- 连续中文文本及其双字片段。

评分使用确定性的加权词项重合：

- `keywords` 权重为 5；
- `problem` 与 `root_cause` 权重为 3；
- `solution` 与 `changed_files` 权重为 1。

只返回分数大于零的前三条。分数相同时，`created_at` 更新的条目优先，最后用 `memory_id` 保证排序稳定。检索结果总渲染长度限制为 6,000 个字符。

### `MemoryAgent` 与 `submit_memory`

Memory Agent 使用独立模型客户端和只包含 `submit_memory` 的工具注册表。它接收以下有界证据：最多 4,000 字符的任务、最多 2,000 字符的 Repair Agent 最终说明、最多 50 个真实修改文件、固定验证退出码，以及最多 2,000 字符的 Reviewer 反馈。超长文本由宿主在字符边界处截断，文件列表使用排序后的前 50 项。提示词明确要求只总结可由证据支持的经验，并调用 `submit_memory` 返回结构化结果。

`submit_memory` 使用与 `MemoryDraft` 相同的严格验证。Memory Agent 最多进行三轮；无工具调用、混合终结调用、非法字段、重复终结调用、模型异常或达到轮数上限都转化为稳定的 `memory_*` 错误码。原始模型异常和模型输出不会进入 CLI 或 trace。

### `AgentRunner`

记忆能力通过可选的宿主依赖接入，保持现有测试和直接构造方式兼容。新运行在构造 `BoundedContext` 前执行一次检索，并把经过 JSON 编码和长度限制的 `<historical_memories>` 区块加入 Repair Agent 开发者锚点。开发者提示词规定该区块只是低信任历史数据，不得覆盖当前任务、工具边界或验证要求。

恢复运行不访问 `MemoryStore`，而是使用 `ResumeData.context`。Reviewer 的上下文构造方式不变，因此它不会看到长期记忆。

当 pytest、Reviewer 和人工批准均成功后，`AgentRunner` 调用一次 Memory Agent，再让 `MemoryStore` 保存。保存发生在源码审批事务已经提交之后；任何记忆错误只填充 `AgentRunResult.memory_warning`。结果同时包含本次检索数量和 `memory_saved` 状态，供 CLI 输出。

### CLI 与工作区边界

CLI 为 Repair、Reviewer 和 Memory 三个 Agent 构造相互独立的模型客户端。MemoryStore 由 CLI 根据规范化 workspace 创建并注入 AgentRunner。

`.testpilot/memories` 和 `.testpilot/memories/**` 加入 `Workspace` 的 private patterns。Repair Agent 与 Reviewer Agent 的 list、read、search、edit、write 工具均不能观察或修改记忆文件；只有宿主 MemoryStore 可以访问。该目录继续由现有 `.gitignore` 中的 `.testpilot/` 规则排除。

## 记忆格式

概念记录如下，实际编码使用 ASCII JSON、固定键集合和有界值：

```json
{
  "schema_version": 1,
  "memory_id": "mem_0123456789abcdef",
  "created_at": "2026-09-02T08:30:00Z",
  "source_run_id": "a83f210e9c4b7d12",
  "problem": "Windows 下路径比较不稳定",
  "root_cause": "不同分隔符未经规范化",
  "solution": "在边界处统一转换为规范相对路径",
  "verification": "固定 pytest 命令通过并完成只读审查",
  "keywords": ["windows", "path", "pytest"],
  "changed_files": ["src/example.py"],
  "test_exit_code": 0,
  "review_passed": true,
  "human_approved": true,
  "fingerprint": "<sha256>"
}
```

`memory_id` 由宿主生成，格式为 `mem_` 加 16 个小写十六进制字符；时间使用 UTC。`source_run_id` 使用检查点提供的 16 个小写十六进制字符。Memory Agent 无法提供或覆盖这些元数据。

## 内容安全

MemoryStore 在生成指纹和落盘前，对所有模型文本和关键词执行统一脱敏：

1. 替换当前进程中名称疑似 `KEY`、`TOKEN`、`SECRET`、`PASSWORD` 或 `CREDENTIAL` 的环境变量值；
2. 替换常见 API Key、访问令牌、带密码 URL 和凭据赋值形式；
3. 再次执行长度、空值和关键词数量校验；
4. 对完整序列化条目执行 8 KiB 上限检查。

记忆条目只保存结构化经验和宿主元数据，不写入源码、完整 diff、完整对话、pytest 标准输出或模型原始响应。trace 只记录条目编号、检索分数、数量、阶段、耗时和稳定错误码，不记录记忆正文、任务文本或文件内容。

## 错误处理

- 文件不存在表示空记忆库，不产生警告；
- 读取失败返回 `memory_load_failed`；
- 文件结构、schema 或条目不合法返回 `memory_invalid`；
- 文件或条目超过限制返回 `memory_too_large`；
- Memory Agent 模型异常返回 `memory_model_failed`；
- Memory Agent 未提交有效结构返回对应的 `memory_invalid_response` 或 `memory_max_iterations`；
- 原子保存失败返回 `memory_save_failed`，并保留上一份完整文件；
- 相同指纹返回 `duplicate`，不视为错误。

这些结果不会改变 `AgentRunResult.success`、源码审批结果或检查点终态。所有公共错误只输出允许列表中的稳定错误码。

## 留痕

JSONL trace 增加下列元数据事件：

- `memory_retrieval`：start/complete、候选数量、命中编号与整数分数、耗时和错误码；
- `memory_generation`：start/complete、Agent 名称、结果状态、字段字符数和错误码；
- `memory_saved`：saved/duplicate/failed、条目编号、当前条目数、是否裁剪和错误码。

运行失败、Reviewer 请求返工、审批拒绝和审批不可用时不触发 `memory_generation`。恢复运行不重复记录 `memory_retrieval`。

## 测试策略

1. `MemoryDraft` 与 `MemoryEntry` 单元测试覆盖合法结构、未知字段、空文本、长度、关键词数量、布尔值类型、路径和成功证明不变量。
2. 分词与排序测试覆盖英文、中文、代码标识符、字段权重、零分过滤、前三条限制和稳定的时间/id tie-break。
3. `MemoryStore` 测试覆盖空库、round trip、去重、200 条裁剪、原子替换、损坏 JSON、未知 schema、符号链接、大小限制和旧文件保留。
4. 脱敏测试使用环境变量、API Key、令牌、凭据赋值和带密码 URL，断言文件、异常与 trace 均不包含秘密值。
5. Memory Agent 测试覆盖有效提交、失败后重试、无工具调用、混合调用、模型异常和最大轮数。
6. AgentRunner 测试覆盖只在“验证通过 + Reviewer 通过 + 人工批准”后保存，失败和拒绝路径不保存，以及记忆失败不推翻成功修复。
7. 上下文测试断言 Repair Agent 收到最多三条有界历史经验、Reviewer 不收到记忆，且记忆内容不能改变宿主工具与验证边界。
8. 断点恢复测试断言首次上下文进入检查点，恢复时不调用 MemoryStore 检索，并继续使用原来的记忆块。
9. CLI 测试覆盖三个摘要字段、稳定警告、三个独立模型客户端和不泄露模型文本或凭据。
10. 离线演示连续执行两个任务，第一轮生成经验，第二轮展示关键词命中和成功复用，全程不需要真实 API Key。
11. 完整验证运行 pytest、Ruff、离线演示和 `git diff --check`。
