# Checkpoint Resume Design

## 目标

为 TestPilot 增加基于安全节点的断点恢复。一次任务获得稳定的 `run_id`，运行时持续保存足以恢复 Repair Agent 的状态。进程退出或模型请求失败后，用户可以在同一工作区通过 `--resume RUN_ID` 继续任务；恢复前必须验证检查点结构、工作区身份和已修改文件的当前状态。

恢复后的流程仍然遵守现有安全门：固定 pytest 验证器、只读 Reviewer Agent 和最终人工审批。旧的验证、审查和审批结果只用于说明历史进度，恢复后必须针对当前文件重新生成证明。

## 用户流程

创建新任务：

```powershell
python -m testpilot --workspace . --verify "python -m pytest -q" "修复这个项目"
```

初始检查点创建成功后，CLI 在第一次模型调用前立即输出：

```text
run_id=a83f210e9c4b7d12
checkpoint=.testpilot/checkpoints/a83f210e9c4b7d12.json
```

最终运行摘要再输出：

```text
resume_available=yes
```

继续已有任务：

```powershell
python -m testpilot --workspace . --resume a83f210e9c4b7d12
```

恢复模式从检查点读取原始任务、固定验证命令和剩余状态。模型连接仍由当前进程的环境配置提供。恢复成功后，CLI 沿用原 `run_id` 和 JSONL trace，并记录一次不含源码内容的 `resume` 事件。

新任务模式要求 `--verify` 和任务文本；恢复模式要求 `--resume`。两种模式都要求 `--workspace`，并在 CLI 配置边界完成互斥校验。`--max-iterations` 表示本次进程允许执行的模型轮数；恢复时默认沿用检查点中的值，也允许用户为本次继续执行提供新的正整数预算。`RunState.iteration` 始终保存整个任务的累计轮数。

## 安全节点

检查点只代表已经完成且内部一致的状态。保存时机为：

1. 第一次模型调用之前；
2. 一个完整的 assistant/tool transaction 加入 `BoundedContext` 之后；
3. 固定验证器和 Reviewer 的结果已经转化为完整 transaction 之后；
4. 可恢复的停止结果返回给 CLI 之前。

同一 assistant turn 中的所有工具调用属于一个 transaction。保存操作发生在整组工具结果齐备之后，确保恢复的消息序列不会出现孤立的 tool message。每次保存都同时记录已修改文件的指纹，使检查点状态和磁盘状态能够配对验证。

## 组件与职责

### `CheckpointStore`

负责检查点路径、严格 JSON 编解码、大小限制和原子替换。它只接受格式固定的随机 `run_id`，并将文件保存在工作区内的：

```text
.testpilot/checkpoints/<run_id>.json
```

写入时先在同一目录生成临时文件，完成 flush 和 fsync 后使用 `os.replace` 替换正式文件。读取时限制文件大小，拒绝符号链接、未知 schema、错误类型、非法相对路径和非有限数字。公开异常只使用稳定错误码，不回显检查点内容。

### `RunCheckpoint`

使用显式 `schema_version = 1`，保存以下白名单字段：

- `run_id`、规范化工作区身份和创建/更新时间；
- 原始任务、规范化验证命令和 trace 相对路径；
- Repair Agent 的 developer/user 锚点和有界 transaction groups；
- `RunState` 的稳定字段；
- 重复调用保护所需的上一批调用签名；
- `ChangeJournal` 的原始快照；
- 每个已记录文件保存时的当前指纹。

模型密钥、访问令牌、环境变量和模型客户端对象不属于序列化字段。加载后的对象由独立的验证函数重建，业务代码不直接信任原始字典。

### `BoundedContext`

增加受校验的导出和恢复接口。恢复接口重新执行现有角色、tool-call id、完整 transaction、JSON 原生类型和内容长度约束，不能绕过 `append_transaction` 已有的不变量。检查点仍只保留配置允许的最近 transaction groups。

### `ChangeJournal`

增加持久化快照接口，记录每个路径在本次任务第一次写入前的：

- 工作区相对路径；
- 原始字节或“运行开始时不存在”的状态；
- 原始文件权限；
- 本次任务创建的缺失父目录。

二进制内容使用 base64 写入 JSON。恢复时重新执行路径归一化和工作区边界检查。这样，即使进程已经重启，人工拒绝仍能调用现有原子 rollback，将工作区恢复到任务开始前。

### `Workspace` 私有边界

`.testpilot/checkpoints/**` 作为 host-private 路径加入工作区边界。Repair Agent 和 Reviewer Agent 的 list、read、search、edit、write 工具都不能观察或修改该目录。`.testpilot/` 继续由 `.gitignore` 排除，检查点由宿主程序直接管理。

### `AgentRunner`

检查点能力作为可选宿主依赖接入，保持现有直接构造 `AgentRunner` 的兼容性。CLI 始终注入检查点会话。新运行创建全新状态；恢复运行注入已经验证并重建的 context、state、journal 和重复调用状态，然后从下一个 iteration 继续。

AgentRunner 在安全节点通知检查点会话保存。检查点失败转化为稳定的停止原因，不能被当作 Agent 工具结果或成功状态忽略。

## 检查点结构

概念结构如下；实际编码只接受列出的字段和受限值：

```json
{
  "schema_version": 1,
  "run_id": "a83f210e9c4b7d12",
  "workspace": {"identity": "..."},
  "request": {
    "task": "...",
    "verifier": ["python", "-m", "pytest", "-q"],
    "max_iterations": 12,
    "trace_path": ".testpilot/traces/run-....jsonl"
  },
  "runtime": {
    "context": {"developer": {}, "user": {}, "groups": []},
    "state": {},
    "last_call_signature": null
  },
  "journal": {"snapshots": []},
  "fingerprints": [],
  "lifecycle": {"status": "active", "safe_point": 3},
  "created_at": "...",
  "updated_at": "..."
}
```

文件指纹由相对路径、存在状态、普通文件类型、权限和 SHA-256 组成。路径集合来自 ChangeJournal，而不是检查点提供的任意扫描范围。

## 恢复校验与状态转换

CLI 按以下顺序恢复：

1. 校验 `run_id`，并安全定位检查点文件；
2. 限量读取并严格解析 schema，确认 lifecycle 仍为 `active`；
3. 确认检查点绑定到当前规范化工作区；
4. 重新规范化验证命令，确认它仍是受限制的 workspace-confined pytest 命令；
5. 恢复 ChangeJournal，并重新计算所有已记录路径的当前指纹；
6. 指纹全部一致后，恢复 Repair Agent context 和 RunState；
7. 清除可直接通向成功的临时证明，再进入下一次模型循环。

第 7 步将 `verified_after_last_edit` 设为 false，并清除当前 `approval_status`。Reviewer 的返工计数继续保留，因此一次恢复不能重置“最多一次返工”的限制。下一次 `finish` 必须重新运行固定 pytest；pytest 通过后重新启动一个新的只读 Reviewer 上下文，之后才进入人工审批。

如果结构、工作区或文件指纹校验失败，CLI 在调用模型之前停止，输出稳定的 `checkpoint_*` 原因，并保留检查点与工作区供用户检查。

## 生命周期与留痕

初始检查点在首次模型调用前创建。每次后续保存覆盖同一个 `run_id` 文件，因此它始终表示最近一个完整安全节点。模型/API 异常、进程正常中止或其他可恢复停止会保留该文件，并在结果中输出 `resume_available=yes`。

人工批准成功后，ChangeJournal 提交；人工拒绝或审批不可用时，先完成 rollback。宿主随后将检查点原子更新为 terminal 状态，再删除敏感检查点，并由 trace 记录终态。恢复入口只接受 `active`，因此即使终态文件清理失败也不能再次执行。rollback 未完成时保留 active 检查点和当前工作区供用户检查。删除失败只输出内容无关的清理告警，不能改变已经确定的业务结果。

trace 继续遵守现有元数据策略。新增事件只包含 `run_id`、schema 版本、安全节点序号、恢复成功与否、稳定错误码和耗时；任务文本、消息、源码、文件内容、完整路径及检查点载荷不会进入 trace。

## 错误处理

- 原子写入失败时保留上一份完整检查点，并以 `checkpoint_save_failed` 停止本次自动运行。
- JSON 损坏或 schema 不兼容返回 `checkpoint_invalid`。
- 工作区身份不匹配返回 `checkpoint_workspace_mismatch`。
- 已记录文件发生变化返回 `checkpoint_workspace_changed`。
- journal、context 或 state 无法通过不变量校验时返回 `checkpoint_invalid`。
- 恢复后的模型、验证、审查和审批错误继续使用各自现有的稳定错误码。

所有检查点错误都在 CLI 公共边界转换成不含用户输入、路径、源码和 SDK 异常文本的输出。

## 测试策略

1. `CheckpointStore` 单元测试覆盖 round trip、原子替换、大小上限、损坏 JSON、未知 schema、符号链接、路径穿越和内容清理。
2. `BoundedContext` 测试覆盖 transaction 恢复、裁剪边界、非法角色、缺失 tool result、重复 tool-call id 和防御性复制。
3. `ChangeJournal` 测试覆盖修改文件、新建文件、二进制内容、权限、缺失父目录，以及重启后的精确 rollback。
4. 指纹测试覆盖内容、存在状态、文件类型和权限变化，并证明不一致发生在模型调用之前。
5. AgentRunner 测试覆盖安全节点保存、模型失败后恢复、累计 iteration、重复调用保护和 Reviewer 返工计数延续。
6. 流程测试覆盖“编辑 → API 失败 → 新进程恢复 → pytest → Reviewer → 人工批准”，并断言恢复后旧验证和旧审查不能直接通向批准。
7. CLI 测试覆盖新任务/恢复模式的参数互斥、稳定摘要、原 trace 续写、终态清理和不泄露配置。
8. 离线演示增加可重复的中断与恢复场景，在没有 API Key 的情况下展示 `INTERRUPTED -> RESUMED -> VERIFIED -> REVIEWED -> APPROVED`。
9. 完整验证运行 pytest、Ruff、离线演示和 `git diff --check`。
