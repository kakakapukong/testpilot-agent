项目名称：TestPilot Agent

公开仓库：https://github.com/kakakapukong/testpilot-agent

运行环境：Python 3.11+

离线演示（不需要 API Key）：
1. pip install -e ".[dev]"
2. python -m testpilot.demo
预期输出：BEFORE=FAIL、INTERRUPTED=CHECKPOINTED、RESUMED=SUCCESS、VERIFIED=PASS、REVIEWED=PASS、APPROVED=SIMULATED、AFTER=PASS、MEMORY_FIRST_SAVED=yes、MEMORY_SECOND_RETRIEVED=1、MEMORY_REUSED=yes。

真实模型运行：
1. pip install -e ".[api,dev]"
2. 设置 OPENAI_API_KEY 与 OPENAI_MODEL 环境变量；使用 DeepSeek 等兼容接口时再设置 OPENAI_BASE_URL。
3. python -m testpilot --workspace <项目目录> --verify "python -m pytest -q" "修复任务描述"
4. 若结果显示 resume_available=yes：python -m testpilot --workspace <项目目录> --resume <run_id>

主要功能：独立实现 Repair/Reviewer/Memory 三 Agent、模型工具调用循环、有界且隔离的上下文、七个 Repair 本地工具、只读 Reviewer、结构化 Memory 提交、工作区路径限制、精确/原子编辑、受限 pytest、固定独立验证门、一次人工审批、JSONL 审计轨迹、带回滚日志和文件指纹校验的断点恢复，以及仓库级长期记忆。记忆只在 pytest、Reviewer 和人工审批全部通过后写入；新任务最多检索 3 条，恢复任务沿用原上下文而不重新检索。最终结果会显示 memories_retrieved、memory_saved 和 memory_warning。

本地数据提醒：.testpilot/checkpoints 可能包含任务、模型上下文和源码写前快照，.testpilot/memories 可能包含结构化任务经验与文件名，.testpilot/traces 保存运行元数据；它们仅用于本机，已被 .gitignore 排除，不要上传到公开仓库、展示正文或打包进提交材料。真实 API 运行会把检索出的结构化记忆发送给所配置的模型端点。

合规说明：未使用 LangChain、OpenAI Agents SDK、AutoGen、CrewAI、Claude Code、Codex、OpenCode、MCP、托管文件工具或托管代码执行。API Key 仅从环境变量读取，不写入仓库。
