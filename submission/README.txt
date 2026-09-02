TestPilot Agent 提交说明与网页使用指南
========================================

项目名称：TestPilot Agent
公开仓库：https://github.com/kakakapukong/testpilot-agent
运行环境：Python 3.11+

一、使用 TestPilot 网页
----------------------

1. 安装

在项目根目录打开 PowerShell，执行：

    pip install -e ".[api,dev]"

其中 api 用于连接真实模型，dev 包含 pytest 和代码检查工具。

2. 配置模型

推荐把模型配置写在用户目录，而不是写进仓库。创建：

    $env:USERPROFILE\.testpilot\web.env

例如使用 DeepSeek 兼容接口时，文件内容为：

    OPENAI_API_KEY=填写自己的API-Key
    OPENAI_MODEL=deepseek-chat
    OPENAI_BASE_URL=https://api.deepseek.com

也可以设置同名环境变量；环境变量优先于 web.env。不要把真实 API Key 提交到 Git。

3. 启动网页

在项目根目录执行：

    python -m testpilot.web

程序会启动只监听本机的服务并自动打开网页。若浏览器没有自动打开，请访问：

    http://127.0.0.1:8765/

页面右上角出现“模型名 · 凭据已加载”，表示配置已读取。停止服务时，回到启动它的 PowerShell 窗口按 Ctrl+C。

4. 发起修复任务

网页只需要填写两项：

    工作目录：待修复 Python 项目的绝对路径，该目录必须已经存在。
    任务：用一句明确的话描述错误、期望行为和限制。

任务示例：

    修改 calculator.py 中的 subtract，使其做减法而不是加法；不要修改 tests。

点击“运行”后，网页会按实际顺序展示历史经验检索、Repair Agent 的模型轮次与工具、宿主 pytest、Reviewer Agent、人工审批和记忆保存。当前步骤展开显示，已完成步骤自动折叠，可以重新点开查看。网页一次只接受一个任务，避免两次运行同时修改同一工作目录。

网页没有“验证命令”输入框。验证命令由后端固定为当前可信 Python 解释器以安全路径模式执行 pytest，网页请求也不能覆盖它；工作区中的同名模块不能冒充 pytest，每次验证也使用全新的字节码缓存。因此模型不能把验证替换成 echo、跳过测试或只口头声称成功。

5. 人工审批

只有 Repair Agent 确实修改过 Python 源码、最后一次修改后的宿主 pytest 退出码为 0、独立 Reviewer 给出 pass 后，页面才会出现审批卡片。

    批准：保留修改，任务可以报告 SUCCESS，随后尝试保存成功经验。
    拒绝并回滚：恢复运行前记录的文件内容和权限，本次任务报告失败。

审批区只展示验证结果、文件名和新增/删除行数，不展示源码、完整 diff、模型回复或 API Key。若本地服务断开，页面会提示连接失败并恢复运行按钮；审批请求失败时审批卡片会保留，避免把“没有提交成功”误当成已经批准。

6. 判断结果

页面底部状态栏显示 STATUS、stop_reason、review、approval、run_id 和 memory_saved 等结果。SUCCESS 不是模型自己输出的文字，而是宿主综合 pytest、Reviewer 和人工批准后生成的状态。若显示 FAILED，应先查看 stop_reason 和最后一个展开步骤，再检查任务描述、测试和模型配置。


二、Agent 的特色功能
-------------------

1. 自主实现的三 Agent 协作

TestPilot 自主实现 Repair Agent、Reviewer Agent 和 Memory Agent 的提示词、上下文、工具注册表、循环、状态转换与终止条件。三个 Agent 可以使用同一个模型配置，但运行时拥有三个独立客户端、三份上下文和三套不同权限。本项目没有用 LangChain、OpenAI Agents SDK、AutoGen 或 CrewAI 完成调度，也没有把 Claude Code、Codex、OpenCode 当作内部执行器。

三者按确定顺序协作：

    仓库记忆检索
          ↓
    Repair Agent 读取并修改源码
          ↓
    宿主运行固定 pytest
          ↓
    Reviewer Agent 只读审查
          ↓
    用户批准或拒绝
          ↓
    Memory Agent 总结成功经验

这种设计不追求“Agent 数量多”，而是让每个角色都有清楚、可验证的职责边界。

2. Repair Agent 的真实工具闭环

Repair Agent 不是一次性让模型生成整段答案，而是重复执行“模型决定下一步 → 调用本地工具 → 工具结果回填 → 模型继续判断”。它拥有七个工具：list_files、read_file、search_text、edit_file、write_file、run_command 和 finish。

文件路径被限制在指定 workspace；命令工具只允许受限的 pytest 形式；测试目录、pytest 配置、显式验证目标、审计轨迹、检查点和记忆目录受到宿主保护。上下文、读取长度、搜索结果、命令输出、重复调用次数和总轮数都有上限。

finish 只代表“申请验证”，不代表成功。Repair 至少要成功修改一次 .py 或 .pyi 源码，并且最后一次修改后的固定 pytest 必须通过，流程才会进入审查。只改说明文档、自己运行命令或在回复中写“测试已通过”都不能制造成功状态。

3. 独立、只读、会退回修改的 Reviewer Agent

Reviewer 使用全新上下文，不继承 Repair 的自我评价，也看不到长期记忆。它只有 list_files、read_file、search_text 和 submit_review 四个工具，没有编辑文件、运行命令、联网和批准权限。Reviewer 必须真正完成至少一次仓库检查，宿主才接受 pass 或 request_changes。

第一次 request_changes 会作为结构化反馈返回 Repair。Repair 必须产生新的源码修改，再通过固定 pytest，才可提交第二次审查；最多返工一次，防止两个 Agent 无限争论。第二次仍未通过时任务失败，不进入人工审批。

对于部分兼容接口“检查后只返回说明文字”的情况，宿主会开启一个只暴露 submit_review 的独立决策轮，要求 Reviewer 提交结构化结论；若仍不提交，系统按稳定错误码失败，而不是猜测它“应该算通过”。

4. 宿主掌握的成功证据链

模型没有最终裁决权。一次真实网页或 CLI 运行必须同时满足：

    Repair 显式调用 finish；
    至少一次规则允许的 Python 源码修改；
    固定 pytest 在最后一次修改之后退出码为 0；
    Reviewer 最终通过；
    用户明确批准。

这些证据由普通 Python 宿主代码维护，而不是从模型文本中提取。验证失败会作为工具结果回填，Repair 可以继续修复；Reviewer、审批或回滚失败都有独立停止原因。因此 TestPilot 是一个有控制平面的 Agent 系统，而不是“调用一次大模型再执行它给出的代码”。

5. 人工审批与精确回滚

Agent 第一次写文件前，ChangeJournal 会记录原始文件字节、权限和必要目录信息。审批时仅显示按路径排序的安全摘要：已有文件记为 M，新文件记为 A，并显示保守统计的新增/删除行数。文件名会转义，控制字符不能伪造审批界面。

批准后，改动成为新的工作区基线；拒绝、输入不可用或无法取得明确批准时，系统用写前快照恢复原文件和权限，并删除本轮创建的文件及空目录。只有确认回滚完整时才给出相应结果，回滚异常不会被隐藏成成功。

6. 可验证的断点恢复

真实任务在首次模型请求前生成 run_id，并把安全节点原子保存到：

    workspace/.testpilot/checkpoints/<run_id>.json

检查点保存有界 Repair 上下文、累计轮数、重复调用保护、Reviewer 轮数与返工计数、文件指纹和 ChangeJournal 写前快照，但不保存 API Key 或模型客户端。写文件前先持久化回滚边界，完整的 assistant/tool 事务结束后才推进对话状态，因此半次工具调用不会冒充成完整进度。

若最终摘要显示 resume_available=yes，可执行：

    python -m testpilot --workspace <原工作目录> --resume <run_id>

恢复前，宿主会校验 workspace 身份、检查点生命周期、修改文件的存在状态、类型、权限和 SHA-256 指纹。若用户或其他程序已经改过文件，系统会拒绝自动恢复，避免用旧状态覆盖新代码。恢复后仍要重新执行 pytest、Reviewer 和人工审批，旧的成功证据不会被直接沿用。

7. 仓库级长期记忆

每个仓库拥有独立记忆库：

    workspace/.testpilot/memories/entries.jsonl

新任务开始时，宿主用本地关键词相关度确定性排序，最多检索三条正相关经验，只注入 Repair 的初始上下文。Reviewer 看不到记忆，保证审查仍然独立。恢复同一 run_id 时沿用原上下文，不在中途重新检索。

只有 pytest 通过、Reviewer 通过、人工批准三项证据同时存在，Memory Agent 才能把经验整理为 problem、root_cause、solution、verification 和 keywords。任务、完成说明、文件名和 Reviewer 反馈在发送给 Memory 模型前会先清理常见凭据，并受到字段、路径和总回复长度限制。宿主再进行字段校验、二次清理、重复指纹判断和原子写入，并附加 run_id、验证退出码和审批状态等不可由模型伪造的证据。记忆生成失败只产生明确 warning，不会推翻已经完成的代码修复。

8. 可演示的留痕与隐私控制

每次运行会生成 JSONL 审计轨迹，记录 Repair 轮次、工具名、参数类型与长度摘要、模型/工具/验证耗时、Reviewer 与 Memory 阶段、审批结果、检索命中数、记忆 ID 和停止原因。它能证明 Agent 真实经历了哪些阶段，也便于答辩时解释失败发生在哪里。

轨迹不保存工具参数原文、提示输入、源码、完整 diff、API Key、Reviewer 反馈或记忆正文；Reviewer 和 Memory 事件只保留必要元数据。文件工具不能覆盖轨迹，常见 key、token、secret、password 和 credential 值会被清理。检查点、记忆和轨迹默认位于 workspace/.testpilot。TestPilot 不会修改待修复仓库的 .gitignore，因此应在目标仓库中自行忽略 .testpilot/，并在提交前确认它没有被暂存。

9. 网页是可观察、可审批的宿主界面

网页不是“第四个会改代码的 Agent”，而是把宿主状态机可视化。它通过事件流展示模型轮次、工具开始/完成、耗时、pytest、Reviewer、审批和记忆阶段；页面只接收安全标题、路径或命令摘要和稳定错误码，不显示源码与密钥。

服务只允许绑定本机回环地址，并校验本机 Host、同源 Origin 和 JSON 请求类型；API Key 只由后端从环境变量或用户目录文件读取。网页后端强制固定 pytest，即使直接构造网页请求也不能替换验证命令。任务互斥、审批状态和网络异常由后端与页面共同处理，降低跨站请求、重复启动或误判审批的风险。


三、无需 API Key 的完整演示
----------------------------

执行：

    pip install -e ".[dev]"
    python -m testpilot.demo

演示使用确定性 FakeModel，但调用的是实际 Agent 调度、工具、pytest、Reviewer、审批、检查点、恢复和记忆代码。它先制造并修复 subtract 错误，在中断后从磁盘恢复并写入一条经验；随后启动全新 difference 任务，检索并复用该经验。预期输出：

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

该演示无需网络和终端输入，适合提交检查、答辩和录屏。MEMORY_REUSED=yes 表示第二个新任务确实检索并注入一条经验并完成脚本化修复，不把它夸大为真实模型效果评测。


四、命令行备用入口
------------------

真实模型新任务：

    python -m testpilot --workspace <项目目录> --verify "python -m pytest -q" "修复任务描述"

恢复中断任务：

    python -m testpilot --workspace <项目目录> --resume <run_id>

命令行与网页使用同一套三 Agent、验证、审查、审批、检查点和记忆实现。网页适合演示与日常操作；命令行额外提供自定义受限 pytest 目标、指定 trace 路径和 run_id 恢复入口。


五、本地数据提醒
----------------

请勿提交真实 API Key，也不要把待修复仓库中的 .testpilot 目录上传到公开仓库。使用真实模型时，任务描述、工具结果和检索出的结构化记忆会发送到所配置的模型端点；只应处理允许发送给该服务的代码与信息。对于不可信仓库，应在容器或虚拟机内运行 pytest。
