# skill 工作原理

> 核心类比：  
> **LLM = 大脑；Agent = 执行人；Tool = 手脚（原子动作）；Skill = 专业SOP/操作手册（怎么做这件事）**
>
> - Tool：**能干什么**（单次原子调用，函数/API）
> - Skill：**该怎么干**（一套流程、约束、最佳实践，由多条Tool调用编排而成）

## 一、Skill 在 Agent 架构里的定位

Skill 是**领域任务的可复用能力包**，不是可直接执行的函数，它是给大模型看的任务说明书，包含：触发条件、执行步骤、分支逻辑、异常处理、输出格式、允许使用哪些 Tool、禁止行为。

### 层级关系（由底层到上层）

1. **Function Calling**：模型原生结构化输出协议，定义怎么输出调用指令
2. **Tool**：原子执行单元，读文件、bash、http 请求，单次执行，输入输出 Schemas
3. **Skill**：上层业务封装，把多步 Tool 调用流程、领域约束打包（SKILL.md 标准格式，YAML 元数据 + Markdown 指令）
4. **Agent**：运行调度主体，Planner 规划器，负责选择 Skill、循环调用 Tool、管理记忆、状态、上下文
5. **Plugin/MCP**：工具分发/连接协议，用于批量导入 Tool 集合

> 关键区分：  
> Tool 有 `call()` 执行函数；Skill **没有执行函数**，输出是提示词/指令注入，指导 Agent 去调用已有的 Tool 完成任务。  
> Skill 解决的痛点：如果把全部任务流程全塞进 System Prompt，token 爆炸、上下文臃肿；Skill 采用**按需加载**，只常驻元数据，匹配成功才加载完整流程内容。

## 二、Skill 标准结构（SKILL.md）

```yaml
---
name: sql-code-review
description: "SQL代码审查，用于检查SQL语法、索引、风险语句；用户提交SQL片段触发"
allowed_tools: ["read_file", "bash"]
risk: low
---
# 执行指引
1. 读取待审查SQL文本
2. 检查SELECT是否缺少索引字段
3. 禁止SELECT *；禁止不带limit的全表扫描
4. 发现风险逐条输出，给出修改建议
5. 输出固定markdown报告格式
```

- YAML frontmatter：**常驻索引**，只存名字、简短描述、触发场景，启动时全部加载进 Agent 索引，token 开销很小
- Markdown 正文：完整 SOP 流程，**不会默认塞入上下文；只有被选中激活之后，才注入会话**

## 三、完整运行工作流程（5 阶段）

### 1. 发现 & 注册（Agent 启动阶段）

Agent 框架扫描 skill 目录，只解析每个 SKILL.md 头部 YAML 元数据，生成 `skill_name + description` 的索引目录，写入 Agent 系统提示词。

> ✅ 此时**不会加载 SKILL.md 完整正文**，避免 token 爆炸，模型只知道“有哪些技能可用”，不知道内部详细步骤。

### 2. 意图匹配 & Skill 选择（推理阶段）

用户输入任务 → Agent Planner（LLM）读取 skill 目录索引，判断当前任务匹配哪一个 Skill。

输出结构化 Action：`activate_skill(skill-name, params)`，请求宿主运行时加载该 Skill 完整内容。

> 两种选择模式：
>
> 1. 静态注册：启动扫描本地目录
> 2. 动态发现：元技能 `find_skill`，从远端 Skill 库检索匹配（解决成千上万个 Skill 场景）

### 3. 加载注入（关键：Progressive Disclosure 渐进式加载）

Agent 运行时读取磁盘/远端完整 SKILL.md，把完整 SOP 指令**注入当前对话上下文**（新增一条 meta 消息）。

> 注意：Skill 本身**不执行任何操作**，只是把“操作手册”交给大模型，告诉模型接下来该怎么干活。

两种注入模式：

1. **Inline 内联（主流）**：直接把指令注入主会话上下文，复用 Agent 现有 Tool 集合，用完之后可以丢弃该段上下文，释放 token；绝大多数 Skill 使用该模式。
2. **Fork 子 Agent**：新建独立隔离会话，子 Agent 加载 Skill，拥有独立上下文窗口；适合重型、高风险任务，执行结束销毁子会话，不污染主对话。

### 4. 驱动 Tool 循环执行

大模型读到注入的 Skill 完整 SOP，按照 Skill 写好的步骤，循环调用底层 Tool（Function Calling）：

> Skill 写的是“第一步调用 read_file，第二步 bash 执行检查脚本，第三步整理结果”，但 Skill 本身不调用工具；**LLM 根据 Skill 指令输出 Tool 调用请求**，由 Agent 执行 Tool 拿到外部结果，回传给模型，多轮迭代直到任务完成。

### 5. 任务结束，卸载 Skill

- Inline 模式：把 Skill 指令消息标记失效，不再参与后续推理；
- Fork 子 Agent：销毁整个子会话实例。

> 不会永久占用上下文 token，下一轮对话重新走匹配逻辑。

## 四、Skill vs Tool 核心对比

| 维度       | Tool（工具）                         | Skill（技能）                                |
| ---------- | ------------------------------------ | -------------------------------------------- |
| 本质       | 原子执行函数，有 call() 逻辑         | 任务 SOP/指令包，无执行函数                  |
| 内容       | 代码、API 调用、Schema               | SKILL.md：元数据 + markdown 流程             |
| 上下文     | 每次调用返回结果，不修改 system 提示 | 匹配成功才注入完整流程指令                   |
| 粒度       | 单步动作（读文件、http 请求）        | 多步业务流程（SQL 审查、代码提交、报表生成） |
| token 开销 | 每次调用只返回结果                   | 仅元数据常驻，完整内容按需加载               |
| 开发方式   | 写代码实现接口                       | 写 markdown，非开发者也可以编写              |

## 五、两种典型 Skill 实现范式

1. **指令式 Skill（Claude Code 主流）**：纯 SKILL.md，不附带脚本；完全靠 LLM 理解 SOP，自主编排 Tool 调用；实现简单，但模型会有偏离流程风险。
2. **脚本增强 Skill**：SKILL.md 附带脚本文件夹，可嵌入 shell/python 片段；Skill 激活时可预执行脚本，生成辅助数据，再交给 LLM；可控性更强。

## 六、现实局限

1. Skill 只是提示词注入，**不能 100% 强制模型遵守流程**，复杂任务依然会出现偏离 SOP；工程上会增加 Harness 执行层做校验、强制回退、重试、权限拦截。
2. Inline 模式依然消耗主会话 token；大量复杂 Skill 优先使用 Fork 子 Agent 模式。
3. Skill 依赖 LLM 语义匹配，描述写得不好，会出现选错 Skill、漏触发。

## 七、链路极简总结

```text
用户输入
    ↓
Agent读取全部Skill【元数据目录】（仅名字+简介）
    ↓
LLM Planner判断：激活skill-xxx
    ↓
运行时读取完整SKILL.md，注入会话上下文
    ↓
LLM读取Skill的SOP流程，循环调用Tool执行多步任务
    ↓
Tool执行返回结果回传给LLM迭代
    ↓
任务完成，卸载Skill，输出最终结果
```



# 八、Skill 的触发机制

### 1. 自动发现

系统自动扫描 `.agents/skills/` 目录下所有 `SKILL.md` 文件，提取 YAML frontmatter 中的 `name` 和 `description`，生成可用 skill 列表注入 Agent 上下文：

```xml
<available_skills>
  <skill>
    <name>tilelang-operator-dev</name>
    <description>NPU TileLang 算子自动开发与优化闭环 skill(触发词...)</description>
    <location>...</location>
  </skill>
B</skill>
  ...
</available_skills>
```

### 2. description 语义匹配

Agent 收到用户请求后，将请求内容与所有 skill 的 description 做语义匹配。当前 `tilelang-operator-dev` 的 description 中定义了触发词：

```yaml
description: >
  当用户要求"开发 TileLang 算子"、"优化 NPU 算子"、"自动生成算子"、
  "算子性能调优"、"TileLang kernel 开发"、"NPU 算子开发"时触发。
```

所以用户请求中包含这些关键词时，Agent 会自动匹配并调用：

```text
用户: "帮我开发一个 RMSNorm 的 NPU 算子"
         ↑↑↑↑↑↑↑                    ↑↑↑
     匹配"开发 TileLang 算子"    匹配"NPU 算子开发"
         → Agent 调用 skill("tilelang-operator-dev")
```

### 3. 实际调用流程

```text
用户请求
    ↓
系统扫描 .agents/skills/，生成可用 skill 列表
    ↓
Agent 读取 skill 列表，匹配 description 触发词
    ↓
匹配成功 → Agent 调用 skill 工具: skill("tilelang-operator-dev")
    ↓
SKILL.md 内容注入 Agent 上下文（705 行方法论知识）
    ↓
Agent 按 §6 的 10 步闭环自动执行
```

### 4、如何确保触发

| 方式     | 操作                                              | 说明                       |
| -------- | ------------------------------------------------- | -------------------------- |
| 自动触发 | 用户请求包含"开发 TileLang 算子"等触发词          | Agent 自动匹配 description |
| 手动触发 | 用户说"使用 tilelang-operator-dev skill 开发算子" | 显式指定 skill 名          |
| 强化触发 | 在 AGENTS.md 中添加 skill 引用                    | 项目级强制 Agent 优先使用  |

如果担心自动匹配不够可靠，可以在 `.codeartsdoer/AGENTS.md` 中添加推荐：

```markdown
## Skill 使用

当涉及 NPU 算子开发或优化时，优先使用 `tilelang-operator-dev` skill。
```

