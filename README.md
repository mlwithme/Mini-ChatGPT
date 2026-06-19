# Mini-ChatGPT 教学版说明

这个仓库用 3 个最小示例讲清一件事：

**一个“像 ChatGPT 的聊天助手”如何管理短期记忆、长期记忆，以及工具调用闭环，即如何从零实现一个Mini-ChatGPT**

<div align="center">
  <img width="760" src="imgs/202606141405.jpg" alt="Mini-ChatGPT"/>
</div>

## 1 这个仓库能学到什么

- 短期记忆是什么，长期记忆是什么，上下文又是什么？
- 如何搭建一套类似 ChatGPT 一轮对话的核心逻辑
- 同一套原理，如何从手写版（`mini-chatgpt`）走到 LangGraph版（`mini-chatgpt-langgraph`），最后再到 Agent 版（`mini-chatgpt-agent`）。

## 2 核心逻辑

一个最小可用的记忆型聊天助手，通常就是下面这条链路：

```text
用户输入
  -> 检索相关长期记忆
  -> 拼接 system prompt + 短期历史 + 当前问题
  -> 调用大模型
  -> 如果模型要调用工具，就执行工具
  -> 把工具结果继续喂回模型
  -> 输出最终回答
```

如果只保留最关键的主干逻辑，可以把这个仓库理解成下面这套流程：

<div align=center><img width="700" src="imgs/202606141439.jpg"/> </div>

这里最重要的 3 个概念：

- 短期记忆：当前会话里的消息历史
- 长期记忆：跨会话仍然有价值的用户信息
- 工具调用：让模型能“写记忆”而不只是“说话”

长期记忆在本仓库里分为 3 类：

- `episodic`：情景记忆，记录经历过的事件
- `semantic`：语义记忆，记录稳定事实、偏好、背景
- `procedural`：程序性记忆，记录长期协作方式或回答偏好

## 3 为什么要区分短期记忆和长期记忆

- 短期记忆服务“当前这次会话”
- 长期记忆服务“未来很多次会话”

也因此通常会区分两个标识：

- `user_id`：这个用户是谁
- `thread_id`：当前这条会话线程是谁

常见设计就是：

- 长期记忆挂在 `user_id`
- 短期记忆挂在 `thread_id`

## 4 三种实现方式

### 4.1 `mini-chatgpt`

最适合入门，主打“手写原理”。

- 短期记忆：Python 进程内的 `history`
- 长期记忆：本地 JSON 文件
- 核心特点：代码最直白，最容易看懂记忆检索和工具调用闭环

建议重点看：

- [`mini-chatgpt/src/agent.py`](mini-chatgpt/src/agent.py)
- [`mini-chatgpt/src/store.py`](mini-chatgpt/src/store.py)
- [`mini-chatgpt/src/tools.py`](mini-chatgpt/src/tools.py)

### 4.2 `mini-chatgpt-langgraph`

主打“显式状态图 + 工程化状态管理”。

- 短期记忆：`PostgresSaver` + `thread_id`
- 长期记忆：`PostgresStore`
- 核心特点：把对话状态、会话线程、持久化管理得更清晰

建议重点看：

- [`mini-chatgpt-langgraph/src/agent.py`](mini-chatgpt-langgraph/src/agent.py)
- [`mini-chatgpt-langgraph/src/store.py`](mini-chatgpt-langgraph/src/store.py)

### 4.3 `mini-chatgpt-agent`

主打“用 `langchain.agents.create_agent` 快速搭 Agent”。

- 短期记忆：`checkpointer` 管理
- 长期记忆：`PostgresStore`
- Agent 构建：`langchain.agents.create_agent(...)`
- 核心特点：比手写循环更省代码，比显式 LangGraph 图更轻，适合理解 LangChain 高层封装

这个版本的关键点是：

- 仍然会先检索长期记忆，再动态拼出 `system_prompt`
- 仍然会把 `upsert_memory` 作为工具交给模型
- 但对话循环、工具执行、状态推进，更多交给 `create_agent` 处理

建议重点看：

- [`mini-chatgpt-agent/src/agent.py`](mini-chatgpt-agent/src/agent.py)
- [`mini-chatgpt-agent/src/store.py`](mini-chatgpt-agent/src/store.py)
- [`mini-chatgpt-agent/src/tools.py`](mini-chatgpt-agent/src/tools.py)

## 5 三个版本对比

| 版本 | 短期记忆 | 长期记忆 | 适合学习什么 |
| --- | --- | --- | --- |
| `mini-chatgpt` | 手动 `history` | 本地 JSON | 从零实现记忆型 ChatGPT 最小原理 |
| `mini-chatgpt-langgraph` | `PostgresSaver` | `PostgresStore` | LangGraph 状态图和工程化状态管理 |
| `mini-chatgpt-agent` | `PostgresSaver` | `PostgresStore` | LangChain 中 `create_agent` 高层封装 |

建议阅读顺序：

1. 先看 `mini-chatgpt`
2. 再看 `mini-chatgpt-langgraph`
3. 最后看 `mini-chatgpt-agent`

这样最容易看出：

- 哪些是 Agent 的不变原理
- 哪些只是不同框架提供的封装方式

## 6 快速运行

### 6.1 本地 JSON 版

```bash
cd mini-chatgpt
python main.py

用户ID = demo-user
模型 = qwen-plus
记忆目录 = .memory
已存储记忆数量 = 2
Commands: /help: 查看帮助  /memories: 查看当前用户所有长期记忆  /clear: 清楚当前用户所有长期记忆 /exit: 退出

[You]: 
```

从当前提示可以看到，本地已经储存有两条记忆信息，并且可以通过命令 `/memories` 查看：

```shell
[You]: /memories
已保存记忆:
1. [semantic / 语义记忆] 用户籍贯为四川，当前常居地为上海 | 用户的地理背景信息，可能影响后续对地域相关话题（如饮食、方言、政策等）的交流
2. [episodic / 情景记忆] 用户曾用5天完成聊天机器人搭建，流程包括：梳理逻辑→绘制结构图→编码实现→调试/注释/修bug→发布 | 体现用户的项目执行能力、结构化思维和工程实践习惯，属于具体的一次性项目经历

[You]: 
```

进一步，可直接输入相关内容与其进行交互，如果识别到新的记忆内容，也会进行保存：

```shell
[You]: 我有一个朋友叫张三，他是北京人，我们是大学同学。对了，今天北京天气怎么样？
有 1 条记忆插入
[Assistant]:
 已为你保存这条语义记忆：  
✅ 朋友张三，北京人，与你是大学同学。
这有助于我未来更自然地理解你们之间的关系背景（比如聊到母校、北京生活、学生时代等话题时）😊
至于**今天北京的天气**——虽然我无法实时联网查询......
欢迎告诉我你的偏好～ 🌤️
----------------------------------------------------------------------

[You]: 
```

根据上述内容可以看出，用户的输入包含有记忆相关内容，智能体对其进行了保存，同时也输出了与北京天气相关的内容。此时，可以再次查看当前保存的记忆内容：

```shell
[You]: /memories
已保存记忆:
1. [semantic / 语义记忆] 用户籍贯为四川，当前常居地为上海 | 用户的地理背景信息，可能影响后续对地域相关话题（如饮食、方言、政策等）的交流
2. [episodic / 情景记忆] 用户曾用5天完成聊天机器人搭建，流程包括：梳理逻辑→绘制结构图→编码实现→调试/注释/修bug→发布 | 体现用户的项目执行能力、结构化思维和工程实践习惯，属于具体的一次性项目经历
3. [semantic / 语义记忆] 朋友张三，北京人，与用户是大学同学 | 属于用户长期人际关系信息，可能影响后续对话中对共同经历、地域背景或社交场景的理解
```

### 6.2 LangGraph 版

```bash
cd mini-chatgpt-langgraph
python main.py --cli  # 命令行交互启动
streamlit run main.py  # Web 端交互启动
```

### 6.3 create_agent 版

```bash
cd mini-chatgpt-agent
python main.py --cli  # 命令行交互启动
streamlit run main.py  # Web 端交互启动
```
