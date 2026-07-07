
# 🛒 电商智能客服助手 v5.1
> 基于 **LangChain + Chroma + DeepSeek** 的电商客服系统，RAG + Agent 双模式
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-green)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-purple)](https://langchain-ai.github.io/langgraph/)
[![Chroma](https://img.shields.io/badge/Chroma-0.5+-orange)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
---
## 🎯 项目简介
一个能回答电商客户常见问题的 AI 客服助手，**RAG + Agent 双模式**：
| 模式 | 能力 | 适用场景 |
|------|------|---------|
| 🧠 **RAG 模式** | 检索知识库 → DeepSeek 生成回答 | 退换货政策、使用问题 |
| 🤖 **Agent 模式** | LLM 自主决策，调用 4 个工具完成任务 | 查订单、查物流、转人工 |
> 电商客服 80% 的问题是重复的售后问题，用 AI 自动回答能让人工专注处理 20% 真正复杂的 case。
---
## 🚀 v5.1 升级亮点
| 维度 | v5.0 | v5.1 |
|------|------|------|
| 核心能力 | RAG（检索+生成） | **RAG + Agent 双模式** |
| 决策方式 | 固定流程 | **LLM 自主选择工具** |
| 工具数量 | 1（知识库检索） | **4（知识库 / 订单 / 物流 / 转人工）** |
| 前端展示 | 检索案例 | **+ Agent 思考过程可视化** |
| Agent 框架 | ❌ | **LangGraph create_react_agent** |
**核心变化**：LLM 从"生成器"升级为"决策者"——自己判断该调哪个工具，ReAct 循环每一步都可在前端查看。
---
## 🧰 技术栈
| 层 | 技术 |
|---|---|
| RAG 框架 | LangChain 1.0+ |
| Agent 框架 | **LangGraph 1.0+** (create_react_agent) |
| 向量库 | Chroma 0.5+（本地持久化） |
| Embedding | ONNX all-MiniLM-L6-v2（384维，CPU 可跑） |
| LLM | DeepSeek-V3（OpenAI 兼容模式） |
| 后端 | FastAPI + Pydantic v2 |
| 前端 | Streamlit |
---
## 📂 项目结构

├── main.py # FastAPI 入口（RAG + Agent 双端点）

├── app.py # Streamlit 前端（双模式切换 + 思考过程展示）

├── llm.py # DeepSeek 调用 + 三档 Fallback

├── vector_store.py # Chroma 向量库封装

├── knowledge_base.py # 数据加载 + 对话格式化

├── config.py # 全局配置（从 .env 读取）

├── test_agent.py # Agent 独立测试脚本（4 工具 + ReAct）

├── data/

│ └── dianshang_sample.json # 100+ 条真实电商客服对话

├── tests/

│ └── eval_retrieval.py # 检索质量评估

├── chroma_db/ # 向量库持久化（自动生成）

├── requirements.txt

├── .env.example

├── .gitignore

└── README.md

plaintext
---
## ⚡ 快速开始
### 1. 克隆 & 装依赖
```bash
git clone https://github.com/songming66/sales-assistant-rag.git
cd sales-assistant-rag
pip install -r requirements.txt

2. 配置 API Key

bash
1
2
3
copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-你的key

申请地址：https://platform.deepseek.com/（注册送额度）

3. 启动后端

bash

python main.py

首次启动自动下载 ONNX 模型（~80MB）+ 灌入知识库，等 30-60 秒。

4. 启动前端（新终端）

bash
python -m streamlit run app.py --server.port 8501

浏览器打开 http://localhost:8501，切换 🤖 Agent 模式 体验。

5. 测试 Agent（可选）

bash
python test_agent.py

📡 API

表格
端点	模式	说明
POST /ask	RAG	检索+生成，返回 {answer, relevant_docs, scores}
POST /agent_ask	Agent	4 工具智能客服，返回 {answer, steps}（含思考过程）
GET /knowledge	通用	查看知识库
GET /docs	通用	Swagger API 文档

🤖 Agent 工具

表格
工具	功能	对接
search_knowledge	搜索客服知识库	Chroma 向量检索
check_order	查询订单状态	Mock 数据（可替换真实数据库）
check_logistics	查询物流进度	Mock 数据（可替换物流 API）
transfer_human	转人工客服	兜底机制

ReAct 循环示例（查订单 12345 的物流）：

plaintext
Thought → Action: check_logistics("12345")
       → Observation: "顺丰快递，杭州分拣中心，预计明天送达"
       → Final Answer: "您的订单通过顺丰配送，预计明天送达 😊"

📊 检索质量评估

bash
python tests/eval_retrieval.py

plaintext
Top-3 检索评估（共 10 个测试用例）
命中率: 9/10 = 90.0%

🌐 部署到 Streamlit Cloud

推代码到 GitHub
打开 https://share.streamlit.io → New app
选仓库 songming66/sales-assistant-rag，主文件 app.py
Secrets 填入：DEEPSEEK_API_KEY = "sk-你的key"
Deploy → 1-2 分钟后得到公网链接

注意：Streamlit Cloud 只跑前端，后端需本地运行或单独部署（推荐 Railway 免费托管）。

📝 简历描述

电商智能客服 RAG + Agent 系统（2026.06 - 2026.07）

技术栈：LangChain · LangGraph · Chroma · DeepSeek · FastAPI · Streamlit

基于 100+ 真实电商客服对话，构建 RAG + Agent 双模式智能客服系统
从 RAG 升级到 Agent：LLM 自主决策调用 4 个工具（知识库/订单/物流/转人工）
实现 ReAct 循环可视化，前端展示完整思考过程（Thought → Action → Observation）
三档 Fallback 机制，DeepSeek 不可用时降级展示检索结果，服务永远 200 响应
检索命中率 90%，GitHub 公开仓库，Streamlit Cloud 部署 Demo

🎤 面试 Q&A

Q1. RAG 和 Agent 的区别？

RAG：固定流程——检索 → 拼 prompt → LLM 生成。只能做"查资料回答问题"这一件事。

Agent：LLM 自主决策——自己判断该调哪个工具，可以多步推理。用户问"查订单 12345 的物流"，Agent 自己决定调 check_logistics，而不是去知识库瞎搜。

一句话：RAG 是"带着参考书回答"，Agent 是"带着工具箱干活"。

Q2. ReAct 循环是什么？

ReAct = Reasoning + Acting。LLM 先想（Thought）、再动手（Action）、看了结果（Observation）、再决定下一步，直到能回答用户。我的项目里用 LangGraph 的 create_react_agent 实现，前端还能把每一步都展示出来。

Q3. 为什么用 LangGraph 而不是手写 Agent？

LangGraph 的 create_react_agent 封装了 ReAct 循环的标准实现，自动处理 tool calling 解析、消息路由、循环终止。手写也能做，但 LangGraph 是 LangChain 生态的官方 Agent 框架，工业界认可度高，面试能讲清楚。

Q4. 工具怎么设计的？怎么替换成真实数据？

工具就是普通 Python 函数 + @tool 装饰器。现在用 Mock 字典演示，生产环境把字典换成数据库查询，工具接口不变，Agent 代码零改动。

Q5. DeepSeek 挂了怎么办？

三档 Fallback：

DeepSeek 正常 → 完整 AI 回答
DeepSeek 超时/报错 → 跳过 LLM，直接展示检索结果
检索也没结果 → 友好提示"换个问法"

核心原则：服务可用性 > 回答质量，用户永远拿 200 响应。

📜 版本历史

v5.1 (2026.07.07) - Agent 升级：4 工具 + ReAct 循环 + 双模式前端
v5.0 (2026.06.28) - 电商客服场景，LangChain 重构，加评估 + 部署
v4.1 (2026.06.28) - Streamlit 前端
v4.0 (2026.06.27) - Chroma 向量库 + DeepSeek 生成 + Fallback
v3.0 (2026.06.26) - DeepSeek API 集成
v2.0 (2026.06.26) - 纯 Python 分词 + 检索
v1.0 (2026.06.25) - TF-IDF 检索

📜 License

MIT

plaintext
把上面内容替换 `README.md`，然后：
```bash
git add README.md
git commit -m "docs: README v5.1 Agent 升级，双模式 + ReAct 可视化"
git push origin main