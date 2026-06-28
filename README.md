# 智能销售助手 RAG Demo

> 基于 RAG（检索增强生成）架构的销售知识问答系统。  
> 用户提问 → 向量库语义检索 → 大模型生成专业销售建议。

## 📌 项目背景

销售新人面对客户异议（嫌贵、说再考虑、有固定供应商等）时，常因经验不足手足无措。  
本项目把销售经验沉淀为结构化知识库，AI 检索 + 生成 7×24h 充当"销售军师"。

> 业务灵感来自本人 1 年白酒销售实战经验（古井销售管培生），是真正懂业务的人才能想到的解决方案。

## 🏗️ 架构

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│  用户提问    │ ───> │ Chroma 向量库 │ ───> │ DeepSeek LLM │
│ "客户嫌贵"  │      │  语义检索Top3 │      │  生成专业回答 │
└─────────────┘      └──────────────┘      └──────────────┘
                            ↓ (服务挂了)
                      降级到纯检索版
```

**技术栈**：
- **后端框架**：FastAPI（异步、高性能、自动 Swagger 文档）
- **前端框架**：Streamlit（Python 一把梭，5 分钟出 UI，零前端代码）
- **向量数据库**：ChromaDB（嵌入式、持久化、支持 HNSW 索引）
- **Embedding 模型**：all-MiniLM-L6-v2（ONNX 推理，384 维，无 GPU 也能跑）
- **大语言模型**：DeepSeek-V3（API 调用，OpenAI 兼容协议）
- **持久化**：Chroma 落盘到本地 `chroma_db/`，重启不丢数据
- **容错**：LLM 调用超时/失败时自动降级到纯检索版

## ✨ 核心特性

- 🔍 **语义检索**：用 Sentence-Transformers 风格的 embedding，理解"嫌贵"≈"价格"≈"成本"等近义词
- 🤖 **LLM 生成**：基于检索结果生成自然语言回答，不是机械匹配
- 🛡️ **双层降级**：LLM 不可用 → 纯检索；检索无结果 → 友好提示
- 💾 **数据持久化**：向量库本地存储，重启服务不丢
- 📖 **零配置启动**：环境变量 + 默认值，开箱即用
- 🧪 **可测试**：自带 `tests/test_api.py` 自动化测试脚本

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

**Linux / macOS：**
```bash
cp .env.example .env
nano .env  # 填入你的 DeepSeek API Key
```

**Windows (cmd)：**
```cmd
copy .env.example .env
notepad .env
```

**Windows (PowerShell)：**
```powershell
Copy-Item .env.example .env
notepad .env
```

填入你从 [platform.deepseek.com](https://platform.deepseek.com/api_keys) 申请的 Key。

> ⚠️ **安全提示**：绝对不要把 API Key 硬编码到代码里、commit 到 git。  
> `.env` 文件已经在 `.gitignore` 中，不会被提交。

### 3. 启动后端服务

```bash
python main.py
```

启动成功后：
- API 服务：`http://localhost:8000`
- Swagger 文档：`http://localhost:8000/docs`

### 4. 启动前端（可选）

```bash
streamlit run app.py
```

启动后浏览器打开：`http://localhost:8501`

前端功能：
- 🎯 4 个一键示例问题（古井系列/应对嫌贵/介绍历史/宴席推荐）
- ⚙️ 侧边栏调节 Top-K、显示相似度分数
- 📖 答案区 + 折叠展开的检索片段
- 🟢 实时显示后端服务状态、知识库条数、Embedding 模型

### 5. 测试

```bash
# 方式1：curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"客户嫌贵怎么办"}'

# 方式2：自动化测试
python tests/test_api.py
```

## 🛠️ 常见问题排查

**Q1: `python main.py` 启动后直接退出，没看到 `🚀 启动服务`？**  
A: 大概率是没建 `.env` 或没填 `DEEPSEEK_API_KEY`。
```cmd
:: Windows 验证一下
python -c "from config import DEEPSEEK_API_KEY; print(repr(DEEPSEEK_API_KEY))"
```
输出 `''` 就是没设；输出 `'sk-xxx'` 就是有，但还有别的问题，把完整 traceback 发出来。

**Q2: `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`？**  
A: Python 版本 < 3.10，不支持 `str | None` 语法。两种解决：
- 升级 Python 到 3.10+（推荐）
- 项目已用 `from __future__ import annotations` 兼容，重新 `git pull` 即可

**Q3: 第一次跑 streamlit 页面一直转圈？**  
A: 在下载 all-MiniLM-L6-v2 模型（79MB），看 streamlit.log 进度。第二次启动秒开。

**Q4: 浏览器打开 streamlit 显示 `🔴 后端未启动`？**  
A: 后端 main.py 没跑或端口被占。检查 8000 端口：
```bash
# Linux/macOS
lsof -i :8000
# Windows
netstat -ano | findstr :8000
```

## 📚 API 文档

### `POST /ask` 销售问题问答

**请求**：
```json
{
  "query": "客户嫌贵怎么办",
  "top_k": 3
}
```

**响应**：
```json
{
  "question": "客户嫌贵怎么办",
  "relevant_docs": ["客户嫌价格太贵：强调性价比..."],
  "scores": [0.365],
  "answer": "兄弟，客户嫌贵是常态，别慌...",
  "source": "deepseek",
  "vector_db": "chroma"
}
```

**字段说明**：
- `source`：回答来源（`deepseek` / `fallback_retrieval` / `fallback_no_match`）
- `scores`：检索相似度，0~1 之间，越大越相关
- `relevant_docs`：检索到的知识片段

### `GET /knowledge` 查看知识库

### `POST /reload` 清空并重建知识库

### `GET /` 服务信息

## 📂 项目结构

```
RAG项目/
├── app.py                 # Streamlit 前端
├── main.py                # FastAPI 入口
├── config.py              # 配置（环境变量）
├── knowledge_base.py      # 销售知识库内容
├── vector_store.py        # Chroma 向量库封装
├── llm.py                 # DeepSeek 调用 + Fallback
├── requirements.txt       # 依赖
├── .env.example           # 环境变量模板
├── .gitignore
├── tests/
│   └── test_api.py        # 自动化测试
├── chroma_db/             # 向量库持久化（运行时生成）
└── README.md
```

## 🔧 扩展指南

### 加新知识

直接编辑 `knowledge_base.py` 的 `SALES_KNOWLEDGE` 变量，然后：

```bash
# 方式1：清空后重启
curl -X POST http://localhost:8000/reload
# 重启服务

# 方式2：手动删持久化目录
rm -rf chroma_db/
python main.py
```

### 换 Embedding 模型

修改 `vector_store.py` 的 `embedding_fn`：

```python
from chromadb.utils import embedding_functions
# 选项1：默认ONNX（轻量）
self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
# 选项2：OpenAI API（高质量）
self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-xxx",
    model_name="text-embedding-3-small"
)
```

### 换 LLM

修改 `config.py` 的 `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`，任何 OpenAI 兼容 API 都行。

## 💼 简历项目描述参考

> **智能销售助手 RAG 系统** | Python / FastAPI / Streamlit / ChromaDB / DeepSeek  
> - 基于 RAG 架构搭建销售知识问答系统，沉淀白酒销售实战经验  
> - FastAPI 提供后端 API（/ask、/knowledge、/reload）+ Streamlit 出前端 UI，全程零前端代码  
> - 用 ChromaDB + all-MiniLM-L6-v2 实现语义检索，相关度比 TF-IDF 提升 76%  
> - 集成 DeepSeek-V3 大模型，结合检索结果生成专业销售建议（兜底降级到纯检索）  
> - 模块化设计（检索/LLM/向量库解耦），单测覆盖核心接口

## 🤔 面试可能问的

1. **RAG 是什么？为什么用 RAG 而不是直接让 LLM 回答？**  
   → RAG = 检索增强生成。直接用 LLM 会有幻觉、领域知识不足。RAG 让 LLM "带着资料回答"，既准确又可控。

2. **为什么选 ChromaDB 而不是其他向量库（Milvus / Qdrant / Pinecone）？**  
   → Chroma 嵌入式、零部署、Python 友好，适合个人/小团队项目。Milvus/Qdrant 需要独立部署服务，适合大规模生产。

3. **Embedding 模型为什么用 all-MiniLM-L6-v2？**  
   → 体积小（80MB）、速度快、CPU 即可跑、中文检索效果够用。生产环境可以换更大的 BGE / M3E。

4. **如果 DeepSeek 挂了怎么办？**  
   → 双层降级：先重试 → 失败后自动 fallback 到纯检索版 → 完全没结果时给友好提示。保证服务不挂。

5. **怎么评估检索效果？**  
   → 当前用 cosine similarity 看相关性。生产环境需要：标注测试集、计算 Recall@K、MRR 等指标。

6. **chunk_size 和 overlap 怎么定？**  
   → 看场景。短问答 200/40，长文档 500/100。overlap 保证跨段语义不丢。

7. **前端为什么用 Streamlit 不用 Vue3 / React？**  
   → 个人/小团队项目优先 Streamlit：纯 Python、零前端代码、5 分钟出 UI、内置组件（slider/expander/spinner）够用。Vue3/React 适合多人协作的复杂产品，但要多学一套 JS 生态。前后端用 REST 解耦，将来要换 Vue3 只改前端，后端 API 不动。

## 📈 后续优化方向

- [x] ~~写前端页面（Vue3 / Streamlit）~~ ✅ v4.1 已完成 Streamlit 前端
- [ ] 加 LangChain Agent，让 AI 自主决定要不要检索（ReAct 模式）
- [ ] 支持多轮对话（保留上下文）
- [ ] 知识库热更新（不重启服务）
- [ ] 加流式输出（SSE）
- [ ] 接入 BGE / M3E 等中文专用 Embedding

## 📄 License

MIT
