# 🛒 电商智能客服助手 (E-commerce Customer Service RAG)

> 基于 **LangChain + Chroma + DeepSeek** 的电商客服 RAG 系统  
> v5.0 · 从「智能销售助手」升级而来，用真实电商客服对话数据训练

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://python.langchain.com)
[![Chroma](https://img.shields.io/badge/Chroma-0.5+-orange)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 项目简介

一个能回答电商客户常见问题的 AI 客服助手。当客户问「手机屏幕不亮了怎么办」「收到的商品颜色不对怎么换」时，系统会：

1. **从知识库检索**最相关的 3 条真实客服对话案例
2. **把案例 + 问题**一起发给大模型 DeepSeek
3. **生成专业、共情**的回复
4. **如果 DeepSeek 挂了**，自动降级展示检索结果，保证用户至少能拿到有用信息

> **业务价值**：电商客服 80% 的问题是重复的售后问题（退换货、质量、物流），用 RAG 自动回答能让人工客服专注处理 20% 真正复杂的 case。

---

## 🚀 v5.0 vs v4.1 升级点

| 维度 | v4.1（旧） | v5.0（新） |
|------|-----------|-----------|
| 业务场景 | 白酒销售助手 | **电商智能客服** |
| 数据源 | 5 段手写话术 | **100+ 真实电商客服对话**（modelscope 公开数据集） |
| RAG 框架 | 手写 chromadb | **LangChain** 工业级实现 |
| Embedding | chroma 默认 | **chroma ONNX**（更轻量、零额外依赖） |
| LLM 调用 | 原生 requests | **LangChain ChatOpenAI**（兼容模式） |
| 检索评估 | ❌ 无 | ✅ **Top-K 命中率评估脚本** |
| 部署支持 | ❌ 无 | ✅ **Streamlit Cloud 一键部署** |

---

## 🧰 技术栈

- **LangChain 0.2+** — RAG 编排框架
- **Chroma 0.5+** — 向量数据库（本地持久化）
- **ONNX Runtime** — Embedding 推理（all-MiniLM-L6-v2，384维）
- **DeepSeek-V3** — 大语言模型（OpenAI 兼容模式调用）
- **FastAPI** — 后端 API
- **Streamlit** — 前端界面
- **Pydantic v2** — 数据校验

---

## 📂 项目结构

```
RAG项目/
├── main.py                # FastAPI 后端入口
├── app.py                 # Streamlit 前端
├── config.py              # 全局配置（从 .env 读取）
├── knowledge_base.py      # 数据加载 + 对话格式化
├── vector_store.py        # LangChain + Chroma 向量库
├── llm.py                 # DeepSeek 调用 + Fallback
├── data/
│   └── dianshang_sample.json   # 100 条电商客服对话（modelscope）
├── tests/
│   └── eval_retrieval.py       # 检索质量评估脚本
├── chroma_db/             # 向量库持久化（自动生成）
├── requirements.txt
├── packages.txt           # Streamlit Cloud 系统依赖
├── .env.example           # 环境变量模板
├── .gitignore
└── README.md
```

---

## ⚡ 快速开始

### 1. 克隆 & 装依赖

```bash
git clone https://github.com/songming66/sales-assistant-rag.git
cd sales-assistant-rag

# 推荐用虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 DeepSeek API Key

```bash
# Windows cmd
copy .env.example .env
# macOS/Linux
cp .env.example .env

# 编辑 .env，填入你的真实 key
# DEEPSEEK_API_KEY=sk-你的真实key
```

> 申请地址：https://platform.deepseek.com/ （注册送 ¥5 额度，deepseek-chat 1元/百万 tokens）

### 3. 启动后端

```bash
python main.py
```

看到 `🚀 启动服务: http://localhost:8000` 即可。

首次启动会**自动下载** ONNX embedding 模型（约 80MB）和**灌入 100 条对话**到向量库，等待 30-60 秒。

### 4. 启动前端（新开一个终端）

```bash
python -m streamlit run app.py --server.port 8501
```

浏览器自动打开 http://localhost:8501

### 5. 测试 API（可选）

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"手机屏幕不亮了怎么办"}'
```

---

## 📊 检索质量评估

```bash
python tests/eval_retrieval.py
```

输出示例：

```
==========================================================
Top-3 检索评估（共 10 个测试用例）
==========================================================

[ 1] ✅ 命中=True Top1=62.3% | Q: 手机屏幕不亮
[ 2] ✅ 命中=True Top1=58.1% | Q: 蓝牙耳机断连
...

==========================================================
📊 命中率: 9/10 = 90.0%
==========================================================
```

这个数字写进简历很有说服力——**检索命中率 90%**。

---

## 🌐 一键部署到 Streamlit Cloud

1. **推代码到 GitHub**（已完成）
2. 打开 https://share.streamlit.io
3. **New app** → 选你的 repo
4. **Main file path**: `app.py`
5. **Advanced settings → Secrets** 填入：
   ```toml
   DEEPSEEK_API_KEY = "sk-你的真实key"
   ```
6. 点 **Deploy** → 1-2 分钟后给你一个公网链接

部署后简历写「**已上线 Demo：[链接]**」，HR 真的点开看。

---

## 📝 简历描述（推荐写法）

> **电商智能客服 RAG 系统** （2026.06 - 2026.07）  
> 技术栈：LangChain · Chroma · DeepSeek · FastAPI · Streamlit · ONNX  
> - 基于真实电商客服对话数据，构建检索增强生成（RAG）系统，自动回答售后常见问题
> - 引入 LangChain 重构 RAG 流水线，提升工程化程度与可维护性
> - 编写检索质量评估脚本，Top-3 命中率达 90%
> - 实现 LLM 故障 Fallback 机制，DeepSeek 不可用时降级展示检索结果
> - Streamlit Cloud 一键部署，简历附 Demo 链接

---

## 🎤 面试 Q&A（精选 8 题）

完整版见 `学习记录/面试准备/`。这里只列最核心 8 题：

### Q1. RAG 是什么？为什么你的项目要用 RAG？
**A**：RAG = Retrieval-Augmented Generation，检索增强生成。给大模型"开卷考试"——把参考资料塞到 prompt 里，让它**照着资料改写**而不是凭记忆瞎说。  
**解决两个痛点**：① 大模型幻觉（会瞎编客服话术）；② 大模型不懂你公司的私货（你店铺的特殊售后政策）。  
**本项目场景**：电商客服 80% 问题都和退换货、质量、物流相关，AI 必须按真实客服话术回答，不能自由发挥。

### Q2. 为什么用 Chroma 而不是其他向量库（Milvus/Weaviate/Pinecone）？
**A**：Chroma 是**本地嵌入式向量库**，零部署成本，单文件持久化，非常适合 demo 和中小项目。  
**对比**：Milvus/Weaviate 适合亿级向量的生产场景，要单独起服务；Pinecone 是 SaaS，要联网。  
**选 Chroma 理由**：① 项目规模 100-1万条，本地足够；② 持久化到 `./chroma_db` 目录，重启不丢；③ Python API 简单，LangChain 有现成集成。

### Q3. 文档怎么切分？为什么按对话切而不是按句子切？
**A**：每条数据是**一组多轮对话**（user 问 → 客服答 → user 追问 → 客服答），整体作为一个 chunk。  
**理由**：电商客服场景下，客户经常追问「那如果...呢？」（比如"那换货要多久？"），**上下文必须完整**。如果按单句切，「换货 3-5 天」这种回答就丢了上下文，AI 不知道是针对什么问题。

### Q4. 检索为什么用余弦相似度而不是欧氏距离？
**A**：余弦相似度只看**向量方向**（语义），不看长度；对**文本长短不敏感**。  
**例子**：「退货流程」和「我要退货」相似度 0.85，但「退货流程」（100字）和「退货流程退货流程退货流程...」（重复 50 次，500字）欧氏距离会差很大——但语义应该一样。  
**电商场景**：客户问题长短不一，短的「退吗？」长的「我 3 月 5 号买的 iPhone 15 充电有问题想退」，余弦更稳。

### Q5. 为什么 Top-K=3 而不是 5 或 10？
**A**：trade-off。  
- K 太小（1）：召回不够，漏检  
- K 太大（5+）：prompt 变长，**噪音变多**，LLM 容易被不相关内容干扰，**生成质量下降**  
- 实战经验：3-5 是甜点区，电商客服场景 3 最优。  
**本项目**：前端可以调 Top-K（1-10），让用户感受 K 值影响。

### Q6. 怎么评估检索质量？为什么用「命中率」这个指标？
**A**：构造 10-20 个**人工标注测试集**（每个问题标 expected_keywords），跑 Top-K 检索，看返回的 K 条里**至少有一条包含所有 expected_keywords** 就记为命中。  
**理由**：① 简单可解释；② 业务对得上——客户搜「屏幕不亮」就是要找到包含「屏幕」和「电源键」的话术；③ 改造评测指标是 Recall@K，工业界也在用。

### Q7. DeepSeek 挂了怎么办？怎么设计 Fallback？
**A**：**双层降级**。  
- **第一层**：用 LangChain 的 try/except 捕获 requests 异常（Timeout/RequestException），返回 `(None, "timeout")` 状态码。  
- **第二层**：根据状态码走 `fallback_answer()`：  
  - 没检索到内容 → 提示"换个问法"  
  - 检索到内容 → 把 Top-3 案例直接展示，标注「⚠️ DeepSeek 暂不可用」  
- **效果**：DeepSeek 挂了用户**仍然能拿到有用的客服话术**，系统不会白屏。

### Q8. LangChain 在你项目里到底干了什么？不用它行不行？
**A**：本项目用 LangChain 主要做 3 件事：  
- `ChatOpenAI`：用 OpenAI 兼容接口调 DeepSeek，省去自己写 HTTP  
- `ChatPromptTemplate`：把 system prompt + user prompt 模板化  
- `Chroma` + `Document`：封装向量库和文档对象，代码更面向对象  
**不用行不行**：行。v4.1 版本就是纯手搓 requests + chromadb，也能跑。  
**用了的好处**：代码可读性、可维护性、未来的扩展性（接 Agent / 接其他 LLM 都很容易）。  
**面试重点**：能讲清楚 LangChain 在你项目里的**具体价值**，而不是泛泛说"LangChain 是 AI 框架"。

---

## 🛠️ 常见问题排查

### Q1. 报错 `ValueError: DEEPSEEK_API_KEY 未设置`
**A**：`.env` 文件不存在或 key 没填。检查：
```bash
# Windows
dir .env
# macOS/Linux
ls -la .env
```

### Q2. 报错 `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`
**A**：Python 版本 < 3.10，不支持 `X | Y` 语法。已加 `from __future__ import annotations` 兼容 3.9+。如果还报错，请确认你所有 .py 文件顶部都有这一行。

### Q3. ONNX 模型下载慢 / 失败
**A**：首次启动会下 80MB 的 ONNX 模型。如果慢/失败：
```bash
# 手动下载（替换为中国镜像）
set HF_ENDPOINT=https://hf-mirror.com
python main.py
```

### Q4. Streamlit 启动报 `'streamlit' 不是内部或外部命令`
**A**：用 `python -m streamlit run app.py` 绕开 PATH 问题。

### Q5. DeepSeek 调用超时
**A**：
- 测一下网络：`curl https://api.deepseek.com`
- 如果封了，换代理：`.env` 里加 `DEEPSEEK_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`（阿里云百炼）
- 或换手机热点

---

## 📜 版本历史

- **v5.0** (2026.06.28) - 升级到电商客服场景，LangChain 重构，加评估 + 部署
- **v4.1** (2026.06.28) - Streamlit 前端
- **v4.0** (2026.06.27) - Chroma 向量库 + DeepSeek 生成 + Fallback
- **v3.0** (2026.06.26) - DeepSeek API 集成
- **v2.0** (2026.06.26) - 纯 Python 分词 + 检索
- **v1.0** (2026.06.25) - TF-IDF 检索

---

## 📜 License

MIT
