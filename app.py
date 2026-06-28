"""Streamlit 前端 - 智能销售助手

启动: streamlit run app.py
要求: FastAPI 服务 (main.py) 在 8000 端口运行
"""
import streamlit as st
import requests

# ============ 配置 ============
API_BASE = "http://127.0.0.1:8000"

# ============ 页面配置 ============
st.set_page_config(
    page_title="智能销售助手",
    page_icon="🍶",
    layout="wide",
)

# ============ 自定义样式 ============
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%); }
    .answer-box {
        background: white;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 5px solid #d4a017;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        font-size: 16px;
        line-height: 1.8;
        color: #2c3e50;
    }
    .source-tag {
        display: inline-block;
        background: #f0f2f6;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: #555;
        margin-right: 6px;
    }
    .score-high { color: #d4a017; font-weight: 600; }
    .score-mid  { color: #5b8def; font-weight: 600; }
    .score-low  { color: #999; }
</style>
""", unsafe_allow_html=True)


# ============ 辅助函数 ============
@st.cache_data(ttl=30)
def get_service_info():
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_knowledge():
    try:
        r = requests.get(f"{API_BASE}/knowledge", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"获取知识库失败: {e}")
        return None


def ask_question(query: str, top_k: int):
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ 请求超时（>30s），DeepSeek 响应慢或挂了")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 连不上后端服务（127.0.0.1:8000），先 `python main.py` 启动")
        return None
    except Exception as e:
        st.error(f"❌ 出错了: {e}")
        return None


def reload_kb():
    try:
        r = requests.post(f"{API_BASE}/reload", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"重载失败: {e}")
        return None


def score_color(score: float) -> str:
    if score >= 0.4:
        return "score-high"
    elif score >= 0.2:
        return "score-mid"
    return "score-low"


# ============ 侧边栏 ============
with st.sidebar:
    st.header("⚙️ 控制台")

    # 服务状态
    info = get_service_info()
    if info:
        st.success(f"🟢 服务运行中 (v{info['version']})")
        st.caption(f"📚 知识库: {info['knowledge_count']} 条")
        st.caption(f"🧠 Embedding: {info['embedding'].split(' ')[0]}")
        st.caption(f"🤖 LLM: {info['llm'].split('-')[-1]}")
    else:
        st.error("🔴 后端未启动")
        st.code("python main.py", language="bash")
        st.stop()

    st.divider()

    # 参数
    top_k = st.slider("检索条数 Top-K", 1, 5, 3)
    show_sources = st.checkbox("显示检索片段", value=True)
    show_scores = st.checkbox("显示相似度分数", value=True)

    st.divider()

    # 知识库管理
    st.subheader("📚 知识库")
    if st.button("👀 查看知识库", use_container_width=True):
        st.session_state.show_kb = True
    if st.button("🔄 重载知识库", use_container_width=True, type="secondary"):
        with st.spinner("清空中..."):
            res = reload_kb()
        if res:
            st.success(f"✅ 已重载: {res.get('count', '?')} 条")
            st.rerun()

    st.divider()
    st.caption("Built with Streamlit + FastAPI + Chroma + DeepSeek")

# ============ 主区 ============
st.title("🍶 智能销售助手")
st.markdown("##### 白酒销售场景 · RAG 知识库问答 · 基于 DeepSeek 生成")

# 知识库查看弹窗
if st.session_state.get("show_kb"):
    with st.expander("📚 知识库内容", expanded=True):
        kb = get_knowledge()
        if kb and kb.get("documents"):
            for i, (doc, meta) in enumerate(zip(kb["documents"], kb["metadatas"]), 1):
                st.markdown(f"**[{i}] {meta.get('title', '无标题')}** · `{meta.get('category', '?')}`")
                st.caption(doc[:200] + ("..." if len(doc) > 200 else ""))
                st.divider()
        if st.button("关闭"):
            st.session_state.show_kb = False
            st.rerun()

# 示例问题
st.markdown("**💡 试试这些问题：**")
example_cols = st.columns(4)
examples = [
    "古井贡酒年份原浆有哪些系列？",
    "客户嫌价格贵怎么应对？",
    "怎么向客户介绍古井贡酒的历史？",
    "宴席场景怎么推荐用酒？",
]
for i, (col, ex) in enumerate(zip(example_cols, examples)):
    if col.button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state.question = ex

# 问题输入
question = st.text_input(
    "🗣️ 你的问题：",
    value=st.session_state.get("question", ""),
    placeholder="比如：古井贡酒年份原浆古20适合什么场合？",
    label_visibility="collapsed",
)

# 提问
ask_btn = st.button("🚀 提问", type="primary", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("🔍 检索知识库 + 🤖 生成回答..."):
        result = ask_question(question.strip(), top_k)

    if result:
        # 答案
        st.markdown("### 💬 回答")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        # 元信息
        st.caption(
            f"来源: `{result.get('source', '?')}` · "
            f"向量库: `{result.get('vector_db', '?')}`"
        )

        # 检索片段
        if show_sources and result.get("relevant_docs"):
            st.markdown("### 📖 检索到的知识片段")
            for i, (doc, meta, score) in enumerate(zip(
                result["relevant_docs"],
                result.get("metadatas", [{}] * len(result["relevant_docs"])),
                result.get("scores", []),
            ), 1):
                with st.expander(
                    f"片段 {i} · {meta.get('title', '?')} · "
                    f"{meta.get('category', '?')}"
                    + (f" · 相似度 {score:.1%}" if show_scores and score else ""),
                    expanded=(i == 1),
                ):
                    if show_scores and score:
                        st.markdown(
                            f"相似度: <span class='{score_color(score)}'>{score:.1%}</span>",
                            unsafe_allow_html=True,
                        )
                    st.caption(doc)

elif ask_btn and not question.strip():
    st.warning("⚠️ 先输入问题")
