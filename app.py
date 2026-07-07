"""
v5.1 Streamlit 前端 - 电商智能客服助手（RAG + Agent 双模式）
启动: python -m streamlit run app.py --server.port 8501
"""
import requests
import streamlit as st

st.set_page_config(page_title="电商智能客服", page_icon="🛒", layout="wide")

st.title("🛒 电商智能客服助手")
st.caption("v5.1 · RAG + Agent 双模式")

# ─── 模式切换 ───
mode = st.radio("模式", ["🧠 RAG 模式", "🤖 Agent 模式"], horizontal=True)
is_agent = mode.startswith("🤖")

# ─── 侧边栏 ───
with st.sidebar:
    st.header("⚙️ 设置")
    api_base = st.text_input("后端 API 地址", value="http://localhost:8000")
    if not is_agent:
        top_k = st.slider("检索条数 Top-K", 1, 10, 3)

    st.divider()
    st.header("💡 示例问题")
    examples = [
        "手机屏幕不亮了怎么办？",
        "帮我查一下订单12345",
        "订单12345的物流到哪了？",
        "我要退款，商品有质量问题",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:8]}"):
            st.session_state.pending_query = ex

    st.divider()
    if st.button("🔄 检查服务状态"):
        try:
            r = requests.get(f"{api_base}/", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ 服务运行中 · 知识库 {data.get('knowledge_count', 0)} 条")
            else:
                st.error(f"❌ 服务异常: {r.status_code}")
        except Exception as e:
            st.error(f"❌ 连接失败\n\n请先启动: `python main.py`\n\n{e}")

# ─── 聊天区 ───
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("steps"):
            with st.expander("🧠 查看 Agent 思考过程", expanded=False):
                for s in msg["steps"]:
                    icon = {"action": "🔧", "observation": "📋", "answer": "💬"}.get(s["type"], "❓")
                    st.caption(f"{icon} **{s['type']}**")
                    if s.get("tool"):
                        st.caption(f"　　工具: `{s['tool']}({s.get('tool_input', '')})`")
                    st.text(s["content"][:300])
                    st.divider()

query = st.chat_input("请输入你的问题...")
if not query and "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        endpoint = "/agent_ask" if is_agent else "/ask"
        payload = {"query": query} if is_agent else {"query": query, "top_k": top_k}

        with st.spinner("🤔 思考中..."):
            try:
                resp = requests.post(f"{api_base}{endpoint}", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                st.write(data["answer"])

                if is_agent:
                    with st.expander("🧠 Agent 思考过程", expanded=True):
                        for s in data.get("steps", []):
                            icon = {"action": "🔧", "observation": "📋", "answer": "💬"}.get(s["type"], "❓")
                            st.caption(f"{icon} **{s['type']}**")
                            if s.get("tool"):
                                st.caption(f"　　工具: `{s['tool']}({s.get('tool_input', '')})`")
                            st.text(s["content"][:300])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "steps": data.get("steps", []),
                })
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接后端，请先启动 `python main.py`")
            except Exception as e:
                st.error(f"❌ 出错了: {e}")