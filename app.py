"""
v5.0 Streamlit 前端 - 电商智能客服助手
启动: python -m streamlit run app.py --server.port 8501
"""
from __future__ import annotations
import requests
import streamlit as st

# ─── 页面配置 ───
st.set_page_config(
    page_title="电商智能客服",
    page_icon="🛒",
    layout="wide",
)

# ─── 标题区 ───
st.title("🛒 电商智能客服助手")
st.caption("v5.0 · LangChain + Chroma + DeepSeek · 基于真实电商客服对话数据训练")

# ─── 侧边栏 ───
with st.sidebar:
    st.header("⚙️ 设置")
    api_base = st.text_input("后端 API 地址", value="http://localhost:8000")
    top_k = st.slider("检索条数 Top-K", min_value=1, max_value=10, value=3)

    st.divider()
    st.header("💡 示例问题")
    examples = [
        "我的手机屏幕突然不亮了怎么办？",
        "收到的商品颜色和图片不一样怎么换？",
        "鞋子穿几天就开胶是不是质量问题？",
        "衣服掉色严重可以退货吗？",
        "蓝牙耳机一直断连怎么处理？",
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
                st.success(f"✅ 服务运行中\n\n知识库: {data.get('knowledge_count', 0)} 条")
            else:
                st.error(f"❌ 服务异常: {r.status_code}")
        except Exception as e:
            st.error(f"❌ 连接失败\n\n请确认后端已启动: `python main.py`\n\n错误: {e}")


# ─── 主聊天区 ───
# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "meta" in msg:
            with st.expander("📎 检索到的相关案例", expanded=False):
                for i, (doc, score) in enumerate(zip(msg["meta"].get("docs", []), msg["meta"].get("scores", [])), 1):
                    st.markdown(f"**【案例 {i}】** 相关度 `{score:.1%}`")
                    st.text(doc[:300] + ("..." if len(doc) > 300 else ""))
                    st.divider()

# 用户输入
query = st.chat_input("请输入你的问题...")
if not query and "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # 调用后端
    with st.chat_message("assistant"):
        with st.spinner("🤔 AI 客服正在思考..."):
            try:
                resp = requests.post(
                    f"{api_base}/ask",
                    json={"query": query, "top_k": top_k},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                st.write(data["answer"])

                # 显示检索片段
                with st.expander("📎 检索到的相关案例", expanded=False):
                    for i, (doc, score) in enumerate(zip(data.get("relevant_docs", []), data.get("scores", [])), 1):
                        st.markdown(f"**【案例 {i}】** 相关度 `{score:.1%}`")
                        st.text(doc[:300] + ("..." if len(doc) > 300 else ""))
                        st.divider()

                # 底部信息
                source = data.get("source", "unknown")
                if source == "deepseek":
                    st.caption(f"🤖 来源: DeepSeek 生成")
                elif source.startswith("fallback"):
                    st.caption(f"⚠️ 来源: {source}（DeepSeek 不可用，检索结果直出）")

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "meta": {
                        "docs": data.get("relevant_docs", []),
                        "scores": data.get("scores", []),
                    }
                })
            except requests.exceptions.Timeout:
                st.error("⏱️ 请求超时（>60s），DeepSeek 响应慢或挂了")
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接后端服务，请确认 `python main.py` 已启动")
            except Exception as e:
                st.error(f"❌ 出错了: {e}")
