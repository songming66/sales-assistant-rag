"""
Day 2: 电商客服 Agent — 4 个工具
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

# ─── 1. LLM ───
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.3,
    max_tokens=800,
)

# ─── 2. 工具 ①：查知识库（接 v5.0 向量库）───
from langchain.tools import tool
from vector_store import VectorStore

store = VectorStore()

@tool
def search_knowledge(query: str) -> str:
    """
    当用户问商品使用、售后政策、退换货规则等需要参考历史客服案例的问题时，
    用此工具搜索知识库。
    参数 query: 用户问题的关键词，如"手机屏幕不亮"
    """
    docs, scores = store.search(query, top_k=3)
    if not docs:
        return "知识库中没有找到相关案例。"
    lines = []
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        lines.append(f"【案例{i}】相关度 {score:.0%}\n{doc.strip()[:200]}")
    return "\n\n".join(lines)


# ─── 3. 工具 ②：查订单 ───
MOCK_ORDERS = {
    "12345": "订单12345｜商品：iPhone 15 Pro Max｜金额：¥9,999｜状态：已发货｜下单时间：7月3日",
    "12346": "订单12346｜商品：AirPods Pro｜金额：¥1,899｜状态：备货中｜下单时间：7月5日",
    "12347": "订单12347｜商品：MacBook Pro 14｜金额：¥14,999｜状态：已完成｜下单时间：6月28日",
    "12348": "订单12348｜商品：iPad Air｜金额：¥5,499｜状态：已取消｜下单时间：7月1日",
}

@tool
def check_order(order_id: str) -> str:
    """
    查询用户的订单状态。
    参数 order_id: 订单号，如"12345"
    """
    info = MOCK_ORDERS.get(order_id)
    if info:
        return info
    return f"未找到订单 {order_id}，请确认订单号是否正确。"


# ─── 4. 工具 ③：查物流 ───
MOCK_LOGISTICS = {
    "12345": "顺丰快递 SF1234567890｜当前位置：杭州分拣中心｜预计送达：明天（7月7日）",
    "12346": "中通快递 ZT9876543210｜当前位置：广州转运中心｜预计送达：7月9日",
    "12347": "已签收｜签收人：本人｜签收时间：7月1日 14:30",
    "12348": "无物流信息（订单已取消）",
}

@tool
def check_logistics(order_id: str) -> str:
    """
    查询订单的物流配送进度。
    参数 order_id: 订单号，如"12345"
    """
    info = MOCK_LOGISTICS.get(order_id)
    if info:
        return f"📦 {info}"
    return f"未找到订单 {order_id} 的物流信息。"


# ─── 5. 工具 ④：转人工 ───
@tool
def transfer_human(reason: str) -> str:
    """
    当问题超出 AI 能力范围，或用户明确要求转人工时调用。
    参数 reason: 转人工的原因，如"用户要求退款但已超7天无理由"
    """
    return (
        f"✅ 已为您转接人工客服，请稍候…\n"
        f"转接原因：{reason}\n"
        f"预计等待时间：约 2 分钟"
    )


# ─── 6. 组装 Agent ───
from langgraph.prebuilt import create_react_agent

tools = [search_knowledge, check_order, check_logistics, transfer_human]
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=tools)

SYSTEM_PROMPT = """你是小蜜，一位电商智能客服助手。规则：
1. 用户问退换货、使用问题 → 先调用 search_knowledge 查案例
2. 用户问订单状态 → 调用 check_order
3. 用户问物流到哪了 → 调用 check_logistics
4. 超出能力范围（如退款、投诉）→ 调用 transfer_human
5. 回答要亲切、具体，200字以内"""


# ─── 7. 测试 ───
if __name__ == "__main__":
    test_questions = [
        "手机屏幕不亮了怎么办？",
        "帮我查一下订单12345",
        "订单12345的物流到哪了？",
        "我要退款，商品有质量问题",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"👤 {q}")
        print(f"{'='*60}")
        result = agent.invoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
        })
        # 只打印最后一条 AI 回复
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                print(f"🤖 {msg.content}")
                break
        print()