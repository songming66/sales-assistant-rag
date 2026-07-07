"""
v5.1 FastAPI 主入口
基于 LangChain 的电商智能客服 RAG 系统 + Agent 模式
"""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

from config import DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, APP_HOST, APP_PORT, validate
from knowledge_base import build_documents
from vector_store import VectorStore
from llm import call_deepseek, fallback_answer

app = FastAPI(
    title="电商智能客服助手",
    description="基于 LangChain + Chroma + DeepSeek 的电商客服 RAG 系统",
    version="5.1"
)

# ─── 启动时初始化向量库 ───
store = VectorStore()
if store.is_empty():
    docs = build_documents()
    store.add_documents(docs)


# ═══════════════════════════════════════════════
# Agent 模式（v5.1）—— 4 工具智能客服
# ═══════════════════════════════════════════════
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

MOCK_ORDERS = {
    "12345": "订单12345｜商品：iPhone 15 Pro Max｜金额：¥9,999｜状态：已发货｜下单时间：7月3日",
    "12346": "订单12346｜商品：AirPods Pro｜金额：¥1,899｜状态：备货中｜下单时间：7月5日",
    "12347": "订单12347｜商品：MacBook Pro 14｜金额：¥14,999｜状态：已完成｜下单时间：6月28日",
    "12348": "订单12348｜商品：iPad Air｜金额：¥5,499｜状态：已取消｜下单时间：7月1日",
}
MOCK_LOGISTICS = {
    "12345": "顺丰快递 SF1234567890｜当前位置：杭州分拣中心｜预计送达：明天",
    "12346": "中通快递 ZT9876543210｜当前位置：广州转运中心｜预计送达：7月9日",
    "12347": "已签收｜签收人：本人｜签收时间：7月1日 14:30",
    "12348": "无物流信息（订单已取消）",
}


@tool
def agent_search_knowledge(query: str) -> str:
    """搜索电商客服知识库，用于退换货、使用问题等场景。参数 query: 问题关键词"""
    docs, scores = store.search(query, top_k=3)
    if not docs:
        return "知识库中没有找到相关案例。"
    return "\n\n".join(
        f"【案例{i+1}】相关度 {s:.0%}\n{doc.strip()[:200]}"
        for i, (doc, s) in enumerate(zip(docs, scores))
    )


@tool
def agent_check_order(order_id: str) -> str:
    """查询订单状态。参数 order_id: 订单号，如12345"""
    return MOCK_ORDERS.get(order_id, f"未找到订单 {order_id}，请确认订单号。")


@tool
def agent_check_logistics(order_id: str) -> str:
    """查询物流进度。参数 order_id: 订单号，如12345"""
    info = MOCK_LOGISTICS.get(order_id)
    return f"📦 {info}" if info else f"未找到订单 {order_id} 的物流信息。"


@tool
def agent_transfer_human(reason: str) -> str:
    """转人工客服。参数 reason: 转接原因"""
    return f"✅ 已转接人工客服，原因：{reason}，预计等待约2分钟。"


AGENT_SYSTEM_PROMPT = """你是小蜜，电商智能客服。规则：
1. 退换货、使用问题 → 先调 search_knowledge
2. 订单状态 → 调 check_order
3. 物流进度 → 调 check_logistics
4. 退款/投诉/超出能力 → 调 transfer_human
5. 回答亲切具体，200字以内"""

agent_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.3,
    max_tokens=800,
)
agent_tools = [agent_search_knowledge, agent_check_order, agent_check_logistics, agent_transfer_human]
agent = create_react_agent(model=agent_llm, tools=agent_tools)


# ─── RAG 请求/响应模型 ───
class Question(BaseModel):
    query: str
    top_k: int = 3


class Answer(BaseModel):
    question: str
    relevant_docs: list
    scores: list
    answer: str
    source: str
    vector_db: str = "chroma"
    framework: str = "langchain"


# ─── Agent 请求/响应模型 ───
class AgentQuestion(BaseModel):
    query: str


class AgentStep(BaseModel):
    type: str       # action / observation / answer
    content: str
    tool: str | None = None
    tool_input: str | None = None


class AgentAnswer(BaseModel):
    question: str
    answer: str
    steps: list[AgentStep]


# ─── RAG API ───
@app.post("/ask", response_model=Answer)
def ask(req: Question):
    """客服问题问答：LangChain 检索 → DeepSeek 生成"""
    docs, scores = store.search(req.query, req.top_k)

    ai_answer, status = call_deepseek(req.query, docs, scores)
    if ai_answer:
        return Answer(
            question=req.query,
            relevant_docs=docs,
            scores=scores,
            answer=ai_answer,
            source="deepseek"
        )

    fb_answer, fb_status = fallback_answer(req.query, docs, scores)
    return Answer(
        question=req.query,
        relevant_docs=docs,
        scores=scores,
        answer=fb_answer,
        source=fb_status
    )


@app.post("/reload")
def reload_knowledge():
    """清空知识库（重启服务后自动重建）"""
    store.clear()
    return {"status": "cleared", "message": "重启服务后会自动重新灌入知识"}


@app.get("/")
def home():
    return {
        "name": "电商智能客服助手",
        "version": app.version,
        "status": "running",
        "framework": "LangChain + Chroma + DeepSeek",
        "vector_db": "Chroma",
        "embedding": "all-MiniLM-L6-v2 (384维)",
        "knowledge_count": store.count(),
        "llm": f"DeepSeek-{DEEPSEEK_MODEL}",
        "endpoints": {
            "POST /ask": "RAG 客服问答",
            "POST /agent_ask": "Agent 智能客服（4工具）",
            "POST /reload": "清空知识库",
            "GET /knowledge": "查看知识库",
            "GET /docs": "Swagger API文档",
        }
    }


@app.get("/knowledge")
def show_knowledge():
    """查看知识库前20条"""
    return {"total": store.count(), "sample": store.get_all(limit=20)}


# ─── Agent API ───
@app.post("/agent_ask", response_model=AgentAnswer)
def agent_ask(req: AgentQuestion):
    """Agent 模式：LLM 自主选择工具，返回完整思考过程"""
    result = agent.invoke({
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": req.query},
        ]
    })
    steps = []
    for msg in result["messages"]:
        cls = msg.__class__.__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append(AgentStep(
                    type="action",
                    content=f"🔧 调用 {tc['name']}",
                    tool=tc["name"],
                    tool_input=str(tc.get("args", {}))
                ))
        elif cls == "ToolMessage":
            steps.append(AgentStep(
                type="observation",
                content=str(msg.content)[:500]
            ))
        elif cls == "AIMessage" and hasattr(msg, "content") and msg.content:
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                steps.append(AgentStep(
                    type="answer",
                    content=msg.content
                ))

    final = next((s.content for s in reversed(steps) if s.type == "answer"), "处理完成")
    return AgentAnswer(question=req.query, answer=final, steps=steps)


if __name__ == "__main__":
    validate()
    import uvicorn
    print(f"\n🚀 启动服务: http://localhost:{APP_PORT}")
    print(f"📖 API文档:   http://localhost:{APP_PORT}/docs")
    print('🧪 RAG 测试:  curl -X POST http://localhost:8000/ask \\\n'
          '        -H "Content-Type: application/json" \\\n'
          '        -d \'{"query":"手机屏幕不亮了怎么办"}\'')
    print('🧪 Agent 测试: curl -X POST http://localhost:8000/agent_ask \\\n'
          '        -H "Content-Type: application/json" \\\n'
          '        -d \'{"query":"查一下订单12345的物流"}\'')
    print()
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)