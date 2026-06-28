"""
FastAPI 主入口
"""
from fastapi import FastAPI
from pydantic import BaseModel

from config import DEEPSEEK_MODEL, APP_HOST, APP_PORT
from knowledge_base import SALES_KNOWLEDGE, split_knowledge
from vector_store import VectorStore
from llm import call_deepseek, fallback_answer

app = FastAPI(
    title="智能销售助手",
    description="基于RAG的销售知识问答，Chroma向量库+DeepSeek生成",
    version="4.0"
)

# ─── 启动时初始化向量库 ───
store = VectorStore()
if store.is_empty():
    chunks = split_knowledge(SALES_KNOWLEDGE)
    print(f"   文档切分为 {len(chunks)} 个片段")
    store.add_documents(chunks)


# ─── 请求/响应模型 ───
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


# ─── API ───
@app.post("/ask", response_model=Answer)
def ask(req: Question):
    """销售问题问答：检索→生成"""
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
        "name": "智能销售助手",
        "version": app.version,
        "status": "running",
        "vector_db": "Chroma",
        "embedding": "all-MiniLM-L6-v2 (ONNX, 384维)",
        "knowledge_count": store.count(),
        "llm": f"DeepSeek-{DEEPSEEK_MODEL}",
        "endpoints": {
            "POST /ask": "销售问题问答",
            "POST /reload": "清空知识库",
            "GET /knowledge": "查看知识库",
            "GET /docs": "Swagger API文档"
        }
    }


@app.get("/knowledge")
def show_knowledge():
    """查看知识库前20条"""
    return {"total": store.count(), "sample": store.get_all(limit=20)}


if __name__ == "__main__":
    validate()  # 启动前检查 API Key
    import uvicorn
    print(f"\n🚀 启动服务: http://localhost:{APP_PORT}")
    print("📖 API文档: http://localhost:8000/docs")
    print('🧪 测试: curl -X POST http://localhost:8000/ask \\\n'
          '        -H "Content-Type: application/json" \\\n'
          '        -d \'{"query":"客户嫌贵怎么办"}\'')
    print()
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
