"""
v5.0 LLM 调用 - 基于 LangChain + DeepSeek（OpenAI 兼容模式）
带 Fallback 机制
"""
from __future__ import annotations
from typing import Tuple

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_TIMEOUT


SYSTEM_PROMPT = """你是"小蜜"，一位经验丰富的电商智能客服助手。你的任务：

1. 严格基于下方"历史客服对话案例"回答，参考真实客服的处理方式
2. 回答要专业、亲切、有同理心，先安抚客户情绪
3. 给出具体可操作的解决步骤，避免空话
4. 如果案例中没有相关信息，明确告诉用户并建议换个问法
5. 回答控制在200字以内，分点说明更易读"""


USER_PROMPT_TEMPLATE = """【历史客服对话案例】
{context}

【客户当前问题】
{question}

请参考上述案例的处理方式，给出你的回复："""


def _build_context(chunks: list, scores: list) -> str:
    if not chunks:
        return "（知识库中没有找到相关案例）"
    return "\n\n---\n\n".join([
        f"【案例{i+1}】（相关度 {score:.0%}）\n{doc.strip()}"
        for i, (doc, score) in enumerate(zip(chunks, scores))
    ])


def get_llm() -> ChatOpenAI:
    """获取 LangChain 封装的 DeepSeek"""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.5,
        max_tokens=500,
        request_timeout=DEEPSEEK_TIMEOUT,
    )


def call_deepseek(question: str, context_chunks: list, scores: list) -> Tuple[str | None, str]:
    """
    调用 DeepSeek 生成回答
    返回: (回答内容, 状态)  状态: deepseek / timeout / error
    """
    try:
        llm = get_llm()
        context = _build_context(context_chunks, scores)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
        ])
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        return response.content, "deepseek"
    except Exception as e:
        err_type = type(e).__name__
        if "Timeout" in err_type or "timeout" in str(e).lower():
            print(f"⏱️  DeepSeek 调用超时: {e}")
            return None, "timeout"
        print(f"❌ DeepSeek 调用失败 [{err_type}]: {e}")
        return None, "error"


def fallback_answer(question: str, docs: list, scores: list) -> Tuple[str, str]:
    """DeepSeek 不可用时的降级回答"""
    if not docs:
        return (
            f"❌ 没找到和「{question}」相关的客服案例，DeepSeek 也暂不可用。\n\n"
            f"💡 试试这些问题：\n"
            f"  • 手机屏幕不亮怎么办\n"
            f"  • 收到商品颜色不对怎么处理\n"
            f"  • 商品有质量问题如何退换",
            "fallback_no_match"
        )
    suggestion = f"⚠️ DeepSeek 暂不可用，以下是知识库中检索到的相似案例（相关度 {scores[0]:.0%} 起）：\n\n"
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        suggestion += f"━━━ 【案例 {i}】相关度 {score:.2%} ━━━\n{doc.strip()}\n\n"
    return suggestion, "fallback_retrieval"
