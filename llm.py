"""
DeepSeek LLM 调用 + Fallback
"""
from __future__ import annotations
import requests

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_TIMEOUT


SYSTEM_PROMPT = """你是"小白"，一位有5年白酒销售经验的智能销售助手。你的任务：
1. 严格基于下方"知识库"内容回答，不编造信息
2. 回答要专业、具体、可操作，避免空话
3. 如果知识库没有相关信息，明确告诉用户并建议换个问法
4. 语气亲切务实，像资深销售老哥分享经验
5. 回答控制在200字以内，分点说明更易读"""


def _build_context(chunks: list, scores: list) -> str:
    if not chunks:
        return "（知识库中没有找到相关内容）"
    return "\n\n".join([
        f"【知识{i+1}（相关度{score:.0%}）】\n{doc.strip()}"
        for i, (doc, score) in enumerate(zip(chunks, scores))
    ])


def call_deepseek(question: str, context_chunks: list, scores: list) -> tuple[str | None, str]:
    """
    调用 DeepSeek 生成回答
    返回: (回答内容, 状态)  状态: deepseek / timeout / error
    """
    context = _build_context(context_chunks, scores)
    user_prompt = f"""【知识库】
{context}

【销售问题】
{question}

请基于知识库给出你的建议："""

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 500
            },
            timeout=DEEPSEEK_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"], "deepseek"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"❌ DeepSeek调用失败: {e}")
        return None, "error"
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return None, "error"


def fallback_answer(question: str, docs: list, scores: list) -> tuple[str, str]:
    """DeepSeek 不可用时的降级回答"""
    if not docs:
        return (
            f"❌ 没找到和「{question}」相关的销售知识，DeepSeek也暂不可用。\n"
            f"试试：客户嫌贵怎么办、报价后多久跟进、怎么给客户分级。",
            "fallback_no_match"
        )
    suggestion = f"⚠️ DeepSeek暂不可用，知识库检索结果（相关度{scores[0]:.0%}起）：\n\n"
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        suggestion += f"━━━ 【相关知识 {i}】相关度 {score:.2%} ━━━\n{doc.strip()}\n\n"
    return suggestion, "fallback_retrieval"
