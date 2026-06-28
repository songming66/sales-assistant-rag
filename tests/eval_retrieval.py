"""
v5.0 检索质量评估脚本
指标：Top-K 命中率（人工构造测试集 + 关键词匹配）
"""
from __future__ import annotations
import json
from typing import List, Dict

from knowledge_base import build_documents
from vector_store import VectorStore


# ─── 人工标注的测试集 ───
# 每条：{question, expected_keywords}  expected_keywords 是判断命中的关键词
TEST_CASES = [
    {"q": "手机屏幕不亮了", "kw": ["屏幕", "电源键"]},
    {"q": "蓝牙耳机断连", "kw": ["蓝牙", "配对"]},
    {"q": "收到的商品颜色不对", "kw": ["颜色", "换"]},
    {"q": "鞋底开胶了", "kw": ["鞋", "质量"]},
    {"q": "衣服掉色", "kw": ["掉色", "质量"]},
    {"q": "充电很慢", "kw": ["充电"]},
    {"q": "包装破损", "kw": ["包裹", "损坏"]},
    {"q": "订单号忘了", "kw": ["订单"]},
    {"q": "平板自动重启", "kw": ["重启", "平板"]},
    {"q": "运动鞋鞋带断了", "kw": ["鞋带"]},
]


def evaluate_hit_at_k(test_cases: List[Dict], store: VectorStore, k: int = 3) -> float:
    """
    计算 Top-K 命中率
    命中定义：检索到的 k 条文档中，至少有一条包含所有 expected_keywords
    """
    hit = 0
    print(f"\n{'='*60}")
    print(f"Top-{k} 检索评估（共 {len(test_cases)} 个测试用例）")
    print(f"{'='*60}\n")

    for i, case in enumerate(test_cases, 1):
        docs, scores = store.search(case["q"], top_k=k)
        if not docs:
            print(f"[{i:2d}] ❌ 无结果 | Q: {case['q']}")
            continue

        # 判断是否命中
        is_hit = False
        best_doc = None
        for doc in docs:
            if all(kw in doc for kw in case["kw"]):
                is_hit = True
                best_doc = doc
                break

        icon = "✅" if is_hit else "❌"
        top_score = scores[0] if scores else 0
        print(f"[{i:2d}] {icon} 命中={is_hit} Top1={top_score:.1%} | Q: {case['q']}")
        if not is_hit:
            print(f"     期望关键词: {case['kw']}")
            print(f"     实际返回: {docs[0][:60]}...")
        if is_hit:
            hit += 1

    rate = hit / len(test_cases)
    print(f"\n{'='*60}")
    print(f"📊 命中率: {hit}/{len(test_cases)} = {rate:.1%}")
    print(f"{'='*60}\n")
    return rate


if __name__ == "__main__":
    print("🚀 启动检索评估...")
    store = VectorStore()

    # 如果向量库是空的，先灌入数据
    if store.is_empty():
        print("⚠️  向量库为空，先灌入数据...")
        docs = build_documents()
        store.add_documents(docs)

    # Top-1, 3, 5 三档评估
    rate_1 = evaluate_hit_at_k(TEST_CASES, store, k=1)
    rate_3 = evaluate_hit_at_k(TEST_CASES, store, k=3)
    rate_5 = evaluate_hit_at_k(TEST_CASES, store, k=5)

    print("📈 综合结果：")
    print(f"  Top-1 命中率: {rate_1:.1%}")
    print(f"  Top-3 命中率: {rate_3:.1%}")
    print(f"  Top-5 命中率: {rate_5:.1%}")
