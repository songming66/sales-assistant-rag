"""
API 自动化测试
运行: python tests/test_api.py
需要先启动服务: python main.py
"""
import requests
import sys

BASE_URL = "http://127.0.0.1:8000"  # 用 IP 避免 IPv6 解析问题

TEST_CASES = [
    {"query": "客户嫌贵怎么办", "expect_match": True},
    {"query": "报价后多久跟进", "expect_match": True},
    {"query": "怎么给客户分级", "expect_match": True},
    {"query": "客户总说再考虑", "expect_match": True},
    {"query": "怎么让老客户介绍新客户", "expect_match": True},
    {"query": "今天天气怎么样", "expect_match": False},  # 知识库外
]


def test_health():
    print("\n=== 测试 / 健康检查 ===")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "running"
    print(f"✅ 服务运行中，向量库 {data['knowledge_count']} 条知识")


def test_ask():
    print("\n=== 测试 / 问答接口 ===")
    passed = 0
    for tc in TEST_CASES:
        r = requests.post(
            f"{BASE_URL}/ask",
            json={"query": tc["query"], "top_k": 3}
        )
        if r.status_code != 200:
            print(f"❌ {tc['query']} - HTTP {r.status_code}")
            continue

        data = r.json()
        has_match = len(data["relevant_docs"]) > 0
        expectation_met = (has_match == tc["expect_match"])

        status = "✅" if expectation_met else "⚠️ "
        print(f"{status} {tc['query']}")
        print(f"   来源: {data['source']}, 命中: {len(data['relevant_docs'])} 条")
        if data["scores"]:
            print(f"   最高相关度: {data['scores'][0]:.2%}")
        print(f"   回答预览: {data['answer'][:80]}...")
        if expectation_met:
            passed += 1
    print(f"\n通过: {passed}/{len(TEST_CASES)}")


def test_knowledge():
    print("\n=== 测试 / 知识库端点 ===")
    r = requests.get(f"{BASE_URL}/knowledge")
    assert r.status_code == 200
    data = r.json()
    print(f"✅ 知识库共 {data['total']} 条，返回 {len(data['sample'])} 条样本")


if __name__ == "__main__":
    print("🧪 开始 API 测试")
    try:
        test_health()
        test_ask()
        test_knowledge()
        print("\n🎉 所有测试完成！")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 连不上服务 {BASE_URL}，请先 python main.py 启动")
        sys.exit(1)
