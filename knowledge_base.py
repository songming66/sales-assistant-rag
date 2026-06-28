"""
v5.0 知识库 - 从电商客服对话数据集加载
数据源：modelscope xuri2004/dianshang_dataset
每条数据是一组多轮对话（user/assistant 交替）
"""
from __future__ import annotations
import json
import os
from typing import List, Dict

DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "dianshang_sample.json")
)


def load_conversations(json_path: str = DATA_PATH) -> List[Dict]:
    """
    加载电商客服对话数据集
    支持两种格式：
      - .json  : 一个 JSON 数组，元素为 {conversations, origin}
      - .jsonl : 每行一个 JSON 对象 {conversations, origin}
    返回：[{"conversations": [...], "origin": "..."}]
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"❌ 数据集文件不存在: {json_path}\n"
            f"请确认 data/dianshang_sample.json 是否在项目根目录"
        )

    raw_items = []
    if json_path.endswith(".jsonl"):
        # JSONL 格式：每行一个 JSON
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))
    else:
        # JSON 格式：一个数组
        with open(json_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

    # 解析 conversations 字段（字符串 → 列表）
    parsed = []
    for item in raw_items:
        try:
            convs = json.loads(item["conversations"])
            if convs and isinstance(convs, list):
                parsed.append({
                    "conversations": convs,
                    "origin": item.get("origin", "[]"),
                })
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️  跳过一条异常数据: {e}")
            continue

    print(f"✅ 已加载 {len(parsed)} 组客服对话")
    return parsed


def conversation_to_text(conv: Dict) -> str:
    """
    把一组对话格式化成可检索的文本
    格式：用户问 / 客服答 交替显示，保留多轮上下文
    """
    lines = []
    for turn in conv["conversations"]:
        role = "客户" if turn["from"] == "user" else "客服"
        lines.append(f"{role}：{turn['value']}")
    return "\n".join(lines)


def build_documents(json_path: str = DATA_PATH) -> List[Dict]:
    """
    构建向量库用的文档列表
    返回：[{"content": "...对话文本...", "metadata": {"conv_id": 0, "turns": 4}}, ...]
    """
    convs = load_conversations(json_path)
    docs = []
    for i, conv in enumerate(convs):
        content = conversation_to_text(conv)
        docs.append({
            "content": content,
            "metadata": {
                "conv_id": i,
                "turns": len(conv["conversations"]),
                "source": "dianshang_dataset",
            }
        })
    print(f"✅ 已构建 {len(docs)} 个对话文档")
    return docs


# 模块测试
if __name__ == "__main__":
    docs = build_documents()
    print(f"\n=== 样例文档（第 1 条）===")
    print(docs[0]["content"])
    print(f"\n元数据: {docs[0]['metadata']}")
