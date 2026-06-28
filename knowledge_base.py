"""
销售知识库 - 集中管理，方便扩展
未来要加新知识直接在这里追加，重启服务即可生效
"""

SALES_KNOWLEDGE = """
白酒销售核心话术

一、客户开发阶段
1. 初次拜访：先送小样品试饮，建立信任关系
2. 需求挖掘：询问客户饮酒习惯、用酒场景、预算范围
3. 产品推荐：根据客户场景推荐对应档次（婚宴/商务/自饮）

二、异议处理
- 客户嫌价格太贵：强调性价比（每场宴席均摊成本）、品牌价值
- 客户担心质量：介绍生产工艺、第三方检测报告
- 客户已有固定供应商：不急着否定，先问对方需求痛点
- 客户说再考虑：约定具体回访时间，不要说"随时联系"

三、跟进节奏
- 首次接触：3天内回访
- 已发样品：1周内问反馈
- 报价后：3天内跟进决策
- 节日旺季前：提前2周开始重点客户拜访

四、促单技巧
1. 限时优惠：月底冲量节点谈政策
2. 赠品策略：配酒杯、酒具等周边
3. 案例分享：同类型客户成功案例（婚宴酒店/商务接待）
4. 客户见证：让老客户带新客户，介绍成功合作

五、客户分级
- A类：月采购量大、回款及时，重点维护
- B类：潜力客户，每月至少1次拜访
- C类：暂维护，节假日群发祝福即可
"""


def split_knowledge(text: str, chunk_size: int = 200, overlap: int = 40) -> list:
    """按段落切分文档"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > chunk_size and current:
            chunks.append(current)
            current = current[-overlap:] + p if overlap > 0 else p
        else:
            current = current + "\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks
