"""
translator.py — 翻譯工具模組（v2 - 桌遊術語強化版）

使用 deep-translator（Google Translate 免費 API）將任何語言的文字翻譯為繁體中文。
不需要 API Key，也不需要下載模型。

v2 改進：
- 內建桌遊專用術語詞典，確保專有名詞翻譯一致
- 支援自動偵測來源語言
- 結果後處理：修正常見的 Google Translate 誤譯
"""

import logging
import re
from functools import lru_cache

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 桌遊術語詞典（英文 → 繁體中文）
# ═══════════════════════════════════════════════════════════════════
# 這些術語會在交給 Google Translate 之前先精準替換，
# 避免 Google Translate 翻譯不一致的問題。

BOARD_GAME_GLOSSARY = {
    # 勝利／分數
    "Victory Point": "勝利點數",
    "Victory Points": "勝利點數",
    "VP": "勝利點數",
    "victory point": "勝利點數",
    "victory points": "勝利點數",
    "victory condition": "勝利條件",
    "victory conditions": "勝利條件",
    "win condition": "勝利條件",
    "win conditions": "勝利條件",
    "scoring": "計分",
    "score": "得分",
    "final score": "最終得分",
    "tie breaker": "平手判定",
    "tie-breaker": "平手判定",

    # 遊戲流程
    "turn": "回合",
    "round": "輪",
    "phase": "階段",
    "setup": "遊戲設置",
    "set up": "遊戲設置",
    "action": "行動",
    "actions": "行動",
    "move": "行動",
    "player turn": "玩家回合",
    "game round": "遊戲輪",
    "preparation phase": "準備階段",
    "action phase": "行動階段",
    "production phase": "生產階段",
    "election phase": "選舉階段",
    "scoring phase": "計分階段",

    # 資源
    "resource": "資源",
    "resources": "資源",
    "money": "金錢",
    "gold": "金幣",
    "coin": "金幣",
    "goods": "貨物",
    "food": "食物",
    "health": "健康",
    "education": "教育",
    "luxury": "奢侈品",
    "influence": "影響力",

    # 工人／階級
    "worker": "工人",
    "workers": "工人",
    "working class": "工人階級",
    "middle class": "中產階級",
    "capitalist class": "資本家階級",
    "state": "國家",
    "labor": "勞工",
    "workforce": "勞動力",
    "unemployed": "失業",
    "unemployed worker": "失業工人",
    "wage": "薪資",
    "wages": "薪資",
    "salary": "薪資",

    # 遊戲機制
    "policy": "政策",
    "policies": "政策",
    "bill": "法案",
    "bills": "法案",
    "election": "選舉",
    "vote": "投票",
    "voting": "投票",
    "legitimacy": "合法性",
    "prosperity": "繁榮",
    "tax": "稅收",
    "taxes": "稅收",
    "fiscal policy": "財政政策",
    "labor market": "勞動市場",
    "trade": "貿易",
    "production": "生產",
    "strike": "罷工",

    # 卡牌
    "action card": "行動卡",
    "action cards": "行動卡",
    "event card": "事件卡",
    "event cards": "事件卡",
    "political agenda": "政治議程",
    "export card": "出口卡",
    "export cards": "出口卡",
    "hand": "手牌",
    "discard": "棄牌",
    "draw": "抽牌",

    # 公司
    "company": "公司",
    "companies": "公司",
    "corporation": "公司",
    "public company": "上市公司",
    "private company": "私人公司",
    "subsidiary": "子公司",
    "share": "股份",

    # 版圖
    "board": "版圖",
    "game board": "遊戲版圖",
    "player board": "玩家版圖",
    "space": "空格",
    "slot": "欄位",
    "track": "軌道",
    "population track": "人口軌道",
    "political table": "政治桌",

    # 標記
    "token": "標記",
    "tokens": "標記",
    "cube": "方塊",
    "cubes": "方塊",
    "marker": "標記",
    "markers": "標記",
    "discontent marker": "不滿標記",
    "legitimacy marker": "合法性標記",
    "strike token": "罷工標記",

    # 行動類型
    "propose a bill": "提出法案",
    "exert political pressure": "施加政治壓力",
    "sell to foreign market": "銷往國外市場",
    "obtain benefits": "獲得福利",
    "pay off loan": "還清貸款",
    "hire worker": "雇用工人",
    "fire worker": "解雇工人",

    # 其他常見桌遊用語
    "player": "玩家",
    "players": "玩家",
    "opponent": "對手",
    "rule": "規則",
    "rules": "規則",
    "rulebook": "規則手冊",
    "component": "配件",
    "components": "配件",
    "dice": "骰子",
    "card": "卡片",
    "cards": "卡片",
    "deck": "牌庫",
    "pile": "牌堆",
    "market": "市場",
    "price": "價格",
    "cost": "花費",
    "effect": "效果",
    "ability": "能力",
    "bonus": "獎勵",
    "penalty": "懲罰",
    "maximum": "上限",
    "minimum": "下限",
    "limit": "限制",
    "requirement": "條件",
    "condition": "條件",
}

# 翻譯後修復規則
# Google Translate 常見的桌遊術語誤譯修正
POST_TRANSLATION_FIXES = {
    # 中文修正
    r'\bVP\b': '勝利點數',
    r'\bvps?\b': '勝利點數',
    r'工人\s*階級': '工人階級',
    r'中產\s*階級': '中產階級',
    r'資本\s*家\s*階級': '資本家階級',
    r'玩家\s*委員會': '玩家版圖',
    r'玩家\s*板': '玩家版圖',
    r'遊戲\s*板': '遊戲版圖',
    r'主機板': '遊戲版圖',
    r'行動\s*卡': '行動卡',
    r'事件\s*卡': '事件卡',
    r'政治\s*議程\s*卡': '政治議程',
    r'出口\s*卡': '出口卡',
    r'影響力\s*點數': '影響力',
    r'正統\s*性': '合法性',
    r'正統\s*標記': '合法性標記',
    r'不滿\s*標記': '不滿標記',
    r'棄牌\s*堆': '棄牌堆',
    r'抽\s*牌\s*堆': '抽牌堆',
}


def _pre_translate_terms(text: str) -> str:
    """在交給 Google Translate 之前，先將桌遊術語精準替換。

    確保專有名詞（VP、Worker、Action 等）翻譯一致。
    """
    # 按長度排序，優先替換較長的詞組（避免 "Victory Point" 被 "Point" 先取代）
    sorted_terms = sorted(BOARD_GAME_GLOSSARY.items(), key=lambda x: len(x[0]), reverse=True)

    result = text
    for eng, chinese in sorted_terms:
        # 全詞匹配（區分大小寫）
        result = re.sub(
            r'\b' + re.escape(eng) + r'\b',
            chinese,
            result,
        )

    return result


def _post_translate_fix(text: str) -> str:
    """翻譯後修正：修正 Google Translate 常見的誤譯。"""
    result = text
    for pattern, replacement in POST_TRANSLATION_FIXES.items():
        result = re.sub(pattern, replacement, result)
    return result


def _contains_chinese(text: str) -> bool:
    """檢查文字是否已包含中文字元。"""
    return any('\u4e00' <= c <= '\u9fff' for c in text)


@lru_cache(maxsize=1024)
def translate_to_traditional_chinese(text: str) -> str:
    """將任何語言的文字翻譯成繁體中文（桌遊術語強化版）。

    使用桌遊術語詞典進行預處理 + Google Translate + 後修正。

    Args:
        text: 要翻譯的文字

    Returns:
        翻譯後的繁體中文文字（若已為中文則保持原樣）
    """
    if not text or not text.strip():
        return text

    # 如果已包含中文字元，視為已是中文
    # 只做後修正，不做翻譯
    if _contains_chinese(text):
        return _post_translate_fix(text)

    try:
        # Step 1: 預處理 — 精準替換桌遊術語
        preprocessed = _pre_translate_terms(text)

        # Step 2: Google Translate 翻譯
        translator = GoogleTranslator(source='auto', target='zh-TW')
        translated = translator.translate(preprocessed)

        # Step 3: 後修正 — 修復常見誤譯
        fixed = _post_translate_fix(translated)

        logger.debug(f"🌐 翻譯完成: {len(text)} 字元 -> {len(fixed)} 字元")
        return fixed

    except Exception as e:
        logger.warning(f"⚠️ 翻譯失敗，保留原文: {e}")
        return text


def batch_translate(texts: list[str]) -> list[str]:
    """批次翻譯多段文字。

    Args:
        texts: 要翻譯的文字列表

    Returns:
        翻譯後的繁體中文文字列表
    """
    return [translate_to_traditional_chinese(t) for t in texts]