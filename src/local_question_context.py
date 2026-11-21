"""Recognize requests to observe live state that a rulebook cannot contain.

This does not classify rule questions or infer a game state. Ambiguous questions
continue to the ordinary evidence pipeline.
"""
import re
import json
from pydantic import BaseModel, ConfigDict, Field


class ConfirmedContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: str = Field(default='', max_length=120)
    players: str = Field(default='', max_length=40)
    stage: str = Field(default='', max_length=200)
    situation: str = Field(default='', max_length=500)


def context_query(question, confirmed_context):
    """User premises can aid retrieval, but are never added as source blocks."""
    context=ConfirmedContext.model_validate(confirmed_context or {}).model_dump()
    context={key:value.strip() for key,value in context.items() if value.strip()}
    return question+'\n使用者確認的情境（不是規則證據）：'+json.dumps(context,ensure_ascii=False) if context else question


def missing_observation(question):
    text=question.strip().lower()
    # Explicitly supplied state belongs to a rule query, not an observation.
    if re.search(r'(?:手牌|手上|棋子|分數|得分)(?:目前|現在)?(?:是|為|有|在)\s*[^哪幾多什甚？?，,。;；]{2,}',text):
        return []
    if re.search(r'(?:my hand is|i (?:currently )?have \d|my (?:score|position) is)',text):
        return []
    if re.search(r'(?:可以|能否|可否|應該|如何|怎麼|抽|棄|交換|draw|discard|may i|can i)',text):
        return []
    personal=bool(re.search(r'我|my|\bi\b',text))
    if not personal: return []
    if re.search(r'(?:手上|手牌)',text) and re.search(r'哪些|幾張|清單|列出|什麼|甚麼',text):
        return ['請提供目前手牌；規則書無法得知實際持有的牌。']
    if re.search(r'(?:which|what|list|how many).*cards.*(?:my hand|i (?:currently )?(?:have|hold))',text):
        return ['請提供目前手牌；規則書無法得知實際持有的牌。']
    if re.search(r'現在|目前|current|right now',text):
        if re.search(r'(?:棋子|角色).*(?:哪裡|哪一格|位置)|where.*(?:piece|pawn|token)',text):
            return ['請提供目前棋盤位置或局面；規則書無法觀察現場狀態。']
        if re.search(r'(?:分數|得分|資源).*(?:多少|幾點|幾個)|how many.*(?:points|resources).*(?:have|my)',text):
            return ['請提供目前分數或資源；規則書無法得知實際遊戲紀錄。']
    return []
