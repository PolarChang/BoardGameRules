"""Shared title-based categories; these labels never change rule content."""
import re
import unicodedata

TAXONOMY = [
    ('overview', '遊戲概覽與配件', r'遊戲目標|遊戲概覽|配件|組件|\boverview\b|\bcomponents?\b|\bcontents\b|\bobjective\b|\bintroduction\b'),
    ('setup', '遊戲準備', r'設置|準備|\bset[ -]?up\b|\bpreparation\b'),
    ('turns', '回合與階段', r'回合|階段|輪替|流程|\bturns?\b|\bphases?\b|\brounds?\b|\bsequence\b'),
    ('actions', '行動與操作', r'行動|移動|建造|交易|戰鬥|\bactions?\b|\bmovement\b|\bcombat\b|\bbuilding\b|\btrading\b'),
    ('resources', '資源與費用', r'資源|費用|成本|收入|支付|消耗|上限|\bresources?\b|\bcosts?\b|\bincome\b|\bpayment\b|\blimits?\b'),
    ('abilities', '卡牌與特殊能力', r'卡牌|能力|效果|\bcards?\b|\babilities\b|\bability\b|\beffects?\b'),
    ('endgame', '結束與計分', r'結束|勝負|計分|平手|\bend[ -]?game\b|\bgame end\b|\bend of the game\b|\bscoring\b|\bwinning\b|\btie.?break'),
    ('exceptions', '例外與變體', r'例外|變體|特殊情況|不同人數|\bexceptions?\b|\bvariants?\b|\boptional\b|\bsolo\b|\bFAQ\b'),
    ('unassigned', '尚未分類', r'(?!)'),
]


def match_title(title):
    text = unicodedata.normalize('NFKC', title)
    international = {
        'overview': r'\b(?:introduzione|componenti|obiettivo del gioco|panoramica|game board|player boards?|the theater|monsters|rooms|doors)\b',
        'setup': r'\bpreparazione iniziale\b',
        'turns': r'\b(?:svolgimento|pianificazione|fase azioni|round successivo|gameplay|turn steps)\b',
        'actions': r'\b(?:move|play a card|adding|acquistare|utilizza|piazza|abbandonare)\b',
        'resources': r'\b(?:costi?|risorsa|risorse|loans?|prices|wages|taxes)\b',
        'abilities': r'\b(?:carte|tessere bonus)\b',
        'endgame': r'\bfine del gioco\b',
        'exceptions': r'\b(?:varianti|casi particolari|casi speciali|campagna e moduli|golden rules)\b',
    }
    return [identifier for identifier, _, pattern in TAXONOMY
            if re.search(pattern, text, re.I) or re.search(international.get(identifier, r'(?!)'), text, re.I)]
