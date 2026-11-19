"""Bounded, source-preserving retrieval for unreviewed import previews."""
import re
import math


CONCEPTS = [
    ('最終', '終局', '遊戲結束', 'endgame', 'game end', 'end of the game', 'fine del gioco'),
    ('計分', '得分', 'scoring', 'score', 'punteggio'), ('貸款', 'loan', 'loans'),
    ('平手', 'tie', 'tied'), ('準備', '設置', 'setup', 'preparazione'),
    ('回合', 'turn', 'turns', 'round', 'turno'), ('行動', 'action', 'actions', 'azione', 'azioni'),
    ('階段', 'phase', 'phases', 'stage', 'stages', 'fase', 'fasi'),
    ('費用', '支付', 'cost', 'costs', 'pay', 'costo', 'pagare'),
    ('移動', 'move', 'movement', 'muovere'), ('抽牌', 'draw', 'pescare'),
    ('能力', 'ability', 'abilities', 'effetto'), ('例外', 'exception', 'exceptions', 'except'),
    ('資源', 'resource', 'resources', 'risorse'), ('結束', 'end', 'fine'),
    ('玩家輔助卡', '玩家輔助', '玩家參考卡', 'player aid', 'player aids', 'player reference card', 'player reference cards'),
    ('玩家圖板', '玩家版圖', 'player board', 'player boards'),
    ('規則輔助卡', '規則摘要卡', 'rule aid', 'rule aids', 'rules summary'),
    ('相鄰', '鄰接', 'adjacent', 'next to', 'adjoining'),
    ('政策', 'policy', 'policies'),
    ('提案', 'propose', 'proposal'),
    ('區段', 'section', 'sections'),
]


def contains(text, term):
    if re.fullmatch(r'[a-z0-9 ]+', term):
        pattern=re.escape(term)
        if re.fullmatch(r'[a-z]{3,}',term) and not term.endswith(('ss','us','is')) and not (len(term)==3 and term.endswith('s')):
            if term.endswith('ies'):
                pattern=re.escape(term[:-3])+r'(?:y|ies)'
            elif term.endswith('y'):
                pattern=re.escape(term[:-1])+r'(?:y|ies)'
            else:
                pattern=re.escape(term[:-1] if term.endswith('s') else term)+r's?'
        return bool(re.search(r'(?<![a-z0-9])' + pattern + r'(?![a-z0-9])', text.lower()))
    return term in text.lower()


def nearby_regions(source, blocks):
    """Additional context stays separate; geometry does not assert a rule dependency."""
    box = source.get('source_location', {}).get('bbox')
    if not box or len(box) != 4:
        return []
    before, after = [], []
    for block in blocks:
        other = block.get('source_location', {}).get('bbox')
        if block['id'] == source['id'] or not other or len(other) != 4:
            continue
        if any(block['fields'].get(k) != source['fields'].get(k) for k in ('來源','PDF頁碼')):
            continue
        # Text bounding boxes describe glyphs, not column width: a heading or
        # list can be much shorter than the paragraph directly above it.
        if abs(box[0]-other[0]) > 12:
            continue
        if 0 <= box[1]-other[3] <= 36:
            before.append((box[1]-other[3],block['id']))
        if 0 <= other[1]-box[3] <= 36:
            after.append((other[1]-box[3],block['id']))
    return [min(items)[1] for items in (before,after) if items]


def retrieve(blocks, question, semantic_ids=None):
    semantic_ids = list(dict.fromkeys(semantic_ids or []))[:12]
    terms = set(re.findall(r'[a-z0-9]+', question.lower()))
    terms.difference_update({'a','an','the','of','to','in','on','at','for','from','with','and','or','is','are','be','can','could','do','does','when','what','which','how','many','there','you','your','it','its','by','as','that','this'})
    for run in re.findall(r'[\u3400-\u9fff]+', question):
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    # General rulebook concepts, independent of any game's roles or page numbers.
    for aliases in CONCEPTS:
        if any(contains(question, term) for term in aliases):
            terms.update(aliases)
    corpus = [b for b in blocks if b['kind']=='規則' and b.get('evidence_usable') is not False]
    weights = {term: 1 + math.log(1 + len(corpus) / (1 + sum(contains(' '.join(b['fields'].get(k, '') for k in ('原文','內容','標題')), term) for b in corpus))) for term in terms}
    candidates = []
    seen_sources = set()
    for block in blocks:
        if block['kind'] != '規則' or block.get('layout_evidence') or block.get('evidence_usable') is False:
            continue
        fields = block['fields']
        source = fields.get('原文') or fields.get('內容', '')
        identity = (fields.get('來源'), fields.get('PDF頁碼'), ' '.join(source.split()))
        if identity in seen_sources:
            continue
        seen_sources.add(identity)
        lines = source.splitlines()
        ranked = []
        for i, line in enumerate(lines):
            score = sum(weights[term] for term in terms if contains(line, term))
            if score:
                ranked.append((score, i))
        context = block.get('chapter_context', {})
        title = ' '.join([fields.get('標題', ''), *context.get('titles', []), *context.get('parents', [])])
        title_score = sum(weights[t] for t in terms if contains(title, t))
        translation_score = sum(weights[t] for t in terms if contains(fields.get('內容', ''), t))
        if not ranked and not (title_score or translation_score) and block['id'] not in semantic_ids:
            continue
        # Keep contiguous context after the strongest heading/hit; never combine
        # disjoint fragments into an apparently continuous quotation.
        score, index = max(ranked, key=lambda item: item[0]) if ranked else (0, 0)
        score += title_score * 2 + translation_score * .5
        start = max(0, index - 1)
        excerpt = source if len(source) <= 1800 else '\n'.join(lines[start:start + 50])
        if len(excerpt) > 1800:
            # Cut only at a complete line; oversized single lines abstain.
            kept = []
            for line in lines[start:start + 50]:
                if len('\n'.join(kept + [line])) > 1800:
                    break
                kept.append(line)
            excerpt = '\n'.join(kept)
        if not excerpt.strip():
            continue
        candidates.append((score, {'id': block['id'], 'kind': '規則', 'fields': {
            '標題': fields.get('標題', ''), '原文': excerpt,
            '來源': fields.get('來源', ''), 'PDF頁碼': fields.get('PDF頁碼', ''),
            '狀態': '待確認', '限制': '檢索節錄，可能缺少其他段落、角色與例外；不可當作完整規則。'}}))
    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    if semantic_ids:
        lexical_ranks = {b['id']:i for i,(score,b) in enumerate(ranked[:12]) if score > 0}
        semantic_ranks = {identifier:i for i,identifier in enumerate(semantic_ids)}
        def fusion(item):
            identifier = item[1]['id']
            return sum(1/(60+ranks[identifier]+1) for ranks in (lexical_ranks,semantic_ranks) if identifier in ranks)
        ranked.sort(key=fusion,reverse=True)
    selected = {block['id']: block for _, block in ranked[:6]}
    originals = {b['id']: b for b in corpus}
    # Explicit extraction links only: never infer adjacency across columns or sources.
    for identifier in list(selected):
        source = originals[identifier]
        neighbors = [*source.get('source_links', {}).values(), *nearby_regions(source, corpus)]
        for neighbor_id in neighbors:
            neighbor = originals.get(neighbor_id)
            if not neighbor or neighbor_id in selected or len(selected) >= 10:
                continue
            if any(neighbor['fields'].get(k) != source['fields'].get(k) for k in ('來源', 'PDF頁碼')):
                continue
            text = neighbor['fields'].get('原文', '')
            if text and len(text) <= 1800:
                selected[neighbor_id] = {'id': neighbor_id, 'kind': '規則', 'fields': dict(neighbor['fields'])}
    pages = {(b['fields'].get('來源', ''), b['fields'].get('PDF頁碼', '')) for b in selected.values()}
    for block in corpus:
        if len(selected) >= 12:
            break
        if block.get('layout_evidence') and (block['fields'].get('來源', ''), block['fields'].get('PDF頁碼', '')) in pages:
            selected[block['id']] = {'id': block['id'], 'kind': block['kind'], 'fields': block['fields']}
    return selected
