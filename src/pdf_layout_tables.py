"""Conservative numeric row alignment from PDF word coordinates, without inference."""
import hashlib
import json
import re
import fitz

ENGINE = 'numeric-layout-v1'


def table_answer(blocks, question):
    """Explicit source-table browsing only; never infer a scoring rule or role."""
    if '來源座標表格' not in question:
        return None
    from src.preview_retrieval import retrieve
    selected = retrieve(blocks, question)
    tables = [b for b in blocks if b.get('layout_evidence') and b['id'] in selected]
    if not tables:
        return None
    return {
        'kind': 'source_table', 'provider': 'deterministic',
        'answer': '以下為相關頁面自動重建的來源表格，保留原文標籤；只列出表格，不代表完整計分規則。\n\n' + '\n\n'.join(
            f"PDF 第 {b['fields']['PDF頁碼']} 頁\n{b['fields']['原文']}" for b in tables),
        'citations': [b['id'] for b in tables],
        'sources': [{'id': b['id'], 'source': b['fields'].get('來源', ''), 'page': b['fields']['PDF頁碼']} for b in tables],
    }


def tables_on_page(page):
    words = [w for w in page.get_text('words') if str(w[4]).strip()]
    numeric = sorted([w for w in words if re.fullmatch(r'\d+(?:\.\d+)?', w[4])], key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows = []
    for word in numeric:
        center = (word[1] + word[3]) / 2
        row = next((r for r in rows if abs(r['y'] - center) <= 2), None)
        if row is None:
            row = {'y': center, 'words': []}
            rows.append(row)
        row['words'].append(word)
    candidates = []
    for row in rows:
        ordered = sorted(row['words'], key=lambda w: w[0])
        # Contiguous runs with a uniform column pitch; ordinary prose is excluded.
        for start in range(len(ordered)):
            run = ordered[start:]
            if len(run) < 4:
                continue
            pitch = (run[1][0] + run[1][2] - run[0][0] - run[0][2]) / 2
            if not 14 <= pitch <= 100:
                continue
            kept = run[:2]
            for word in run[2:]:
                gap = (word[0] + word[2] - kept[-1][0] - kept[-1][2]) / 2
                if abs(gap - pitch) > 3:
                    break
                kept.append(word)
            if len(kept) >= 4 and not any(set(tuple(w) for w in kept) <= set(tuple(w) for w in r['words']) for r in candidates):
                candidates.append({'y': row['y'], 'words': kept})
    result, used = [], set()
    for i, upper in enumerate(candidates):
        if i in used:
            continue
        for j, lower in enumerate(candidates):
            if j <= i or j in used or not 12 <= lower['y'] - upper['y'] <= 45 or len(upper['words']) != len(lower['words']):
                continue
            if any(abs((a[0] + a[2] - b[0] - b[2]) / 2) > 4 for a, b in zip(upper['words'], lower['words'])):
                continue
            rows_out = []
            cells = upper['words'] + lower['words']
            for row in (upper, lower):
                left = row['words'][0][0]
                labels = sorted([w for w in words if left - 75 <= w[2] < left - 5 and w[2] - w[0] <= 75 and abs((w[1] + w[3]) / 2 - row['y']) <= 11 and abs((w[3] - w[1]) - (row['words'][0][3] - row['words'][0][1])) < .8 and not re.fullmatch(r'\d+(?:\.\d+)?', w[4])], key=lambda w: (round(w[1] / 4), w[0]))
                if not labels:
                    break
                cells.extend(labels)
                rows_out.append([' '.join(w[4] for w in labels), *[w[4] for w in row['words']]])
            if len(rows_out) != 2:
                continue
            box = [min(w[0] for w in cells), min(w[1] for w in cells), max(w[2] for w in cells), max(w[3] for w in cells)]
            # Reject cells from merged prose: each numeric cell must contain only
            # its expected token within its original word rectangle.
            if any(page.get_textbox(fitz.Rect(w[:4])).strip() != w[4] for w in upper['words'] + lower['words']):
                continue
            result.append({'rows': rows_out, 'bbox': box,
                           'cells': [{'text': w[4], 'bbox': list(w[:4])} for w in cells]})
            used.update((i, j))
            break
    return result


def supplement(data):
    digest = hashlib.sha256(data).hexdigest()
    blocks = []
    with fitz.open(stream=data, filetype='pdf') as pdf:
        for number, page in enumerate(pdf, 1):
            for index, table in enumerate(tables_on_page(page), 1):
                text = '\n'.join(' | '.join(row) for row in table['rows'])
                blocks.append({'kind': '規則', 'id': f'LAYOUT-P{number:03d}-T{index:03d}', 'fields': {
                    '標題': '來源座標表格：' + table['rows'][0][0], '大項': '其他規則',
                    '內容': text, '原文': text, 'PDF頁碼': str(number), '來源': 'LOCAL-SOURCE',
                    '狀態': '待確認', '條件': 'PDF 文字座標重建，僅保留兩列對齊的數字表格；不推論未顯示的列。',
                    '例外': '不辨識資源圖示，不代表此頁其他內容皆正確。'},
                    'layout_evidence': {'engine': ENGINE, 'source_digest': digest, **table}})
    return blocks


def cached_supplement(job_id, source_key, digest):
    from src.database import _connect
    from src.editorial_visuals import read_bytes
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS import_layout_sources (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        key = hashlib.sha256(f'{job_id}:{digest}:{ENGINE}'.encode()).hexdigest()
        cur.execute('SELECT payload FROM import_layout_sources WHERE id=?', (key,))
        row = cur.fetchone()
        conn.commit()
        if row:
            return json.loads(row['payload'])
        data = read_bytes(source_key)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Source digest mismatch')
        blocks = supplement(data)
        cur.execute('INSERT INTO import_layout_sources VALUES (?,?) ON CONFLICT (id) DO NOTHING', (key, json.dumps(blocks, ensure_ascii=False)))
        conn.commit()
        return blocks
    finally:
        conn.close()
