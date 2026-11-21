"""Immutable native-text fallback independent of translation worker progress."""
import hashlib
import json

import fitz

from src.database import _connect
from src.editorial_visuals import read_bytes
from src.source_segments import split_source

ENGINE = 'native-source-v2'


def source_pages(job, max_new_pages=None):
    key = hashlib.sha256(f"{job['id']}:{job['digest']}:{ENGINE}".encode()).hexdigest()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS import_native_sources (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        cur.execute('SELECT payload FROM import_native_sources WHERE id=?', (key,))
        cached = cur.fetchone()
        conn.commit()
        if cached:
            return json.loads(cached['payload'])
        data = read_bytes(job['source_key'])
        if hashlib.sha256(data).hexdigest() != job['digest']:
            raise ValueError('Source digest mismatch')
        pages = {}
        with fitz.open(stream=data, filetype='pdf') as pdf:
            processed = 0
            complete = True
            for number, page in enumerate(pdf, 1):
                page_key = key + ':' + str(number)
                cur.execute('SELECT payload FROM import_native_sources WHERE id=?', (page_key,))
                saved = cur.fetchone()
                if saved:
                    result = json.loads(saved['payload'])
                    if result['rules']:
                        pages[str(number)] = result
                    continue
                if max_new_pages is not None and processed >= max_new_pages:
                    complete = False
                    continue
                rules = []
                text_blocks = [(index, block) for index, block in enumerate(page.get_text('blocks', sort=True))
                               if len(block) > 6 and block[6] == 0 and block[4].strip()]
                # Append formerly omitted numbers so existing text IDs stay stable.
                # Keep physical reading order separately; numbers are not assumed
                # to be page numbers or assigned table meanings without evidence.
                text_blocks.sort(key=lambda item: item[1][4].strip().isdigit())
                for order, block in text_blocks:
                    text = block[4]
                    for segment in split_source(text, 1600):
                        rules.append({'title': f'PDF 第 {number} 頁原文 {len(rules) + 1}',
                                      'original': segment['text'], 'translation': '',
                                      'category': '其他規則',
                                      'source_location': {'bbox': list(block[:4]), 'granularity': 'text_blocks',
                                                          'reading_order': order, 'text_start': segment['start'],
                                                          'text_end': segment['end']},
                                      'extraction': ENGINE})
                result = {'rules': rules, 'warnings': ['原文先行擷取，尚在分類及翻譯；圖片與閱讀順序可能尚未完整處理。']}
                cur.execute('INSERT INTO import_native_sources VALUES (?,?) ON CONFLICT (id) DO NOTHING', (page_key,json.dumps(result,ensure_ascii=False)))
                conn.commit()
                processed += 1
                if rules:
                    pages[str(number)] = result
        if not complete:
            return pages
        encoded = json.dumps(pages, ensure_ascii=False)
        cur.execute('INSERT INTO import_native_sources VALUES (?,?) ON CONFLICT (id) DO NOTHING', (key, encoded))
        conn.commit()
        return pages
    finally:
        conn.close()
