"""Resumable local PDF-to-editorial draft import. Never publishes or replaces a human draft."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import uuid
import fitz
from src.gemini_pdf import PIPELINE_VERSION, check_page, extract_pdf_rules
from src.gemini_usage import setting
from src.editorial_validation import validate_draft


def build_document(name, filename, total, results):
    blocks = [
        {'kind': '文件', 'id': 'META', 'fields': {'遊戲': name, '版本': 'Gemini 自動擷取・待人工校對'}},
        {'kind': '來源', 'id': 'AUTO-SOURCE', 'fields': {'名稱': filename, '版本': '匯入原檔', 'PDF頁數': str(total)}},
    ]
    groups = {'setup': '開始遊戲', 'basic_info': '開始遊戲', 'components': '卡牌與配件',
              'cards': '卡牌與配件', 'tiles': '卡牌與配件', 'tokens': '卡牌與配件',
              'scoring': '計分與結束', 'end_game': '計分與結束', 'exceptions': '例外與 FAQ', 'faq': '例外與 FAQ'}
    for page in range(1, total + 1):
        result = results.get(page)
        ids = []
        if result:
            for index, rule in enumerate(result.rules, 1):
                identifier = f'AUTO-P{page:03d}-R{index:03d}'
                ids.append(identifier)
                blocks.append({'kind': '規則', 'id': identifier, 'fields': {
                    '大項': groups.get(rule.category.value, '進行遊戲'), '標題': rule.title,
                    '內容': rule.simplified_text, '原文': rule.rule_text,
                    '角色': '未確認', '人數': '未確認', '擴充': '未確認',
                    '條件': '未確認，請核對內容與原文', '例外': '未確認，請核對內容與原文',
                    '來源': 'AUTO-SOURCE', 'PDF頁碼': str(page) if rule.page_end == page else f'{page}-{rule.page_end}',
                    '狀態': '待確認'}})
        blocks.append({'kind': '涵蓋', 'id': f'AUTO-C{page:03d}', 'fields': {
            '章節': f'PDF 第 {page} 頁', '來源': 'AUTO-SOURCE', 'PDF頁碼': str(page),
            '規則': '、'.join(ids), '狀態': '待確認',
            '說明': ('AI 已擷取，仍須核對完整性。' + result.excluded_content + '；'.join(result.warnings)) if result else '尚未擷取此頁'}})
    document = {'blocks': blocks}
    if validate_draft(document):
        raise ValueError('Invalid generated draft')
    return document


def markdown(document):
    lines = ['# 規則整理稿 v1']
    for block in document['blocks']:
        lines.extend(['', f"## {block['kind']} {block['id']}"])
        for key, value in block['fields'].items():
            # Keep model-controlled text from creating headings or new fields.
            lines.append(key + '：' + ' '.join(value.splitlines()))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--game-id', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--max-pages', type=int, default=1)
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 40:
        parser.error('--max-pages must be 1..40')
    from src.config import DATA_DIR
    digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    model = setting('RULE_EXTRACTION_MODEL', 'gemini-3.8-flash')
    cache_key = hashlib.sha256(f'{digest}:{model}:{PIPELINE_VERSION}'.encode()).hexdigest()
    cache = DATA_DIR / 'extraction-cache'
    directory = cache / cache_key
    with fitz.open(args.pdf) as pdf:
        total = len(pdf)
    attempted = 0
    failure = None
    for page in range(1, total + 1):
        if args.export_only or (directory / f'{page:04d}.json').exists():
            continue
        if attempted >= args.max_pages:
            break
        attempted += 1
        try:
            extract_pdf_rules(str(args.pdf), digest, args.game_id, cache_dir=str(cache), pages=[page])
        except ValueError:
            # A cover can have a valid checkpoint but no playable rules.
            if not (directory / f'{page:04d}.json').exists():
                failure = f'Page {page}: extraction validation failed'
                break
        except Exception as error:
            http = re.search(r'Gemini HTTP (\d{3})', str(error))
            reason = f'HTTP {http.group(1)}' if http else type(error).__name__
            failure = f'Page {page}: {reason}; paused'
            break
        print(f'Completed page {page}/{total}', flush=True)
    results = {page: check_page(json.loads(path.read_text(encoding='utf-8')), page, total)
               for page in range(1, total + 1) if (path := directory / f'{page:04d}.json').exists()}
    output = DATA_DIR / 'editorial-imports' / cache_key
    output.mkdir(parents=True, exist_ok=True)
    # Immutable snapshots preserve every prior output and never touch user drafts.
    snapshot = output / (uuid.uuid4().hex + '.md')
    snapshot.write_text(markdown(build_document(args.name, args.pdf.name, total, results)), encoding='utf-8')
    snapshot.with_suffix('.json').write_text(json.dumps({
        'game_id': args.game_id, 'model': model, 'source_sha256': digest,
        'completed_pages': len(results), 'total_pages': total,
        'paused': failure, 'snapshot_id': snapshot.stem,
    }, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'completed_pages': len(results), 'total_pages': total, 'draft': str(snapshot), 'paused': failure}, ensure_ascii=True))


if __name__ == '__main__':
    main()
