"""Fixed categories from source headings or existing generated titles."""
import hashlib
import statistics
import re
from collections import defaultdict
import fitz
from src.editorial_visuals import read_bytes
from src.rule_taxonomy import TAXONOMY, match_title


def page_headings(page):
    lines = [line for block in page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks'] for line in block.get('lines', [])]
    sizes = [span['size'] for line in lines for span in line['spans'] for _ in range(min(len(span['text']), 100))]
    body_size = statistics.median(sizes) if sizes else 0
    headings = []
    for line in lines:
        spans = line['spans']
        text = ''.join(s['text'] for s in spans).strip()
        prefix = ''
        for span in spans:
            if not span['flags'] & 16:
                break
            prefix += span['text']
        if len(spans) > 1 and prefix.strip().endswith((':', '：')) and len(prefix.strip()) <= 80:
            headings.append({'title': prefix.strip(), 'bbox': list(line['bbox']), 'size': max(s['size'] for s in spans), 'body_size': body_size})
            continue
        if not text or len(text) > 100 or text.endswith(('.', '。', '!', '?')) or text.isdigit():
            continue
        if any(s['size'] >= body_size * 1.2 for s in spans) or all(s['flags'] & 16 for s in spans):
            headings.append({'title': text, 'bbox': list(line['bbox']), 'size': max(s['size'] for s in spans), 'body_size': body_size})
    return headings


def single_column(page):
    """Only inherit across pages with an unambiguous aligned text column."""
    boxes = [line['bbox'] for block in page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks'] for line in block.get('lines', [])
             if (text := ''.join(s['text'] for s in line['spans']).strip()) and not text.isdigit()]
    return bool(boxes) and max(b[0] for b in boxes) - min(b[0] for b in boxes) <= 20


def printed_contents(pdf, headings):
    """Accept printed page references only when the target heading corroborates them."""
    entries = []
    normalize = lambda text: ' '.join(text.casefold().split()).strip(' :：')
    for page in pdf:
        lines = page.get_text().splitlines()
        if not any(normalize(line) in {'contents', 'table of contents', '目錄', '目录', 'indice', 'sommario'} for line in lines):
            continue
        for index, line in enumerate(lines):
            match = re.fullmatch(r'\s*(.+?)\s*\.{2,}\s*(\d+)\s*', line)
            if match:
                title, number = match.groups()
            elif index + 1 < len(lines) and lines[index + 1].strip().isdigit():
                title, number = line.strip(), lines[index + 1].strip()
            else:
                continue
            number = int(number)
            if title and any(normalize(h['title']) == normalize(title) for h in headings.get(number, [])):
                entry = (1, title.strip(), number)
                if entry not in entries:
                    entries.append(entry)
    return entries


def categories(job, blocks):
    data = read_bytes(job['source_key'])
    if hashlib.sha256(data).hexdigest() != job['digest']:
        raise ValueError('Source digest mismatch')
    with fitz.open(stream=data, filetype='pdf') as pdf:
        toc = [(depth, title.strip(), page) for depth, title, page in pdf.get_toc()
               if title.strip() and 1 <= page <= len(pdf)]
        headings = {i + 1: page_headings(page) for i, page in enumerate(pdf)}
        printed = printed_contents(pdf, headings) if not toc else []
        # Repeated marginal text is a running header/footer, not a new chapter.
        marginal = defaultdict(set)
        for number, page in enumerate(pdf, 1):
            for h in headings[number]:
                if h['size'] <= h['body_size'] * 1.25 and (h['bbox'][3] < page.rect.height * .10 or h['bbox'][1] > page.rect.height * .90):
                    marginal[' '.join(h['title'].split())].add(number)
        repeated = {title for title, pages in marginal.items() if len(pages) >= 2}
        for number, page in enumerate(pdf, 1):
            headings[number] = [h for h in headings[number] if not (
                ' '.join(h['title'].split()) in repeated and h['size'] <= h['body_size'] * 1.25 and
                (h['bbox'][3] < page.rect.height * .10 or h['bbox'][1] > page.rect.height * .90))]
        aligned = {i + 1: single_column(page) for i, page in enumerate(pdf)}
        inherited, previous = {}, None
        for page in range(1, len(pdf) + 1):
            if not aligned[page]:
                previous = None
                continue
            if previous:
                inherited[page] = previous
            # An unknown heading also terminates the previous chapter.
            if headings[page]:
                previous = max(headings[page], key=lambda h: h['bbox'][3])
    groups = {identifier: {'id': identifier, 'name': name, 'method': 'title-taxonomy-v1',
                          'rule_ids': [], 'assignments': []} for identifier, name, _ in TAXONOMY}
    from src.local_category_worker import cached
    for block in blocks:
        if block['kind'] != '規則':
            continue
        fields = block['fields']
        value = fields.get('PDF頁碼', '')
        page = int(value) if value.isdigit() else 0
        titles, method = [], 'unassigned'
        original = ' '.join(fields.get('原文', '').split())
        local_heads = [h for h in headings.get(page, []) if ' '.join(h['title'].split()) in original]
        strongest = max((h['size'] for h in local_heads), default=0)
        local = [h['title'] for h in local_heads if h['size'] >= strongest - .5]
        box = block.get('source_location', {}).get('bbox')
        if not local and box:
            preceding_headings = [h for h in headings.get(page, [])
                                  if h['bbox'][3] <= box[1] + 2 and abs(h['bbox'][0] - box[0]) <= 20]
            if preceding_headings:
                local = [max(preceding_headings, key=lambda h: h['bbox'][3])['title']]
        if local:
            titles, method = local, 'source_heading'
        else:
            preceding = [(d, t, p) for d, t, p in toc if p <= page]
            # Printed contents are corroborated page-local hints, not inferred ancestry.
            printed_titles = [t for _, t, p in printed if p == page]
            if not preceding and len(printed_titles) == 1:
                titles, method = printed_titles, 'printed_contents_page'
            if preceding:
                latest = max(p for _, _, p in preceding)
                latest_titles = [t for _, t, p in preceding if p == latest]
                if len(latest_titles) == 1:
                    titles = latest_titles
                    method = 'pdf_bookmark_page'
            elif not titles and box and page in inherited and not any(h['bbox'][1] <= box[3] for h in headings.get(page, [])):
                titles, method = [inherited[page]['title']], 'source_heading_continuation'
        parents = []
        for depth, title, toc_page in toc:
            if toc_page > page:
                continue
            if toc_page == page and sum(p == page for _, _, p in toc) > 1:
                continue
            parents = [(d,t) for d,t in parents if d < depth]
            parents.append((depth,title))
        selected_heads = [h for h in headings.get(page, []) if h['title'] in local]
        if selected_heads:
            anchor = min(selected_heads, key=lambda h: h['bbox'][1])
            larger = [h for h in headings.get(page, [])
                      if h['size'] > anchor['size'] + .5 and h['bbox'][3] <= anchor['bbox'][1] + 2
                      and abs(h['bbox'][0] - anchor['bbox'][0]) <= 20]
            if larger:
                parent_heading = max(larger, key=lambda h: h['bbox'][3])
                if parent_heading['title'] not in [t for _, t in parents]:
                    parents.append((len(parents)+1, parent_heading['title']))
        block['chapter_context'] = {'titles': titles, 'parents': [t for _,t in parents]}
        matches = list(dict.fromkeys(identifier for title in titles for identifier in match_title(title)))
        if not matches:
            for _, parent in reversed(parents):
                matches = match_title(parent)
                if matches:
                    titles, method = [parent], 'parent_chapter'
                    break
        if block.get('evidence_usable') is False:
            matches, titles, method = [], [], 'unassigned'
            groups['unassigned']['rule_ids'].append(block['id'])
            groups['unassigned']['assignments'].append({'rule_id':block['id'],'titles':[],'method':method})
            continue
        if not matches and not block.get('extraction') and not block.get('layout_evidence'):
            title = fields.get('標題', '')
            if not title.startswith(('待辨識區塊', '來源內容')):
                matches = match_title(title)
                if matches:
                    titles, method = [title], 'existing_model_title'
        for identifier in matches or ['unassigned']:
            groups[identifier]['rule_ids'].append(block['id'])
            groups[identifier]['assignments'].append({'rule_id': block['id'], 'titles': titles, 'method': method})
    from src.local_category_worker import prepare_units
    prepare_units(blocks)
    model_results = cached(job, blocks)
    for assignment in list(groups['unassigned']['assignments']):
        identifier = assignment['rule_id']
        labels = model_results.get(identifier, [])
        if not labels or 'unassigned' in labels:
            continue
        block = next(b for b in blocks if b['id'] == identifier)
        if block.get('evidence_usable') is False:
            continue
        groups['unassigned']['rule_ids'].remove(identifier)
        groups['unassigned']['assignments'].remove(assignment)
        for label in labels:
            groups[label]['rule_ids'].append(identifier)
            groups[label]['assignments'].append({**assignment, 'method':'local_model_category'})
    return list(groups.values())


def chapter_manifest(blocks):
    """Expose source chapter ancestry and member IDs for navigation, not QA scope."""
    chapters = {}
    for block in blocks:
        if block['kind'] != '規則':
            continue
        context = block.get('chapter_context', {})
        def add(title, parent):
            key = (block['fields'].get('來源'), block['fields'].get('PDF頁碼'), parent, title)
            identifier = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
            node = chapters.setdefault(identifier, {'id':identifier, 'title':title, 'parent_id':parent,
                'page':block['fields'].get('PDF頁碼'), 'rule_ids':[]})
            if block['id'] not in node['rule_ids']:
                node['rule_ids'].append(block['id'])
            return identifier
        parent = None
        for title in dict.fromkeys(context.get('parents', [])):
            parent = add(title, parent)
        # Same-level source headings are siblings, never an invented ancestry.
        for title in dict.fromkeys(context.get('titles', [])):
            if title not in context.get('parents', []):
                add(title, parent)
    return list(chapters.values())
