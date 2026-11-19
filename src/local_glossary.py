"""Read editorial terminology and flag potential inconsistencies without rewriting drafts."""
import re


def read_glossary(path):
    text = path.read_text(encoding='utf-8')
    entries = {}
    for block in re.split(r'^#{2,4}\s+', text, flags=re.M):
        if not block.startswith('術語 '):
            continue
        fields = dict(re.findall(r'^(中文|原文|別名)：(.*)$', block, re.M))
        original = fields.get('原文', '').strip()
        chinese = re.sub(r'\s*[（(]暫定譯名[）)]', '', fields.get('中文', '')).strip()
        if not original or not chinese or max(len(original), len(chinese)) > 160:
            raise ValueError('Invalid glossary entry')
        aliases = [s.strip() for s in fields.get('別名', '').split('、') if s.strip() not in ('', '無')]
        if original in entries:
            raise ValueError('Duplicate glossary term')
        entries[original] = {'chinese': chinese, 'aliases': aliases}
    if not entries or len(entries) > 100:
        raise ValueError('Glossary must contain 1 to 100 terms')
    return entries


def matches(source, term):
    return re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', source, flags=re.I) is not None


def prompt_terms(glossary, source, image=False):
    terms = [f'{term}＝{entry["chinese"]}' for term, entry in glossary.items() if image or matches(source, term)]
    return '\n本遊戲暫定術語對照（僅在來源出現時使用，不增添規則）：\n' + '\n'.join(terms) if terms else ''


def audit_rules(glossary, rules):
    findings = []
    for index, rule in enumerate(rules, 1):
        for term, entry in glossary.items():
            # English aliases are not evidence of a Chinese translation.
            accepted = [entry['chinese']] + [a for a in entry['aliases'] if re.search(r'[\u3400-\u9fff]', a)]
            if matches(rule['original'], term) and not any(a in rule['translation'] for a in accepted):
                findings.append({'block': index, 'term': term, 'expected': entry['chinese'], 'status': 'needs_review'})
    return findings
