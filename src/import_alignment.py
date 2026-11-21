"""Conservative exact-source alignment; no fuzzy deletion of rule conditions."""
from collections import defaultdict


def align(blocks):
    native, derived = defaultdict(list), defaultdict(list)
    segments = defaultdict(list)
    def identity(block):
        fields=block['fields']
        return (fields.get('來源'),fields.get('PDF頁碼'),' '.join(fields.get('原文','').split()))
    for block in blocks:
        if block['kind'] != '規則' or block.get('evidence_usable') is False:
            continue
        if block.get('extraction'):
            native[identity(block)].append(block)
            location = block.get('source_location', {})
            segments[(block['fields'].get('PDF頁碼'), location.get('reading_order'))].append(block)
        elif block.get('derivation'):
            derived[identity(block)].append(block)
    for parts in segments.values():
        parts.sort(key=lambda b: b.get('source_location', {}).get('text_start', 0))
        for index, part in enumerate(parts):
            part['source_links'] = {
                'previous': parts[index-1]['id'] if index else None,
                'next': parts[index+1]['id'] if index+1 < len(parts) else None,
            }
    aliases={}
    for key, candidates in derived.items():
        sources=native.get(key,[])
        # Ambiguous repeated text or multiple translations remain separate.
        if len(sources)!=1 or len(candidates)!=1 or not key[2]:
            continue
        source, candidate=sources[0],candidates[0]
        source['derivatives']=[candidate]
        source['fields']['內容']=candidate['fields']['內容']
        aliases[candidate['id']]=source['id']
    result=[block for block in blocks if block['id'] not in aliases]
    for block in result:
        if block['kind']=='涵蓋':
            ids=block['fields'].get('規則','').split('、')
            block['fields']['規則']='、'.join(dict.fromkeys(aliases.get(identifier,identifier) for identifier in ids))
    return result, aliases
