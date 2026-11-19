"""Public local-answer structure; source matching is not semantic proof."""


def response_schema(identifiers):
    source_id = {'type': 'string', 'enum': list(identifiers)}
    return {'type': 'object', 'properties': {
        'answer': {'type': 'string'},
        'rule_explanation': {'type': 'string'},
        'scope_limit': {'type': 'string'},
        'citations': {'type': 'array', 'items': source_id},
        'cannot_determine': {'type': 'boolean'},
    }, 'required': ['answer', 'rule_explanation', 'scope_limit', 'citations', 'cannot_determine'],
        'additionalProperties': False}


def render_answer(result):
    for key in ('answer', 'rule_explanation', 'scope_limit'):
        if not isinstance(result.get(key), str):
            raise ValueError('Invalid rule explanation contract')
    if not result.get('cannot_determine') and not result['rule_explanation'].strip():
        raise ValueError('Missing rule explanation')
    # Do not manufacture reasons or a scope disclaimer when a model omits them.
    return '\n'.join(dict.fromkeys(result[key].strip() for key in
        ('answer', 'rule_explanation', 'scope_limit') if result[key].strip()))


def source_evidence(citations, selected):
    evidence = []
    for identifier in dict.fromkeys(citations):
        if identifier not in selected:
            raise ValueError('Unknown rule source')
        quote = selected[identifier]['fields'].get('原文')
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError('Missing original rule text')
        evidence.append({'id': identifier, 'quote': quote})
    return evidence
