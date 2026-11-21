"""Loopback-only QA transport using the existing editorial response contract."""
import json
import hashlib
import urllib.request
import uuid
import re
import time

from src.local_usage import record, document_work_id
from src.local_runtime import open_local


def evidence_consistent(answer, rules):
    """Conservative veto for known high-impact rounding errors, not a proof."""
    source = ' '.join(r['fields'].get('原文', '') for r in rules)
    if re.search(r'\bround(?:ed|ing)?[\s-]+up\b', source, re.I):
        if any(term in answer for term in ('四捨五入', '最接近', '向下取整', '無條件捨去')):
            return False
    if re.search(r'\bround(?:ed|ing)?[\s-]+down\b', source, re.I):
        if any(term in answer for term in ('四捨五入', '最接近', '向上取整', '無條件進位')):
            return False
    return True


def grounded(result, selected):
    evidence = result.get('evidence', [])
    if not isinstance(evidence, list):
        return False
    supported = set()
    for item in evidence:
        if not isinstance(item, dict):
            return False
        identifier, quote = item.get('id'), item.get('quote')
        if not isinstance(identifier, str) or identifier not in selected or not isinstance(quote, str) or len(quote.strip()) < 4:
            return False
        fields = selected[identifier]['fields']
        if len(quote.strip()) < 12 and not any(quote.strip() == line.strip() for key in ('原文', '內容') for line in fields.get(key, '').splitlines()):
            return False
        if not any(' '.join(quote.split()) in ' '.join(fields.get(key, '').split()) for key in ('原文', '內容')):
            return False
        supported.add(identifier)
    return bool(supported) and set(result.get('citations', [])) <= supported


def generate(model, payload, *, game_id, operation, timeout=35, response_schema=None, on_inference=None, thinking=False):
    identifier = uuid.uuid4().hex
    schema = {'type': 'object', 'properties': {
        'answer': {'type': 'string'},
        'citations': {'type': 'array', 'items': {'type': 'string'}},
        'cannot_determine': {'type': 'boolean'}},
        'required': ['answer', 'citations', 'cannot_determine'], 'additionalProperties': False}
    schema['properties']['evidence'] = {'type': 'array', 'items': {'type': 'object', 'properties': {'id': {'type': 'string'}, 'quote': {'type': 'string'}}, 'required': ['id', 'quote'], 'additionalProperties': False}}
    schema['required'].append('evidence')
    # Restrict generated references to the exact retrieved source identifiers.
    try:
        rules = json.loads(payload['contents'][0]['parts'][0]['text'])['rules']
        identifiers = [rule['id'] for rule in rules]
    except (ValueError, KeyError, TypeError):
        identifiers = []
    if identifiers:
        schema['properties']['citations']['items']['enum'] = identifiers
        schema['properties']['evidence']['items']['properties']['id']['enum'] = identifiers
    if response_schema is not None:
        schema = response_schema
    request = urllib.request.Request('http://127.0.0.1:11434/api/chat',
        data=json.dumps({'model': model, 'stream': False, 'think': thinking,
            'messages': [
                {'role': 'system', 'content': payload['systemInstruction']['parts'][0]['text'] + ('' if response_schema is not None else ' 另輸出 evidence 陣列，每項 id 與 quote，quote 必須逐字引用該規則原文或內容中支持結論的完整句，至少 12 字元，或完整的一行配件標籤（至少 4 字元）；找不到證據則 cannot_determine=true。')},
                {'role': 'user', 'content': payload['contents'][0]['parts'][0]['text']}],
            'format': schema, 'keep_alive': '5m',
            'options': {'temperature': 0, 'num_ctx': 8192, 'num_predict': 3072 if thinking else 1600}}).encode(),
        headers={'Content-Type': 'application/json'})
    from src.local_inference_queue import inference_slot
    # Private desktop previews have a 180-second proxy window. Keep public
    # requests within their shorter timeout while allowing an active import to finish.
    with inference_slot('qa', wait_seconds=60 if operation == 'editorial-preview-qa' else 5):
        from src.local_import_budget import reserve,settle
        reservation=reserve(game_id,operation,8192+(3072 if thinking else 1600),timeout,family='qa')
        started=time.monotonic()
        if on_inference is not None: on_inference()
        record(identifier, game_id, model, operation,
               work_id=document_work_id(game_id,hashlib.sha256(request.data).hexdigest()))
        try:
            with open_local(request, timeout=timeout) as response:
                result = json.load(response)
            settle(reservation,result.get('prompt_eval_count'),result.get('eval_count'),time.monotonic()-started,family='qa')
        except Exception:
            record(identifier, game_id, model, operation, outcome='failed')
            raise
    record(identifier, game_id, model, operation, {
        'prompt_tokens': result.get('prompt_eval_count'), 'output_tokens': result.get('eval_count')})
    from src.local_usage import output_validation
    with output_validation(identifier,game_id,model,operation):
        if result.get('done_reason') != 'stop':
            raise ValueError('Incomplete local answer')
        # A full prompt near the context ceiling may have displaced source text.
        if result.get('prompt_eval_count', 0) >= 6400:
            raise ValueError('Local context capacity exceeded')
        return {'candidates': [{'finishReason': 'STOP', 'content': {
            'parts': [{'text': result['message']['content']}]}}]}
