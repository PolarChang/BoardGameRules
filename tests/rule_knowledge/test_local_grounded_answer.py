import json
import pytest
from src import local_grounded_answer as qa


SELECTED={'R':{'id':'R','fields':{'原文':'Pay 5 coins unless the effect says otherwise.','PDF頁碼':'2','來源':'S'}}}
CLAIM={'text':'支付5枚硬幣，除非效果另有說明。','evidence':[{'id':'R','quote':'Pay 5 coins unless the effect says otherwise.'}]}


def response(value):
    return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(value)}]}}]}


def test_claim_requires_real_quote_and_no_invented_number():
    assert qa.claim_valid(CLAIM,SELECTED)
    assert not qa.claim_valid({**CLAIM,'text':'支付99枚硬幣。'},SELECTED)
    assert not qa.claim_valid({**CLAIM,'text':'支付九十九枚硬幣。'},SELECTED)
    assert qa.claim_valid({**CLAIM,'text':'支付五枚硬幣，除非效果另有說明。'},SELECTED)
    assert not qa.claim_valid({**CLAIM,'evidence':[{'id':'OTHER','quote':CLAIM['evidence'][0]['quote']}]},SELECTED)


def test_missing_exception_coverage_removes_only_affected_claim():
    selected={**SELECTED,'E':{'fields':{'原文':'The action is skipped during the final round.'}},
              'N':{'fields':{'原文':'There are five rounds.'}}}
    default={'text':'執行此行動。','evidence':[{'id':'R','quote':SELECTED['R']['fields']['原文']}]}
    number={'text':'共有五回合。','evidence':[{'id':'N','quote':'There are five rounds.'}]}
    requirements=[{'requirement':'最後回合跳過行動','evidence':[{'id':'E','quote':selected['E']['fields']['原文']}],'claim_indexes':[0]}]
    kept,checks,missing=qa.check_coverage(requirements,[default,number],[default,number],selected)
    assert kept==[number]
    assert checks[0]['covered'] is False and missing==['最後回合跳過行動']


def test_coverage_accepts_a_repair_only_when_its_own_quote_covers_requirement():
    bad={**CLAIM,'evidence':[]}
    requirement={'requirement':'費用及例外','evidence':CLAIM['evidence'],'claim_indexes':[1]}
    kept,checks,missing=qa.check_coverage([requirement],[bad,CLAIM],[CLAIM],SELECTED)
    assert kept==[CLAIM] and checks[0]['covered'] and missing==[]
    with pytest.raises(ValueError):
        qa.check_coverage([{**requirement,'claim_indexes':[99]}],[bad,CLAIM],[CLAIM],SELECTED)
    with pytest.raises(ValueError):
        qa.check_coverage([{**requirement,'evidence':[{'id':'R','quote':'Invented exception.'}]}],[bad,CLAIM],[CLAIM],SELECTED)


def test_true_completeness_cannot_override_an_uncovered_source_condition(monkeypatch):
    selected={**SELECTED,'E':{'fields':{'原文':'The action is skipped during the final round.'}}}
    replies=iter([response({'status':'complete','claims':[CLAIM],'missing_context':[],'limitations':[]}),
                  response({'supported':[0],'repairs':[],'fully_answers_question':True,'missing_conditions':[],'conflict':False,
                            'requirements':[{'requirement':'最後回合例外','evidence':[{'id':'E','quote':selected['E']['fields']['原文']}],'claim_indexes':[0]}]})])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:next(replies))
    answer=qa.answer_from_evidence('費用？',selected,'test')
    assert answer['status']=='insufficient' and answer['claims']==[]
    assert answer['coverage_checks'][0]['covered'] is False


def test_inventory_runs_before_answer_and_no_draft_can_anchor_it(monkeypatch):
    calls=[]
    inventory={'units':{'R':{'relevance':'required','uncertainty':''}}}
    replies=iter([inventory,{'status':'complete','claims':[{'text':CLAIM['text'],'source_ids':['R']}],'missing_context':[],'limitations':[]}])
    def generate(model,body,**kwargs):
        parsed=json.loads(body['contents'][0]['parts'][0]['text'])
        calls.append(kwargs['operation'])
        if len(calls)==1:
            assert 'claims' not in parsed and 'source_requirements' not in parsed
            units=kwargs['response_schema']['properties']['units']
            assert units['required']==['R'] and units['additionalProperties'] is False
            assert units['properties']['R']['properties']['relevance']['enum']==['required','irrelevant','unresolved']
        else: assert parsed['source_requirements']==[{'requirement':SELECTED['R']['fields']['原文'],'evidence':CLAIM['evidence']}]
        return response(next(replies))
    monkeypatch.setattr(qa,'generate',generate)
    answer=qa.answer_from_evidence('費用？',SELECTED,'test',inventory_first=True)
    assert calls==['qa-source-inventory','qa-claims']
    assert answer['status']=='complete' and answer['coverage_checks'][0]['covered']


@pytest.mark.parametrize('units',[{'OTHER':None},{},{'R':True},{'R':''},[{'id':'R','relevant':True}]])
def test_incomplete_inventory_stops_before_generating_answer(monkeypatch,units):
    calls=[]
    def generate(*args,**kwargs):
        calls.append(kwargs['operation'])
        return response({'units':units,'requirements':[]})
    monkeypatch.setattr(qa,'generate',generate)
    diagnostics={}
    with pytest.raises(ValueError): qa.answer_from_evidence('費用？',SELECTED,'test',inventory_first=True,diagnostics=diagnostics)
    assert calls==['qa-source-inventory']
    assert diagnostics['source_inventory_raw']['units']==units


def test_inventory_attaches_only_snapshot_text_and_keeps_exceptions(monkeypatch):
    selected={**SELECTED,'E':{'fields':{'原文':'Skip this action in the last round.\n'}},
              'TITLE':{'fields':{'原文':'Overview'}}}
    raw={'units':{'R':{'relevance':'required','uncertainty':''},'E':{'relevance':'required','uncertainty':''},'TITLE':{'relevance':'irrelevant','uncertainty':''}}}
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(raw))
    inventory=qa.source_inventory('費用？',selected,'test','unused',None,None)
    assert [r['evidence'][0]['id'] for r in inventory['requirements']]==['R','E']
    for requirement in inventory['requirements']:
        quote=requirement['evidence'][0]
        assert quote['quote']==selected[quote['id']]['fields']['原文']
    assert inventory['requirements'][1]['requirement']==selected['E']['fields']['原文']


def test_exception_cannot_disappear_when_another_claim_is_filtered():
    assert not qa.claim_valid({**CLAIM,'text':'支付5枚硬幣。'},SELECTED)
    assert qa.claim_valid(CLAIM,SELECTED)


def test_ordinary_rule_does_not_require_an_invented_exception():
    source={'R':{'fields':{'原文':'Pay 5 coins.'}}}
    assert qa.claim_valid({'text':'支付5枚硬幣。','evidence':[{'id':'R','quote':'Pay 5 coins.'}]},source)


def test_repeating_a_question_is_not_a_supported_answer_claim():
    assert not qa.claim_valid({**CLAIM,'text':'是否需要支付5枚硬幣，除非另有說明？'},SELECTED)


def test_pdf_control_characters_restore_exact_source_instead_of_false_rejection():
    source={'R':{'fields':{'原文':'•\t\x07In a 2-player game, use these roles.'}}}
    claim={'text':'兩人遊戲使用指定角色。','evidence':[{'id':'R','quote':'• In a 2-player game, use these roles.'}]}
    restored=qa.restore_evidence(claim,source)
    assert restored['evidence'][0]['quote']==source['R']['fields']['原文']
    assert qa.claim_valid(restored,source)


def test_quote_restoration_cannot_add_or_remove_a_negation():
    restored=qa.restore_evidence({'text':'可以支付。','evidence':[{'id':'R','quote':'You may pay.'}]}, {'R':{'fields':{'原文':'You may not pay.'}}})
    assert restored['evidence'][0]['quote']=='You may pay.'
    assert not qa.claim_valid(restored,{'R':{'fields':{'原文':'You may not pay.'}}})


def test_split_sentence_restores_separate_raw_quotes_with_correct_region_ids():
    selected={key:{'fields':{'原文':text,'來源':'S','PDF頁碼':'7'}} for key,text in [
        ('A','Effects are not optional unless'),('B','Unrelated column.'),('C','explicitly stated (using “may” or “up to”).')]}
    claim={'text':'除非另有說明，效果不可跳過。','evidence':[{'id':'A','quote':'Effects are not optional unless explicitly stated (using "may" or "up to").'}]}
    restored=qa.restore_evidence(claim,selected)
    assert [e['id'] for e in restored['evidence']]==['A','C']
    assert restored['evidence'][1]['quote']==selected['C']['fields']['原文']
    assert qa.claim_valid(restored,selected)
    selected['C']['fields']['來源']='OTHER'
    assert not qa.claim_valid(qa.restore_evidence(claim,selected),selected)


def test_split_quote_alignment_cannot_skip_negation_or_missing_words():
    selected={'A':{'fields':{'原文':'Effects are not optional unless','來源':'S','PDF頁碼':'1'}},'B':{'fields':{'原文':'explicitly stated.','來源':'S','PDF頁碼':'1'}}}
    assert qa.align_split_quote('Effects are optional unless explicitly stated.','A',selected) is None


def test_failed_review_drops_claim_instead_of_returning_unsupported_answer(monkeypatch):
    replies=iter([response({'status':'complete','claims':[CLAIM],'missing_context':[],'limitations':[]}),
                  response({'supported':[],'missing_conditions':['需要核對作用對象'],'conflict':False})])
    calls=[]
    def generate(*args,**kwargs):
        calls.append(kwargs['operation']); return next(replies)
    monkeypatch.setattr(qa,'generate',generate)
    result=qa.answer_from_evidence('費用是多少？',SELECTED,'test')
    assert result['status']=='insufficient' and not result['claims'] and not result['citations']
    assert '支付5枚' not in result['answer']
    assert calls==['qa-claims','qa-verify']


def test_missing_condition_produces_partial_answer(monkeypatch):
    replies=iter([response({'status':'complete','claims':[CLAIM],'missing_context':[],'limitations':[]}),
                  response({'supported':[0],'missing_conditions':['沒有提供折扣效果'],'conflict':False})])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:next(replies))
    result=qa.answer_from_evidence('費用是多少？',SELECTED,'test')
    assert result['status']=='partial' and result['citations']==['R']


def test_missing_live_context_is_not_invented_and_skips_verification(monkeypatch):
    calls=[]
    def generate(*args,**kwargs):
        calls.append(kwargs['operation'])
        return response({'status':'needs_context','claims':[],'missing_context':['請提供目前手牌'],'limitations':[]})
    monkeypatch.setattr(qa,'generate',generate)
    result=qa.answer_from_evidence('我手上有哪些牌？',SELECTED,'test')
    assert result['status']=='needs_context' and calls==['qa-claims']


def test_insufficient_status_cannot_leak_model_claims_or_citations(monkeypatch):
    replies=iter([response({'status':'insufficient','claims':[CLAIM],'missing_context':[],'limitations':[]}),
                  response({'supported':[],'missing_conditions':[],'conflict':False})])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:next(replies))
    result=qa.answer_from_evidence('費用？',SELECTED,'test')
    assert result['status']=='insufficient' and result['claims']==[] and result['citations']==[]
    assert '支付5枚' not in result['answer']


def test_one_review_can_restore_omitted_exception_without_a_third_generation(monkeypatch):
    bad={**CLAIM,'text':'支付5枚硬幣。'}
    replies=iter([response({'status':'complete','claims':[bad],'missing_context':[],'limitations':[]}),
                  response({'supported':[],'repairs':[CLAIM],'missing_conditions':[],'conflict':False})])
    calls=[]
    def generate(*a,**k): calls.append(k['operation']); return next(replies)
    monkeypatch.setattr(qa,'generate',generate)
    result=qa.answer_from_evidence('費用？',SELECTED,'test')
    assert result['claims']==[CLAIM] and result['status']=='partial'
    assert calls==['qa-claims','qa-verify']


def test_conflict_is_preserved(monkeypatch):
    replies=iter([response({'status':'complete','claims':[CLAIM],'missing_context':[],'limitations':[]}),
                  response({'supported':[0],'missing_conditions':[],'conflict':True})])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:next(replies))
    assert qa.answer_from_evidence('費用？',SELECTED,'test')['status']=='conflict'


def test_malformed_context_fails_closed(monkeypatch):
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response({'status':'complete','claims':[CLAIM],'missing_context':'not a list','limitations':[]}))
    with pytest.raises(ValueError): qa.answer_from_evidence('費用？',SELECTED,'test')


def test_native_regions_are_presented_in_reading_order_without_merging_ids(monkeypatch):
    captured=[]
    def generate(model,body,**kwargs):
        captured.extend(json.loads(body['contents'][0]['parts'][0]['text'])['rules'])
        return response({'status':'insufficient','claims':[],'missing_context':[],'limitations':[]})
    monkeypatch.setattr(qa,'generate',generate)
    selected={i:{'id':i,'fields':{'原文':'Source text','來源':'s'}} for i in ['NATIVE-P010-R002','NATIVE-P002-R003','NATIVE-P002-R001']}
    qa.answer_from_evidence('問題',selected,'test')
    assert [b['id'] for b in captured]==['NATIVE-P002-R001','NATIVE-P002-R003','NATIVE-P010-R002']


@pytest.mark.parametrize('ordinal',['settimo','settima','settimi','settime','SETTIMO'])
def test_translated_italian_round_ordinal_preserves_numeric_validation(ordinal):
    original=f'Dopo la fase azioni del {ordinal} round, il gioco termina.'
    selected={'R':{'fields':{'原文':original}}}
    claim={'text':'遊戲在第七回合的行動階段後結束。','evidence':[{'id':'R','quote':original}]}
    assert qa.claim_valid(claim,selected)
    assert not qa.claim_valid({**claim,'text':'遊戲在第六回合的行動階段後結束。'},selected)


def test_italian_ordinal_matching_does_not_treat_substrings_as_numbers():
    assert qa.numeric_values('secondario primordiale ottaviano')==set()
    assert qa.numeric_values('non termina')==set()


def test_coverage_recomputes_after_dependent_claim_is_removed():
    selected={key:{'fields':{'原文':text}} for key,text in [('A','Normal action.'),('B','Second action.'),('C','Except on the last round.'),('D','There are five rounds.')]}
    def evidence(key): return {'id':key,'quote':selected[key]['fields']['原文']}
    claims=[{'text':key,'evidence':[evidence(key)]} for key in ('A','B','D')]
    requirements=[{'requirement':'Sequence','evidence':[evidence('A'),evidence('B')],'claim_indexes':[0,1]},
                  {'requirement':'Exception','evidence':[evidence('C')],'claim_indexes':[0]},
                  {'requirement':'Rounds','evidence':[evidence('D')],'claim_indexes':[2]}]
    kept,checks,missing=qa.check_coverage(requirements,claims,claims,selected)
    assert kept==[claims[2]]
    assert [c['covered'] for c in checks]==[False,False,True]
    assert missing==['Sequence','Exception']


@pytest.mark.parametrize('ids',[['FOREIGN'],[],['R']*9,[None]])
def test_candidate_cannot_reference_outside_snapshot(monkeypatch,ids):
    replies=iter([{'units':{'R':{'relevance':'required','uncertainty':''}}},
                  {'status':'complete','claims':[{'text':CLAIM['text'],'source_ids':ids}],'missing_context':[],'limitations':[]}])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(next(replies)))
    with pytest.raises(ValueError,match='Invalid claim source IDs'):
        qa.answer_from_evidence('費用？',SELECTED,'test',inventory_first=True)


@pytest.mark.parametrize('phrase',['下一回合','上一回合','每一回合'])
def test_relative_round_reference_is_not_the_number_one(phrase):
    assert qa.numeric_values(phrase)==set()
    assert qa.numeric_values('第三階段：'+phrase)=={'3'}
    assert qa.numeric_values('第十三階段')=={'13'}


def test_unverified_inventory_statement_is_not_rendered_as_answer(monkeypatch):
    replies=iter([{'units':{'R':{'relevance':'required','uncertainty':''}}},
                  {'status':'insufficient','claims':[],'missing_context':[],'limitations':[]}])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(next(replies)))
    answer=qa.answer_from_evidence('費用？',SELECTED,'test',inventory_first=True)
    assert '99' not in answer['answer']
    assert '來源 R' in answer['answer']
    assert answer['claims']==[]


@pytest.mark.parametrize('thinking',[False,True])
def test_reasoning_policy_reaches_inventory_and_generation_without_extra_calls(monkeypatch,thinking):
    replies=iter([{'units':{'R':{'relevance':'required','uncertainty':''}}},
                  {'status':'complete','claims':[{'text':CLAIM['text'],'source_ids':['R']}],'missing_context':[],'limitations':[]}])
    calls=[]
    def generate(*a,**k):
        assert k['thinking'] is thinking
        assert k['timeout']==(120 if thinking else 60)
        calls.append(k['operation'])
        return response(next(replies))
    monkeypatch.setattr(qa,'generate',generate)
    result=qa.answer_from_evidence('費用？',SELECTED,'test',inventory_first=True,thinking=thinking)
    assert result['status']=='complete'
    assert calls==['qa-source-inventory','qa-claims']



def test_provisional_name_label_requires_paired_chinese_and_quoted_original():
    claim={'text':'規劃（Pianificazione）、行動（Fase azioni）、未知（Other）、數值（x）',
           'evidence':[{'id':'R','quote':'Pianificazione Fase azioni x'}]}
    labelled=qa.mark_provisional_names(claim)
    assert labelled['text']=='規劃（Pianificazione；暫譯）、行動（Fase azioni；暫譯）、未知（Other）、數值（x）'
    assert labelled['evidence']==claim['evidence']
    assert qa.mark_provisional_names(labelled)==labelled
    assert claim['text'].startswith('規劃（Pianificazione）')



def test_last_round_is_relative_reference_not_an_additional_numeric_one():
    assert qa.numeric_values('最後一回合（第5回合）結束後')=={'5'}
    assert qa.numeric_values('第一回合')=={'1'}
    assert qa.numeric_values('最後一回合花費一枚硬幣')=={'1'}



@pytest.mark.parametrize('missing',[False,True])
def test_dependent_claim_keeps_heading_quote_or_rejects_missing_context(monkeypatch,missing):
    selected={'H':{'id':'H','fields':{'原文':'Preparation Phase','來源':'s','PDF頁碼':'1'}},
              'C':{'id':'C','fields':{'原文':'This phase is skipped during round 1.','來源':'s','PDF頁碼':'1'},'context_ids':['H']}}
    if missing: selected.pop('H')
    units={key:{'relevance':'required' if key=='C' else 'irrelevant','uncertainty':''} for key in selected}
    replies=iter([{'units':units},{'status':'complete','claims':[{'text':'第1回合跳過準備階段。','source_ids':['C']}],'missing_context':[],'limitations':[]}])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(next(replies)))
    if missing:
        with pytest.raises(ValueError,match='Required source context unavailable'): qa.answer_from_evidence('階段？',selected,'test',inventory_first=True)
    else:
        answer=qa.answer_from_evidence('階段？',selected,'test',inventory_first=True)
        assert [e['id'] for e in answer['claims'][0]['evidence']]==['C','H']
        assert answer['claims'][0]['evidence'][1]['quote']=='Preparation Phase'


def test_running_header_cannot_be_inventory_rule_fact(monkeypatch):
    selected={'H':{'id':'H','fields':{'原文':'MAIN RULES 11'},'evidence_role':'running_header'}}
    def generate(*args,**kwargs):
        assert kwargs['response_schema']['properties']['units']['properties']['H']['properties']['relevance']['enum']==['irrelevant']
        return response({'units':{'H':{'relevance':'required','uncertainty':''}}})
    monkeypatch.setattr(qa,'generate',generate)
    with pytest.raises(ValueError,match='Running header cannot'):
        qa.source_inventory('幾回合？',selected,'test','unused',None,None)



def test_terminology_labels_only_names_with_own_cited_source_and_merges_exact_duplicates():
    selected={'R':{'fields':{'原文':'Preparation Phase Action Phase'}},'OTHER':{'fields':{'原文':'Scoring Phase'}}}
    evidence=[{'id':'R','quote':selected['R']['fields']['原文']}]
    claims=[{'text':'準備階段，接著行動階段。','evidence':evidence}]*2+[{'text':'準備階段除第一回合外均執行。','evidence':evidence}]
    terms=[{'original':'Preparation Phase','translated':'準備階段','source_id':'R'},
           {'original':'Action Phase','translated':'行動階段','source_id':'R'},
           {'original':'Scoring Phase','translated':'計分階段','source_id':'OTHER'}]
    result=qa.render_terminology(claims,terms,selected)
    assert len(result)==2
    assert result[0]['text']=='準備階段（Preparation Phase；暫譯），接著行動階段（Action Phase；暫譯）。'
    assert '第一回合' in result[1]['text']
    assert result[0]['evidence']==evidence
    assert 'Scoring' not in str(result)
    assert claims[0]['text']=='準備階段，接著行動階段。'


@pytest.mark.parametrize('term',[{'original':'Invented','translated':'名稱','source_id':'R'},
                                {'original':'Pay','translated':'支付','source_id':'OTHER'},
                                {'original':'Pay','translated':'pay','source_id':'R'}])
def test_terminology_rejects_unavailable_or_untranslated_names(term):
    with pytest.raises(ValueError): qa.render_terminology([CLAIM],[term],SELECTED)



def test_mixed_language_term_is_not_accepted_as_finished_translation():
    with pytest.raises(ValueError,match='Terminology not grounded'):
        qa.render_terminology([CLAIM],[{'original':'Pay','translated':'Pay支付','source_id':'R'}],SELECTED)



def test_overlapping_names_are_rendered_once_without_rewriting_inserted_labels():
    source='Working Class Worker'
    claims=[{'text':'工人階級與工人。','evidence':[{'id':'R','quote':source}]}]
    terms=[{'original':'Working Class','translated':'工人階級','source_id':'R'},
           {'original':'Worker','translated':'工人','source_id':'R'}]
    result=qa.render_terminology(claims,terms,{'R':{'fields':{'原文':source}}})
    assert result[0]['text']=='工人階級（Working Class；暫譯）與工人（Worker；暫譯）。'
    assert qa.render_terminology(result,terms,{'R':{'fields':{'原文':source}}})==result


def test_same_name_in_multiple_sources_preserves_each_claim_own_citation():
    selected={i:{'fields':{'原文':'Working Class'}} for i in ('A','B')}
    claims=[{'text':'工人階級'+suffix,'evidence':[{'id':i,'quote':'Working Class'}]} for i,suffix in [('A','。'),('B','可行動。')]]
    terms=[{'original':'Working Class','translated':'工人階級','source_id':i} for i in ('A','B')]
    result=qa.render_terminology(claims,terms,selected)
    assert all('Working Class；暫譯' in c['text'] for c in result)
    assert [c['evidence'][0]['id'] for c in result]==['A','B']


@pytest.mark.parametrize('original,source,valid',[('Round','Each round ends.',True),('Round','Surround the board.',False),('US','Send us a card.',False)])
def test_terminology_recovers_source_case_without_substring_or_acronym_guess(original,source,valid):
    terms=[{'original':original,'translated':'回合','source_id':'R'}]
    claims=[{'text':'回合結束。','evidence':[{'id':'R','quote':source}]}]
    selected={'R':{'fields':{'原文':source}}}
    if valid:
        result=qa.render_terminology(claims,terms,selected)
        assert '（round；暫譯）' in result[0]['text']
        assert result[0]['evidence'][0]['quote']==source
    else:
        with pytest.raises(ValueError,match='Terminology not grounded'):
            qa.render_terminology(claims,terms,selected)


@pytest.mark.parametrize('capacity',[True,False])
def test_support_audit_filters_unsupported_addition_without_exceeding_budget(monkeypatch,capacity):
    selected={'R':{'id':'R','fields':{'原文':'Pay 5 coins.','來源':'s','PDF頁碼':'1'}},
              'B':{'id':'B','fields':{'原文':'The game has 8 rounds.','來源':'s','PDF頁碼':'1'}}}
    replies=[{'units':{i:{'relevance':'required','uncertainty':''} for i,text in [('R','支付5枚硬幣'),('B','共8回合')]}},
             {'status':'complete','claims':[{'text':'支付5枚硬幣。','source_ids':['R']},{'text':'還可以免費取牌。','source_ids':['R']}],'terminology':[],'missing_context':[],'limitations':[]}]
    if capacity:
        replies.append({'claims':{'0':{'accepted':True,'reason':'Exact cost'},'1':{'accepted':False,'reason':'No card rule in own quote'}},
                        'requirements':{'0':{'accepted':True,'reason':'Asked cost'},'1':{'accepted':False,'reason':'Round count unrelated to cost'}}})
    operations=[]
    def generate(*args,**kwargs):
        operations.append(kwargs['operation'])
        return response(replies.pop(0))
    monkeypatch.setattr(qa,'generate',generate)
    answer=qa.answer_from_evidence('費用是多少？',selected,'test',inventory_first=True,support_audit=True,can_audit=lambda:capacity)
    assert len(operations)==(3 if capacity else 2)
    assert answer['status']==('partial' if capacity else 'insufficient')
    assert [c['text'] for c in answer['claims']]==(['支付5枚硬幣。'] if capacity else [])


@pytest.mark.parametrize('background_applies',[False, True])
def test_claim_must_serve_an_audited_requirement_even_with_correct_source(monkeypatch, background_applies):
    selected={'R':{'id':'R','fields':{'原文':'Pay 5 coins.','來源':'s','PDF頁碼':'1'}},
              'B':{'id':'B','fields':{'原文':'The game has 8 rounds.','來源':'s','PDF頁碼':'1'}}}
    replies=iter([
        {'units':{i:{'relevance':'required','uncertainty':''} for i in selected}},
        {'status':'complete','claims':[{'text':'支付5枚硬幣。','source_ids':['R']},
                                        {'text':'遊戲共8回合。','source_ids':['B']}],
         'terminology':[],'missing_context':[],'limitations':[]},
        {'claims':{str(i):{'accepted':True,'reason':'Own quote supports the statement'} for i in range(2)},
         'requirements':{'0':{'accepted':True,'reason':'Asked cost'},
                         '1':{'accepted':background_applies,'reason':'Round count scope checked'}}},
    ])
    monkeypatch.setattr(qa,'generate',lambda *args,**kwargs:response(next(replies)))
    diagnostics={}
    question='費用是多少？遊戲有幾回合？' if background_applies else '只問費用是多少？'
    answer=qa.answer_from_evidence(question,selected,'test',inventory_first=True,support_audit=True,diagnostics=diagnostics)
    assert answer['status']=='complete'
    assert answer['answer']==('支付5枚硬幣。\n遊戲共8回合。' if background_applies else '支付5枚硬幣。')
    assert len(diagnostics['out_of_scope_claims'])==(0 if background_applies else 1)


@pytest.mark.parametrize('original,valid',[('Working Class',True),('WorkingClass',False),('Working Classes',False)])
def test_terminology_handles_pdf_line_wrap_without_merging_words(original,valid):
    source='The Working \nClass acts.'
    selected={'R':{'fields':{'原文':source}}}
    claims=[{'text':'勞動階級。','evidence':[{'id':'R','quote':source}]}]
    terms=[{'original':original,'translated':'勞動階級','source_id':'R'}]
    if valid:
        result=qa.render_terminology(claims,terms,selected)
        assert 'Working Class；暫譯' in result[0]['text']
        assert result[0]['evidence'][0]['quote']==source
    else:
        with pytest.raises(ValueError,match='Terminology not grounded'):
            qa.render_terminology(claims,terms,selected)


@pytest.mark.parametrize('extra_claim_generated',[False, True])
def test_inventory_candidate_pool_allows_omitting_or_filtering_different_timing_general_rule(monkeypatch, extra_claim_generated):
    selected={
        'R005':{'id':'R005','fields':{'原文':'At the start of the Production Phase, discard the Strike token from all Companies with Level 3 Wages.','來源':'S1','PDF頁碼':'16'}},
        'R009':{'id':'R009','fields':{'原文':'If a Company has a Strike token, it cannot produce in any way, including during the Action Phase.','來源':'S1','PDF頁碼':'16'}},
    }
    generated_claims=[{'text':'生產階段（Production Phase；暫譯）開始時，第3級工資的公司會棄掉罷工標記，不會因該枚罷工標記跳過生產。','source_ids':['R005']}]
    if extra_claim_generated:
        generated_claims.append({'text':'若公司有罷工標記則不能在行動階段（Action Phase；暫譯）生產。','source_ids':['R009']})
    audit_claims={'0':{'accepted':True,'reason':'Directly answers whether the Strike token prevents production at Level 3 Wages'}}
    if extra_claim_generated:
        audit_claims['1']={'accepted':True,'reason':'Accurate quote of R009'}
    replies=iter([
        {'units':{'R005':{'relevance':'required','uncertainty':''},'R009':{'relevance':'required','uncertainty':''}}},
        {'status':'complete','claims':generated_claims,
         'terminology':[{'original':'Production Phase','translated':'生產階段','source_id':'R005'}],
         'missing_context':[],'limitations':[]},
        {'claims':audit_claims,
         'requirements':{'0':{'accepted':True,'reason':'Establishes token removal at Production Phase start'},
                         '1':{'accepted':False,'reason':'General rule for active Strike token is moot after token is discarded'}}},
    ])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(next(replies)))
    diagnostics={}
    answer=qa.answer_from_evidence(
        '生產階段開始時工資為第3級，還會因罷工標記跳過生產嗎？',
        selected,'hegemony',inventory_first=True,support_audit=True,diagnostics=diagnostics
    )
    assert answer['status']=='complete'
    assert len(answer['claims'])==1
    assert answer['citations']==['R005']
    assert '行動階段' not in answer['answer']
    assert len(diagnostics['out_of_scope_claims'])==(1 if extra_claim_generated else 0)


def test_applicable_same_timing_exception_in_inventory_cannot_be_omitted(monkeypatch):
    selected={
        'R1':{'id':'R1','fields':{'原文':'Companies produce goods during the Production Phase.','來源':'S1','PDF頁碼':'16'}},
        'R2':{'id':'R2','fields':{'原文':'Companies with an active Strike token skip production.','來源':'S1','PDF頁碼':'16'}},
    }
    replies=iter([
        {'units':{'R1':{'relevance':'required','uncertainty':''},'R2':{'relevance':'required','uncertainty':''}}},
        {'status':'complete','claims':[{'text':'公司在生產階段生產商品。','source_ids':['R1']}],
         'terminology':[],'missing_context':[],'limitations':[]},
        {'claims':{'0':{'accepted':True,'reason':'States default production'}},
         'requirements':{'0':{'accepted':True,'reason':'Default production rule'},
                         '1':{'accepted':True,'reason':'Applicable same-timing Strike exception'}}},
    ])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(next(replies)))
    answer=qa.answer_from_evidence('公司在生產階段一定會生產嗎？',selected,'hegemony',inventory_first=True,support_audit=True)
    assert answer['status']!='complete'
    assert any(not c['covered'] for c in answer['coverage_checks'])


def test_claim_mixing_main_fact_and_rejected_background_number_is_rejected_after_evidence_trim(monkeypatch):
    selected={
        'R':{'id':'R','fields':{'原文':'Pay 5 coins.','來源':'s','PDF頁碼':'1'}},
        'B':{'id':'B','fields':{'原文':'The game has 8 rounds.','來源':'s','PDF頁碼':'1'}},
    }
    replies=iter([
        {'units':{'R':{'relevance':'required','uncertainty':''},'B':{'relevance':'required','uncertainty':''}}},
        {'status':'complete','claims':[{'text':'支付5枚硬幣，且遊戲共有8回合。','source_ids':['R','B']}],
         'terminology':[],'missing_context':[],'limitations':[]},
        {'claims':{'0':{'accepted':True,'reason':'Both quotes match'}},
         'requirements':{'0':{'accepted':True,'reason':'Asked cost'},
                         '1':{'accepted':False,'reason':'Round count is unrelated background'}}},
    ])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(next(replies)))
    answer=qa.answer_from_evidence('只問費用是多少？',selected,'test',inventory_first=True,support_audit=True)
    assert answer['status']=='insufficient'
    assert answer['claims']==[]
    assert '8回合' not in answer['answer']


@pytest.mark.parametrize('mixed_claim',[
    '支付5枚硬幣，之後必須跳過行動階段。',
    '支付5枚硬幣後跳過行動階段。',
])
def test_claim_mixing_main_fact_and_rejected_non_numeric_assertion_is_rejected_not_trimmed_into_complete(monkeypatch,mixed_claim):
    selected={
        'R':{'id':'R','fields':{'原文':'Pay 5 coins.','來源':'s','PDF頁碼':'1'}},
        'B':{'id':'B','fields':{'原文':'Afterwards, you must skip the Action Phase.','來源':'s','PDF頁碼':'1'}},
    }
    replies=iter([
        {'units':{'R':{'relevance':'required','uncertainty':''},'B':{'relevance':'required','uncertainty':''}}},
        {'status':'complete','claims':[{'text':mixed_claim,'source_ids':['R','B']}],
         'terminology':[],'missing_context':[],'limitations':[]},
        {'claims':{'0':{'accepted':True,'reason':'Both quotes match'}},
         'requirements':{'0':{'accepted':True,'reason':'Asked cost'},
                         '1':{'accepted':False,'reason':'Action Phase rule is rejected by audit'}}},
    ])
    monkeypatch.setattr(qa,'generate',lambda *a,**k:response(next(replies)))
    answer=qa.answer_from_evidence('只問費用是多少？',selected,'test',inventory_first=True,support_audit=True)
    assert answer['status']=='insufficient'
    assert answer['claims']==[]
    assert '跳過行動階段' not in answer['answer']
