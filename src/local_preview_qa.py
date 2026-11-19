"""Local-only preview pipeline, bounded to translation + answer + verification."""
import threading
from fastapi import HTTPException
from src.local_import_budget import BudgetExceeded

_lock=threading.Lock()
_engines=None


def reserve_budget():
    from src.local_qa_budget import reserve
    return reserve()


def refund_budget(reservation,unused):
    # Durable dispatch count is authoritative, including after a process restart.
    from src.local_qa_budget import finish
    finish(reservation)


def ask(snapshot,question,game_id,*,diagnostics=None,confirmed_context=None):
    global _engines
    # Serialize model/index mutation as well as inference: concurrent snapshots
    # must never share an in-flight candidate set.
    if not _lock.acquire(blocking=False):
        raise HTTPException(409,'另一筆本機問答正在處理，請稍後再試。')
    day=None; spent=0; stage='source_tools'
    def consume():
        nonlocal spent
        if spent>=3: raise ValueError('Generation budget exceeded')
        from src.local_qa_budget import consume as consume_reserved
        consume_reserved(day)
        spent+=1
    try:
        from src.local_question_context import missing_observation, context_query
        retrieval_question=context_query(question,confirmed_context)
        from src.local_grounded_answer import context_answer
        missing=missing_observation(question)
        if missing:
            return {**context_answer(missing),'version':snapshot['version'],'retrieved_ids':[],
                    'pipeline':'local-claims-v1','budget':{'max_generation_calls':3,'generation_attempts':0}}
        from src.source_rule_tools import natural_tools
        tool_answer=natural_tools(snapshot['document']['blocks'],question) if not confirmed_context else None
        if tool_answer is not None:
            return {**tool_answer,'version':snapshot['version'],'pipeline':'source-tools-v1'}
        day=reserve_budget()
        stage='load_models'
        from src.local_semantic import SemanticIndex
        from src.local_evidence_retrieval import Reranker, retrieve_evidence
        from src.local_grounded_answer import answer_from_evidence, context_answer
        if _engines is None:
            _engines=(SemanticIndex('minilm',game_id),Reranker(game_id))
        index,reranker=_engines
        index.game_id=reranker.game_id=game_id
        blocks=snapshot['document']['blocks']
        stage='embedding_index'
        index.prepare(blocks)
        metadata={}
        stage='retrieval'
        # Multilingual embeddings and lexical aliases do not consume a text
        # generation. Reserve all three calls for inventory, answer and audit,
        # independently of whether this question has a translation cache hit.
        selected=retrieve_evidence(blocks,retrieval_question,index,reranker,game_id,translate=False,metadata=metadata,on_inference=consume)
        metadata['query_policy']='direct-multilingual-no-generation-v1'
        if diagnostics is not None:
            diagnostics.update(retrieval=metadata,selected=selected)
        if metadata.get('missing_context'):
            return {**context_answer(metadata['missing_context']),'version':snapshot['version'],'retrieved_ids':[],
                    'pipeline':'local-claims-v1','budget':{'max_generation_calls':3,'generation_attempts':spent}}
        tool_answer=natural_tools([b for b in blocks if b['id'] in selected],question,table_query='\n'.join(metadata['queries'])) if not confirmed_context else None
        if tool_answer is not None:
            return {**tool_answer,'version':snapshot['version'],'pipeline':'source-tools-v1'}
        stage='answer_verification'
        from src.gemini_usage import setting
        thinking=setting('LOCAL_QA_REASONING_ENABLED','false').lower()=='true'
        answer_options={'on_inference':consume,'inventory_first':True,'thinking':thinking,
                        'support_audit':True,'can_audit':lambda:spent<3}
        if diagnostics is not None:
            diagnostics['generation_policy']={'model':'qwen3:8b','thinking':thinking,'stage_timeout_seconds':120 if thinking else 60,'max_generation_calls':3}
        if confirmed_context:
            answer_options['confirmed_context']=confirmed_context
        if diagnostics is not None:
            answer_options['diagnostics']=diagnostics.setdefault('answer',{})
        answer=answer_from_evidence(question,selected,game_id,**answer_options)
        return {**answer,'version':snapshot['version'],'retrieved_ids':list(selected),
                'coverage_gaps':answer.get('limitations',[]),'pipeline':'local-claims-v1',
                'budget':{'max_generation_calls':3,'generation_attempts':spent}}
    except BudgetExceeded as error:
        return {'status':'insufficient','kind':'abstained','answer':error.user_message,'stop_reason':error.reason,
                'citations':[],'retrieved_ids':[],'version':snapshot['version'],'pipeline':'local-claims-v1'}
    except HTTPException:
        raise
    except Exception as error:
        return {'status':'insufficient','kind':'abstained','answer':'本機查詢未能完成；來源仍可閱讀，請稍後重試。',
                'citations':[],'retrieved_ids':[],'failure_stage':stage,'failure_type':type(error).__name__,
                'version':snapshot['version'],'pipeline':'local-claims-v1'}
    finally:
        try:
            if day is not None: refund_budget(day,3-spent)
        finally: _lock.release()
