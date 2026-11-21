"""Operational state derived from durable work; readiness is not content quality."""
def describe(*,source_state,source_complete,readable_pages,total,model_state,
             category_pending=0,category_waiting=0,retries=0,repair_pending=0,repair_gaps=0,gaps=False):
    searchable=readable_pages>0
    wait=None
    if source_state=='waiting_retry': wait='source_retry'
    elif category_waiting: wait='local_service'
    elif model_state=='paused': wait='paused'
    elif model_state=='queued': wait='worker_queue'
    elif repair_pending: wait='repair_queue'
    if source_state=='failed' and not searchable:
        stage='failed'
    elif not searchable:
        stage='extracting' if source_state in {'extracting','done'} or model_state=='processing' else 'queued'
    elif not source_complete and not (source_state=='failed' and model_state=='review'):
        stage='searchable'
    elif model_state=='review' and not category_pending and not category_waiting and not retries and not repair_pending:
        stage='ready_with_gaps' if gaps or repair_gaps or readable_pages<total or source_state=='failed' else 'ready'
    else:
        stage='enriching'
    return {'stage':stage,'searchable':searchable,'readable_pages':readable_pages,
            'total_pages':total,'wait_reason':wait,'repair_pending':repair_pending,'repair_gaps':repair_gaps,'content_verified':False}
