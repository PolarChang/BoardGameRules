"""Explicit, local-only consumer for durable private question jobs."""
import argparse
import os
import time
import multiprocessing
import threading

from src import import_question_jobs as jobs


def execute(lease):
    from src.import_jobs import PreviewQuestion, preview_ask
    from src.gemini_usage import setting
    payload = {k:v for k,v in lease['question'].items() if k != 'timeout_seconds'}
    key = setting('KNOWLEDGE_ADMIN_KEY')
    if key and not os.getenv('KNOWLEDGE_ADMIN_KEY'):
        os.environ['KNOWLEDGE_ADMIN_KEY'] = key
    return preview_ask(lease['game_id'], lease['job_id'], PreviewQuestion(**payload), key)


def child_entry(lease, paths, sender, target):
    # Spawn avoids sharing mutable model state and works on Windows. No secret
    # or lease token is placed on a shell command line.
    from pathlib import Path
    from src import database, local_inference_queue
    database.DB_PATH = Path(paths['database'])
    local_inference_queue.QUEUE_PATH = Path(paths['queue'])
    remaining = max(0, lease['deadline'] - time.time())
    cutoff = time.monotonic() + remaining
    parent = multiprocessing.parent_process()
    def watchdog():
        while True:
            if time.monotonic() >= cutoff or (parent and not parent.is_alive()):
                os._exit(70)
            time.sleep(0.1)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        from src.local_usage import execution_scope
        with execution_scope(lease['game_id'], lease['id']):
            result = target(lease)
        sender.send({'result': result, 'error': result.get('failure_type') or result.get('stop_reason') or ''})
    except Exception as error:
        sender.send({'result': None, 'error': type(error).__name__})
    finally:
        sender.close()


def supervise(lease, target=execute):
    from src import database, local_inference_queue
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=child_entry, args=(lease, {
        'database': str(database.DB_PATH), 'queue': str(local_inference_queue.QUEUE_PATH),
    }, sender, target))
    cutoff = time.monotonic() + max(0, lease['deadline'] - time.time())
    def interrupted():
        # EOF can arrive after the child's deadline watchdog exits but before
        # the supervisor's next deadline check. Exit 70 is reserved for that
        # watchdog; its parent is still alive here.
        process.join(timeout=0.2)
        if process.exitcode == 70 or time.monotonic() >= cutoff:
            jobs.timeout_execution(lease['id'], lease['token'])
        else:
            jobs.finish(lease['id'], lease['token'], error='worker_process_interrupted')
    try:
        process.start(); sender.close()
        while True:
            status = jobs.execution_status(lease['id'], lease['token'])
            if time.monotonic() >= cutoff:
                jobs.timeout_execution(lease['id'], lease['token'])
                break
            if status != 'running':
                break
            if receiver.poll(0.1):
                try:
                    message = receiver.recv()
                except EOFError:
                    interrupted()
                else:
                    jobs.finish(lease['id'], lease['token'], **message)
                break
            if not process.is_alive():
                interrupted()
                break
    finally:
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill(); process.join(timeout=5)
            if process.is_alive(): raise RuntimeError('Unable to stop question process')
            process.close()
        sender.close(); receiver.close()
    from src.local_usage import interrupt_execution
    from src.local_qa_budget import finish_execution
    interrupt_execution(lease['game_id'], lease['id'])
    finish_execution(lease['game_id'], lease['id'])
    # A cancel request may race with result delivery. The conditional update
    # leaves completed/failed/timed-out states intact and settles that race.
    jobs.acknowledge_cancel(lease['id'], lease['token'])


def run_once():
    from src.gemini_usage import setting
    key = setting('KNOWLEDGE_ADMIN_KEY')
    if not key:
        raise RuntimeError('KNOWLEDGE_ADMIN_KEY is required')
    lease = jobs.claim_next()
    if lease is None:
        return False
    try:
        supervise(lease)
    except Exception as error:
        # Do not persist provider exception messages, which may contain credentials.
        jobs.finish(lease['id'], lease['token'], error=type(error).__name__)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Keep consuming queued questions; default processes at most one')
    args = parser.parse_args()
    while True:
        worked = run_once()
        if not args.watch:
            return
        if not worked:
            time.sleep(2)


if __name__ == '__main__':
    main()
