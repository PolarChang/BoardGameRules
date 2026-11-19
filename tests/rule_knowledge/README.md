# Model-free verification dependency slice

This branch stages the Python source closure needed by the isolated publication/source fixture and its focused tests. No book, database, environment file, cache, virtualenv, model weight, credential or deployment setting is included. The regular app entry point and production requirements are unchanged.

Install only `requirements-ci.txt`, then run `python -m pytest tests/rule_knowledge -q`. For frontend E2E, launch `integration_server.py` with KNOWLEDGE_INTEGRATION_BACKEND_PORT set to a free loopback port. It creates new temporary SQLite/source storage, disables local environment loading and model calls, and serves generated rules/PDF images. The health response is an explicit isolation assertion.

Frontend CI must pin this repository to an immutable tested commit. This source slice is not evidence for the full RAG application, cloud storage, live models or production compatibility; those require separate review. Do not merge or deploy merely to run the isolated fixture.
