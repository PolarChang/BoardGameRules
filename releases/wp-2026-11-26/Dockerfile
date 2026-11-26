FROM us-central1-docker.pkg.dev/boardgame-rules-507516/cloud-run-source-deploy/boardgame-rules-api@sha256:9aa2d0630b41f9c282592111643c86e818edfd3d9e1f9cb089982e676786f33b
COPY releases/wp-2026-11-26/check.py /tmp/wp-base-check.py
RUN python /tmp/wp-base-check.py
COPY src/api.py src/game_catalog_api.py src/local_grounded_answer.py /app/src/
RUN python -c "import src.api,src.local_grounded_answer"
