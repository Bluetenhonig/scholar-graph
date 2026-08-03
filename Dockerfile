# syntax=docker/dockerfile:1
#
# Multi-stage: dependencies resolve once in a layer that only changes when the
# lockfile does, so an application edit rebuilds in seconds.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, without the project itself: this layer is cached until
# pyproject.toml or the lockfile changes.
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra panel || \
    uv sync --no-install-project --no-dev --extra panel

COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra panel


FROM python:3.12-slim AS runtime

# Run as a non-root user. The agent executes no untrusted code, but a container
# that does not need root should not have it.
RUN useradd --create-home --uid 10001 scholar

WORKDIR /app

COPY --from=builder --chown=scholar:scholar /app/.venv /app/.venv
COPY --from=builder --chown=scholar:scholar /app/src /app/src
COPY --chown=scholar:scholar cassettes ./cassettes
COPY --chown=scholar:scholar evals ./evals

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SCHOLAR_GRAPH_LOG_FORMAT=json \
    SCHOLAR_GRAPH_CHECKPOINT_DB=/data/checkpoints.sqlite

RUN mkdir -p /data && chown scholar:scholar /data
VOLUME ["/data"]

USER scholar
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

CMD ["uvicorn", "scholar_graph.api:app", "--host", "0.0.0.0", "--port", "8000"]
