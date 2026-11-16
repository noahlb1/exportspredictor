# One image serves both the API and the Streamlit app; compose.yaml picks the command.
# Artefacts are not baked in: compose mounts ./artefacts as a volume.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /srv/exportspredictor

# Dependencies first, so code edits don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY api ./api
COPY app ./app
RUN uv sync --frozen --no-dev

ENV PATH="/srv/exportspredictor/.venv/bin:$PATH"

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
