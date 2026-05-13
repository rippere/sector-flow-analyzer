FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install uv && uv pip install --system -e ".[dev]"

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:////data/sector_flow.db

ENTRYPOINT ["sector-flow"]
