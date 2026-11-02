FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-install-project

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "-m", "main"]
