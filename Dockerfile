FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-install-project

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "-m", "src.optimization.squad_optimizer"]
