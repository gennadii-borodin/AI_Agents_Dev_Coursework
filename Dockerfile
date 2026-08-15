FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip

COPY pyproject.toml .

RUN pip install .

COPY . .

RUN pip install -e ".[dev]"

CMD ["python", "-m", "src.main"]
