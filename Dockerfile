# Description: Dockerfile for the FastAPI application
FROM python:3.11.0-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /code/requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY ./app /code/app

COPY ./.env.local /code/.env.local

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
