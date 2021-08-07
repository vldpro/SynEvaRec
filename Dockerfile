FROM --platform=linux/amd64 python:3.8

WORKDIR /app
RUN pip install --upgrade pip && pip install poetry==1.1.7
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install
