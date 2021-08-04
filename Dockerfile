FROM --platform=linux/amd64 python:3.8

WORKDIR /app
COPY requirements.txt . 
