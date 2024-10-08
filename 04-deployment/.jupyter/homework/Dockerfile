# FROM python:3.10.13-slim
# 
# WORKDIR /app
# COPY [ "model.bin", "./"]

FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY [ "starter.py", "Pipfile", "Pipfile.lock", "./" ]

RUN pip install pipenv
RUN pipenv install --system --deploy

CMD python3 starter.py