# FROM python:3.10.13-slim
# 
# WORKDIR /app
# COPY [ "model2.bin", "model.bin" ]
# 
# RUN pipenv install --system --deploy

FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY ["starter.py", "Pipfile", "Pipfile.lock"]
RUN pipenv install --system --deploy

CMD ["starter"]