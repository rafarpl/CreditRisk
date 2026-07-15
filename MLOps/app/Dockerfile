# Dockerfile — serviço de predição (API FastAPI).
#
# Contexto de build esperado: raiz do repositório (..), não esta pasta —
# ver MLOps/docker-compose.yml (`build.context: ..`). É por isso que os
# caminhos de COPY abaixo começam com "MLOps/".

FROM python:3.11-slim

WORKDIR /opt/project

# libgomp1 fornece libgomp.so.1 (runtime OpenMP), exigido em tempo de
# execução pela lib nativa do LightGBM — sem isso, `import lightgbm` falha
# com "OSError: libgomp.so.1: cannot open shared object file", mesmo com o
# pacote lightgbm instalado via pip (a wheel não embute essa dependência de
# sistema; no macOS o equivalente é o `libomp` do Homebrew).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY MLOps/app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# O código do projeto (Model/, config.py, MLOps/app/) e os artefatos
# (abt.csv, Model/model.pkl) entram via volume no docker-compose, não via
# COPY — evita reconstruir a imagem a cada mudança de código/dado e evita
# inflar a imagem com arquivos grandes que nem sempre existem no momento do
# build (abt.csv não é versionado).

EXPOSE 8000

CMD ["uvicorn", "MLOps.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
