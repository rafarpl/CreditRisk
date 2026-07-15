"""
pipeline_orchestration.py — DAG do Airflow que orquestra o pipeline de
produção: sanitização -> construção da ABT -> treino do modelo final.

Entregável da etapa individual (arquitetura de deploy).

Como o Airflow encontra esta DAG:
    MLOps/docker-compose.yml monta este arquivo direto em
    /opt/airflow/dags/pipeline_orchestration.py (bind mount de um único
    arquivo) — não precisa copiar/duplicar em MLOps/dags/.

Por que PythonOperator chamando as funções run() dos scripts, em vez de
BashOperator + `python script.py`?
    Cada script (data_sanitization.py, abt_transform.py, Model/train.py) já
    expõe uma função run() como ponto de entrada único, pensada exatamente
    para isso — reaproveita a mesma lógica usada na execução local (CLI),
    sem duplicar código nem passar por uma camada de shell.

O que esta DAG NÃO faz (de propósito):
    Não expõe o modelo como serviço — isso é responsabilidade da API
    (MLOps/app/main.py), que roda como um serviço à parte, sempre no ar,
    lendo o Model/model.pkl mais recente gerado por esta DAG. Orquestração
    de treino e serviço de predição são responsabilidades diferentes.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# DATA_DIR é setado no docker-compose (aponta pra raiz do projeto montada
# no container, ex.: /opt/project). Sem ele, os imports abaixo falham.
PROJECT_ROOT = os.environ.get("DATA_DIR", "/opt/project")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _run_sanitization():
    from DataPipeline.data_sanitization import run
    run()


def _run_abt_transform():
    from DataPipeline.abt_transform import run
    run()


def _run_train():
    from Model.train import run
    # TRAIN_SAMPLE_ROWS (opcional, via variável de ambiente do container)
    # permite rodar a DAG numa amostra em ambientes com pouca RAM/CPU — ver
    # docstring de Model.train.run(). Em produção, essa env var não deve
    # estar setada (treina no dataset completo).
    run()


default_args = {
    "owner": "rafa",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="credit_risk_pipeline",
    description="Sanitização -> ABT -> Treino do modelo de risco de crédito.",
    default_args=default_args,
    schedule=None,  # disparo manual — ver MLOps/README.md sobre frequência de retreino
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["credit-risk", "tcc-individual"],
) as dag:

    sanitize = PythonOperator(
        task_id="sanitize_data",
        python_callable=_run_sanitization,
    )

    build_abt = PythonOperator(
        task_id="build_abt",
        python_callable=_run_abt_transform,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=_run_train,
    )

    sanitize >> build_abt >> train
