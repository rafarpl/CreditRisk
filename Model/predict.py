"""
predict.py — Serviço de predição (entregável da etapa individual).

Contrato de entrada (decisão registrada com o autor em 14/07/2026):
    O modelo usa ~346 features agregadas de 6 tabelas (application, bureau,
    previous_application, POS_CASH, installments, credit_card) por
    SK_ID_CURR — não é viável pedir esses campos brutos num formulário/JSON.
    predict.py e a API (MLOps/app) recebem apenas o SK_ID_CURR do cliente e
    reaproveitam as features já calculadas na ABT (abt.csv). Isso simula o
    cenário real: no momento da solicitação de crédito, o banco já consulta
    o histórico do cliente (bureau interno + histórico de operações) por ID
    — não pede para o cliente digitar 348 campos manualmente.

    Em produção, essa leitura viria de um feature store (baixa latência),
    não de um CSV de 600MB. Aqui lemos a ABT em chunks (sem carregar o
    arquivo inteiro em memória) como uma aproximação razoável dentro do
    escopo do projeto — ver MLOps/README.md para a proposta de arquitetura
    completa.

Uso via CLI:
    python -m Model.predict --sk-id-curr 100001
    python -m Model.predict --sk-id-curr 100001 --threshold 0.3

Uso programático (API, notebooks, Airflow):
    from Model.predict import predict_by_id
    predict_by_id(100001)
"""

import argparse
import json
import re
import sys, os

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ABT_DATA_PATH, MODEL_PATH, ID_COLUMN, TARGET_COLUMN

# Ponto de corte padrão para a decisão binária (aprovar/negar).
# Decisão de negócio, não estatística — instituições reais calibram esse
# valor de acordo com o apetite de risco da carteira (ver MLOps/README.md,
# seção de governança). Fica configurável de propósito, tanto na função
# quanto no --threshold da CLI.
DEFAULT_THRESHOLD = 0.5


def _sanitize(col: str) -> str:
    """
    Mesma sanitização de nomes de coluna aplicada em Model/train.py
    (clean_cols) e DataPipeline/data_sanitization.py (sanitize_column_names):
    LightGBM não aceita [ ] { } nos nomes de feature.

    Por que repetir aqui em vez de importar de train.py?
        Os nomes salvos em metadata["features"] (Model/model.pkl) já estão
        sanitizados — mas o abt.csv em disco ainda tem os nomes originais
        (a sanitização final só acontece dentro de kfold_lightgbm, em
        memória, não é persistida de volta no CSV). Sem repetir essa regra
        aqui, o predict.py não conseguiria casar as colunas do abt.csv com
        a lista de features que o modelo espera.
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", col)


def load_model(model_path: str = MODEL_PATH) -> dict:
    """Carrega o artefato salvo por Model/train.py: dict com model + features + metadados."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Rode `python -m Model.train` primeiro."
        )
    return joblib.load(model_path)


def fetch_client_features(
    sk_id_curr: int,
    features: list,
    abt_path: str = ABT_DATA_PATH,
    chunksize: int = 50_000,
) -> pd.Series:
    """
    Busca a linha de um cliente na ABT sem carregar o arquivo inteiro em memória.

    Percorre abt.csv em chunks e para assim que encontra o SK_ID_CURR — evita
    ler as ~350 colunas x centenas de milhares de linhas por completo só para
    achar 1 cliente (relevante rodando num container com recursos limitados).
    """
    # Descobre o cabeçalho real do CSV e mapeia nome-sanitizado -> nome-em-disco,
    # pois metadata["features"] guarda os nomes já sanitizados (ver _sanitize).
    header = pd.read_csv(abt_path, nrows=0).columns.tolist()
    clean_to_raw = {_sanitize(c): c for c in header}

    id_raw = clean_to_raw.get(ID_COLUMN, ID_COLUMN)
    target_raw = clean_to_raw.get(TARGET_COLUMN, TARGET_COLUMN)

    wanted_raw = [clean_to_raw[f] for f in features if f in clean_to_raw]
    missing = [f for f in features if f not in clean_to_raw]
    if missing:
        print(
            f"[predict] aviso: {len(missing)} feature(s) do modelo não "
            f"encontradas na ABT atual (viram NaN): {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    usecols = list(dict.fromkeys(wanted_raw + [id_raw, target_raw]))

    for chunk in pd.read_csv(abt_path, usecols=usecols, chunksize=chunksize):
        chunk.columns = [_sanitize(c) for c in chunk.columns]
        match = chunk[chunk[ID_COLUMN] == sk_id_curr]
        if not match.empty:
            return match.iloc[0]

    raise KeyError(f"SK_ID_CURR={sk_id_curr} não encontrado em {abt_path}.")


def predict_by_id(
    sk_id_curr: int,
    threshold: float = DEFAULT_THRESHOLD,
    model_path: str = MODEL_PATH,
) -> dict:
    """
    Retorna a probabilidade de inadimplência (TARGET=1) para um cliente.

    threshold: corte de decisão (default 0.5). Ver docstring de
    DEFAULT_THRESHOLD sobre por que isso é uma escolha de negócio.
    """
    artifact = load_model(model_path)
    model = artifact["model"]
    features = artifact["features"]

    row = fetch_client_features(sk_id_curr, features)

    # reindex (não seleção direta) tolera features ausentes na ABT atual —
    # viram NaN, que o LightGBM trata nativamente sem quebrar a predição.
    x = row.reindex(features).to_frame().T.apply(pd.to_numeric, errors="coerce")

    proba = float(model.predict_proba(x)[:, 1][0])
    decisao = "NEGAR" if proba >= threshold else "APROVAR"

    target_real = row.get(TARGET_COLUMN)
    target_real = None if target_real is None or pd.isna(target_real) else int(target_real)

    return {
        "sk_id_curr": sk_id_curr,
        "probabilidade_inadimplencia": round(proba, 6),
        "threshold": threshold,
        "decisao_sugerida": decisao,
        "target_real": target_real,  # None quando o cliente é do conjunto de teste (sem rótulo)
        "model_auc_oof": artifact.get("auc_oof"),
        "model_trained_at": artifact.get("trained_at"),
        "model_is_sample": artifact.get("is_sample", False),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prediz a probabilidade de inadimplência de um cliente pelo SK_ID_CURR."
    )
    parser.add_argument(
        "--sk-id-curr", type=int, required=True,
        help="ID do cliente (SK_ID_CURR) presente em abt.csv.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Corte de decisão aprovar/negar (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    try:
        resultado = predict_by_id(args.sk_id_curr, args.threshold, args.model_path)
    except (FileNotFoundError, KeyError) as e:
        print(f"[predict] erro: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
