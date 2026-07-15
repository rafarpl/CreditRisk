"""
data_sanitization.py — Limpeza e padronização dos dados brutos.

Responsabilidade: ler os CSVs brutos, aplicar limpezas básicas (valores
sentinela, tipos incorretos, registros inválidos) e salvar clean_data.csv.

AirFlow via PythonOperator:
    task = PythonOperator(task_id="sanitize", python_callable=run)
"""

import gc
import re

import numpy as np
import pandas as pd

# Importa configurações centralizadas
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DATA, CLEAN_DATA_PATH, NUM_ROWS, SENTINEL_VALUE, PREV_APP_DATE_COLS


# ============================================================
# UTILITÁRIOS
# ============================================================

def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    """
    Converte colunas categóricas (dtype 'object') em colunas binárias 0/1.

    Por que: LightGBM e a maioria dos modelos não aceitam strings.
    nan_as_category=True cria coluna extra '_nan' para ausências,
    pois o fato de um valor estar ausente pode ser informativo por si só.

    Retorna: (df_transformado, lista_de_novas_colunas)
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove caracteres especiais dos nomes de colunas.

    LightGBM não aceita [ ] { } nos nomes das features porque usa JSON
    internamente. Ocorre naturalmente depois do get_dummies (ex:
    'NAME_CONTRACT_STATUS_[Approved]'). Substitui tudo que não seja
    letra/número/underscore por underscore.
    """
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in df.columns]
    return df


# ============================================================
# CARREGAMENTO E LIMPEZA DA TABELA PRINCIPAL
# ============================================================

def load_application(num_rows=None) -> pd.DataFrame:
    """
    Carrega application_train.csv + application_test.csv e aplica limpezas básicas.

    Estratégia de concatenação: une treino e teste antes do encoding para
    garantir que as mesmas colunas dummies existam nos dois conjuntos.
    """
    df      = pd.read_csv(RAW_DATA["application_train"], nrows=num_rows)
    test_df = pd.read_csv(RAW_DATA["application_test"],  nrows=num_rows)
    print(f"Train: {len(df)} linhas | Test: {len(test_df)} linhas")

    df = pd.concat([df, test_df], axis=0).reset_index(drop=True)

    # Remove 4 registros com CODE_GENDER = 'XNA' (valor inválido)
    df = df[df["CODE_GENDER"] != "XNA"]

    # Encoding binário para features com exatamente 2 categorias
    # pd.factorize → 0/1 | economiza colunas vs get_dummies que criaria 2
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col], _ = pd.factorize(df[col])

    # One-hot nas demais categóricas
    df, _ = one_hot_encoder(df, nan_as_category=False)

    # Trata valor sentinela: 365243 em DAYS_EMPLOYED significa "sem emprego"
    # Manter causaria o modelo interpretar o cliente como empregado há ~1000 anos
    #
    # NOTA sobre Copy-on-Write (pandas 2.x/3 com CoW ativado):
    # df["col"].replace(..., inplace=True) opera numa cópia intermediária e
    # NUNCA atualiza o df original — falha silenciosamente (gera apenas o
    # ChainedAssignmentError como aviso). A forma correta é reatribuir a coluna
    # ou usar df.replace({"col": valor}, inplace=True).
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(SENTINEL_VALUE, np.nan)

    del test_df
    gc.collect()
    return df


# ============================================================
# LIMPEZA DAS TABELAS SECUNDÁRIAS
# ============================================================

def load_bureau(num_rows=None) -> pd.DataFrame:
    """
    Carrega bureau.csv e bureau_balance.csv e aplica encoding.
    Substitui sentinel em datas e remove SK_ID_BUREAU após o join.
    """
    bureau = pd.read_csv(RAW_DATA["bureau"],         nrows=num_rows)
    bb     = pd.read_csv(RAW_DATA["bureau_balance"], nrows=num_rows)

    bb,     _ = one_hot_encoder(bb,     nan_as_category=True)
    bureau, _ = one_hot_encoder(bureau, nan_as_category=True)

    return bureau, bb


def load_previous_applications(num_rows=None) -> pd.DataFrame:
    """
    Carrega previous_application.csv, aplica encoding e trata sentinelas de datas.
    Colunas de datas preenchidas com 365243 indicam que o evento não ocorreu.
    """
    prev, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["previous_app"], nrows=num_rows),
        nan_as_category=True,
    )

    for col in PREV_APP_DATE_COLS:
        # Mesma correção do load_application(): reatribuir em vez de inplace=True
        prev[col] = prev[col].replace(SENTINEL_VALUE, np.nan)

    return prev


def load_pos_cash(num_rows=None) -> pd.DataFrame:
    pos, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["pos_cash"], nrows=num_rows),
        nan_as_category=True,
    )
    return pos


def load_installments(num_rows=None) -> pd.DataFrame:
    ins, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["installments"], nrows=num_rows),
        nan_as_category=True,
    )
    return ins


def load_credit_card(num_rows=None) -> pd.DataFrame:
    cc, _ = one_hot_encoder(
        pd.read_csv(RAW_DATA["credit_card"], nrows=num_rows),
        nan_as_category=True,
    )
    # SK_ID_PREV é chave de contrato, não feature — remove antes de agregar
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)
    return cc


# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def run():
    """
    Executa a sanitização completa e salva clean_data.csv.
    Chamável diretamente (python data_sanitization.py) ou via AirFlow PythonOperator.
    """
    print("=== Iniciando sanitização dos dados ===")

    df = load_application(NUM_ROWS)
    df = sanitize_column_names(df)

    # Converte colunas object/string remanescentes para numérico.
    # select_dtypes(include=["object", "str"]) quebra em pandas >= 2.2 com
    # TypeError ("numpy string dtypes are not allowed"), mesmo dataframe sem
    # nenhuma coluna de texto — mesmo bug já corrigido em abt_transform.py.
    # pd.api.types.is_string_dtype cobre "object" e o dtype "str"/"string"
    # nativo sem essa incompatibilidade, então nenhuma coluna de texto passa
    # intacta para o clean_data.csv (o que quebraria modelos lineares como a
    # Regressão Logística mais adiante no pipeline).
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    for col in text_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"clean_data.csv salvo em {CLEAN_DATA_PATH} | shape: {df.shape}")
    gc.collect()


if __name__ == "__main__":
    run()