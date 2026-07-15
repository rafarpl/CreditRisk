"""
train.py — Treinamento do modelo LightGBM com K-Fold Cross Validation.

Por que K-Fold?
    Usa TODOS os dados de treino para treinar e validar (em folds diferentes),
    dando uma estimativa mais robusta da performance real.
    A média das predições dos N folds também reduz variância.

Modelo final (entrega individual):
    O K-Fold é usado só para VALIDAÇÃO (estimar o AUC honestamente e escolher
    quantas árvores treinar via early stopping). Nenhum dos 5 modelos de fold
    é salvo para produção — eles só existiram para gerar a métrica.
    Depois da validação, treina-se UM modelo final com TODOS os dados
    rotulados (train_df inteiro), usando o número médio de árvores
    (best_iteration) encontrado nos folds. Esse é o modelo serializado em
    Model/model.pkl e consumido por predict.py / MLOps/app.

AirFlow via PythonOperator:
    task = PythonOperator(task_id="train", python_callable=run)
"""

import gc
import re
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import json

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ABT_DATA_PATH, DATA_DIR, SUBMISSION_PATH, MODEL_PATH,
    NUM_FOLDS, STRATIFIED, RANDOM_STATE,
    NON_FEATURE_COLS, LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS, LOG_PERIOD,
)

# Caminho do JSON gerado pelo tune.py 
BEST_PARAMS_JSON = os.path.join(DATA_DIR, "best_params.json")


# ============================================================
# CARREGAMENTO DE PARÂMETROS (JSON → config.py)
# ============================================================

def load_lgbm_params() -> dict:
    """
    Carrega os parâmetros do LightGBM com a seguinte prioridade:

    1. best_params.json (gerado pelo tune.py) — se existir e for válido
    2. LGBM_PARAMS do config.py              — fallback padrão

    Por que JSON em vez de importar direto do tune.py?
        Desacopla a descoberta de parâmetros (Optuna) do treino (LightGBM).
        O train.py não precisa saber nada sobre Optuna — só lê um arquivo.
        Para forçar o uso do config.py, basta deletar o JSON.
    """
    if os.path.exists(BEST_PARAMS_JSON):
        try:
            with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)

            params = data.get("params", {})
            meta   = data.get("meta", {})

            if not params:
                raise ValueError("Chave 'params' ausente ou vazia no JSON.")

            print(f"[train] Parâmetros carregados de: {BEST_PARAMS_JSON}")
            print(f"        AUC estimado (1 fold): {meta.get('best_auc_1fold', '?')}")
            print(f"        Gerado em:             {meta.get('tuned_at', '?')}")
            return params

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[train] Aviso: falha ao ler {BEST_PARAMS_JSON} ({e}).")
            print(f"        Usando LGBM_PARAMS do config.py como fallback.")

    else:
        print(f"[train] {BEST_PARAMS_JSON} não encontrado.")
        print(f"        Usando LGBM_PARAMS do config.py.")

    return LGBM_PARAMS


# ============================================================
# TREINAMENTO COM K-FOLD CROSS VALIDATION
# ============================================================

def kfold_lightgbm(df: pd.DataFrame):
    """
    Treina o LightGBM com K-Fold CV e retorna a importância de features.

    Parâmetros carregados via load_lgbm_params():
      → best_params.json (Optuna) se existir
      → LGBM_PARAMS do config.py como fallback

    Retorna uma tupla (feature_importance_df, resultado):
      - feature_importance_df: DataFrame com colunas feature/importance/fold.
      - resultado: dict com o necessário para treinar o modelo final —
        auc (float), best_iterations (list[int], um por fold), feats (list[str]),
        train_df/test_df já limpos (colunas sanitizadas).
    """
    params = load_lgbm_params()
    # Separa treino (TARGET preenchido) de teste (TARGET = NaN)
    train_df = df[df["TARGET"].notnull()].copy()
    test_df  = df[df["TARGET"].isnull()].copy()

    # Sanitiza nomes de colunas: LightGBM não aceita [ ] { } nos nomes
    # (gerados pelo pd.get_dummies — ex: 'NAME_CONTRACT_STATUS_[Approved]')
    def clean_cols(frame):
        frame.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in frame.columns]
        return frame

    train_df = clean_cols(train_df)
    test_df  = clean_cols(test_df)

    print(f"Treino: {train_df.shape} | Teste: {test_df.shape}")
    del df; gc.collect()

    # Seleciona features (exclui IDs e target)
    feats = [c for c in train_df.columns if c not in NON_FEATURE_COLS]

    # Escolha da estratégia de fold
    if STRATIFIED:
        # StratifiedKFold: mantém proporção do TARGET em cada fold
        # Recomendado quando o dataset é muito desbalanceado
        folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    else:
        # KFold padrão: divisão aleatória sem considerar proporção do TARGET
        folds = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Arrays de predição
    # oof_preds: predição de cada amostra quando estava no fold de validação
    # (estimativa honesta sem data leakage)
    oof_preds = np.zeros(train_df.shape[0])

    # sub_preds: média das predições do teste nos N folds (ensemble)
    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()
    best_iterations = []  # [ITEM] guarda o best_iteration_ de cada fold p/ o modelo final

    # ---- Loop de validação cruzada ----
    for fold_n, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[feats], train_df["TARGET"])
    ):
        train_x = train_df[feats].iloc[train_idx]
        train_y = train_df["TARGET"].iloc[train_idx]
        valid_x = train_df[feats].iloc[valid_idx]
        valid_y = train_df["TARGET"].iloc[valid_idx]

        clf = LGBMClassifier(**params)
        clf.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric="auc",
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                # Para se AUC não melhorar em N rodadas — evita overfitting
                log_evaluation(period=LOG_PERIOD),
                # Imprime métricas a cada LOG_PERIOD árvores
            ],
        )

        # Predição OOF: usa o melhor checkpoint (early stopping), não o último
        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]

        # Acumula predições de teste dividindo pelo número de folds (para a média)
        sub_preds += clf.predict_proba(
            test_df[feats], num_iteration=clf.best_iteration_
        )[:, 1] / folds.n_splits

        # Registra importância das features neste fold
        fold_imp = pd.DataFrame({
            "feature":    feats,
            "importance": clf.feature_importances_,
            "fold":       fold_n + 1,
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)

        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        print(f"Fold {fold_n + 1:2d} | AUC: {fold_auc:.6f} | best iter: {clf.best_iteration_}")

        best_iterations.append(clf.best_iteration_)

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # AUC final: concatena todas as predições OOF — estimativa mais honesta
    full_auc = roc_auc_score(train_df["TARGET"], oof_preds)
    print(f"\nAUC total (OOF): {full_auc:.6f}")

    # Salva submissão
    test_df["TARGET"] = sub_preds
    test_df[["SK_ID_CURR", "TARGET"]].to_csv(SUBMISSION_PATH, index=False)
    print(f"Submissão salva em: {SUBMISSION_PATH}")

    resultado = {
        "auc": full_auc,
        "best_iterations": best_iterations,
        "feats": feats,
        "train_df": train_df,
        "params": params,
    }
    return feature_importance_df, resultado


# ============================================================
# MODELO FINAL (treinado em 100% dos dados rotulados)
# ============================================================

def train_final_model(resultado: dict) -> dict:
    """
    Treina o modelo que efetivamente vai para produção.

    Por que não reaproveitar um dos 5 modelos do K-Fold?
        Cada modelo de fold viu só ~80% dos dados rotulados. Depois que o
        K-Fold já nos deu uma estimativa honesta de AUC (resultado['auc']),
        o modelo final deve aproveitar TODOS os dados rotulados — mais dado
        tende a generalizar melhor, e não há mais necessidade de reservar
        uma fatia para validação (isso já foi feito).

    Por que n_estimators = média dos best_iteration_ dos folds, sem early
    stopping de novo?
        Não sobra um conjunto de validação "não visto" para o early stopping
        monitorar (usamos tudo para treinar). A média do número de árvores
        em que os 5 modelos do K-Fold pararam é a melhor estimativa
        disponível de quantas árvores evitam overfitting neste dataset.

    Retorna o dict de metadados salvo em Model/model.pkl junto com o modelo.
    """
    feats = resultado["feats"]
    train_df = resultado["train_df"]
    params = dict(resultado["params"])

    avg_best_iteration = int(round(np.mean(resultado["best_iterations"])))
    params["n_estimators"] = avg_best_iteration

    print(f"\n[train_final_model] Treinando modelo final com {len(train_df)} "
          f"linhas rotuladas e n_estimators={avg_best_iteration} "
          f"(média dos {len(resultado['best_iterations'])} folds).")

    final_model = LGBMClassifier(**params)
    final_model.fit(train_df[feats], train_df["TARGET"])

    metadata = {
        "model": final_model,
        "features": feats,
        "target_column": "TARGET",
        "id_column": "SK_ID_CURR",
        "auc_oof": resultado["auc"],
        "n_estimators": avg_best_iteration,
        "n_folds": len(resultado["best_iterations"]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "lgbm_params": params,
    }

    # A serialização final (joblib.dump) acontece em run(), depois que
    # metadata["is_sample"] é definido — evita salvar o arquivo duas vezes.
    return metadata


# ============================================================
# PONTO DE ENTRADA (VS Code e AirFlow)
# ============================================================

def run(sample_rows: int | None = None):
    """
    Lê a ABT, treina o modelo e salva a submissão.
    Chamável diretamente (python train.py) ou via AirFlow PythonOperator.

    sample_rows:
        Se definido (ou via env var TRAIN_SAMPLE_ROWS), treina numa AMOSTRA
        do abt.csv em vez do dataset completo — só para checagem rápida do
        pipeline em máquinas com pouca memória/CPU (ex.: sandbox de CI, sem
        recursos para as ~356 mil linhas x 348 colunas do dataset real).
        O model.pkl gerado nesse modo fica marcado com
        metadata["is_sample"] = True e NÃO deve ser usado como entrega final
        — para a entrega, rode `python -m Model.train` sem essa variável.
    """
    sample_rows = sample_rows or (
        int(os.environ["TRAIN_SAMPLE_ROWS"]) if os.environ.get("TRAIN_SAMPLE_ROWS") else None
    )

    print("=== Iniciando treinamento ===")

    if sample_rows:
        # Amostra linhas de treino (início do arquivo) + uma fatia do final
        # (onde ficam as linhas de teste, TARGET nulo) para exercitar também
        # o caminho de geração de submission.csv.
        df_train_part = pd.read_csv(ABT_DATA_PATH, nrows=sample_rows)
        total_rows = sum(1 for _ in open(ABT_DATA_PATH)) - 1
        test_tail = max(min(sample_rows // 5, total_rows - sample_rows), 1)
        df_test_part = pd.read_csv(
            ABT_DATA_PATH, skiprows=range(1, max(total_rows - test_tail, 1))
        )
        df = pd.concat([df_train_part, df_test_part], ignore_index=True)
        print(f"[run] AMOSTRA de {len(df)} linhas (TRAIN_SAMPLE_ROWS={sample_rows}) "
              f"de {total_rows} totais — NÃO é o dataset completo.")
    else:
        df = pd.read_csv(ABT_DATA_PATH)

    # Converte colunas object/string remanescentes (segurança antes do LightGBM)
    # select_dtypes(include=["object", "str"]) quebra em pandas >= 2.2 com
    # TypeError ("numpy string dtypes are not allowed") mesmo sem colunas de
    # texto — mesmo bug já corrigido em abt_transform.py. Usamos
    # pd.api.types.is_string_dtype, que cobre tanto "object" quanto o dtype
    # "str"/"string" nativo, sem essa incompatibilidade.
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    for col in text_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feat_importance, resultado = kfold_lightgbm(df)

    # Salva importância de features para uso no evaluation.ipynb
    feat_importance.to_csv("feature_importance.csv", index=False)
    print("Importância de features salva em feature_importance.csv")

    # Treina o modelo final (em memória) e só então serializa em disco
    metadata = train_final_model(resultado)
    metadata["is_sample"] = bool(sample_rows)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(metadata, MODEL_PATH)
    print(f"[run] Modelo salvo em: {MODEL_PATH}")
    if sample_rows:
        print("[run] AVISO: model.pkl foi treinado em AMOSTRA. Rode sem "
              "TRAIN_SAMPLE_ROWS para gerar o modelo final da entrega.")


if __name__ == "__main__":
    run()