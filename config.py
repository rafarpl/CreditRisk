"""
config.py — Variáveis, parâmetros e metadados do pipeline de dados.

Centralizar configurações aqui evita hardcoding espalhado pelo código:
altere um valor aqui e todos os scripts herdam automaticamente.
"""

import os

# ============================================================
# CAMINHOS DE DADOS
# ============================================================
# Ajuste DATA_DIR para o diretório onde seus CSVs estão localmente.
DATA_DIR = os.environ.get("DATA_DIR", "/Users/ronaldo.costa/dev/CreditRisk")

RAW_DATA = {
    "application_train": os.path.join(DATA_DIR, "raw_data", "application_train.csv"),
    "application_test":  os.path.join(DATA_DIR, "raw_data", "application_test.csv"),
    "bureau":            os.path.join(DATA_DIR, "raw_data", "bureau.csv"),
    "bureau_balance":    os.path.join(DATA_DIR, "raw_data", "bureau_balance.csv"),
    "previous_app":      os.path.join(DATA_DIR, "raw_data", "previous_application.csv"),
    "pos_cash":          os.path.join(DATA_DIR, "raw_data", "POS_CASH_balance.csv"),
    "installments":      os.path.join(DATA_DIR, "raw_data", "installments_payments.csv"),
    "credit_card":       os.path.join(DATA_DIR, "raw_data", "credit_card_balance.csv"),
}

CLEAN_DATA_PATH  = os.path.join(DATA_DIR, "clean_data.csv")
ABT_DATA_PATH    = os.path.join(DATA_DIR, "abt.csv")

# ============================================================
# PARÂMETROS DO PIPELINE
# ============================================================
# num_rows=None carrega tudo; defina um inteiro (ex: 10000) para debug rápido.
NUM_ROWS = None
NAN_AS_CATEGORY = True  # Cria coluna "_nan" para ausências em variáveis categóricas

# Valor sentinela usado no dataset original para "sem data" / "sem emprego"
SENTINEL_VALUE = 365243

# ============================================================
# METADADOS DO PROJETO
# ============================================================
PROJECT_NAME    = "Home Credit Default Risk"
TARGET_COLUMN   = "TARGET"
ID_COLUMN       = "SK_ID_CURR"
RANDOM_STATE    = 1001

# ============================================================
# SAÍDA / SUBMISSÃO
# ============================================================
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission.csv")

# ============================================================
# VALIDAÇÃO CRUZADA
# ============================================================
NUM_FOLDS  = 5
STRATIFIED = True   # usa StratifiedKFold (recomendado p/ target desbalanceado)

# ============================================================
# FEATURES
# ============================================================
# Colunas que NÃO são features (IDs, target, índices auxiliares)
NON_FEATURE_COLS = [ID_COLUMN, TARGET_COLUMN, "SK_ID_BUREAU", "SK_ID_PREV", "index"]

# ============================================================
# LIGHTGBM
# ============================================================
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 24,
    "max_depth": 8,
    "min_child_samples": 100,
    "subsample": 0.87,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.15,
    "reg_lambda": 0.20,
    "n_estimators": 10000,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 200
LOG_PERIOD            = 100

# ============================================================
# REGRESSÃO LOGÍSTICA (BASELINE)
# ============================================================
# Usado por baseline.py para comparar o ganho real do LightGBM.
# max_iter alto porque, com centenas de features, a convergência é mais lenta.
LOGREG_PARAMS = {
    "max_iter": 1000,
    "C": 0.1,          # regularização forte — útil com tantas features (evita overfit)
    "random_state": RANDOM_STATE,
}

BASELINE_SUBMISSION_PATH = os.path.join(DATA_DIR, "submission_baseline.csv")

# ============================================================
# CONFIGURAÇÕES: PREVIOUS APPLICATIONS
# ============================================================
PREV_APP_DATE_COLS = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION",
]

PREV_APP_NUM_AGG = {
    "AMT_ANNUITY":             ["min", "max", "mean"],
    "AMT_APPLICATION":         ["min", "max", "mean"],
    "AMT_CREDIT":              ["min", "max", "mean"],
    "APP_CREDIT_PERC":         ["min", "max", "mean", "var"],
    "AMT_DOWN_PAYMENT":        ["min", "max", "mean"],
    "AMT_GOODS_PRICE":         ["min", "max", "mean"],
    "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
    "RATE_DOWN_PAYMENT":       ["min", "max", "mean"],
    "DAYS_DECISION":           ["min", "max", "mean"],
    "CNT_PAYMENT":             ["mean", "sum"],
}

# ============================================================
# CONFIGURAÇÕES: BUREAU E BUREAU BALANCE
# ============================================================
BUREAU_SCORE_MAP = {
    "STATUS_0": 0, "STATUS_1": 1, "STATUS_2": 2,
    "STATUS_3": 3, "STATUS_4": 4, "STATUS_5": 5
}

BUREAU_BALANCE_AGG_SPEC = {
    "MONTHS_BALANCE":       ["min", "max", "size"],
    "STATUS_SCORE":         ["max", "mean"],
    "STATUS_SCORE_RECENT6": ["mean"]
}

BUREAU_NUM_AGG = {
    "DAYS_CREDIT":            ["min", "max", "mean", "var"],
    "DAYS_CREDIT_ENDDATE":    ["min", "max", "mean"],
    "DAYS_CREDIT_UPDATE":     ["mean"],
    "CREDIT_DAY_OVERDUE":     ["max", "mean"],
    "AMT_CREDIT_MAX_OVERDUE": ["mean"],
    "AMT_CREDIT_SUM":         ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_DEBT":    ["max", "mean", "sum"],
    "AMT_CREDIT_SUM_OVERDUE": ["mean"],
    "AMT_CREDIT_SUM_LIMIT":   ["mean", "sum"],
    "AMT_ANNUITY":            ["max", "mean"],
    "CNT_CREDIT_PROLONG":     ["sum"],
    "MONTHS_BALANCE_MIN":     ["min"],
    "MONTHS_BALANCE_MAX":     ["max"],
    "MONTHS_BALANCE_SIZE":    ["mean", "sum"],
  # [ITEM 1] Propaga severidade/recência/tendência do nível crédito → cliente
    "STATUS_SCORE_MAX":       ["max", "mean"],
    "STATUS_SCORE_MEAN":      ["mean"],
    "STATUS_SCORE_RECENT6_MEAN": ["mean"],
    "STATUS_SCORE_TREND":     ["mean", "max"],
}

# ============================================================
# CONFIGURAÇÕES: POS CASH BALANCE
# ============================================================
POS_CASH_AGG_BASE = {
    "MONTHS_BALANCE": ["max", "mean", "size"],
    "SK_DPD":         ["max", "mean"],   # max = pior atraso; mean = comportamento médio
    "SK_DPD_DEF":     ["max", "mean"],   # versão "default" do atraso (threshold mais rígido) 
}

# ============================================================
# CONFIGURAÇÕES: INSTALLMENTS PAYMENTS
# ============================================================
INSTALLMENTS_EXPLICIT_COLS = {
    "NUM_INSTALMENT_VERSION", "DPD", "DBD", "PAYMENT_PERC",
    "PAYMENT_DIFF", "AMT_INSTALMENT", "AMT_PAYMENT", "DAYS_ENTRY_PAYMENT"
}

INSTALLMENTS_AGG_BASE = {
    "NUM_INSTALMENT_VERSION": ["nunique"], 
    "DPD":                    ["max", "mean", "sum"],
    "DBD":                    ["max", "mean", "sum"],
    "PAYMENT_PERC":           ["max", "mean", "sum", "var"],
    "PAYMENT_DIFF":           ["max", "mean", "sum", "var"],
    "AMT_INSTALMENT":         ["max", "mean", "sum"],
    "AMT_PAYMENT":            ["min", "max", "mean", "sum"],
    "DAYS_ENTRY_PAYMENT":     ["max", "mean", "sum"],
}

# ============================================================
# CONFIGURAÇÕES: CREDIT CARD BALANCE
# ============================================================
CC_AGG_FUNCS = ["min", "max", "mean", "sum", "var"]