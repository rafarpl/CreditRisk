# Home Credit Default Risk

Repositório: https://github.com/rafarpl/CreditRisk

Modelo de classificação binária para prever inadimplência de clientes de crédito.
Utiliza **LightGBM** com **K-Fold Cross Validation** e features engenheiradas a partir de 8 tabelas do dataset Home Credit.

---

## 📋 Descrição do projeto

Este projeto implementa um pipeline completo de ciência de dados para risco de crédito, cobrindo desde a limpeza dos dados brutos até o treinamento, tuning de hiperparâmetros e avaliação de um modelo de classificação. O pipeline é dividido em três grandes etapas: **sanitização dos dados**, **construção da ABT (Analytical Base Table)** e **modelagem** (baseline, tuning e treino final), com notebooks dedicados à análise exploratória e à avaliação de performance do modelo.

---

## 🎯 Objetivo de negócio

Identificar clientes com maior probabilidade de **não pagar um empréstimo** (`TARGET = 1`), permitindo que a instituição financeira tome decisões de crédito mais seguras, reduza a inadimplência e, ao mesmo tempo, mantenha a concessão de crédito inclusiva para bons pagadores.

---

## 🧭 Metodologia (resumo)

1. **Sanitização** (`DataPipeline/data_sanitization.py`): carrega os CSVs brutos (`raw_data/`), remove registros inválidos, aplica encoding binário e one-hot, trata valores sentinela (365243) e salva `clean_data.csv`.

2. **Construção da ABT** (`DataPipeline/abt_transform.py`): agrega as tabelas secundárias por cliente com estatísticas descritivas (min, max, mean, sum, var) e cria features derivadas (razões de renda, taxas de pagamento, dias de atraso). Une tudo em `abt.csv`.

3. **Modelagem** (`Model/`):
   - `baseline.py`: treina um modelo LightGBM baseline, sem tuning, para servir de referência de performance (`submission_baseline.csv`, `feature_importance_baseline.csv`).
   - `tune.py`: realiza a otimização de hiperparâmetros do LightGBM e salva o melhor conjunto encontrado em `best_params.json`.
   - `train.py`: treina o modelo final com os hiperparâmetros otimizados, usando K-Fold Cross Validation. Salva predições de teste (`submission.csv`) e a importância de features (`feature_importance.csv`). O gráfico `lgbm_importances.png` é gerado à parte, pelo notebook `evaluation.ipynb`.

4. **Avaliação e análise** (`Analysis/`):
   - `exp_analysis.ipynb`: análise exploratória dos dados limpos.
   - `evaluation.ipynb` e `evaluation_part2.ipynb`: avaliação do modelo (AUC por fold, curva ROC, calibração e importância de features).
   - `kpi_analysis.ipynb`: análise de KPIs de negócio derivados das predições do modelo.

---

## 📁 Estrutura do projeto

```
CreditRisk/
├── Analysis/
│   ├── evaluation.ipynb           → avaliação do modelo (AUC, ROC, calibração) + gera lgbm_importances.png
│   ├── evaluation_part2.ipynb     → avaliação complementar / interpretabilidade
│   ├── exp_analysis.ipynb         → análise exploratória dos dados limpos
│   └── kpi_analysis.ipynb         → análise de KPIs de negócio
│
├── DataPipeline/
│   ├── data_sanitization.py       → limpeza e padronização dos dados brutos
│   └── abt_transform.py           → construção da ABT com features agregadas
│
├── Model/
│   ├── baseline.py                → treino do modelo baseline (sem tuning)
│   ├── tune.py                    → otimização de hiperparâmetros
│   ├── train.py                   → treino final com K-Fold Cross Validation + serializa model.pkl
│   ├── predict.py                 → predição de um cliente pelo SK_ID_CURR (entregável individual)
│   └── model.pkl                  → artefato do modelo final (gerado por train.py)
│
├── raw_data/                       → CSVs originais do dataset (não versionado)
│
├── MLOps/                          → entregável individual (deploy e arquitetura)
│   ├── README.md                  → arquitetura da solução + monitoramento + agentes de IA
│   ├── docker-compose.yml         → Airflow + Postgres + API + Streamlit
│   ├── pipeline_orchestration.py  → DAG do Airflow (sanitize → build_abt → train)
│   └── app/
│       ├── main.py                → API FastAPI (POST /predict, GET /health)
│       ├── streamlit_app.py       → interface de demo (consome a API)
│       ├── Dockerfile             → imagem da API
│       ├── Dockerfile.streamlit   → imagem do Streamlit
│       └── requirements.txt       → dependências específicas do deploy
│
├── config.py                      → variáveis, caminhos e parâmetros globais do projeto
├── requirements.txt                → dependências do projeto
├── best_params.json                → hiperparâmetros otimizados (gerado por tune.py)
├── .env.example                    → template de variáveis de ambiente (docker-compose)
└── README.md
```

---

## ⚙️ Instalação

```bash
# Clone o repositório e entre na pasta do projeto
cd CreditRisk

# Crie e ative um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Instale as dependências
pip install -r requirements.txt
```

---

## 🚀 Como treinar o modelo

> ⚠️ **Importante**: como `config.py` está na raiz do projeto, todos os scripts devem ser executados **a partir da pasta raiz `CreditRisk/`**, usando o modo módulo (`-m`), para que os imports funcionem corretamente.
>
> `DATA_DIR` (usado para localizar `raw_data/`, `clean_data.csv`, `abt.csv` e
> `Model/model.pkl`) tem como padrão a própria pasta onde `config.py` está — não
> precisa configurar nada para rodar localmente. Só defina a variável de ambiente
> `DATA_DIR` manualmente se quiser apontar para um caminho diferente (é o que o
> `docker-compose.yml` faz, apontando para `/opt/project` dentro dos containers).

### 1. Sanitizar os dados brutos
```bash
python -m DataPipeline.data_sanitization
```
Gera o arquivo `clean_data.csv`.

### 2. Construir a ABT (Analytical Base Table)
```bash
python -m DataPipeline.abt_transform
```
Gera o arquivo `abt.csv`, usado como entrada para a modelagem.

### 3. (Opcional) Treinar o modelo baseline
```bash
python -m Model.baseline
```
Gera `submission_baseline.csv` e `feature_importance_baseline.csv`, servindo como referência de performance.

### 4. (Opcional) Otimizar hiperparâmetros
```bash
python -m Model.tune
```
Gera/atualiza o arquivo `best_params.json` com os melhores hiperparâmetros encontrados.

### 5. Treinar o modelo final
```bash
python -m Model.train
```
Lê `abt.csv` e `best_params.json`, treina o LightGBM com K-Fold Cross Validation (usado
só para **validação** — estima o AUC honestamente e o nº de árvores via early stopping)
e depois treina **um modelo final** em 100% dos dados rotulados. Gera:
- `submission.csv` — predições do conjunto de teste
- `feature_importance.csv` — importância das features
- `Model/model.pkl` — modelo final serializado (usado por `Model/predict.py` e pela API) — ver seção 7

> `lgbm_importances.png` (gráfico) **não** é gerado por este comando — é produzido à
> parte pelo notebook `Analysis/evaluation.ipynb` (seção 6, abaixo).

> Para rodar rápido em máquinas com pouca RAM/CPU (ex.: debug), defina a variável de
> ambiente `TRAIN_SAMPLE_ROWS` (ex.: `TRAIN_SAMPLE_ROWS=15000 python -m Model.train`)
> para treinar numa amostra. **Para a entrega final, rode sem essa variável** — o
> `model.pkl` fica marcado com `metadata["is_sample"] = True` quando treinado em amostra.

### 6. Avaliar o modelo
Abra e execute os notebooks em `Analysis/`:
```bash
jupyter notebook Analysis/evaluation.ipynb
jupyter notebook Analysis/evaluation_part2.ipynb
jupyter notebook Analysis/kpi_analysis.ipynb
```

---

## 🔮 Como executar o serviço de predição (entregável individual)

Pré-requisito: `Model/model.pkl` precisa existir (ver seção 5, "Como treinar o modelo").
O modelo é consultado por `SK_ID_CURR` (ID do cliente já presente em `abt.csv`) — não
por campos digitados num formulário. O motivo está documentado na docstring de
`Model/predict.py` e na seção 1 de `MLOps/README.md`: o modelo usa 346 features vindas
de 6 tabelas (histórico de bureau, operações anteriores, comportamento de pagamento),
que não estão disponíveis num formulário de solicitação preenchido na hora.

### Opção A — CLI direta (sem servidor)

```bash
python -m Model.predict --sk-id-curr 100002
python -m Model.predict --sk-id-curr 100002 --threshold 0.3   # ajusta o corte aprovar/negar
```

### Opção B — API FastAPI (local, sem Docker)

```bash
pip install -r MLOps/app/requirements.txt
uvicorn MLOps.app.main:app --reload --port 8000
```

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sk_id_curr": 100002, "threshold": 0.5}'
```

Documentação interativa (Swagger UI): http://localhost:8000/docs

### Opção C — Stack completa via docker-compose (Airflow + API + Streamlit)

```bash
cd MLOps
cp ../.env.example ../.env      # ajuste AIRFLOW_UID (rode `id -u` no Linux/macOS)
docker-compose --env-file ../.env up -d
```

O `airflow-webserver`/`airflow-scheduler` esperam automaticamente o `airflow-init`
terminar (migração do banco + criação do usuário) antes de subir — não precisa rodar
o init manualmente.

- **Airflow** (orquestra sanitize → build_abt → train): http://localhost:8080 (`airflow` / `airflow`)
- **API**: http://localhost:8000/docs
- **Streamlit** (interface de demo, consome a API): http://localhost:8501

Detalhes de arquitetura, monitoramento em produção e proposta de agentes de IA:
[`MLOps/README.md`](MLOps/README.md).

---

## 📦 Principais arquivos gerados

| Arquivo | Descrição |
|---|---|
| `clean_data.csv` | Dados limpos e padronizados da tabela principal |
| `abt.csv` | Tabela analítica completa (todas as features) |
| `best_params.json` | Hiperparâmetros otimizados do LightGBM |
| `Model/model.pkl` | Modelo final serializado + features + metadados (AUC, data de treino, `is_sample`) |
| `submission.csv` | Predições do modelo final para o conjunto de teste |
| `submission_baseline.csv` | Predições do modelo baseline |
| `feature_importance.csv` | Importância de features do modelo final |
| `feature_importance_baseline.csv` | Importância de features do modelo baseline |