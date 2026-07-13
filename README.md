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
   - `train.py`: treina o modelo final com os hiperparâmetros otimizados, usando K-Fold Cross Validation. Salva predições OOF e de teste (`submission.csv`), além da importância de features (`feature_importance.csv`, `lgbm_importances.png`).

4. **Avaliação e análise** (`Analysis/`):
   - `exp_analysis.ipynb`: análise exploratória dos dados limpos.
   - `evaluation.ipynb` e `evaluation_part2.ipynb`: avaliação do modelo (AUC por fold, curva ROC, calibração e importância de features).
   - `kpi_analysis.ipynb`: análise de KPIs de negócio derivados das predições do modelo.

---

## 📁 Estrutura do projeto

```
CreditRisk/
├── Analysis/
│   ├── evaluation.ipynb           → avaliação do modelo (AUC, ROC, calibração)
│   ├── evaluation_part2.ipynb     → avaliação complementar / interpretabilidade
│   ├── exp_analysis.ipynb         → análise exploratória dos dados limpos
│   └── kpi_analysis.ipynb         → análise de KPIs de negócio
│
├── DataPipeline/
│   ├── data_sanitization.py       → limpeza e padronização dos dados brutos
│   ├── abt_transform.py           → construção da ABT com features agregadas
│   └── mnt/                       → ponto de montagem / dados auxiliares
│
├── Model/
│   ├── baseline.py                → treino do modelo baseline (sem tuning)
│   ├── tune.py                    → otimização de hiperparâmetros
│   ├── train.py                   → treino final com K-Fold Cross Validation
│   └── raw_data/                  → CSVs originais do dataset
│
├── config.py                      → variáveis, caminhos e parâmetros globais do projeto
├── requirements.txt                → dependências do projeto
├── best_params.json                → hiperparâmetros otimizados (gerado por tune.py)
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
Lê `abt.csv` e `best_params.json`, treina o LightGBM com K-Fold Cross Validation e gera:
- `submission.csv` — predições do conjunto de teste
- `feature_importance.csv` — importância das features
- `lgbm_importances.png` — gráfico de importância das features

### 6. Avaliar o modelo
Abra e execute os notebooks em `Analysis/`:
```bash
jupyter notebook Analysis/evaluation.ipynb
jupyter notebook Analysis/evaluation_part2.ipynb
jupyter notebook Analysis/kpi_analysis.ipynb
```

---

## 📦 Principais arquivos gerados

| Arquivo | Descrição |
|---|---|
| `clean_data.csv` | Dados limpos e padronizados da tabela principal |
| `abt.csv` | Tabela analítica completa (todas as features) |
| `best_params.json` | Hiperparâmetros otimizados do LightGBM |
| `submission.csv` | Predições do modelo final para o conjunto de teste |
| `submission_baseline.csv` | Predições do modelo baseline |
| `feature_importance.csv` | Importância de features do modelo final |
| `feature_importance_baseline.csv` | Importância de features do modelo baseline |