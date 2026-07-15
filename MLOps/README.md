# MLOps — Arquitetura de Deploy (Entregável Individual)

Este documento cobre os 4 itens da etapa individual do Projeto Final (slide 6/11 do
enunciado): arquitetura funcional completa, infraestrutura via docker-compose,
monitoramento em produção e ações automatizadas/agentes de IA.

---

## 1. Arquitetura da solução

```
raw_data/                    DataPipeline/                         Model/
┌──────────────┐   ┌─────────────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ 8 CSVs Kaggle │──▶│ data_sanitization.py│──▶│ abt_transform.py │──▶│    train.py       │
│ (application, │   │  limpeza, encoding, │   │  agrega 6 tabelas│   │  K-Fold CV (valid.)│
│  bureau, prev,│   │  trata sentinelas   │   │  + features      │   │  + treino final    │
│  POS, install.,│  │                     │   │  cruzadas        │   │  (100% dos dados)  │
│  credit_card) │   └─────────┬───────────┘   └────────┬─────────┘   └─────────┬─────────┘
└──────────────┘              │                        │                       │
                               ▼                        ▼                       ▼
                        clean_data.csv               abt.csv               Model/model.pkl
                                                    (346 features)      (LGBMClassifier + metadados)
                                                                                 │
                     Orquestrado pelo Airflow                                   ▼
                  (pipeline_orchestration.py,                          Model/predict.py
                   DAG "credit_risk_pipeline":                    (busca features do cliente
                   sanitize ▸ build_abt ▸ train)                   na ABT por SK_ID_CURR)
                                                                                 │
                                                                                 ▼
                                                                    MLOps/app/main.py (FastAPI)
                                                                     POST /predict {sk_id_curr}
                                                                                 │
                                                                                 ▼
                                                                 MLOps/app/streamlit_app.py
                                                                  (UI de demo, consome a API)
```

### Componentes

| Componente | Responsabilidade | Onde |
|---|---|---|
| **Ingestão** | 8 CSVs brutos do Kaggle (Home Credit Default Risk) | `raw_data/` |
| **Sanitização** | Limpeza, encoding categórico, tratamento de sentinelas (`DAYS_EMPLOYED=365243`) | `DataPipeline/data_sanitization.py` |
| **ABT** | Agregação de bureau, previous_application, POS_CASH, installments, credit_card por cliente + features cruzadas (346 no total) | `DataPipeline/abt_transform.py` |
| **Treino** | LightGBM com Stratified K-Fold (5 folds) para **validação honesta** (AUC OOF) + treino de **um modelo final** em 100% dos dados rotulados, usando o nº médio de árvores dos folds | `Model/train.py` |
| **Artefato do modelo** | `model.pkl` — dict com o modelo LightGBM, lista de features, AUC de validação e metadados de treino | `Model/model.pkl` |
| **Serviço de predição (lógica)** | Busca as features de um cliente (por `SK_ID_CURR`) na ABT e aplica o modelo | `Model/predict.py` |
| **Serviço de predição (HTTP)** | API FastAPI — `POST /predict`, `GET /health` | `MLOps/app/main.py` |
| **Interface de demo** | Streamlit, consome a API via HTTP (não acessa o modelo direto) | `MLOps/app/streamlit_app.py` |
| **Orquestração** | Airflow — DAG `credit_risk_pipeline` (sanitize → build_abt → train), disparo manual | `MLOps/pipeline_orchestration.py` |
| **Infraestrutura** | docker-compose: Postgres (metadados do Airflow) + Airflow (webserver/scheduler) + API + Streamlit | `MLOps/docker-compose.yml` |

### Por que o serviço de predição recebe `SK_ID_CURR` e não os dados brutos do cliente?

O modelo usa 346 features vindas de 6 tabelas — a maior parte delas (histórico no
bureau de crédito, operações anteriores, comportamento de pagamento) não está
disponível num formulário de solicitação preenchido na hora. A decisão de arquitetura
(registrada com o autor em 14/07/2026) foi: o serviço recebe o ID do cliente e busca as
features já calculadas na ABT — simulando o cenário real, em que o banco já consulta o
histórico do cliente (bureau interno + histórico de operações) no momento da
solicitação, em vez de pedir esses dados digitados. Ver a docstring de
`Model/predict.py` para o detalhe completo dessa decisão, incluindo a limitação
assumida (leitura da ABT em CSV, não de um feature store de baixa latência — ver
seção 3 abaixo sobre isso como próximo passo).

### Como executar

```bash
cd MLOps
cp ../.env.example ../.env      # ajuste AIRFLOW_UID (rode `id -u`)
docker-compose --env-file ../.env up -d
```

`airflow-webserver`/`airflow-scheduler` têm `depends_on: airflow-init` com
`condition: service_completed_successfully` — o próprio `up -d` roda o init
(migração do banco + criação do usuário admin) na ordem certa.

- Airflow: http://localhost:8080 (usuário/senha: `airflow` / `airflow`) — dispare a DAG `credit_risk_pipeline`.
- API: http://localhost:8000/docs (Swagger UI interativo)
- Streamlit: http://localhost:8501

Instruções detalhadas de uso da API/CLI estão no `README.md` da raiz do projeto.

> **Nota de verificação**: este docker-compose foi validado por sintaxe (YAML e
> Dockerfiles revisados) e cada peça (API, Streamlit, `predict.py`) foi testada
> individualmente fora de container, rodando de verdade contra `Model/model.pkl` e
> `abt.csv` reais. O `docker-compose up` completo (Airflow + Postgres subindo juntos)
> **não pôde ser testado neste ambiente** — não há Docker disponível aqui. Rode
> `docker-compose up -d` localmente antes da apresentação para confirmar que os 6
> serviços sobem sem erro.

---

## 2. Modelo — fundamentação (resumo; detalhe completo no README raiz)

**LightGBM** (Gradient Boosting de árvores, com crescimento *leaf-wise*): cada árvore
nova é treinada para corrigir o erro residual das anteriores, priorizando a folha que
mais reduz a função de perda a cada split — mais eficiente que o crescimento
*level-wise* (usado por XGBoost/árvores tradicionais) para datasets grandes e tabulares
como este. Controle de overfitting via `early_stopping` (monitorando AUC de validação),
`reg_alpha`/`reg_lambda` (regularização L1/L2) e `subsample`/`colsample_bytree`
(amostragem de linhas/colunas por árvore). Hiperparâmetros otimizados via Optuna
(`Model/tune.py`) e versionados em `best_params.json`.

---

## 3. Monitoramento em produção

Conforme orientação da disciplina (aula de 02/07/2026), três frentes precisam ser
endereçadas — nem sempre as três são viáveis ao mesmo tempo, mas todas devem estar
mapeadas:

### 3.1 Dados de entrada (data drift) — sempre monitorável

Monitorar a distribuição das variáveis identificadas como mais importantes na análise
de interpretabilidade (`feature_importance.csv` / SHAP, `Analysis/evaluation.ipynb`),
comparando produção vs. o corte temporal usado no treino. No modelo atual, as
top features por importância são `EXT_SOURCE_1/2/3` (scores externos de crédito),
`PAYMENT_RATE`, `BURO_DAYS_CREDIT_MAX`, `CREDIT_GOODS_PRICE_RATIO` e `DAYS_BIRTH` — são
essas as primeiras candidatas ao monitoramento de drift.

- **Métrica sugerida**: PSI (*Population Stability Index*) ou teste KS por feature,
  comparando a distribuição da safra em produção contra a distribuição de treino.
- **Gatilho de alerta**: PSI > 0.1 (mudança moderada) ou > 0.25 (mudança relevante —
  já é prática de mercado em modelos de crédito).
- **Cadência**: o monitoramento cobre os dados a partir do corte temporal do treino em
  diante (ex.: se o treino usa dados até uma determinada safra, o monitoramento
  acompanha as safras seguintes) — nunca dados anteriores ao corte.

### 3.2 Performance do modelo — depende de ter o "real" disponível

Comparar a predição do modelo (probabilidade de inadimplência) com o desfecho real
(o cliente pagou ou não), quando esse dado existir.

- **Atraso do dado real**: só se sabe se um cliente entrou em default depois de um
  prazo de observação (tipicamente 90+ dias de atraso, no critério do próprio dataset
  Home Credit) — a performance medida hoje reflete decisões de meses atrás, não as de
  hoje. Isso precisa ser explicitado no reporte de performance, não escondido.
- **Problema do grupo de controle**: se o modelo já dispara uma ação (ex.: negar
  crédito, ou oferecer renegociação — ver seção 4) para quem tem propensão a não
  pagar, deixa de existir um grupo "não afetado" para comparar contrafactualmente —
  não há como saber se aquele cliente pagaria se a ação não tivesse sido tomada. Mitigar
  isso exigiria reservar uma fatia pequena e aleatória de clientes sem a ação do modelo
  (grupo de controle), o que tem custo direto (mais inadimplência nesse grupo) e precisa
  ser uma decisão consciente de negócio, não só técnica.
- **Métricas de acompanhamento**: AUC-ROC e KS recalculados por safra (mês de
  concessão), taxa de inadimplência real da carteira aprovada vs. prevista.

### 3.3 Monitoramento de negócio — alternativa quando o "real" não está disponível

Quando não é viável obter o desfecho real em tempo hábil, acompanhar o impacto de
negócio diretamente:

- **Perda financeira evitada**: comparar a perda por inadimplência da carteira antes e
  depois do modelo entrar em produção (um dos KPIs já definidos no escopo do projeto —
  ver README raiz, seção "Objetivo de Negócio").
- **Taxa de aprovação segura**: % de aprovações que não aumentam o risco médio da
  carteira, acompanhada mês a mês.
- Essa frente não substitui o monitoramento de performance (3.2) quando o dado real
  está disponível — é o plano B para quando não está.

### Como isso seria implementado (próximo passo — não implementado neste projeto)

Um job agendado (nova DAG do Airflow, paralela à `credit_risk_pipeline`) rodando
periodicamente: calcula PSI das top features entre a safra mais recente e a base de
treino, calcula AUC/KS quando o desfecho real estiver disponível, e escreve as métricas
num dashboard (ex.: Grafana sobre um banco de métricas, ou um notebook agendado). Alertas
disparados quando os gatilhos da seção 3.1 forem ultrapassados.

---

## 4. Ações automatizadas e agentes de IA

"O agente nasce do modelo" — a predição de probabilidade de inadimplência é o gatilho;
o agente orquestra a ação. Proposta (não implementada neste projeto — descrição
conceitual, conforme pedido no enunciado):

1. **Modelo prediz** a probabilidade de inadimplência de um cliente no momento da
   solicitação de crédito (via `POST /predict`).
2. **Roteamento por faixa de risco** (não é uma decisão binária só aprovar/negar):
   - **Risco baixo** (ex.: probabilidade < 0.2): aprovação automática.
   - **Risco médio** (ex.: entre 0.2 e 0.5): um agente de IA monta uma oferta
     alternativa (crédito menor, prazo diferente, ou uma garantia adicional) usando as
     variáveis explicativas do SHAP daquele cliente específico — ex.: se
     `ANNUITY_INCOME_PERC` (peso da parcela na renda) for a variável que mais empurrou
     o risco para cima, a oferta prioriza reduzir o valor da parcela.
   - **Risco alto** (probabilidade ≥ 0.5): nega a operação e o agente dispara uma
     comunicação (e-mail/SMS/WhatsApp) explicando o motivo em linguagem acessível e
     oferecendo um caminho alternativo (ex.: educação financeira, produto de crédito
     com garantia).
3. **Agente de acompanhamento**: para clientes já aprovados que entram numa janela de
   atraso inicial (ex.: `INSTAL_DPD` começando a subir nos meses recentes — sinal já
   presente na ABT), um agente prioriza a fila de cobrança/negociação, sugerindo o
   melhor canal e a melhor oferta de renegociação com base no histórico do cliente,
   em vez de um fluxo de cobrança genérico.
4. **Human-in-the-loop**: nenhuma decisão de risco alto é 100% automática sem
   possibilidade de revisão humana — o agente prepara a recomendação e a justificativa
   (via SHAP), mas a política de crédito da instituição define em que faixas a ação é
   automática e em que faixas precisa de aprovação humana.

Essa proposta conecta diretamente com o monitoramento da seção 3: as mesmas variáveis
usadas para explicar a decisão do agente (SHAP) são as que devem ser monitoradas quanto
a data drift, porque uma mudança na distribuição delas muda tanto a qualidade da
predição quanto a qualidade da ação automatizada.

---

## 5. Limitações conhecidas e próximos passos

- Antes da apresentação, confirme que `Model/model.pkl` foi gerado com
  `python -m Model.train` **sem** a variável `TRAIN_SAMPLE_ROWS` — ou seja, treinado no
  dataset completo (356 mil linhas), não numa amostra de debug. O artefato guarda
  `metadata["is_sample"]`; `False` confirma que é o modelo final.
- O `docker-compose up` completo não foi testado de ponta a ponta em ambiente de
  desenvolvimento (sem Docker disponível ali) — validar localmente antes da banca.
- A leitura da ABT em `Model/predict.py` é por varredura em chunks de um CSV de ~600MB
  — funciona para o escopo do projeto, mas não é a arquitetura de baixa latência que se
  usaria em produção real (um feature store, ex. Feast, Redis ou uma tabela indexada por
  `SK_ID_CURR` num banco relacional/colunar).
- `_PIP_ADDITIONAL_REQUIREMENTS` no `docker-compose.yml` instala as libs do pipeline
  toda vez que o container do Airflow sobe — aceitável para este projeto, mas o padrão
  recomendado para produção é build de uma imagem Docker customizada com as
  dependências já embutidas.
