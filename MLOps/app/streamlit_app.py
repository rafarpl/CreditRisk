"""
streamlit_app.py — Interface de demo (consome a API via HTTP, não o modelo direto).

Por que não carregar o modelo direto aqui, como um protótipo típico faria?
    Separação de camadas: a API (main.py) é o serviço de predição "real"
    (contrato HTTP, validação via Pydantic, pode ser chamado por qualquer
    cliente — Streamlit, curl, outro backend). O Streamlit é só uma
    interface de demonstração para a apresentação individual, e roda num
    container separado no docker-compose. Ver MLOps/README.md, seção de
    arquitetura.

Layout: 3 colunas iguais, numa única visualização (sem scroll durante a
demo ao vivo):
    1. Identificação do cliente + threshold + 4 variáveis.
    2. Demais 5 variáveis + botão de calcular.
    3. Resultado (probabilidade, decisão, JSON detalhado).
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Rótulos das features ajustáveis (API_FEATURES em Model/predict.py) — a UI
# não hardcoda a lista, só os rótulos amigáveis; se a feature não vier no
# snapshot da API, o campo simplesmente não aparece.
FEATURE_LABELS = {
    "PAYMENT_RATE": "Taxa parcela / crédito",
    "ANNUITY_INCOME_PERC": "Parcela sobre a renda",
    "INCOME_CREDIT_PERC": "Renda sobre o crédito",
    "DEBT_INCOME_RATIO": "Dívida sobre a renda",
    "DAYS_EMPLOYED_PERC": "Tempo empregado / idade",
    "DAYS_BIRTH": "Idade (anos)",
    "EXT_SOURCE_1": "Score externo 1",
    "EXT_SOURCE_2": "Score externo 2",
    "EXT_SOURCE_3": "Score externo 3",
}

# DAYS_BIRTH é salvo como dias negativos (ex.: -14600). Convertida pra
# idade em anos só na UI — mais intuitivo pra simular "e se o cliente
# fosse mais velho?" do que pedir pra digitar um número negativo de dias.
DAYS_PER_YEAR = 365.25

DEFAULT_SK_ID = 100002

# Distribuição das 9 API_FEATURES nas colunas 1 e 2, na ordem pedida.
COL1_FEATURES = ["PAYMENT_RATE", "INCOME_CREDIT_PERC", "DAYS_EMPLOYED_PERC", "ANNUITY_INCOME_PERC"]
COL2_FEATURES = ["DAYS_BIRTH", "DEBT_INCOME_RATIO", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

st.set_page_config(page_title="Credit Risk — Demo", page_icon="💳", layout="wide")

# Paleta alinhada à apresentação da banca (fundo navy escuro + acentos
# teal/azul do slide "Por que LightGBM?").
st.markdown(
    """
    <style>
    :root {
        --bg: #0a1420;
        --card-bg: #101f33;
        --border: #1c3a52;
        --accent-teal: #2dd4bf;
        --accent-blue: #38bdf8;
        --text: #e5edf5;
        --text-muted: #8fa3ba;
        --success: #34d399;
        --danger: #f87171;
    }
    .stApp { background-color: var(--bg); color: var(--text); }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    h1, h2, h3, h4 { color: var(--text) !important; }
    p, span, label { color: var(--text); }
    [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }
    [data-testid="stWidgetLabel"] p { color: var(--accent-blue) !important; font-weight: 600; }
    .stButton > button {
        background-color: var(--accent-teal);
        color: #06231f;
        border: none;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover { background-color: #24b8a4; color: #06231f; }
    [data-testid="stMetricValue"] { color: var(--accent-teal); }
    [data-testid="stMetricLabel"] { color: var(--text-muted); }
    .decisao-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .decisao-aprovar { background-color: rgba(52, 211, 153, 0.15); color: var(--success); border: 1px solid var(--success); }
    .decisao-negar { background-color: rgba(248, 113, 113, 0.15); color: var(--danger); border: 1px solid var(--danger); }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_client():
    """Callback do number_input do SK_ID_CURR — busca as features reais assim
    que o usuário troca o ID, sem precisar de um botão separado."""
    sk_id_curr = st.session_state.sk_id_curr_input
    try:
        resp = requests.get(f"{API_URL}/client/{int(sk_id_curr)}", timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.session_state["fetch_error"] = str(e)
        st.session_state.pop("baseline", None)
        return
    st.session_state["fetch_error"] = None
    st.session_state["fetched_id"] = int(sk_id_curr)
    st.session_state["baseline"] = resp.json()["features"]
    st.session_state.pop("result", None)  # cliente novo invalida resultado anterior


def render_feature_input(feature: str, baseline: dict):
    """Renderiza o campo ajustável de uma feature, pré-preenchido com o
    valor real do cliente. DAYS_BIRTH é o único caso especial (vira "Idade
    (anos)" na UI — ver DAYS_PER_YEAR)."""
    label = FEATURE_LABELS[feature]
    real_value = baseline.get(feature)

    if feature == "DAYS_BIRTH":
        default_age = round(-real_value / DAYS_PER_YEAR, 1) if real_value is not None else 40.0
        age = st.number_input(label, min_value=18.0, max_value=100.0, value=default_age, step=1.0)
        return -age * DAYS_PER_YEAR

    return st.number_input(label, value=float(real_value) if real_value is not None else 0.0, format="%.4f")


st.title("Credit Risk — Probabilidade de Inadimplência")
st.caption(f"Consumindo API em: {API_URL}")

# Primeira carga da página: busca o cliente default automaticamente, antes
# mesmo do widget SK_ID_CURR existir (ele é criado dentro da coluna 1).
if "baseline" not in st.session_state and "fetch_error" not in st.session_state:
    st.session_state["sk_id_curr_input"] = DEFAULT_SK_ID
    fetch_client()

if st.session_state.get("fetch_error"):
    st.error(f"Falha ao buscar cliente na API ({API_URL}): {st.session_state['fetch_error']}")

baseline = st.session_state.get("baseline", {})
overrides = {}

inputs_col, result_col = st.columns([2, 1], gap="large")

with inputs_col:
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "SK_ID_CURR do cliente",
                min_value=1,
                value=DEFAULT_SK_ID,
                step=1,
                key="sk_id_curr_input",
                on_change=fetch_client,
            )
            threshold = st.slider("Threshold (aprovar/negar)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            for feature in COL1_FEATURES:
                if feature in baseline:
                    overrides[feature] = render_feature_input(feature, baseline)

        with col2:
            for feature in COL2_FEATURES:
                if feature in baseline:
                    overrides[feature] = render_feature_input(feature, baseline)

            calcular = st.button("Calcular predição", disabled=not baseline)
            if calcular:
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "sk_id_curr": st.session_state["fetched_id"],
                            "threshold": threshold,
                            "overrides": overrides,
                        },
                        timeout=30,
                    )
                except requests.exceptions.RequestException as e:
                    st.session_state["result"] = None
                    st.error(f"Falha ao conectar na API ({API_URL}): {e}")
                else:
                    if resp.status_code == 200:
                        st.session_state["result"] = resp.json()
                    else:
                        st.session_state["result"] = None
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except ValueError:
                            detail = resp.text
                        st.error(f"Erro {resp.status_code}: {detail}")

with result_col:
    with st.container(border=True):
        st.markdown("#### Resultado")
        data = st.session_state.get("result")
        if data:
            st.metric("Probabilidade de inadimplência", f"{data['probabilidade_inadimplencia']:.1%}")

            badge_class = "decisao-aprovar" if data["decisao_sugerida"] == "APROVAR" else "decisao-negar"
            st.markdown(
                f'<span class="decisao-badge {badge_class}">{data["decisao_sugerida"]}</span>',
                unsafe_allow_html=True,
            )

            if data.get("model_is_sample"):
                st.warning(
                    "Modelo treinado numa AMOSTRA dos dados (ambiente de "
                    "desenvolvimento) — não é o modelo final da entrega."
                )

            with st.expander("Detalhes (JSON)"):
                st.json(data)
        else:
            st.caption("Ajuste as variáveis e clique em \"Calcular predição\".")

st.divider()
st.caption(
    "O cliente precisa existir na ABT (abt.csv) — o serviço busca as "
    "features já calculadas pelo pipeline (bureau, previous_application etc.), "
    "não recebe dados brutos digitados no formulário. Ver Model/predict.py "
    "para a justificativa dessa decisão de arquitetura."
)
