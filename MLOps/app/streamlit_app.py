"""
streamlit_app.py — Interface de demo (consome a API via HTTP, não o modelo direto).

Por que não carregar o modelo direto aqui, como um protótipo típico faria?
    Separação de camadas: a API (main.py) é o serviço de predição "real"
    (contrato HTTP, validação via Pydantic, pode ser chamado por qualquer
    cliente — Streamlit, curl, outro backend). O Streamlit é só uma
    interface de demonstração para a apresentação individual, e roda num
    container separado no docker-compose. Ver MLOps/README.md, seção de
    arquitetura.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk — Demo", page_icon="💳")
st.title("Credit Risk — Probabilidade de Inadimplência")
st.caption(f"Consumindo API em: {API_URL}")

sk_id_curr = st.number_input("SK_ID_CURR do cliente", min_value=1, value=100002, step=1)
threshold = st.slider("Threshold de decisão (aprovar/negar)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Consultar predição"):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"sk_id_curr": int(sk_id_curr), "threshold": threshold},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Falha ao conectar na API ({API_URL}): {e}")
    else:
        if resp.status_code == 200:
            data = resp.json()
            col1, col2 = st.columns(2)
            col1.metric("Probabilidade de inadimplência", f"{data['probabilidade_inadimplencia']:.1%}")
            col2.metric("Decisão sugerida", data["decisao_sugerida"])
            st.json(data)
            if data.get("model_is_sample"):
                st.warning(
                    "Este modelo foi treinado numa AMOSTRA dos dados "
                    "(ambiente de desenvolvimento) — não é o modelo final da entrega."
                )
        else:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            st.error(f"Erro {resp.status_code}: {detail}")

st.divider()
st.caption(
    "O cliente precisa existir na ABT (abt.csv) — o serviço busca as "
    "features já calculadas pelo pipeline (bureau, previous_application etc.), "
    "não recebe dados brutos digitados no formulário. Ver Model/predict.py "
    "para a justificativa dessa decisão de arquitetura."
)
