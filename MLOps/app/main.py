"""
main.py — API de predição (FastAPI). Entregável da etapa individual.

Expõe o modelo treinado (Model/model.pkl) como um serviço HTTP:
    POST /predict {"sk_id_curr": 100002, "threshold": 0.5}
    GET  /health

Reaproveita a lógica de Model/predict.py (predict_by_id) em vez de duplicar
a leitura da ABT/modelo — um único ponto de verdade para a predição, usado
tanto pela CLI (`python -m Model.predict`) quanto por esta API.
"""

import os
import sys
from typing import Optional

# Garante que o repositório (raiz, onde ficam config.py e Model/) está no
# sys.path, mesmo se a API for iniciada de um cwd diferente. Em produção
# (docker-compose), DATA_DIR já aponta pra raiz montada no container.
_REPO_ROOT = os.environ.get(
    "DATA_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Model.predict import (
    API_FEATURES,
    DEFAULT_THRESHOLD,
    MODEL_PATH,
    get_client_snapshot,
    predict_by_id,
)

app = FastAPI(
    title="Credit Risk — Serviço de Predição",
    description=(
        "Probabilidade de inadimplência (Home Credit Default Risk) por "
        "SK_ID_CURR. Ver Model/predict.py para a justificativa do contrato "
        "de entrada (por que ID e não campos brutos)."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    sk_id_curr: int = Field(..., description="ID do cliente (SK_ID_CURR), presente na ABT (abt.csv).")
    threshold: float = Field(
        DEFAULT_THRESHOLD, ge=0.0, le=1.0,
        description="Corte de decisão aprovar/negar (default 0.5).",
    )
    overrides: Optional[dict[str, float]] = Field(
        None,
        description=(
            "Sobrescreve valores de features do cliente para simular cenários "
            f"hipotéticos (ex.: {API_FEATURES}). Features fora dessa lista "
            "também são aceitas, mas não têm suporte na UI de demo."
        ),
    )


class ClientSnapshotResponse(BaseModel):
    sk_id_curr: int
    features: dict[str, Optional[float]]


class PredictResponse(BaseModel):
    sk_id_curr: int
    probabilidade_inadimplencia: float
    threshold: float
    decisao_sugerida: str
    target_real: Optional[int] = None
    model_auc_oof: Optional[float] = None
    model_trained_at: Optional[str] = None
    model_is_sample: bool = False


@app.get("/health")
def health():
    """Checagem simples: API no ar + modelo presente no caminho esperado."""
    model_ok = os.path.exists(MODEL_PATH)
    return {
        "status": "ok" if model_ok else "modelo_ausente",
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return predict_by_id(req.sk_id_curr, req.threshold, overrides=req.overrides)
    except FileNotFoundError as e:
        # Modelo não treinado ainda — erro de disponibilidade do serviço, não do cliente.
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError as e:
        # SK_ID_CURR não existe na ABT.
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/client/{sk_id_curr}", response_model=ClientSnapshotResponse)
def client_snapshot(sk_id_curr: int):
    """
    Valores reais atuais das API_FEATURES de um cliente — usado pela UI de
    demo para pré-preencher os campos ajustáveis antes de simular um
    cenário (POST /predict com `overrides`).
    """
    try:
        return {"sk_id_curr": sk_id_curr, "features": get_client_snapshot(sk_id_curr)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
