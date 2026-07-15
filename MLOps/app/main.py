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

from Model.predict import DEFAULT_THRESHOLD, MODEL_PATH, predict_by_id

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
        return predict_by_id(req.sk_id_curr, req.threshold)
    except FileNotFoundError as e:
        # Modelo não treinado ainda — erro de disponibilidade do serviço, não do cliente.
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError as e:
        # SK_ID_CURR não existe na ABT.
        raise HTTPException(status_code=404, detail=str(e))
