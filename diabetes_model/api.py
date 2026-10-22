from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import FEATURES, MODEL_PATH


if not MODEL_PATH.exists():
    raise RuntimeError("Modelo não encontrado. Execute `python -m diabetes_model.training` primeiro.")

modelo = joblib.load(MODEL_PATH)
app = FastAPI(
    title="API educativa do modelo de diabetes",
    description="Recebe as features do quiz e retorna uma previsão do GaussianNB.",
    version="1.0.0",
)


class RespostasQuiz(BaseModel):
    idade: int = Field(ge=0)
    imc: float = Field(gt=0)
    glicemia: float = Field(gt=0)
    sede_excessiva: Literal[0, 1]
    urina_frequente: Literal[0, 1]
    fadiga: Literal[0, 1]
    historico_familiar: Literal[0, 1]
    atividade_fisica: int = Field(ge=0, le=7)
    pressao_sistolica: float = Field(gt=0)


class ResultadoPrevisao(BaseModel):
    classe_prevista: Literal[0, 1]
    probabilidade_modelo: float = Field(ge=0, le=1)
    aviso: str


@app.get("/health")
def health():
    return {"status": "ok", "modelo": "GaussianNB"}


@app.post("/predict", response_model=ResultadoPrevisao)
def predict(respostas: RespostasQuiz):
    entrada = pd.DataFrame([respostas.model_dump()], columns=FEATURES)
    classe = int(modelo.predict(entrada)[0])
    indice_diabetes = list(modelo.classes_).index(1)
    probabilidade = float(modelo.predict_proba(entrada)[0][indice_diabetes])

    return ResultadoPrevisao(
        classe_prevista=classe,
        probabilidade_modelo=probabilidade,
        aviso="Resultado educativo; não é diagnóstico médico.",
    )
