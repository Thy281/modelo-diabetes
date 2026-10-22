from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "dados" / "dados_diabetes.xlsx"
MODEL_PATH = ROOT_DIR / "modelo_diabetes.joblib"
FEATURES = [
    "idade",
    "imc",
    "glicemia",
    "sede_excessiva",
    "urina_frequente",
    "fadiga",
    "historico_familiar",
    "atividade_fisica",
    "pressao_sistolica",
]
