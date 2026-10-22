import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from .config import DATA_PATH, FEATURES, MODEL_PATH


def main():
    df = pd.read_excel(DATA_PATH).drop_duplicates()
    colunas_necessarias = FEATURES + ["diabetes"]
    colunas_ausentes = [coluna for coluna in colunas_necessarias if coluna not in df.columns]

    if colunas_ausentes:
        raise ValueError(f"Colunas ausentes na planilha: {colunas_ausentes}")

    if df[colunas_necessarias].isna().any().any():
        raise ValueError("A planilha contém valores vazios nas features ou no alvo diabetes.")

    if not set(df["diabetes"].unique()).issubset({0, 1}):
        raise ValueError("A coluna diabetes deve conter somente 0 e 1.")

    modelo = GaussianNB()
    modelo.fit(df[FEATURES], df["diabetes"])
    joblib.dump(modelo, MODEL_PATH)
    print(f"Modelo treinado com {len(df)} registros e salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
