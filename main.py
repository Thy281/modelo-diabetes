# %%

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

import pandas as pd

df = pd.read_excel("dados/dados_diabetes.xlsx")
df.drop_duplicates()
# %%

fetures = [
    "sede_acima_normal",
    "vontade_banheiro_acima_normal",
    "historico_familiar",
    "imc_alto",
    "glicose_alta",
]

X = df[fetures]
y = df["diabetes"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size = 0.25,
    random_state = 42,
    stratify = y,
)

model = BernoulliNB()
model.fit(X_treino, y_treino)

predict = model.predict(X_teste)

# %%
