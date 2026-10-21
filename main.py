# %%

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score

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
    test_size = 0.25, # 25% do test
    random_state = 42,
    stratify = y,
)

model = BernoulliNB()
model.fit(X_treino, y_treino)

predict = model.predict(X_teste)
accuracy = accuracy_score(y_teste, predict)
print(f"Acurácia: {accuracy:.2%}")

# %%

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_teste,
    predict,
    display_labels = ["Sem diabetes", "Com diabetes"],
    cmap = "Blues",
)

plt.title("Matriz de confusão")
plt.show()

# %%
