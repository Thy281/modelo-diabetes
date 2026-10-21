# %%

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_excel("dados/dados_diabetes.xlsx")
linhas_antes = len(df)
df.drop_duplicates()
print(f"Duplicatas removidas: {linhas_antes - len(df)}")
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

print(f"Dados carregados: {len(df)}")
print(f"Dados de treino: {len(X_treino)}")
print(f"Dados de teste: {len(X_teste)}")
print(f"Casos na matriz: {len(y_teste)}")

model = BernoulliNB()
model.fit(X_treino, y_treino)

predict = model.predict(X_teste)
accuracy = accuracy_score(y_teste, predict)

# %%

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

valores_matriz = confusion_matrix(y_teste, predict, labels = [0, 1]).T
matriz = ConfusionMatrixDisplay(
    confusion_matrix = valores_matriz,
    display_labels = ["Sem diabetes", "Com diabetes"],
)
matriz.plot(
    cmap = "Blues",
    values_format = "d",
    colorbar = False,
)

matriz.ax_.set_title(f"Matriz de confusão\nAcurácia: {accuracy:.2%}")
matriz.ax_.xaxis.set_label_position("top")
matriz.ax_.xaxis.tick_top()
matriz.ax_.set_xlabel("Classe real")
matriz.ax_.yaxis.set_label_position("left")
matriz.ax_.yaxis.tick_left()
matriz.ax_.set_ylabel("Classe prevista pelo modelo")
plt.figtext(0.5, 0.01, "Diagonal = acertos | Fora da diagonal = erros", ha="center")
plt.tight_layout(rect = [0, 0.04, 1, 1])
plt.show()

# %%
