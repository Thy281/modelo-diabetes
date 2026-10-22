# %%

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

import pandas as pd

df = pd.read_excel("dados/dados_diabetes_100.xlsx")
linhas_antes = len(df)
df = df.drop_duplicates()
print(f"Duplicatas removidas: {linhas_antes - len(df)}")

print(pd.crosstab(df["glicose_alta"], df["diabetes"]))

print(
    pd.crosstab(
        df["glicose_alta"],
        df["diabetes"],
        normalize="index"
    )
)
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

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)
modelo = BernoulliNB()

scores = cross_val_score(
    modelo,
    X_treino,
    y_treino,
    cv = cv,
    scoring = "accuracy"
)

print("Acurácias por divisão:", scores)
print(f"Acurácia média: {scores.mean():.2%}")

print("==========")

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)
modelo_gau = GaussianNB()

scores_gau = cross_val_score(
    modelo_gau,
    X_treino,
    y_treino,
    cv = cv,
    scoring = "accuracy"
)

print("Acurácias por divisão:", scores_gau)
print(f"Acurácia média: {scores_gau.mean():.2%}")

model_dummy = DummyClassifier(strategy = "most_frequent")
scores_dummy = cross_val_score(
    model_dummy,
    X_treino,
    y_treino,
    cv = cv,
    scoring = "accuracy"
)

print("==========")

print("Acurácias da previsão simples:", scores_dummy)
print(f"Média da previsão simples: {scores_dummy.mean():.2%}")

print(f"Dados carregados: {len(df)}")
print(f"Dados de treino: {len(X_treino)}")
print(f"Dados de teste: {len(X_teste)}")
print(f"Casos na matriz: {len(y_teste)}")

model = BernoulliNB()
model.fit(X_treino, y_treino)

predict = model.predict(X_teste)
accuracy = accuracy_score(y_teste, predict)
predict_treino = model.predict(X_treino)
accuracy_treino = accuracy_score(y_treino, predict_treino)

model_gau = GaussianNB()
model_gau.fit(X_treino, y_treino)

predict_gau = model_gau.predict(X_teste)
accuracy_gau = accuracy_score(y_teste, predict_gau)
predict_treino_gau = model_gau.predict(X_treino)
accuracy_treino_gau = accuracy_score(y_treino, predict_treino_gau)

print(f"Acurácia de treino BernoulliNB: {accuracy_treino:.2%}")
print(f"Acurácia de teste BernoulliNB: {accuracy:.2%}")
print(f"Acurácia de treino GaussianNB: {accuracy_treino_gau:.2%}")
print(f"Acurácia de teste GaussianNB: {accuracy_gau:.2%}")

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

matriz.ax_.set_title(f"Matriz de confusão BernoulliNB\nAcurácia: {accuracy:.2%}")
matriz.ax_.xaxis.set_label_position("top")
matriz.ax_.xaxis.tick_top()
matriz.ax_.set_xlabel("Classe real")
matriz.ax_.yaxis.set_label_position("left")
matriz.ax_.yaxis.tick_left()
matriz.ax_.set_ylabel("Classe prevista pelo modelo")
plt.figtext(0.5, 0.01, "Diagonal = acertos | Fora da diagonal = erros", ha="center")
plt.tight_layout(rect = [0, 0.04, 1, 1])
plt.show()

# ----- \\ ----- \\

valores_matriz_gau = confusion_matrix(y_teste, predict_gau, labels = [0, 1]).T
matriz_gau = ConfusionMatrixDisplay(
    confusion_matrix = valores_matriz_gau,
    display_labels = ["Sem diabetes", "Com diabetes"],
)
matriz_gau.plot(
    cmap = "Blues",
    values_format = "d",
    colorbar = False,
)

matriz_gau.ax_.set_title(f"Matriz de confusão GaussianNB\nAcurácia: {accuracy_gau:.2%}")
matriz_gau.ax_.xaxis.set_label_position("top")
matriz_gau.ax_.xaxis.tick_top()
matriz_gau.ax_.set_xlabel("Classe real")
matriz_gau.ax_.yaxis.set_label_position("left")
matriz_gau.ax_.yaxis.tick_left()
matriz_gau.ax_.set_ylabel("Classe prevista pelo modelo")
plt.figtext(0.5, 0.01, "Diagonal = acertos | Fora da diagonal = erros", ha="center")
plt.tight_layout(rect = [0, 0.04, 1, 1])
plt.show()

# %%
