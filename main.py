# %%

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

import pandas as pd

df = pd.read_excel("dados/dados_diabetes_1000.xlsx")
linhas_antes = len(df)
df = df.drop_duplicates()
print(f"Duplicatas removidas: {linhas_antes - len(df)}")

# %%

features = [
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

X = df[features]
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
modelo_gau = GaussianNB()

scores_gau = cross_val_score(
    modelo_gau,
    X_treino,
    y_treino,
    cv = cv,
    scoring = "accuracy"
)

recalls_gau = cross_val_score(
    modelo_gau,
    X_treino,
    y_treino,
    cv = cv,
    scoring = "recall"
)

print("==========")

print("Acurácias por divisão GaussianNB:", scores_gau)
print(f"Acurácia média GaussianNB: {scores_gau.mean():.2%}")
print("Recalls por divisão GaussianNB:", recalls_gau)
print(f"Recall médio GaussianNB: {recalls_gau.mean():.2%}")

print(f"Dados carregados: {len(df)}")
print(f"Dados de treino: {len(X_treino)}")
print(f"Dados de teste: {len(X_teste)}")
print(f"Casos na matriz: {len(y_teste)}")

modelo_gau.fit(X_treino, y_treino)

previsoes_gau = modelo_gau.predict(X_teste)
accuracy_gau = accuracy_score(y_teste, previsoes_gau)
recall_gau = recall_score(y_teste, previsoes_gau, pos_label = 1)
previsoes_treino_gau = modelo_gau.predict(X_treino)
accuracy_treino_gau = accuracy_score(y_treino, previsoes_treino_gau)

print(f"Acurácia de treino GaussianNB: {accuracy_treino_gau:.2%}")
print(f"Acurácia de teste GaussianNB: {accuracy_gau:.2%}")
print(f"Recall de teste GaussianNB (diabetes): {recall_gau:.2%}")

# %%

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

valores_matriz_gau = confusion_matrix(y_teste, previsoes_gau, labels = [0, 1]).T
matriz_gau = ConfusionMatrixDisplay(
    confusion_matrix = valores_matriz_gau,
    display_labels = ["Sem diabetes", "Com diabetes"],
)
matriz_gau.plot(
    cmap = "Blues",
    values_format = "d",
    colorbar = False,
)

matriz_gau.ax_.set_title(f"Matriz de confusão GaussianNB\nAcurácia: {accuracy_gau:.2%} | Recall diabetes: {recall_gau:.2%}")
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
