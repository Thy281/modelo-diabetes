# %%

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd

FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
]

df = pd.read_csv("dados/dados_diabetes.csv")
linhas_antes = len(df)
df = df.drop_duplicates()
print(f"Duplicatas removidas: {linhas_antes - len(df)}")

# %%

features = FEATURES

X = df[features]
y = df["Diabetes_binary"]

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
modelos = {
    "GaussianNB": GaussianNB(),
    "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter = 1000)),
    "Árvore de decisão": DecisionTreeClassifier(random_state = 42),
    "Floresta aleatória": RandomForestClassifier(n_estimators = 200, random_state = 42, n_jobs = -1),
}

resultados = {}
for nome, modelo in modelos.items():
    validacao = cross_validate(
        modelo,
        X_treino,
        y_treino,
        cv=cv,
        scoring={"accuracy": "accuracy", "recall": "recall"},
    )
    print(f"\n{nome}")
    print("Acurácias por divisão:", validacao["test_accuracy"])
    print(f"Acurácia média na validação cruzada: {validacao['test_accuracy'].mean():.2%}")
    print("Recalls por divisão:", validacao["test_recall"])
    print(f"Recall médio na validação cruzada: {validacao['test_recall'].mean():.2%}")
    resultados[nome] = modelo

print(f"\nDados carregados: {len(df)}")
print(f"Dados de treino: {len(X_treino)}")
print(f"Dados de teste: {len(X_teste)}")
print(f"Casos na matriz: {len(y_teste)}")

fig, eixos = plt.subplots(2, 2, figsize=(12, 10))
for (nome, modelo), eixo in zip(resultados.items(), eixos.ravel()):
    modelo.fit(X_treino, y_treino)
    previsoes_teste = modelo.predict(X_teste)
    previsoes_treino = modelo.predict(X_treino)
    accuracy_teste = accuracy_score(y_teste, previsoes_teste)
    recall_teste = recall_score(y_teste, previsoes_teste, pos_label=1)
    accuracy_treino = accuracy_score(y_treino, previsoes_treino)

    print(f"\n{nome}")
    print(f"Acurácia de treino: {accuracy_treino:.2%}")
    print(f"Acurácia de teste: {accuracy_teste:.2%}")
    print(f"Recall de teste (diabetes): {recall_teste:.2%}")

    valores_matriz = confusion_matrix(y_teste, previsoes_teste, labels=[0, 1]).T
    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=valores_matriz,
        display_labels=["Sem diabetes", "Com diabetes"],
    )
    exibicao.plot(cmap="Blues", values_format="d", colorbar=False, ax=eixo)
    eixo.set_title(f"{nome}\nAcurácia: {accuracy_teste:.2%} | Recall: {recall_teste:.2%}")
    eixo.xaxis.set_label_position("top")
    eixo.xaxis.tick_top()
    eixo.set_xlabel("Classe real")
    eixo.set_ylabel("Classe prevista")

# %%

plt.figtext(0.5, 0.01, "Diagonal = acertos | Fora da diagonal = erros", ha="center")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

# %%
