# Modelo educativo de diabetes

O treinamento e a avaliação didática estão em `main.py`. A API Python carrega o modelo salvo e pode receber as respostas do quiz por JSON.

## Treinar e iniciar a API

```bash
source .venv/bin/activate
python -m pip install -r requirements-api.txt
python -m diabetes_model.training
python -m uvicorn diabetes_model.api:app --reload
```

O código da API, do treinamento e das configurações fica no pacote `diabetes_model/`.

A documentação interativa fica em `http://127.0.0.1:8000/docs`, e a verificação da API em `http://127.0.0.1:8000/health`.

## Prever

Envie um `POST` para `http://127.0.0.1:8000/predict` com JSON contendo `idade`, `imc`, `glicemia`, `sede_excessiva`, `urina_frequente`, `fadiga`, `historico_familiar`, `atividade_fisica` e `pressao_sistolica`. Os campos de sintomas e histórico usam `0` ou `1`; atividade física usa a escala de `0` a `7` existente na planilha.

A resposta inclui `classe_prevista` (`0` ou `1`) e `probabilidade_modelo`, entre `0` e `1`. A previsão é apenas educativa e não é diagnóstico médico.

O backend Java deve chamar essa API com HTTP `POST` e enviar o JSON do quiz. Não importe o arquivo Python diretamente no Java.
