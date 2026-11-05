# Prediccion de Series de Tiempo - Beijing PM2.5

Forecasting de la concentracion horaria de material particulado PM2.5 en Beijing utilizando redes neuronales recurrentes (RNN y LSTM) en PyTorch.
El proyecto implementa un flujo MLOps reproducible desde el control de versiones con Git y DVC hasta el despliegue con Docker.

## Como reproducir localmente

1. Descargar datos o sincronizarlos con DVC:
   `dvc pull`
2. Ejecutar preprocesamiento:
   `dvc repro`
3. Entrenar modelos y ejecutar el barrido de hiperparametros con MLflow:
   `python -m src.training.sweep`
4. Resumir resultados del barrido:
   `python -m src.training.summarize`
5. Seleccionar, verificar y exportar el modelo campeon a `model_export/`:
   `python -m src.training.select_best`
   `python -m src.training.verify_champion`
   `python -m src.training.export_champion`

## Entrenamiento con Docker Compose (GPU)

1. Construir las imagenes:
   `docker compose build`

2. Entrenar el modelo campeon por defecto (con registro de commit):
   - En PowerShell:
     `$env:GIT_COMMIT=(git rev-parse --short HEAD); docker compose run --rm train`
   - En Bash / Linux:
     `GIT_COMMIT=$(git rev-parse --short HEAD) docker compose run --rm train`

3. Sobrescribir argumentos de entrenamiento:
   `docker compose run --rm train --model rnn --hidden_size 64 --num_layers 2 --dropout 0.2`

4. Ver el servidor y UI de MLflow:
   - Levantar servidor en segundo plano:
     `docker compose up -d mlflow`
   - Acceder en el navegador a:
     `http://localhost:5000`
   - Detener servicios preservando el volumen de datos:
     `docker compose down`
