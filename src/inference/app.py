import os
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.inference.predictor import Predictor
from src.inference.schemas import PredictRequest, PredictResponse


def create_app(model_dir: Path | str) -> FastAPI:
    predictor = Predictor(model_dir)
    app = FastAPI(title="PM2.5 Forecasting Service")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        return predictor.get_info()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if len(req.records) != predictor.window:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Se requieren exactamente {predictor.window} registros, "
                    f"se recibieron {len(req.records)}"
                ),
            )

        pred_val = predictor.predict(req.records)
        last_timestamp = req.records[-1].timestamp
        predicted_for = last_timestamp + timedelta(hours=predictor.horizon)

        return PredictResponse(
            prediction_pm25=pred_val,
            unit="ug/m3",
            horizon_hours=predictor.horizon,
            predicted_for=predicted_for,
            model=predictor.get_model_metadata(),
        )

    return app


def create_app_from_env() -> FastAPI:
    model_dir = os.environ.get("MODEL_DIR", "model_export/champion")
    return create_app(model_dir)
