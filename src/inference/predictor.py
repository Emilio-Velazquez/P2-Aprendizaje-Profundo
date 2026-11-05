import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.features import add_features
from src.inference.schemas import Record
from src.models import create_model


class Predictor:
    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)

        # Cargar metadatos del entorno y modelo
        with open(self.model_dir / "env_info.json", "r", encoding="utf-8") as f:
            self.env_info: dict[str, Any] = json.load(f)

        # Cargar scaler
        with open(self.model_dir / "scaler.json", "r", encoding="utf-8") as f:
            scaler_data = json.load(f)
        self.mean = np.array(scaler_data["mean"], dtype=np.float32)
        self.std = np.array(scaler_data["std"], dtype=np.float32)
        self.feature_names: list[str] = scaler_data["feature_names"]

        # Verificar orden y cantidad de columnas respecto a add_features
        dummy_df = pd.DataFrame(
            {
                "pm2.5": [0.0],
                "DEWP": [0.0],
                "TEMP": [0.0],
                "PRES": [1000.0],
                "cbwd": ["NE"],
                "Iws": [0.0],
                "Is": [0.0],
                "Ir": [0.0],
            },
            index=pd.date_range("2026-01-01", periods=1, freq="h"),
        )
        expected_cols = list(add_features(dummy_df, target="pm2.5").columns)
        if len(self.feature_names) != 15 or self.feature_names != expected_cols:
            raise ValueError(
                f"Las caracteristicas del scaler no coinciden con add_features. "
                f"Esperadas: {expected_cols}, obtenidas: {self.feature_names}"
            )

        self.model_config = self.env_info["model_config"]
        self.window = int(self.model_config["window"])
        self.horizon = int(
            self.model_config.get("horizon", self.env_info.get("horizon", 1))
        )

        # Cargar pesos y crear modelo
        weights_path = self.model_dir / "weights.pt"
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model = create_model(
            name=self.model_config["model_type"],
            input_size=len(self.feature_names),
            hidden_size=int(self.model_config["hidden_size"]),
            num_layers=int(self.model_config["num_layers"]),
            dropout=float(self.model_config["dropout"]),
            output_size=1,
        )
        self.model.load_state_dict(weights)
        self.model.eval()

    def build_window(self, records: list[Record]) -> np.ndarray:
        # Armar DataFrame a partir de los registros
        data = {
            "pm2.5": [r.pm25 for r in records],
            "DEWP": [r.DEWP for r in records],
            "TEMP": [r.TEMP for r in records],
            "PRES": [r.PRES for r in records],
            "cbwd": [r.cbwd for r in records],
            "Iws": [r.Iws for r in records],
            "Is": [r.Is for r in records],
            "Ir": [r.Ir for r in records],
        }
        index = pd.to_datetime([r.timestamp for r in records])
        df = pd.DataFrame(data, index=index)

        df_feat = add_features(df, target="pm2.5")
        df_feat = df_feat[self.feature_names]

        # Normalizar con media y desviacion estandar
        window_norm = (df_feat.values.astype(np.float32) - self.mean) / self.std
        return window_norm

    def predict(self, records: list[Record]) -> float:
        if len(records) != self.window:
            raise ValueError(
                f"Se requieren exactamente {self.window} registros, se recibieron {len(records)}"
            )

        window_arr = self.build_window(records)
        tensor = torch.from_numpy(window_arr).unsqueeze(0).float()

        with torch.no_grad():
            pred_norm = self.model(tensor).item()

        # Desnormalizar prediccion
        pred_val = pred_norm * float(self.std[0]) + float(self.mean[0])
        return max(0.0, float(pred_val))

    def get_info(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "window": self.window,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "torch_version": self.env_info.get("torch_version", torch.__version__),
            "registered_model_version": self.env_info.get(
                "registered_model_version", "unknown"
            ),
            "run_id": self.env_info.get("run_id", "unknown"),
        }

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_config["model_type"],
            "registered_model_version": self.env_info.get(
                "registered_model_version", "unknown"
            ),
            "run_id": self.env_info.get("run_id", "unknown"),
        }
