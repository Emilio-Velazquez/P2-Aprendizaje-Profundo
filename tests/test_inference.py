import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.data.features import add_features
from src.inference.app import create_app
from src.inference.predictor import Predictor
from src.inference.schemas import Record
from src.models import create_model


@pytest.fixture
def mock_model_dir(tmp_path: Path):
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
    feature_names = list(add_features(dummy_df, target="pm2.5").columns)

    model = create_model(
        "lstm", input_size=15, hidden_size=8, num_layers=1, dropout=0.0
    )
    weights_path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weights_path)

    scaler_data = {
        "mean": [0.0] * 15,
        "std": [1.0] * 15,
        "feature_names": feature_names,
    }
    with open(tmp_path / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_data, f)

    env_data = {
        "torch_version": torch.__version__,
        "registered_model_version": "test-v1",
        "run_id": "test-run-123",
        "horizon": 1,
        "model_config": {
            "model_type": "lstm",
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "window": 24,
            "horizon": 1,
            "n_features": 15,
        },
    }
    with open(tmp_path / "env_info.json", "w", encoding="utf-8") as f:
        json.dump(env_data, f)

    return tmp_path


@pytest.fixture
def client(mock_model_dir: Path):
    app = create_app(mock_model_dir)
    return TestClient(app)


def test_health_and_info(client: TestClient):
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    res_info = client.get("/info")
    assert res_info.status_code == 200
    data = res_info.json()
    assert data["window"] == 24
    assert data["n_features"] == 15
    assert len(data["feature_names"]) == 15


def test_predict_valid_payload(client: TestClient):
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    records = []
    for i in range(24):
        records.append(
            {
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "pm25": 50.0 + i,
                "DEWP": 5.0,
                "TEMP": 10.0,
                "PRES": 1013.0,
                "Iws": 2.0,
                "Is": 0.0,
                "Ir": 0.0,
                "cbwd": "NW",
            }
        )

    res = client.post("/predict", json={"records": records})
    assert res.status_code == 200
    data = res.json()
    assert "prediction_pm25" in data
    assert np.isfinite(data["prediction_pm25"])
    assert data["prediction_pm25"] >= 0.0
    assert data["unit"] == "ug/m3"
    assert data["horizon_hours"] == 1
    expected_pred_for = (base_time + timedelta(hours=24)).isoformat()
    assert data["predicted_for"] == expected_pred_for
    assert data["model"]["model_type"] == "lstm"


def test_predict_invalid_payloads(client: TestClient):
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    valid_record = {
        "timestamp": base_time.isoformat(),
        "pm25": 50.0,
        "DEWP": 5.0,
        "TEMP": 10.0,
        "PRES": 1013.0,
        "Iws": 2.0,
        "Is": 0.0,
        "Ir": 0.0,
        "cbwd": "NW",
    }

    # 1. 23 registros (menos de 24)
    rec_23 = []
    for i in range(23):
        r = dict(valid_record)
        r["timestamp"] = (base_time + timedelta(hours=i)).isoformat()
        rec_23.append(r)
    res_23 = client.post("/predict", json={"records": rec_23})
    assert res_23.status_code == 422

    # 2. cbwd invalido
    rec_bad_cbwd = []
    for i in range(24):
        r = dict(valid_record)
        r["timestamp"] = (base_time + timedelta(hours=i)).isoformat()
        if i == 5:
            r["cbwd"] = "INVALID"
        rec_bad_cbwd.append(r)
    res_cbwd = client.post("/predict", json={"records": rec_bad_cbwd})
    assert res_cbwd.status_code == 422

    # 3. timestamps no consecutivos (salto de 2 horas)
    rec_gap = []
    for i in range(24):
        r = dict(valid_record)
        step = i if i < 10 else i + 2
        r["timestamp"] = (base_time + timedelta(hours=step)).isoformat()
        rec_gap.append(r)
    res_gap = client.post("/predict", json={"records": rec_gap})
    assert res_gap.status_code == 422

    # 4. campo faltante
    rec_missing = []
    for i in range(24):
        r = dict(valid_record)
        r["timestamp"] = (base_time + timedelta(hours=i)).isoformat()
        if i == 0:
            del r["TEMP"]
        rec_missing.append(r)
    res_missing = client.post("/predict", json={"records": rec_missing})
    assert res_missing.status_code == 422


def test_feature_parity_with_training():
    raw_path = Path("data/raw/beijing_pm25.csv")
    dataset_path = Path("data/processed/dataset.npz")
    scaler_path = Path("data/processed/scaler.json")
    meta_path = Path("data/processed/meta.json")

    if (
        not raw_path.exists()
        or not dataset_path.exists()
        or not scaler_path.exists()
        or not meta_path.exists()
    ):
        pytest.skip(
            "Archivos de datos de entrenamiento no disponibles para prueba de paridad."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n_train = meta["rows_per_split"]["train"]
    n_val = meta["rows_per_split"]["val"]

    dataset = np.load(dataset_path)
    x_test = dataset["X_test"]

    # Crear predictor temporal apuntando a deploy/champion
    champ_dir = Path("deploy/champion")
    if not (champ_dir / "weights.pt").exists():
        pytest.skip("Modelo champion no exportado.")

    predictor = Predictor(champ_dir)

    df_raw = pd.read_csv(raw_path)
    test_slice = df_raw.iloc[n_train + n_val :].reset_index(drop=True)

    required_cols = ["pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    valid_indices = []
    for i in range(len(test_slice) - 24):
        win = test_slice.iloc[i : i + 24]
        next_row = test_slice.iloc[i + 24]
        if not win[required_cols].isna().any().any() and not pd.isna(next_row["pm2.5"]):
            valid_indices.append(i)
            if len(valid_indices) == 5:
                break

    assert len(valid_indices) == 5

    # Comprobar que cada ventana construida coincide con alguna ventana de X_test[:800]
    for idx in valid_indices:
        win_rows = test_slice.iloc[idx : idx + 24]
        records = []
        for _, row in win_rows.iterrows():
            dt = datetime(
                int(row["year"]),
                int(row["month"]),
                int(row["day"]),
                int(row["hour"]),
            )
            records.append(
                Record(
                    timestamp=dt,
                    pm25=float(row["pm2.5"]),
                    DEWP=float(row["DEWP"]),
                    TEMP=float(row["TEMP"]),
                    PRES=float(row["PRES"]),
                    Iws=float(row["Iws"]),
                    Is=float(row["Is"]),
                    Ir=float(row["Ir"]),
                    cbwd=str(row["cbwd"]),
                )
            )

        built_window = predictor.build_window(records)
        # Buscar coincidencia en X_test[:800]
        matched = False
        for x_win in x_test[:800]:
            if np.allclose(built_window, x_win, atol=1e-5):
                matched = True
                break
        assert (
            matched
        ), f"Ventana en indice {idx} no coincide con ninguna ventana de X_test[:800]"
