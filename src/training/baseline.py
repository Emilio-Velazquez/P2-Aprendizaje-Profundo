import json
import os
from pathlib import Path

import mlflow
import numpy as np
import yaml

from src.training.metrics import denormalize, mae, rmse


def main():
    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_dir = base_dir / params["data"]["processed_dir"]
    dataset = np.load(data_dir / "dataset.npz")
    with open(data_dir / "scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)

    mean0 = scaler["mean"][0]
    std0 = scaler["std"][0]

    # Prediccion de persistencia con ultimo valor pm2.5 de la ventana
    x_val = dataset["X_val"]
    y_val = dataset["y_val"]
    x_test = dataset["X_test"]
    y_test = dataset["y_test"]

    pred_val = denormalize(x_val[:, -1, 0:1], mean0, std0)
    true_val = denormalize(y_val, mean0, std0)
    pred_test = denormalize(x_test[:, -1, 0:1], mean0, std0)
    true_test = denormalize(y_test, mean0, std0)

    val_rmse_val = rmse(true_val, pred_val)
    val_mae_val = mae(true_val, pred_val)
    test_rmse_val = rmse(true_test, pred_test)
    test_mae_val = mae(true_test, pred_test)

    print(f"Baseline Val RMSE: {val_rmse_val:.2f}, Val MAE: {val_mae_val:.2f}")
    print(f"Baseline Test RMSE: {test_rmse_val:.2f}, Test MAE: {test_mae_val:.2f}")

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment"])

    with mlflow.start_run(run_name="persistence"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.log_params(
            {
                "window": params["preprocess"]["window"],
                "horizon": params["preprocess"]["horizon"],
            }
        )
        mlflow.log_metrics(
            {
                "val_rmse": val_rmse_val,
                "val_mae": val_mae_val,
                "test_rmse": test_rmse_val,
                "test_mae": test_mae_val,
            }
        )


if __name__ == "__main__":
    main()
