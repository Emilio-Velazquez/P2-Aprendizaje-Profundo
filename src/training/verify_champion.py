import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import numpy as np
import torch
import yaml

from src.training.metrics import denormalize, mae, rmse


def predict_batches(model, x, batch_size, device):
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i : i + batch_size]).to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)

    # 1. Cargar modelo PyTorch registrado
    model_uri = "models:/pm25-forecaster@champion"
    model = mlflow.pytorch.load_model(model_uri)
    device = torch.device("cpu")
    if hasattr(model, "eval"):
        try:
            model.eval()
        except NotImplementedError:
            pass

    # 2. Cargar datos de prueba y scaler
    data_dir = base_dir / params["data"]["processed_dir"]
    dataset = np.load(data_dir / "dataset.npz")
    with open(data_dir / "scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)

    mean0 = scaler["mean"][0]
    std0 = scaler["std"][0]

    x_test = dataset["X_test"]
    y_test = dataset["y_test"]

    # Comprobacion 1: Consistencia de lotes
    pred_1 = predict_batches(model, x_test, 1, device)
    pred_7 = predict_batches(model, x_test, 7, device)
    pred_64 = predict_batches(model, x_test, 64, device)
    pred_full = predict_batches(model, x_test, len(x_test), device)

    diffs = [
        np.max(np.abs(pred_1 - pred_7)),
        np.max(np.abs(pred_1 - pred_64)),
        np.max(np.abs(pred_1 - pred_full)),
        np.max(np.abs(pred_7 - pred_64)),
        np.max(np.abs(pred_7 - pred_full)),
        np.max(np.abs(pred_64 - pred_full)),
    ]
    max_batch_diff = float(max(diffs))
    print(f"Diferencia maxima entre lotes: {max_batch_diff:.6e}")
    if max_batch_diff < 1e-4:
        print("Consistencia de lotes: PASS")
    else:
        print("Consistencia de lotes: FAIL")
        sys.exit(1)

    # Comprobacion 2: Desnormalizacion y comparacion con metricas logueadas
    mv = client.get_model_version_by_alias("pm25-forecaster", "champion")
    run_id = mv.tags.get("source_run_id") or mv.run_id
    source_run = client.get_run(run_id)

    logged_test_rmse = float(source_run.data.metrics.get("test_rmse"))
    logged_test_mae = float(source_run.data.metrics.get("test_mae"))

    pred_denorm = denormalize(pred_64, mean0, std0)
    true_denorm = denormalize(y_test, mean0, std0)

    calc_test_rmse = rmse(true_denorm, pred_denorm)
    calc_test_mae = mae(true_denorm, pred_denorm)

    diff_rmse = abs(calc_test_rmse - logged_test_rmse)
    diff_mae = abs(calc_test_mae - logged_test_mae)

    print(
        f"Recalculado test_rmse: {calc_test_rmse:.2f}, Logueado: {logged_test_rmse:.2f}, Dif: {diff_rmse:.4f}"
    )
    print(
        f"Recalculado test_mae: {calc_test_mae:.2f}, Logueado: {logged_test_mae:.2f}, Dif: {diff_mae:.4f}"
    )

    if diff_rmse < 1e-2 and diff_mae < 1e-2:
        print("Comparacion con metricas logueadas: PASS")
    else:
        print("Comparacion con metricas logueadas: FAIL")
        sys.exit(1)

    # Comprobacion 3: PyTorch vs PyFunc en X_test[:64]
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    x_sample = x_test[:64]
    pyfunc_pred = pyfunc_model.predict(x_sample)

    if hasattr(pyfunc_pred, "to_numpy"):
        pyfunc_pred_arr = pyfunc_pred.to_numpy()
    else:
        pyfunc_pred_arr = np.asarray(pyfunc_pred)

    pytorch_sample_pred = pred_64[:64]
    diff_pyfunc = float(np.max(np.abs(pytorch_sample_pred - pyfunc_pred_arr)))
    print(f"Diferencia maxima PyTorch vs PyFunc: {diff_pyfunc:.6e}")

    if diff_pyfunc < 1e-4:
        print("Consistencia PyTorch vs PyFunc: PASS")
    else:
        print("Consistencia PyTorch vs PyFunc: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
