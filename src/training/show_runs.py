import os
from pathlib import Path

import mlflow
import pandas as pd
import yaml


def format_val(val, fmt=None):
    if pd.isna(val) or val is None:
        return "None"
    if fmt:
        return fmt.format(val)
    return str(val)


def main():
    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = params["mlflow"]["experiment"]
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        print(f"Experimento {exp_name} no encontrado.")
        return

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        print("No se encontraron runs.")
        return

    # Normalizar columna para ordenar incluyendo baseline
    if "metrics.val_rmse" in runs.columns:
        if "metrics.best_val_rmse" in runs.columns:
            runs["sort_metric"] = runs["metrics.best_val_rmse"].fillna(
                runs["metrics.val_rmse"]
            )
        else:
            runs["sort_metric"] = runs["metrics.val_rmse"]
    else:
        runs["sort_metric"] = runs.get("metrics.best_val_rmse", None)

    runs_sorted = runs.sort_values(by="sort_metric", ascending=True)

    header = "run_name | model_type | best_epoch | best_val_rmse | test_rmse | test_mae | n_parameters | train_time_s"
    print(header)
    for _, row in runs_sorted.iterrows():
        r_name = str(row.get("tags.mlflow.runName", "None"))
        m_type = str(row.get("tags.model_type", "None"))
        b_epoch = row.get("metrics.best_epoch", None)
        b_epoch_s = format_val(b_epoch, "{:.0f}")
        b_val = row.get("sort_metric", None)
        b_val_s = format_val(b_val, "{:.2f}")
        t_rmse = row.get("metrics.test_rmse", None)
        t_rmse_s = format_val(t_rmse, "{:.2f}")
        t_mae = row.get("metrics.test_mae", None)
        t_mae_s = format_val(t_mae, "{:.2f}")
        n_p = str(row.get("params.n_parameters", "None"))
        t_time = row.get("metrics.train_time_seconds", None)
        t_time_s = format_val(t_time, "{:.1f}")

        print(
            f"{r_name} | {m_type} | {b_epoch_s} | {b_val_s} | {t_rmse_s} | {t_mae_s} | {n_p} | {t_time_s}"
        )


if __name__ == "__main__":
    main()
