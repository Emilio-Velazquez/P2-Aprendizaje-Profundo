import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml


def format_stat(val):
    if pd.isna(val) or val is None:
        return "nan"
    return f"{val:.2f}"


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

    # Filtrar corridas de barrido y semillas terminadas
    if "tags.stage" not in runs.columns:
        print("No hay runs con tag stage.")
        return

    valid_runs = runs[
        runs["tags.stage"].isin(["sweep", "seeds"]) & (runs["status"] == "FINISHED")
    ].copy()

    if valid_runs.empty:
        print("No se encontraron runs completados de sweep o seeds.")
        return

    valid_runs["hidden_size"] = valid_runs["params.hidden_size"].astype(int)
    valid_runs["num_layers"] = valid_runs["params.num_layers"].astype(int)
    valid_runs["dropout"] = valid_runs["params.dropout"].astype(float)
    valid_runs["n_parameters"] = valid_runs["params.n_parameters"].astype(int)
    valid_runs["model_type"] = valid_runs["tags.model_type"]
    valid_runs["best_val_rmse"] = valid_runs["metrics.best_val_rmse"].astype(float)
    valid_runs["test_rmse"] = valid_runs["metrics.test_rmse"].astype(float)
    valid_runs["test_mae"] = valid_runs["metrics.test_mae"].astype(float)

    summary_rows = []
    group_cols = ["model_type", "hidden_size", "num_layers", "dropout"]
    for (m_type, h_size, n_layers, drop), group in valid_runs.groupby(group_cols):
        n_runs = len(group)
        val_rmse_mean = group["best_val_rmse"].mean()
        val_rmse_std = group["best_val_rmse"].std() if n_runs > 1 else np.nan
        test_rmse_mean = group["test_rmse"].mean()
        test_rmse_std = group["test_rmse"].std() if n_runs > 1 else np.nan
        test_mae_mean = group["test_mae"].mean()
        test_mae_std = group["test_mae"].std() if n_runs > 1 else np.nan
        n_params = group["n_parameters"].iloc[0]

        summary_rows.append(
            {
                "model_type": m_type,
                "hidden_size": h_size,
                "num_layers": n_layers,
                "dropout": drop,
                "n_parameters": n_params,
                "n_runs": n_runs,
                "val_rmse_mean": val_rmse_mean,
                "val_rmse_std": val_rmse_std,
                "test_rmse_mean": test_rmse_mean,
                "test_rmse_std": test_rmse_std,
                "test_mae_mean": test_mae_mean,
                "test_mae_std": test_mae_std,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by="val_rmse_mean", ascending=True, inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    header = "model_type | hidden_size | num_layers | dropout | n_parameters | n_runs | val_rmse_mean | val_rmse_std | test_rmse_mean | test_rmse_std | test_mae_mean | test_mae_std"
    print(header)
    for _, row in summary_df.iterrows():
        line = (
            f"{row['model_type']} | {int(row['hidden_size'])} | {int(row['num_layers'])} | {row['dropout']:.1f} | "
            f"{int(row['n_parameters'])} | {int(row['n_runs'])} | "
            f"{row['val_rmse_mean']:.2f} | {format_stat(row['val_rmse_std'])} | "
            f"{row['test_rmse_mean']:.2f} | {format_stat(row['test_rmse_std'])} | "
            f"{row['test_mae_mean']:.2f} | {format_stat(row['test_mae_std'])}"
        )
        print(line)

    # Imprimir baseline de persistencia
    base_runs = runs[runs["tags.model_type"] == "baseline"]
    if not base_runs.empty:
        b_val = base_runs["metrics.val_rmse"].iloc[0]
        b_test = base_runs["metrics.test_rmse"].iloc[0]
        print(
            f"Baseline (persistence) - val_rmse: {b_val:.2f}, test_rmse: {b_test:.2f}"
        )

    # Guardar en artifacts
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(artifacts_dir / "sweep_summary.csv", index=False)


if __name__ == "__main__":
    main()
