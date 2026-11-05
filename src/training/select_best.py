import argparse
import os
import sys
from pathlib import Path

import mlflow
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect",
        type=str,
        default=None,
        help="Configuracion esperada: model,hidden,layers,dropout",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)

    exp_name = params["mlflow"]["experiment"]
    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        print(f"Experimento {exp_name} no encontrado.")
        sys.exit(1)

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty or "tags.stage" not in runs.columns:
        print("No se encontraron runs validos.")
        sys.exit(1)

    # Filtrar corridas completadas de sweep y seeds
    valid_runs = runs[
        runs["tags.stage"].isin(["sweep", "seeds"]) & (runs["status"] == "FINISHED")
    ].copy()

    if valid_runs.empty:
        print("No hay runs completados en sweep o seeds.")
        sys.exit(1)

    valid_runs["hidden_size"] = valid_runs["params.hidden_size"].astype(int)
    valid_runs["num_layers"] = valid_runs["params.num_layers"].astype(int)
    valid_runs["dropout"] = valid_runs["params.dropout"].astype(float)
    valid_runs["model_type"] = valid_runs["tags.model_type"]
    valid_runs["best_val_rmse"] = valid_runs["metrics.best_val_rmse"].astype(float)
    valid_runs["test_rmse"] = valid_runs["metrics.test_rmse"].astype(float)
    valid_runs["test_mae"] = valid_runs["metrics.test_mae"].astype(float)

    # Agrupar y filtrar configs con al menos 4 runs
    group_cols = ["model_type", "hidden_size", "num_layers", "dropout"]
    rankings = []
    grouped = valid_runs.groupby(group_cols)
    for (m_type, h_size, n_layers, drop), group in grouped:
        n_runs = len(group)
        if n_runs >= 4:
            val_mean = group["best_val_rmse"].mean()
            rankings.append(
                {
                    "model_type": m_type,
                    "hidden_size": h_size,
                    "num_layers": n_layers,
                    "dropout": drop,
                    "n_runs": n_runs,
                    "val_rmse_mean": val_mean,
                    "group_df": group,
                }
            )

    if not rankings:
        print("No hay configuraciones con al menos 4 runs.")
        sys.exit(1)

    rankings.sort(key=lambda x: x["val_rmse_mean"])

    # Imprimir ranking simple
    print("Ranking de configuraciones (n_runs >= 4):")
    for r in rankings:
        cfg_str = f"{r['model_type']}_h{r['hidden_size']}_l{r['num_layers']}_d{r['dropout']:.1f}"
        print(
            f"{cfg_str} | n_runs: {r['n_runs']} | val_rmse_mean: {r['val_rmse_mean']:.2f}"
        )

    champ_cfg = rankings[0]
    actual_cfg = (
        champ_cfg["model_type"],
        champ_cfg["hidden_size"],
        champ_cfg["num_layers"],
        champ_cfg["dropout"],
    )

    if args.expect:
        parts = [p.strip() for p in args.expect.split(",")]
        expected = (parts[0], int(parts[1]), int(parts[2]), float(parts[3]))
        if actual_cfg != expected:
            print(
                f"Error: la configuracion elegida {actual_cfg} no coincide con la esperada {expected}."
            )
            sys.exit(1)

    # Seleccionar el run de menor best_val_rmse dentro de la config campeona
    champ_group = champ_cfg["group_df"].sort_values(by="best_val_rmse", ascending=True)
    best_run = champ_group.iloc[0]
    best_run_id = best_run["run_id"]
    best_run_name = best_run["tags.mlflow.runName"]
    best_val_rmse = best_run["best_val_rmse"]
    test_rmse = best_run["test_rmse"]
    test_mae = best_run["test_mae"]

    print(f"Configuracion campeona: {actual_cfg}")
    print(
        f"Run elegido: {best_run_name} (run_id: {best_run_id}, best_val_rmse: {best_val_rmse:.2f})"
    )

    # Registrar en Model Registry
    model_name = "pm25-forecaster"
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    # Asignar alias champion
    client.set_registered_model_alias(model_name, "champion", mv.version)

    # Agregar tags descriptivos
    tags = {
        "source_run_id": str(best_run_id),
        "model_type": str(champ_cfg["model_type"]),
        "hidden_size": str(champ_cfg["hidden_size"]),
        "num_layers": str(champ_cfg["num_layers"]),
        "dropout": str(champ_cfg["dropout"]),
        "best_val_rmse": str(best_val_rmse),
        "test_rmse": str(test_rmse),
        "test_mae": str(test_mae),
        "selection_rule": "min mean best_val_rmse, n_runs>=4",
    }
    for k, v in tags.items():
        client.set_model_version_tag(model_name, mv.version, k, v)

    print(
        f"Modelo registrado exitosamente: {model_name} version {mv.version} con alias @champion"
    )


if __name__ == "__main__":
    main()
