import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml


def is_finished(client, exp_id, run_name):
    runs = client.search_runs(experiment_ids=[exp_id])
    for r in runs:
        if r.info.run_name == run_name and r.info.status == "FINISHED":
            return True, r.data.metrics.get("best_val_rmse")
    return False, None


def get_best_grid_configs(client, exp_id, model_name, top_k=2):
    runs = client.search_runs(experiment_ids=[exp_id])
    candidates = []
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        if r.data.tags.get("stage") != "sweep":
            continue
        if r.data.tags.get("model_type") != model_name:
            continue
        val_rmse = r.data.metrics.get("best_val_rmse")
        if val_rmse is None:
            continue
        hidden = int(r.data.params.get("hidden_size"))
        layers = int(r.data.params.get("num_layers"))
        dropout = float(r.data.params.get("dropout"))
        candidates.append((val_rmse, hidden, layers, dropout))

    candidates.sort(key=lambda x: x[0])
    seen = set()
    best_configs = []
    for _, hidden, layers, dropout in candidates:
        cfg = (hidden, layers, dropout)
        if cfg not in seen:
            seen.add(cfg)
            best_configs.append(cfg)
            if len(best_configs) == top_k:
                break
    return best_configs


def run_training(model, hidden, layers, dropout, seed, stage, run_name, client, exp_id):
    finished, _ = is_finished(client, exp_id, run_name)
    if finished:
        return "skipped"

    cmd = [
        sys.executable,
        "-m",
        "src.training.train",
        "--model",
        model,
        "--hidden_size",
        str(hidden),
        "--num_layers",
        str(layers),
        "--dropout",
        str(dropout),
        "--seed",
        str(seed),
        "--stage",
        stage,
        "--run_name",
        run_name,
    ]

    res = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        print(f"{run_name}: FAILED", flush=True)
        return "failed"

    _, best_val = is_finished(client, exp_id, run_name)
    val_str = f"{best_val:.2f}" if best_val is not None else "None"
    print(f"{run_name} - best_val_rmse: {val_str}", flush=True)
    return "executed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_grid", action="store_true")
    parser.add_argument("--only_seeds", action="store_true")
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
        exp_id = client.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id

    n_executed = 0
    n_skipped = 0
    n_failed = 0

    run_grid = not args.only_seeds
    run_seeds = not args.only_grid

    # 1. Rejilla
    if run_grid:
        models = ["rnn", "lstm"]
        hidden_sizes = [32, 64, 128]
        layer_dropout_configs = [(1, 0.0), (2, 0.0), (2, 0.2)]
        grid_seed = 42

        for model in models:
            for hidden in hidden_sizes:
                for layers, dropout in layer_dropout_configs:
                    run_name = f"{model}_h{hidden}_l{layers}_d{dropout}_s{grid_seed}"
                    status = run_training(
                        model,
                        hidden,
                        layers,
                        dropout,
                        grid_seed,
                        "sweep",
                        run_name,
                        client,
                        exp_id,
                    )
                    if status == "executed":
                        n_executed += 1
                    elif status == "skipped":
                        n_skipped += 1
                    elif status == "failed":
                        n_failed += 1

    # 2. Semillas
    if run_seeds:
        seed_list = [1, 2, 3]
        for model in ["rnn", "lstm"]:
            best_configs = get_best_grid_configs(client, exp_id, model, top_k=2)
            for hidden, layers, dropout in best_configs:
                for seed in seed_list:
                    run_name = f"{model}_h{hidden}_l{layers}_d{dropout}_s{seed}"
                    status = run_training(
                        model,
                        hidden,
                        layers,
                        dropout,
                        seed,
                        "seeds",
                        run_name,
                        client,
                        exp_id,
                    )
                    if status == "executed":
                        n_executed += 1
                    elif status == "skipped":
                        n_skipped += 1
                    elif status == "failed":
                        n_failed += 1

    print(
        f"Ejecutados: {n_executed}, Saltados: {n_skipped}, Fallidos: {n_failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
