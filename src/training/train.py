import argparse
import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import torch
import yaml
from mlflow.models.signature import infer_signature
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.factory import count_parameters, create_model
from src.training.engine import evaluate, train_one_epoch
from src.training.metrics import denormalize, mae, rmse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--stage", type=str, default="single")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_info():
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit.strip()
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:  # noqa: BLE001
        commit = "unknown"
    return commit


def get_dataset_hashes(base_dir: Path):
    raw_md5 = "unknown"
    processed_md5 = "unknown"
    raw_dvc_path = base_dir / "data" / "raw" / "beijing_pm25.csv.dvc"
    try:
        with open(raw_dvc_path, "r", encoding="utf-8") as f:
            raw_dvc = yaml.safe_load(f)
            raw_md5 = raw_dvc.get("outs", [{}])[0].get("md5", "unknown")
    except Exception:  # noqa: BLE001
        raw_md5 = "unknown"

    lock_path = base_dir / "dvc.lock"
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            lock_data = yaml.safe_load(f)
            outs = lock_data.get("stages", {}).get("preprocess", {}).get("outs", [])
            for out in outs:
                if "dataset.npz" in out.get("path", ""):
                    processed_md5 = out.get("md5", "unknown")
                    break
    except Exception:  # noqa: BLE001
        processed_md5 = "unknown"
    return raw_md5, processed_md5


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Parametros con override de CLI
    train_cfg = params["train"]
    model_cfg = params["model"]
    prep_cfg = params["preprocess"]
    mlflow_cfg = params["mlflow"]

    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    lr = args.lr if args.lr is not None else train_cfg["lr"]
    hidden_size = (
        args.hidden_size if args.hidden_size is not None else model_cfg["hidden_size"]
    )
    num_layers = (
        args.num_layers if args.num_layers is not None else model_cfg["num_layers"]
    )
    dropout = args.dropout if args.dropout is not None else model_cfg["dropout"]
    batch_size = (
        args.batch_size if args.batch_size is not None else train_cfg["batch_size"]
    )
    weight_decay = train_cfg["weight_decay"]
    patience = train_cfg["patience"]
    scheduler_patience = train_cfg["scheduler_patience"]
    scheduler_factor = train_cfg["scheduler_factor"]
    grad_clip = train_cfg["grad_clip"]
    seed = args.seed if args.seed is not None else train_cfg["seed"]
    run_name = args.run_name if args.run_name is not None else args.model

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cargar datos procesados
    data_dir = base_dir / params["data"]["processed_dir"]
    dataset = np.load(data_dir / "dataset.npz")
    with open(data_dir / "scaler.json", "r", encoding="utf-8") as f:
        scaler = json.load(f)

    mean0 = scaler["mean"][0]
    std0 = scaler["std"][0]

    x_train, y_train = dataset["X_train"], dataset["y_train"]
    x_val, y_val = dataset["X_val"], dataset["y_val"]
    x_test, y_test = dataset["X_test"], dataset["y_test"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Crear modelo
    input_size = x_train.shape[2]
    model = create_model(
        name=args.model,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=1,
    ).to(device)

    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment"])

    raw_md5, processed_md5 = get_dataset_hashes(base_dir)
    start_time = time.time()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model_type": args.model,
                "git_commit": get_git_info(),
                "stage": args.stage,
                "raw_dataset_md5": raw_md5,
                "processed_dataset_md5": processed_md5,
            }
        )
        mlflow.log_params(
            {
                "model_name": args.model,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "scheduler_patience": scheduler_patience,
                "scheduler_factor": scheduler_factor,
                "grad_clip": grad_clip,
                "seed": seed,
                "window": prep_cfg["window"],
                "horizon": prep_cfg["horizon"],
                "max_interp_gap": prep_cfg["max_interp_gap"],
                "n_parameters": n_params,
                "device": str(device),
                "input_size": input_size,
            }
        )

        best_val_rmse = float("inf")
        best_epoch = -1
        best_weights = None
        patience_counter = 0

        train_loss_history = []
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            t_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, grad_clip
            )
            v_loss, val_preds_norm, val_targets_norm = evaluate(
                model, val_loader, criterion, device
            )

            val_preds = denormalize(val_preds_norm, mean0, std0)
            val_targets = denormalize(val_targets_norm, mean0, std0)
            val_rmse_val = rmse(val_targets, val_preds)
            val_mae_val = mae(val_targets, val_preds)

            curr_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(v_loss)

            train_loss_history.append(t_loss)
            val_loss_history.append(v_loss)

            mlflow.log_metrics(
                {
                    "train_loss": t_loss,
                    "val_loss": v_loss,
                    "val_rmse": val_rmse_val,
                    "val_mae": val_mae_val,
                    "lr": curr_lr,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d} - train_loss: {t_loss:.4f} - val_loss: {v_loss:.4f} - val_rmse: {val_rmse_val:.2f} - lr: {curr_lr:.6f}"
            )

            if val_rmse_val < best_val_rmse:
                best_val_rmse = val_rmse_val
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping en epoch {epoch}")
                    break

        train_time = time.time() - start_time

        # Evaluar mejor modelo en test
        model.load_state_dict(best_weights)
        _, test_preds_norm, test_targets_norm = evaluate(
            model, test_loader, criterion, device
        )
        test_preds = denormalize(test_preds_norm, mean0, std0)
        test_targets = denormalize(test_targets_norm, mean0, std0)
        test_rmse_val = rmse(test_targets, test_preds)
        test_mae_val = mae(test_targets, test_preds)

        mlflow.log_metrics(
            {
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
                "test_rmse": test_rmse_val,
                "test_mae": test_mae_val,
                "train_time_seconds": train_time,
            }
        )

        # Guardar y registrar curvas de perdida
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_loss_history, label="train_loss")
        ax.plot(val_loss_history, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        curves_path = base_dir / "curves.png"
        fig.savefig(curves_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(curves_path))
        if curves_path.exists():
            curves_path.unlink()

        # Grafica de prediccion en test (primeras 168 horas disponibles)
        # Nota: ventanas validas disponibles en test
        n_points = min(168, len(test_targets))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(test_targets[:n_points], label="Real", color="black")
        ax.plot(test_preds[:n_points], label="Predicho", color="crimson")
        ax.set_xlabel("Paso temporal")
        ax.set_ylabel("PM2.5 (ug/m3)")
        ax.legend()
        forecast_path = base_dir / "forecast_test.png"
        fig.savefig(forecast_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(forecast_path))
        if forecast_path.exists():
            forecast_path.unlink()

        # Registrar artefactos de datos y config
        mlflow.log_artifact(str(data_dir / "scaler.json"))
        mlflow.log_artifact(str(data_dir / "feature_names.json"))
        mlflow.log_artifact(str(params_path))

        # Registrar modelo con firma inferida usando name="model"
        input_example = x_train[:1]
        sample_output = (
            model(torch.from_numpy(input_example).to(device)).detach().cpu().numpy()
        )
        signature = infer_signature(input_example, sample_output)
        mlflow.pytorch.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="pickle",
        )


if __name__ == "__main__":
    main()
