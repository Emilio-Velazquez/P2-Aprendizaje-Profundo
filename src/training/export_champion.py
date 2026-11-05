import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml


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

    # 1. Obtener metadatos del campeon registrado
    model_name = "pm25-forecaster"
    mv = client.get_model_version_by_alias(model_name, "champion")
    run_id = mv.tags.get("source_run_id") or mv.run_id

    champion_dir = base_dir / "model_export" / "champion"
    model_dest = champion_dir / "model"
    if champion_dir.exists():
        shutil.rmtree(champion_dir)
    champion_dir.mkdir(parents=True, exist_ok=True)
    model_dest.mkdir(parents=True, exist_ok=True)

    # Descargar artefactos del modelo registrado a model_export/champion/model/
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}@champion", dst_path=str(model_dest)
    )

    # Guardar state_dict en CPU
    loaded_model = mlflow.pytorch.load_model(str(model_dest))
    torch.save(loaded_model.cpu().state_dict(), champion_dir / "weights.pt")

    # 2. Descargar scaler.json, feature_names.json y params.yaml del run origen
    client.download_artifacts(run_id, "scaler.json", dst_path=str(champion_dir))
    client.download_artifacts(run_id, "feature_names.json", dst_path=str(champion_dir))
    client.download_artifacts(run_id, "params.yaml", dst_path=str(champion_dir))

    with open(champion_dir / "feature_names.json", "r", encoding="utf-8") as f:
        features_data = json.load(f)
    feature_list = features_data.get("feature_names", features_data)

    horizon = int(params["preprocess"]["horizon"])

    # Escribir env_info.json
    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mlflow_version": mlflow.__version__,
        "numpy_version": np.__version__,
        "run_id": run_id,
        "registered_model_version": str(mv.version),
        "horizon": horizon,
        "model_config": {
            "model_type": mv.tags.get("model_type", "lstm"),
            "hidden_size": int(mv.tags.get("hidden_size", 64)),
            "num_layers": int(mv.tags.get("num_layers", 2)),
            "dropout": float(mv.tags.get("dropout", 0.0)),
            "window": int(params["preprocess"]["window"]),
            "horizon": horizon,
            "n_features": len(feature_list),
        },
    }
    with open(champion_dir / "env_info.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # 3. Verificar en subproceso limpio que create_model + weights.pt reproduce el modelo registrado
    verify_script = (
        "import mlflow.pytorch, numpy as np, torch, sys\n"
        "from src.models import create_model\n"
        f"model_reg = mlflow.pytorch.load_model('models:/{model_name}@champion')\n"
        f"weights = torch.load(r'{champion_dir / 'weights.pt'}', map_location='cpu', weights_only=True)\n"
        f"cfg = {env_info['model_config']}\n"
        "model_clean = create_model(cfg['model_type'], cfg['n_features'], cfg['hidden_size'], cfg['num_layers'], cfg['dropout'])\n"
        "model_clean.load_state_dict(weights)\n"
        "model_clean.eval()\n"
        f"data = np.load(r'{base_dir / 'data' / 'processed' / 'dataset.npz'}')\n"
        "x_sample = torch.from_numpy(data['X_test'][:64])\n"
        "with torch.no_grad():\n"
        "    pred_clean = model_clean(x_sample).cpu().numpy()\n"
        "    pred_reg = model_reg(x_sample).cpu().numpy()\n"
        "diff = float(np.max(np.abs(pred_clean - pred_reg)))\n"
        "print(f'Diferencia weights.pt vs registrado: {diff:.6e}')\n"
        "if diff >= 1e-5:\n"
        "    print('Verificacion weights.pt: FAIL')\n"
        "    sys.exit(1)\n"
        "print('Verificacion weights.pt: PASS')\n"
    )

    res = subprocess.run(
        [sys.executable, "-c", verify_script],
        check=False,
    )
    if res.returncode != 0:
        print("Error al verificar exportacion local.")
        sys.exit(1)

    print("Exportacion de modelo completada en model_export/champion/")


if __name__ == "__main__":
    main()
