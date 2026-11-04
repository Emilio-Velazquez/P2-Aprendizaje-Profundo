import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.preprocess import (
    add_features,
    impute,
    load_raw,
    split_chronological,
)


@pytest.fixture(scope="session")
def project_paths():
    root = Path(__file__).resolve().parents[1]
    return {
        "params": root / "params.yaml",
        "dataset": root / "data" / "processed" / "dataset.npz",
        "scaler": root / "data" / "processed" / "scaler.json",
        "meta": root / "data" / "processed" / "meta.json",
        "raw": root / "data" / "raw" / "beijing_pm25.csv",
    }


@pytest.fixture(scope="session")
def loaded_data(project_paths):
    with open(project_paths["params"], "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open(project_paths["scaler"], "r", encoding="utf-8") as f:
        scaler = json.load(f)
    with open(project_paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    npz_data = np.load(project_paths["dataset"])
    return params, scaler, meta, npz_data


def test_shapes_and_consistency(loaded_data):
    params, _, meta, npz = loaded_data
    window = params["preprocess"]["window"]
    n_features = meta["n_features"]

    # Verificar 3 dimensiones en X y concordancia de ventana
    for split in ["train", "val", "test"]:
        x = npz[f"X_{split}"]
        y = npz[f"y_{split}"]
        assert x.ndim == 3
        assert y.ndim == 2
        assert x.shape[1] == window
        assert x.shape[2] == n_features
        assert y.shape[1] == 1
        assert len(x) == len(y)


def test_no_nan_or_inf(loaded_data):
    _, _, _, npz = loaded_data
    # Comprobar ausencia estricta de NaN e inf en los 6 arreglos
    for split in ["train", "val", "test"]:
        x = npz[f"X_{split}"]
        y = npz[f"y_{split}"]
        assert not np.isnan(x).any()
        assert not np.isinf(x).any()
        assert not np.isnan(y).any()
        assert not np.isinf(y).any()


def test_no_data_leakage(loaded_data, project_paths):
    params, scaler, _, npz = loaded_data
    x_train = npz["X_train"]
    x_val = npz["X_val"]
    x_test = npz["X_test"]

    # Columna 0 en train normalizada a media ~ 0 y std ~ 1
    col0_train = x_train[:, :, 0]
    assert np.isclose(col0_train.mean(), 0.0, atol=1e-2)
    assert np.isclose(col0_train.std(), 1.0, atol=1e-2)

    # Val y test no deben tener media exactamente 0
    assert abs(x_val[:, :, 0].mean()) > 1e-4
    assert abs(x_test[:, :, 0].mean()) > 1e-4

    # Verificar que el scaler proviene exclusivamente del conjunto train
    raw_df = load_raw(project_paths["raw"])
    feat_df = add_features(raw_df, target=params["preprocess"]["target"])
    train_df, _, _ = split_chronological(
        feat_df,
        params["preprocess"]["train_frac"],
        params["preprocess"]["val_frac"],
    )
    train_imp, _ = impute(
        train_df,
        params["preprocess"]["max_interp_gap"],
        target=params["preprocess"]["target"],
    )

    expected_mean = np.nanmean(train_imp.values, axis=0)
    expected_std = np.nanstd(train_imp.values, axis=0, ddof=0)
    expected_std = np.where(expected_std == 0.0, 1.0, expected_std)

    assert np.allclose(scaler["mean"], expected_mean)
    assert np.allclose(scaler["std"], expected_std)


def test_temporal_order_and_split_limits(loaded_data):
    params, _, meta, npz = loaded_data
    window = params["preprocess"]["window"]
    horizon = params["preprocess"]["horizon"]

    rows = meta["rows_per_split"]
    total_rows = rows["train"] + rows["val"] + rows["test"]
    assert total_rows == 43824

    # Proporciones 70/15/15 sobre filas
    assert rows["train"] == int(total_rows * params["preprocess"]["train_frac"])
    assert rows["val"] == int(total_rows * params["preprocess"]["val_frac"])
    assert rows["test"] == total_rows - rows["train"] - rows["val"]

    # Ninguna ventana debe cruzar el limite de su split
    for split in ["train", "val", "test"]:
        x = npz[f"X_{split}"]
        max_possible = rows[split] - window - horizon + 1
        assert len(x) <= max_possible
        assert len(x) == meta["final_windows"][split]
