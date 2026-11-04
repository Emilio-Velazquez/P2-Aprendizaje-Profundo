import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Construir indice horario
    dt = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.index = dt
    df = df.sort_index()
    # Descartar columnas temporales crudas e identificador
    cols_to_drop = ["No", "year", "month", "day", "hour"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def add_features(df: pd.DataFrame, target: str = "pm2.5") -> pd.DataFrame:
    df_feat = df.copy()

    # One-hot encoding de direccion de viento con orden determinista
    categories = ["NE", "NW", "SE", "cv"]
    for cat in categories:
        df_feat[f"cbwd_{cat}"] = (df_feat["cbwd"] == cat).astype(float)
    df_feat = df_feat.drop(columns=["cbwd"])

    # Caracteristicas ciclicas temporales
    hours = df_feat.index.hour
    months = df_feat.index.month - 1
    df_feat["hour_sin"] = np.sin(2 * np.pi * hours / 24.0).astype(float)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hours / 24.0).astype(float)
    df_feat["month_sin"] = np.sin(2 * np.pi * months / 12.0).astype(float)
    df_feat["month_cos"] = np.cos(2 * np.pi * months / 12.0).astype(float)

    # Reordenar para que el target sea la columna 0
    other_cols = [c for c in df_feat.columns if c != target]
    return df_feat[[target] + other_cols]


def split_chronological(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def impute(split_df: pd.DataFrame, max_gap: int, target: str = "pm2.5"):
    df_out = split_df.copy()
    s = df_out[target].copy()
    is_na = s.isna()
    n_interpolated = 0

    if is_na.any():
        block_ids = (is_na != is_na.shift()).cumsum()
        for _, group in is_na.groupby(block_ids):
            if not group.iloc[0]:
                continue
            run_len = len(group)
            start_pos = s.index.get_loc(group.index[0])
            end_pos = start_pos + run_len

            # Interpolar solo si la racha cabe en max_gap y no toca bordes
            if run_len <= max_gap and start_pos > 0 and end_pos < len(s):
                y0 = s.iloc[start_pos - 1]
                y1 = s.iloc[end_pos]
                if pd.notna(y0) and pd.notna(y1):
                    steps = run_len + 1
                    interp_vals = [
                        y0 + (y1 - y0) * (k + 1) / steps for k in range(run_len)
                    ]
                    s.iloc[start_pos:end_pos] = interp_vals
                    n_interpolated += run_len

    df_out[target] = s
    return df_out, n_interpolated


def fit_scaler(train_df: pd.DataFrame):
    mean = np.nanmean(train_df.values, axis=0)
    std = np.nanstd(train_df.values, axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_scaler(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (df.values - mean) / std


def make_windows(array: np.ndarray, window: int, horizon: int):
    t_steps = len(array)
    total_possible = t_steps - window - horizon + 1
    if total_possible <= 0:
        empty_x = np.empty((0, window, array.shape[1]), dtype=np.float32)
        empty_y = np.empty((0, 1), dtype=np.float32)
        return empty_x, empty_y, 0, 0, 0

    x_list = []
    y_list = []
    discarded = 0

    for i in range(total_possible):
        x_win = array[i : i + window]
        y_val = array[i + window - 1 + horizon, 0:1]
        # Descartar ventana si contiene algun NaN
        if np.isnan(x_win).any() or np.isnan(y_val).any():
            discarded += 1
        else:
            x_list.append(x_win)
            y_list.append(y_val)

    if x_list:
        x_arr = np.array(x_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
    else:
        x_arr = np.empty((0, window, array.shape[1]), dtype=np.float32)
        y_arr = np.empty((0, 1), dtype=np.float32)

    final = len(x_arr)
    return x_arr, y_arr, total_possible, discarded, final


def main():
    base_dir = Path(__file__).resolve().parents[2]
    params_path = base_dir / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_cfg = params["data"]
    prep_cfg = params["preprocess"]

    raw_path = base_dir / data_cfg["raw_path"]
    processed_dir = base_dir / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    window = int(prep_cfg["window"])
    horizon = int(prep_cfg["horizon"])
    train_frac = float(prep_cfg["train_frac"])
    val_frac = float(prep_cfg["val_frac"])
    target = str(prep_cfg["target"])
    max_gap = int(prep_cfg["max_interp_gap"])

    # Carga y features
    df = load_raw(raw_path)
    df = add_features(df, target=target)
    feature_names = list(df.columns)

    # Split cronologico
    train_df, val_df, test_df = split_chronological(df, train_frac, val_frac)
    rows_per_split = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }

    # Imputacion por split
    train_imp, n_tr = impute(train_df, max_gap, target=target)
    val_imp, n_va = impute(val_df, max_gap, target=target)
    test_imp, n_te = impute(test_df, max_gap, target=target)
    interpolated_nulls = {"train": n_tr, "val": n_va, "test": n_te}

    # Ajuste de escalador solo sobre train
    mean, std = fit_scaler(train_imp)

    # Escalado
    train_scaled = apply_scaler(train_imp, mean, std)
    val_scaled = apply_scaler(val_imp, mean, std)
    test_scaled = apply_scaler(test_imp, mean, std)

    # Ventanas temporales
    x_train, y_train, pos_tr, disc_tr, fin_tr = make_windows(
        train_scaled, window, horizon
    )
    x_val, y_val, pos_va, disc_va, fin_va = make_windows(val_scaled, window, horizon)
    x_test, y_test, pos_te, disc_te, fin_te = make_windows(test_scaled, window, horizon)

    possible_windows = {"train": pos_tr, "val": pos_va, "test": pos_te}
    discarded_windows = {"train": disc_tr, "val": disc_va, "test": disc_te}
    final_windows = {"train": fin_tr, "val": fin_va, "test": fin_te}

    # Guardar dataset.npz
    np.savez_compressed(
        processed_dir / "dataset.npz",
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
    )

    # Guardar scaler.json
    scaler_data = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_names": feature_names,
    }
    with open(processed_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler_data, f, indent=2)

    # Guardar feature_names.json
    with open(processed_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)

    # Guardar meta.json
    meta_data = {
        "window": window,
        "horizon": horizon,
        "n_features": len(feature_names),
        "target": target,
        "max_interp_gap": max_gap,
        "shapes": {
            "X_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(x_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(x_test.shape),
            "y_test": list(y_test.shape),
        },
        "rows_per_split": rows_per_split,
        "possible_windows": possible_windows,
        "discarded_windows": discarded_windows,
        "final_windows": final_windows,
        "interpolated_nulls": interpolated_nulls,
    }
    with open(processed_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    # Imprimir resumen
    print("Filas por split:")
    for split_name, count in rows_per_split.items():
        print(f"  {split_name}: {count}")

    print("Ventanas por split (posibles / descartadas / finales):")
    for s_name in ["train", "val", "test"]:
        p = possible_windows[s_name]
        d = discarded_windows[s_name]
        fn = final_windows[s_name]
        print(f"  {s_name}: posibles={p}, descartadas={d}, finales={fn}")

    print("Formas de arreglos:")
    print(f"  X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {x_test.shape}, y_test: {y_test.shape}")


if __name__ == "__main__":
    main()
