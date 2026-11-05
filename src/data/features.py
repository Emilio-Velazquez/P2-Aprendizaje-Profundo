import numpy as np
import pandas as pd

CBWD_CATEGORIES = ["NE", "NW", "SE", "cv"]


def add_features(df: pd.DataFrame, target: str = "pm2.5") -> pd.DataFrame:
    df_feat = df.copy()

    # One-hot encoding de direccion de viento con orden determinista
    for cat in CBWD_CATEGORIES:
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
