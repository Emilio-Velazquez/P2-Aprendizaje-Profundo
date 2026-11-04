from pathlib import Path

import pandas as pd


def inspect_gaps(csv_path: Path):
    df = pd.read_csv(csv_path)

    # Construccion de indice datetime horario
    dt = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["datetime"] = dt

    # 1. Verificacion de continuidad temporal
    n_duplicates = dt.duplicated().sum()
    expected_range = pd.date_range(start=dt.iloc[0], end=dt.iloc[-1], freq="h")
    is_continuous = (len(dt) == len(expected_range)) and (
        dt.equals(pd.Series(expected_range))
    )

    print(f"Indice horario continuo: {is_continuous}")
    print(f"Horas duplicadas: {n_duplicates}")
    print(f"Rango: {dt.iloc[0]} a {dt.iloc[-1]} (Total: {len(dt)} filas)")

    # 2. Conteo de nulos por columna
    print("Nulos por columna:")
    for col, count in df.drop(columns=["datetime"]).isna().sum().items():
        print(f"  {col}: {count}")

    # 3. Analisis detallado para pm2.5
    pm = df["pm2.5"]
    is_null = pm.isna()
    blocks = (is_null != is_null.shift()).cumsum()
    null_runs = is_null.groupby(blocks).sum()
    null_runs = null_runs[null_runs > 0]

    first_valid_idx = pm.first_valid_index()
    initial_nulls = first_valid_idx if first_valid_idx is not None else 0

    runs_over_6 = null_runs[null_runs > 6]
    nulls_over_6 = runs_over_6.sum()

    runs_over_24 = null_runs[null_runs > 24]
    nulls_over_24 = runs_over_24.sum()
    pct_over_24 = (nulls_over_24 / len(df)) * 100

    print("Analisis de nulos en pm2.5:")
    print(f"  Total nulos: {is_null.sum()}")
    print(f"  Cantidad de rachas de nulos: {len(null_runs)}")
    print(f"  Longitud maxima de racha: {int(null_runs.max())}")
    print(f"  Longitud mediana de racha: {float(null_runs.median())}")
    print(f"  Nulos en filas iniciales: {initial_nulls}")
    print(
        f"  Nulos en rachas > 6 horas: {int(nulls_over_6)} ({len(runs_over_6)} rachas)"
    )
    print(
        f"  Nulos en rachas > 24 horas: {int(nulls_over_24)} ({len(runs_over_24)} rachas)"
    )
    print(f"  Porcentaje de nulos en rachas > 24 horas: {pct_over_24:.2f}%")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    raw_csv = base_dir / "data" / "raw" / "beijing_pm25.csv"
    inspect_gaps(raw_csv)
