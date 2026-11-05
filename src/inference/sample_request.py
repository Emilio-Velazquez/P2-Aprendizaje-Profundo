import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def main():
    base_dir = Path(__file__).resolve().parents[2]
    raw_path = base_dir / "data" / "raw" / "beijing_pm25.csv"
    meta_path = base_dir / "data" / "processed" / "meta.json"
    examples_dir = base_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_train = meta["rows_per_split"]["train"]
    n_val = meta["rows_per_split"]["val"]
    window_size = int(meta["window"])

    df = pd.read_csv(raw_path)
    test_slice = df.iloc[n_train + n_val :].reset_index(drop=True)

    required_cols = ["pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]

    found_idx = None
    for i in range(len(test_slice) - window_size):
        win = test_slice.iloc[i : i + window_size]
        next_row = test_slice.iloc[i + window_size]
        if not win[required_cols].isna().any().any() and not pd.isna(next_row["pm2.5"]):
            found_idx = i
            break

    if found_idx is None:
        raise RuntimeError(
            "No se encontro una ventana valida sin NaNs en el split de test."
        )

    win_df = test_slice.iloc[found_idx : found_idx + window_size]
    next_val = float(test_slice.iloc[found_idx + window_size]["pm2.5"])

    records = []
    for _, row in win_df.iterrows():
        dt = datetime(
            int(row["year"]),
            int(row["month"]),
            int(row["day"]),
            int(row["hour"]),
        )
        records.append(
            {
                "timestamp": dt.isoformat(),
                "pm25": float(row["pm2.5"]),
                "DEWP": float(row["DEWP"]),
                "TEMP": float(row["TEMP"]),
                "PRES": float(row["PRES"]),
                "Iws": float(row["Iws"]),
                "Is": float(row["Is"]),
                "Ir": float(row["Ir"]),
                "cbwd": str(row["cbwd"]),
            }
        )

    payload = {"records": records}
    out_path = examples_dir / "sample_request.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Generado {out_path} con {len(records)} registros.")
    print(f"Valor real pm2.5 siguiente hora: {next_val}")


if __name__ == "__main__":
    main()
