import sys
from pathlib import Path
import pandas as pd


def download_beijing(raw_dir: Path) -> Path:
    target_file = raw_dir / "beijing_pm25.csv"
    if target_file.exists():
        print(f"El archivo ya existe en {target_file}")
        return target_file

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Intento 1: descarga mediante ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=381)
        frames = []
        if hasattr(dataset.data, "ids") and dataset.data.ids is not None:
            frames.append(dataset.data.ids)
        frames.append(dataset.data.features)
        frames.append(dataset.data.targets)
        df = pd.concat(frames, axis=1)

        expected_order = [
            "No",
            "year",
            "month",
            "day",
            "hour",
            "pm2.5",
            "DEWP",
            "TEMP",
            "PRES",
            "cbwd",
            "Iws",
            "Is",
            "Ir",
        ]
        cols = [c for c in expected_order if c in df.columns]
        df = df[cols]

        df.to_csv(target_file, index=False)
        print("Dataset descargado exitosamente via ucimlrepo")
        return target_file
    except Exception as e:
        print(f"Fallo descarga con ucimlrepo: {e}")

    # Intento 2: respaldo directo desde URL de UCI
    try:
        backup_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
        df_backup = pd.read_csv(backup_url)
        df_backup.to_csv(target_file, index=False)
        print("Dataset descargado exitosamente via URL de respaldo")
        return target_file
    except Exception as e:
        print(f"Fallo descarga via URL de respaldo: {e}")

    # Fallo total
    print(
        "Descarga fallida. Descargue manualmente PRSA_data_2010.1.1-2014.12.31.csv "
        "desde UCI Machine Learning Repository y coloquelo en data/raw/beijing_pm25.csv"
    )
    sys.exit(1)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    raw_path = base_dir / "data" / "raw"
    csv_path = download_beijing(raw_path)

    # Verificacion basica del dataset
    df_check = pd.read_csv(csv_path)
    print(f"Filas: {len(df_check)}")
    print(f"Columnas: {list(df_check.columns)}")
    print(f"Nulos en pm2.5: {df_check['pm2.5'].isna().sum()}")
