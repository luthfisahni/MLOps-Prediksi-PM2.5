import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/raw/pm25_stt_satyabhakti.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "pm25_stt_satyabhakti_clean.csv"


def preprocess_data():
    print("=== OpenAQ PM2.5 Preprocessing ===")

    if not INPUT_FILE.exists():
        print(f"ERROR: File tidak ditemukan: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Data awal: {len(df)} baris")

    required_columns = [
        "parameter",
        "datetime_local",
        "value",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(f"ERROR: Kolom tidak ditemukan: {missing_columns}")
        return

    before = len(df)

    df = df[
        df["parameter"].astype(str).str.lower() == "pm25"
    ].copy()

    parameter_removed = before - len(df)
    before = len(df)

    df = df.drop_duplicates(
        subset=["datetime_local"]
    )

    duplicate_removed = before - len(df)

    df["datetime"] = pd.to_datetime(
        df["datetime_local"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    before = len(df)

    df = df.dropna(
        subset=["datetime", "value"]
    )
    missing_removed = before - len(df)

    before = len(df)
    df = df[df["value"] >= 0].copy()

    invalid_removed = before - len(df)

    df = df.sort_values("datetime")

    df = df[
        ["datetime", "value"]
    ].reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n=== HASIL PREPROCESSING ===")
    print(f"Parameter bukan PM2.5   : {parameter_removed}")
    print(f"Duplikasi dihapus       : {duplicate_removed}")
    print(f"Missing value dihapus   : {missing_removed}")
    print(f"Nilai PM2.5 tidak valid : {invalid_removed}")
    print(f"Data akhir              : {len(df)} baris")
    print(f"Output                  : {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()