
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# Konfigurasi dataset dan eksperimen
DATA_FILE = Path("data/processed/pm25_stt_satyabhakti_clean.csv")
EXPERIMENT_NAME = "Prediksi PM2.5 - Random Forest"

LAG_PERIODS = [1, 2, 3]
TEST_RATIO = 0.2
RANDOM_STATE = 42

EXPERIMENTS = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]


def load_data():
    """Memuat dataset dan membuat fitur untuk prediksi periode berikutnya."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {"datetime", "value"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset harus memiliki kolom: {required_columns}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce"
    )
    df["value"] = pd.to_numeric(
        df["value"], errors="coerce"
    )

    df = (
        df.dropna(subset=["datetime", "value"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )

    # Fitur dari nilai PM2.5 pada periode sebelumnya.
    for lag in LAG_PERIODS:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # Target adalah nilai PM2.5 pada baris/periode berikutnya.
    df["target"] = df["value"].shift(-1)

    # Baris tanpa fitur lag atau target dihapus.
    feature_columns = [f"lag_{lag}" for lag in LAG_PERIODS]
    df = df.dropna(
        subset=feature_columns + ["target"]
    ).reset_index(drop=True)

    return df, feature_columns


def main():
    df, feature_columns = load_data()

    X = df[feature_columns]
    y = df["target"]

    # Pembagian kronologis: data lama untuk training,
    # data terbaru untuk testing.
    split_index = int(len(df) * (1 - TEST_RATIO))

    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            "Jumlah data tidak cukup untuk train/test split."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Seluruh eksperimen menggunakan dataset dan split yang sama.
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=== Pelatihan Model Prediksi PM2.5 ===")
    print(f"Dataset       : {DATA_FILE}")
    print(f"Jumlah sampel : {len(df)}")
    print(f"Data training: {len(X_train)}")
    print(f"Data testing : {len(X_test)}")
    print(f"Fitur        : {feature_columns}\n")

    results = []

    for config in EXPERIMENTS:
        run_name = (
            f"RF_estimators_{config['n_estimators']}"
            f"_depth_{config['max_depth']}"
        )

        with mlflow.start_run(run_name=run_name):
            model = RandomForestRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            rmse = mean_squared_error(
                y_test, predictions
            ) ** 0.5
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Catat parameter eksperimen.
            mlflow.log_param(
                "model_type", "RandomForestRegressor"
            )
            mlflow.log_param(
                "n_estimators", config["n_estimators"]
            )
            mlflow.log_param(
                "max_depth", config["max_depth"]
            )
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("lag_periods", str(LAG_PERIODS))
            mlflow.log_param("test_ratio", TEST_RATIO)
            mlflow.log_param(
                "feature_columns", str(feature_columns)
            )
            mlflow.log_param("target", "next_period_pm25")

            # Catat metrik evaluasi.
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Simpan model sebagai artefak run di MLflow.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="random_forest_model",
                input_example=X_train.head(3),
            )

            run_id = mlflow.active_run().info.run_id

            results.append({
                "run_id": run_id,
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            })

            print(f"Run       : {run_name}")
            print(f"Run ID    : {run_id}")
            print(f"RMSE      : {rmse:.4f}")
            print(f"MAE       : {mae:.4f}")
            print(f"R²        : {r2:.4f}")
            print("-" * 40)

    # RMSE terkecil menjadi kriteria pemilihan model terbaik.
    best_result = min(results, key=lambda result: result["rmse"])

    print("\n=== Model Terbaik Berdasarkan RMSE ===")
    print(f"Run ID      : {best_result['run_id']}")
    print(f"n_estimators: {best_result['n_estimators']}")
    print(f"max_depth   : {best_result['max_depth']}")
    print(f"RMSE        : {best_result['rmse']:.4f}")
    print(f"MAE         : {best_result['mae']:.4f}")
    print(f"R²          : {best_result['r2']:.4f}")


if __name__ == "__main__":
    main()
