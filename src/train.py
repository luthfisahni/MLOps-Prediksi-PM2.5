from pathlib import Path
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
import os
import json

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
)
DATA_PATH = ROOT_DIR / "data/processed/pm25_stt_satyabhakti_clean.csv"

EXPERIMENT_NAME = "Prediksi PM2.5 - Random Forest"
MODEL_NAME = "PM25_RandomForest_Best"
MODEL_ARTIFACT = "random_forest_model"

LAG_PERIODS = [1, 2, 3]
TEST_RATIO = 0.2
RANDOM_STATE = 42

EXPERIMENTS = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None},
]

VERSION_2_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
}

def prepare_data():
    df = pd.read_csv(DATA_PATH)

    for lag in LAG_PERIODS:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    df["target"] = df["value"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    features = [f"lag_{lag}" for lag in LAG_PERIODS]

    X = df[features]
    y = df["target"]

    split_index = int(len(df) * (1 - TEST_RATIO))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Jumlah data: {len(df)}")
    print(f"Data training: {len(X_train)}")
    print(f"Data testing: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def train_model(params, run_name, register=False):
    X_train, X_test, y_train, y_test = prepare_data()

    model = RandomForestRegressor(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_params({
            **params,
            "random_state": RANDOM_STATE,
            "test_ratio": TEST_RATIO,
            "lag_periods": str(LAG_PERIODS)
        })

        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            name=MODEL_ARTIFACT,
            input_example=X_train.head(3)
        )

        run_id = run.info.run_id

    print("\nHasil Training")
    print("Run ID:", run_id)
    print("Parameter:", params)
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE : {mae:.5f}")
    print(f"R2  : {r2:.5f}")

    if register:
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT}"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        print("\nModel berhasil didaftarkan!")
        print("Nama Model:", MODEL_NAME)
        print("Version:", registered_model.version)
        print("Run ID:", run_id)

    return {
        "run_id": run_id,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register-v2",
        action="store_true",
        help="Menjalankan konfigurasi LK-07 dan mendaftarkannya sebagai versi baru"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.register_v2:
        print("Menjalankan eksperimen LK-07: Version 2")

        train_model(
            params=VERSION_2_PARAMS,
            run_name="RF_estimators_100_depth_5_v2",
            register=True
        )

    else:
        print("Menjalankan seluruh eksperimen LK-06")

        results = []

        for i, params in enumerate(EXPERIMENTS, start=1):
            result = train_model(
                params=params,
                run_name=f"RF_Experiment_{i}"
            )
            results.append({**params, **result})

        best = min(results, key=lambda item: item["rmse"])

        print("\nEksperimen terbaik berdasarkan RMSE:")
        print(best)

        # Simpan hasil eksperimen terbaik untuk tahap evaluasi
        result_path = ROOT_DIR / "training_result.json"

        with open(result_path, "w") as file:
            json.dump(best, file, indent=4)

        print("\nHasil eksperimen terbaik disimpan:", result_path)

if __name__ == "__main__":
    main()
