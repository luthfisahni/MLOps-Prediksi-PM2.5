from pathlib import Path

import mlflow
from mlflow import MlflowClient

# Gunakan database MLflow yang sama dengan eksperimen training
DATABASE_PATH = Path("mlflow.db").resolve()
mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")

RUN_ID = "25c5a2a53a2d4418b3de91dfc9fd4072"
MODEL_NAME = "PM25_RandomForest_Best"
MODEL_ARTIFACT = "random_forest_model"

def main():
    client = MlflowClient()

    # Pastikan run yang akan didaftarkan tersedia
    run = client.get_run(RUN_ID)

    print("Run ID :", RUN_ID)
    print("RMSE   :", run.data.metrics.get("rmse"))
    print("MAE    :", run.data.metrics.get("mae"))
    print("R2     :", run.data.metrics.get("r2"))

    # Lokasi model yang sudah disimpan pada run tersebut
    model_uri = f"runs:/{RUN_ID}/{MODEL_ARTIFACT}"

    # Daftarkan model ke Model Registry
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\n=== MODEL BERHASIL DIDAFTARKAN ===")
    print("Nama model :", registered_model.name)
    print("Versi      :", registered_model.version)
    print("Run ID     :", registered_model.run_id)
    print("Model URI  :", model_uri)

if __name__ == "__main__":
    main()
