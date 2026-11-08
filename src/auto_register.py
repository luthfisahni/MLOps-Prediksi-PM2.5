import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT_DIR = Path(__file__).resolve().parent.parent

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
)

MODEL_NAME = "PM25_RandomForest_Best"
MODEL_ARTIFACT = "random_forest_model"


def main():
    result_path = ROOT_DIR / "evaluation_result.json"

    with open(result_path) as file:
        result = json.load(file)

    if result["validation_status"] != "PASSED":
        raise SystemExit(
            "Model tidak lolos validasi. Registrasi dibatalkan."
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    run_id = result["run_id"]
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT}"

    try:
        client.create_registered_model(MODEL_NAME)
        print("Registered Model berhasil dibuat.")
    except Exception:
        client.get_registered_model(MODEL_NAME)

    registered_version = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id,
        description="Model hasil automated training yang lolos validasi LK-01",
    )

    print("Model berhasil didaftarkan.")
    print("Model:", MODEL_NAME)
    print("Version:", registered_version.version)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_version.version,
        stage="Staging",
    )

    print("Model berhasil dipindahkan ke Staging.")


if __name__ == "__main__":
    main()