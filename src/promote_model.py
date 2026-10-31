
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = ROOT_DIR / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")

client = MlflowClient()

MODEL_NAME = "PM25_RandomForest_Best"
SELECTED_VERSION = "2"

# Memastikan versi model tersedia
model_version = client.get_model_version(
    name=MODEL_NAME,
    version=SELECTED_VERSION
)

print("Model:", MODEL_NAME)
print("Version:", model_version.version)
print("Run ID:", model_version.run_id)

# None -> Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=SELECTED_VERSION,
    stage="Staging"
)

print("\nModel berhasil dipindahkan ke Staging.")

# Staging -> Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=SELECTED_VERSION,
    stage="Production",
    archive_existing_versions=True
)

print("Model berhasil dipindahkan ke Production.")

# Verifikasi status
updated_version = client.get_model_version(
    name=MODEL_NAME,
    version=SELECTED_VERSION
)

print("\nStatus akhir:", updated_version.current_stage)
