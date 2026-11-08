
from pathlib import Path

import mlflow
import pandas as pd

# Konfigurasi
ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = ROOT_DIR / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")

MODEL_NAME = "PM25_RandomForest_Best"

# Memuat model yang berstatus Production
model_uri = f"models:/{MODEL_NAME}@production"

print("Memuat model:", model_uri)

model = mlflow.pyfunc.load_model(model_uri)

# Memuat dataset
DATA_PATH = ROOT_DIR / "data/processed/pm25_stt_satyabhakti_clean.csv"
df = pd.read_csv(DATA_PATH)

# Mengambil tiga observasi terakhir
last_values = df["value"].tail(3).tolist()
input_data = pd.DataFrame([{
    "lag_1": last_values[-1],
    "lag_2": last_values[-2],
    "lag_3": last_values[-3]
}])

# Melakukan prediksi
prediction = model.predict(input_data)

print("\nInput:")
print(input_data)

print("\nHasil prediksi PM2.5:")
print(f"{float(prediction[0]):.2f} µg/m³")
