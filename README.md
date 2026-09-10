# Prediksi Konsentrasi PM2.5 Berbasis Data Kualitas Udara Dinamis dengan Continual Learning

## Deskripsi Proyek

Proyek ini merupakan perancangan sistem MLOps untuk memprediksi konsentrasi PM2.5 menggunakan data kualitas udara yang bersifat dinamis dari OpenAQ API.

Sistem menggunakan pendekatan Continual Learning melalui proses monitoring data dan performa model secara berkala. Apabila terdeteksi data drift atau penurunan performa model, proses retraining akan dilakukan agar model dapat beradaptasi terhadap perubahan karakteristik data.

## Tujuan

Tujuan proyek ini adalah:

1. Mengambil data kualitas udara secara dinamis melalui OpenAQ API.
2. Melakukan preprocessing dan feature engineering pada data PM2.5.
3. Membangun model regresi untuk memprediksi konsentrasi PM2.5.
4. Melakukan eksperimen dan pencatatan model menggunakan MLflow.
5. Menerapkan monitoring terhadap data dan performa model.
6. Menerapkan mekanisme retraining ketika terjadi drift atau penurunan performa.

## Model Machine Learning

Model baseline yang digunakan dalam proyek ini adalah Random Forest Regressor.

Model digunakan untuk melakukan prediksi konsentrasi PM2.5 pada periode berikutnya berdasarkan fitur yang diperoleh dari data kualitas udara historis dan hasil feature engineering.

Metrik evaluasi yang digunakan meliputi:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)

## Struktur Direktori

```text
MLOps-Prediksi-PM2.5/
├── .devcontainer/
│   └── devcontainer.json
├── config/
├── data/
├── models/
├── notebooks/
├── src/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Penjelasan Direktori

| Direktori/File | Fungsi |
|---|---|
| `.devcontainer/` | Konfigurasi environment GitHub Codespaces |
| `config/` | Menyimpan konfigurasi proyek |
| `data/` | Menyimpan data proyek |
| `models/` | Menyimpan model machine learning |
| `notebooks/` | Menyimpan notebook untuk eksplorasi dan eksperimen |
| `src/` | Menyimpan source code utama proyek |
| `requirements.txt` | Daftar dependency Python yang digunakan |
| `.gitignore` | Menentukan file yang tidak perlu disimpan dalam Git |
| `LICENSE` | Lisensi proyek |
| `README.md` | Dokumentasi proyek |

## Environment Development

Proyek dikembangkan menggunakan GitHub Codespaces dengan environment Python yang telah dikonfigurasi melalui Dev Container.

Versi Python:
Python 3.12.14

Dependency utama:
- pandas
- numpy
- scikit-learn
- requests
- matplotlib
- seaborn
- mlflow
  
## Cara Menjalankan di GitHub Codespaces

1. Buka repository pada GitHub.
2. Pilih **Code → Codespaces**.
3. Buat atau buka Codespace pada branch `main`.
4. Environment akan menggunakan konfigurasi yang telah ditentukan pada `.devcontainer/devcontainer.json`.
5. Install dependency dengan perintah:

```bash
python -m pip install -r requirements.txt

