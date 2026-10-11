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
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── ingest_data.py
│   └── preprocess.py
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
```

## Data Ingestion dan Preprocessing

### Sumber Data

Data kualitas udara diperoleh secara dinamis dari OpenAQ API dengan konfigurasi:

* Lokasi: STT Satyabhakti
* Location ID: `6144741`
* Sensor ID: `14739443`
* Parameter: PM2.5
* Interval pengukuran: 1 jam
* Satuan: µg/m³

### Data Ingestion

Proses pengambilan data diimplementasikan pada:

```text
src/ingest_data.py
```

API key disimpan menggunakan environment variable `OPENAQ_API_KEY` agar tidak ditulis langsung di dalam source code.

Set environment variable:

```bash
export OPENAQ_API_KEY="API_KEY_ANDA"
```

Kemudian jalankan:

```bash
python src/ingest_data.py
```

Data mentah disimpan pada:

```text
data/raw/pm25_stt_satyabhakti.csv
```

Script dapat dijalankan berulang kali untuk melakukan simulasi pengambilan data secara berkala. Data yang sudah tersimpan tidak akan dimasukkan kembali sehingga duplikasi berdasarkan waktu pengukuran dapat dihindari.

### Preprocessing

Proses preprocessing diimplementasikan pada:

```text
src/preprocess.py
```

Jalankan dengan:

```bash
python src/preprocess.py
```

Tahapan preprocessing meliputi:

1. Mengubah waktu pengukuran menjadi format `datetime`.
2. Mengubah nilai PM2.5 menjadi tipe numerik.
3. Menghapus data duplikat berdasarkan waktu pengukuran.
4. Menangani missing value pada waktu dan nilai PM2.5.
5. Menghapus nilai PM2.5 yang tidak valid.
6. Mengurutkan data berdasarkan waktu.
7. Memilih kolom `datetime` dan `value` untuk tahap feature engineering.

Hasil preprocessing disimpan pada:

```text
data/processed/pm25_stt_satyabhakti_clean.csv
```

Dataset hasil preprocessing selanjutnya digunakan sebagai input untuk tahap feature engineering dan pengembangan model.

## Data Versioning dengan DVC

### Tujuan

Data Version Control (DVC) digunakan untuk mengelola versi dataset PM2.5 yang diperoleh melalui proses data ingestion. Dengan DVC, perubahan dataset dapat dilacak sehingga versi data sebelum dan sesudah pembaruan dapat dibandingkan.

Git digunakan untuk menyimpan konfigurasi DVC dan file pointer `.dvc`, sedangkan DVC mengelola isi dataset melalui cache lokal.

### Inisialisasi DVC

DVC diinstal dan diinisialisasi pada repository menggunakan perintah:

```bash
python -m pip install dvc
dvc init
```

Inisialisasi menghasilkan konfigurasi pada direktori `.dvc/` dan file `.dvcignore`. Konfigurasi tersebut disimpan dalam Git agar dapat digunakan bersama oleh pengembang proyek.

### Pelacakan Dataset

Dataset mentah dari proses ingestion LK-04 dilacak menggunakan perintah:

```bash
dvc add data/raw/pm25_stt_satyabhakti.csv
```

Perintah tersebut menghasilkan file pointer:

```text
data/raw/pm25_stt_satyabhakti.csv.dvc
```

DVC juga membuat aturan pada `data/raw/.gitignore` agar file CSV tidak dilacak langsung oleh Git. File pointer `.dvc` tetap disimpan dalam Git untuk mencatat metadata dataset.

### Simulasi Pembaruan Dataset

Pembaruan dataset disimulasikan dengan menjalankan kembali proses ingestion untuk mengambil data terbaru dari OpenAQ API.

```bash
python src/ingest_data.py
dvc add data/raw/pm25_stt_satyabhakti.csv
```

Pada pengujian ini, proses ingestion menerima 719 observasi dari API, menemukan 704 baris data lama, dan melaporkan 344 data baru. Setelah penggabungan data dan penghapusan duplikasi, dataset berisi 1.048 baris data atau 1.049 baris termasuk header CSV.

Setelah dataset diperbarui, perintah `dvc add` dijalankan kembali untuk memperbarui metadata pada file pointer `.dvc`.

### Audit dan Perbandingan Versi Dataset

Perubahan metadata dataset dapat diperiksa menggunakan perintah Git:

```bash
git diff -- data/raw/pm25_stt_satyabhakti.csv.dvc
```

Hasil perbandingan metadata dataset pada pengujian LK-05 adalah sebagai berikut:

| Metadata | Versi Awal | Versi Setelah Ingestion |
|---|---|---|
| MD5 | `174146b294ea2ec06fc9a9c2272b7c3c` | `b823abf04658e25325fd4054c9bba117` |
| Ukuran | 57.783 byte | 85.995 byte |

Perubahan hash dan ukuran menunjukkan bahwa isi dataset telah berubah setelah proses ingestion ulang.

Perbandingan ant versi dataset yang tersimpan pada commit Git dilakukan menggunakan:

```bash
dvc diff HEAD~1 HEAD
```

Hasil pengujian menunjukkan bahwa `data/raw/pm25_stt_satyabhakti.csv` terdeteksi sebagai dataset yang dimodifikasi.

Status dataset saat ini dapat diperiksa menggunakan:

```bash
dvc status
```

Apabila dataset sesuai dengan metadata DVC terbaru, perintah tersebut menampilkan status `Data and pipelines are up to date.`

### Penyimpanan dan Reproduksibilitas

DVC menyimpan dataset dalam cache lokal, sedangkan Git menyimpan file pointer `.dvc` yang mencatat metadata dataset. Kombinasi keduanya memungkinkan perubahan dataset dilacak dan versi yang berbeda dibandingkan.