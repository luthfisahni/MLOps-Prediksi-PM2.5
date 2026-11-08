import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API_URL = (
    "https://api.openaq.org/v3/sensors/14739443/measurements"
)

LOCATION_ID = 6144741
SENSOR_ID = 14739443
PARAMETER = "pm25"
HOURS_TO_FETCH = 24 * 30
LIMIT = 1000
OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "pm25_stt_satyabhakti.csv"

def fetch_data():
    api_key = os.getenv("OPENAQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAQ_API_KEY belum tersedia."
        )

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(
        hours = HOURS_TO_FETCH
    )

    headers = {
        "X-API-Key": api_key
    }

    params = {
        "datetime_from": start_time.isoformat(),
        "datetime_to": now.isoformat(),
        "limit": LIMIT
    }

    try:
        response = requests.get(
            API_URL,
            headers=headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Gagal mengambil data dari OpenAQ API: {error}"
        )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            "Respons API bukan JSON yang valid."
        )

    results = data.get("results", [])

    if not results:
        raise RuntimeError(
            "Tidak ada data yang dikembalikan API."
        )

    rows = []

    for item in results:

        period = item.get("period", {})
        datetime_from = period.get(
            "datetimeFrom", {}
        )

        datetime_utc = datetime_from.get("utc")
        datetime_local = datetime_from.get("local")

        value = item.get("value")

        if datetime_utc is None or value is None:
            continue

        rows.append({
            "location_id": LOCATION_ID,
            "sensor_id": SENSOR_ID,
            "parameter": PARAMETER,
            "datetime_utc": datetime_utc,
            "datetime_local": datetime_local,
            "value": value,
            "unit": "µg/m³"
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Tidak ada data valid yang dapat disimpan."
        )

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.sort_values(
        "datetime_utc"
    )

    df = df.dropna(
        subset=["datetime_utc"]
    )

    df = df.drop_duplicates(
        subset=["datetime_utc"],
        keep="last"
    )

    return df

def save_data(new_data):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FILE.exists():

        old_data = pd.read_csv(
            OUTPUT_FILE
        )

        print(
            f"Data lama ditemukan: "
            f"{len(old_data)} baris"
        )

    else:

        old_data = pd.DataFrame()

    old_count = len(old_data)

    combined = pd.concat(
        [old_data, new_data],
        ignore_index=True
    )

    combined["datetime_utc"] = pd.to_datetime(
        combined["datetime_utc"],
        utc=True,
        errors="coerce"
    )

    combined = combined.dropna(
        subset=["datetime_utc"]
    )

    before = len(combined)

    combined = combined.drop_duplicates(
        subset=["datetime_utc"],
        keep="last"
    )

    duplicate_removed = (
        before - len(combined)
    )

    combined = combined.sort_values(
        "datetime_utc"
    )

    combined["datetime_utc"] = (
        combined["datetime_utc"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    new_count = len(combined) - old_count

    print(
        f"\nData berhasil disimpan ke: "
        f"{OUTPUT_FILE}"
    )
    print(
        f"Total data        : {len(combined)}"
    )
    print(
        f"Data baru         : {max(new_count, 0)}"
    )
    print(
        f"Duplikasi dihapus : "
        f"{duplicate_removed}"
    )

def main():

    print(
        "=== OpenAQ PM2.5 Data Ingestion ==="
    )

    print(
        f"Location ID : {LOCATION_ID}"
    )

    print(
        f"Sensor ID   : {SENSOR_ID}"
    )

    print(
        f"Parameter   : {PARAMETER}"
    )

    print(
        f"Periode     : "
        f"{HOURS_TO_FETCH // 24} hari terakhir"
    )

    try:
        new_data = fetch_data()

        print(
            f"\nData diterima dari API: "
            f"{len(new_data)}"
        )

        save_data(new_data)

    except RuntimeError as error:

        print(
            f"\nERROR: {error}"
        )

        return

if __name__ == "__main__":
    main()