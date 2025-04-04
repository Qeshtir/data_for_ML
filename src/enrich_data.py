import dill
import pandas as pd
import numpy as np
import re
import dagster as dg

OUTPUT_PATH = "../artifacts/cian_dataset.dill"
LOAD_PATH = "../artifacts/cian_dataset_enriched.dill"


def extract_rooms(title):
    if pd.isna(title):
        return np.nan
    match = re.search(r'(\d+)-комн', title)
    return int(match.group(1)) if match else np.nan


def extract_sq_meters(title):
    if pd.isna(title):
        return np.nan
    match = re.search(r'(\d+[,.]\d+|\d+)\s*м²', title)
    if match:
        return float(match.group(1).replace(',', '.'))
    return np.nan


def extract_floors(title):
    if pd.isna(title):
        return np.nan, np.nan
    match = re.search(r'(\d+)/(\d+)\s*этаж', title)
    if match:
        return int(match.group(1)), int(match.group(2))
    return np.nan, np.nan


def main(OUTPUT_PATH, LOAD_PATH):
    with open(OUTPUT_PATH, "rb") as f:
        try:
            df = dill.load(f)
        except Exception as e:
            print(f"Ошибка загрузки существующих данных: {e}")

    geo_split = df['geo'].str.split(',', expand=True)
    geo_split.columns = ['geo_' + str(i) for i in range(geo_split.shape[1])]
    df = pd.concat([df, geo_split], axis=1)

    df["price"] = (
        df["price"]
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", np.nan)
        # .dropna()  # Пропуски пока оставим
        .astype(float)
        .astype("Int64")  # Делаем тип данных целым, но поддерживающим NaN
    )
    df["meter_price"] = df["meter_price"].str.replace(r"[^\d]", "", regex=True).astype(int)
    df['rooms'] = df['object_title'].apply(extract_rooms)
    df['sq_meters'] = df['object_title'].apply(extract_sq_meters)
    df['floor'], df['total_floors'] = zip(*df['object_title'].apply(extract_floors))

    with open(LOAD_PATH, "wb") as f:
        try:
            dill.dump(df, f)
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")