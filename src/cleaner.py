import dill
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

OUTPUT_PATH = "../artifacts/cian_dataset.dill"
SAVE_PATH = "../artifacts/cian_dataset_cleaned.dill"


def extract_quarter_year(text):
    """Извлекает квартал и год из текста с помощью регулярного выражения"""
    if pd.isna(text):
        return None, None

    # Ищем паттерны типа "X кв. YYYY" или "кв. YYYY"
    match = re.search(r'(\d+)\s*кв\.\s*(\d{4})', str(text))
    if match:
        return match.group(1), match.group(2)
    return None, None


def extract_rooms(text):
    if pd.isna(text):
        return np.nan

    text = text.lower()  # Приводим к нижнему регистру для унификации

    # Сначала проверяем студии
    studio_pattern = r'\b(апартаменты-)?студи(я|и|й|ю|ей|ям?|ями|ях)\b'
    if re.search(studio_pattern, text):
        return 0.5

    # Затем проверяем обычные комнаты
    match = re.search(r'(\d+)[-]?(?:комн|к| комнат?)\.?', text)
    return int(match.group(1)) if match else np.nan


def extract_sq_meters(title):
    if pd.isna(title):
        return np.nan
    # Ищем числа с разделителями . или , перед "м²" или "м2"
    match = re.search(r'(\d+[.,]?\d*)\s*(?:м²|м2|м\^2)', title)
    if match:
        value = match.group(1).replace(',', '.')
        return float(value)
    return np.nan


def get_embeddings(texts, model_name='cointegrated/rubert-tiny2'):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)


def extract_floors(title):
    if pd.isna(title):
        return (np.nan, np.nan)
    # Ищем паттерны этажей: "X/Y этаж", "X из Y этаж"
    match = re.search(r'(\d+)\s*[/из]+\s*(\d+)\s*этаж', title)
    return (int(match.group(1)), int(match.group(2))) if match else (np.nan, np.nan)


def clean(OUTPUT_PATH, SAVE_PATH):
    with open(OUTPUT_PATH, "rb") as f:
        try:
            df = dill.load(f)
        except Exception as e:
            print(f"Ошибка загрузки существующих данных: {e}")

    missing_values = ["No_content"]
    df = df.replace(missing_values, np.nan)

    FILLER = "dummy"

    # 1. object_subtitle и jk_name - заполним dummy филлером.
    df['object_subtitle'] = df['object_subtitle'].fillna(FILLER)
    df['jk_name'] = df['jk_name'].fillna(FILLER)

    # 2. deadline - попытаемся вычленить из subtitle, иначе заполним числовым филлером


    # Применяем функцию к обоим столбцам
    df[['deadline_q', 'deadline_y']] = df['deadline'].apply(
        lambda x: pd.Series(extract_quarter_year(x))
    )

    df[['subtitle_q', 'subtitle_y']] = df['object_subtitle'].apply(
        lambda x: pd.Series(extract_quarter_year(x))
    )

    # Объединяем результаты с приоритетом deadline
    df['quarter'] = df['deadline_q'].combine_first(df['subtitle_q'])
    df['year'] = df['deadline_y'].combine_first(df['subtitle_y'])

    # Удаляем временные столбцы
    df = df.drop(['deadline_q', 'deadline_y', 'subtitle_q', 'subtitle_y'], axis=1)

    # Конвертируем типы данных (если нужно)
    df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    df = df.drop(columns=['deadline'])

    # 438 пустых записей. Рискнём заполнить квартал - средним, а год - максимальным
    mean_quarter = round(df['quarter'].mean())
    max_year = df['year'].max()
    df['quarter'] = df['quarter'].fillna(mean_quarter)
    df['year'] = df['year'].fillna(max_year)

    df['quarter'] = df['quarter'].astype(int)
    df['year'] = df['year'].astype(int)

    # 3. Пустой geo - филлер + филлер для всех значений geo1-7
    df.loc[df['geo'].isna(), ['geo', 'geo_0', 'geo_1', 'geo_2', 'geo_3']] = FILLER

    # 4. geo4-7 - удалить столбцы, не забыть в конце удалить geo
    df = df.drop(columns=['geo', 'geo_4', 'geo_5', 'geo_6', 'geo_7'])

    # 5. rooms/floors/total_floors - подкрутим функции-парсеры

    # Сначала пробуем извлечь из object_subtitle
    df[['floor_sub', 'total_floors_sub']] = pd.DataFrame(
        df['object_subtitle'].apply(extract_floors).tolist(),
        index=df.index
    )
    df['rooms_sub'] = df['object_subtitle'].apply(extract_rooms)
    df['sq_meters_sub'] = df['object_subtitle'].apply(extract_sq_meters)

    # Затем из object_title
    df[['floor_title', 'total_floors_title']] = pd.DataFrame(
        df['object_title'].apply(extract_floors).tolist(),
        index=df.index
    )
    df['rooms_title'] = df['object_title'].apply(extract_rooms)
    df['sq_meters_title'] = df['object_title'].apply(extract_sq_meters)

    # Объединяем с приоритетом subtitle
    df['floor'] = df['floor_sub'].combine_first(df['floor_title'])
    df['total_floors'] = df['total_floors_sub'].combine_first(df['total_floors_title'])
    df['rooms'] = df['rooms_sub'].combine_first(df['rooms_title'])
    df['sq_meters'] = df['sq_meters_sub'].combine_first(df['sq_meters_title'])

    # Удаляем временные колонки
    df.drop(columns=[
        'floor_sub', 'total_floors_sub', 'rooms_sub', 'sq_meters_sub',
        'floor_title', 'total_floors_title', 'rooms_title', 'sq_meters_title'
    ], inplace=True)

    # План меняется, у нас почти не осталось пропусков. Заполним свободные планировки как 0
    df.loc[df['rooms'].isna(), ['rooms']] = 0

    # Теперь посмотрим экстремальные значения метро. Там нет метро. Поставим филлер.
    df.loc[df['geo_3'].isna(), ['geo_3']] = FILLER

    # 7. price - при наличии метража - заполним рассчётно, иначе - удалим объект. У нас всегда есть метраж.
    df['price'] = df.apply(
        lambda row: (row['meter_price'] * row['sq_meters'])
        if pd.isna(row['price'])
        else row['price'],
        axis=1
    )
    df["price"] = df["price"].astype("int64")

    le_jk = LabelEncoder()
    le_g1 = LabelEncoder()
    le_g2 = LabelEncoder()
    le_g3 = LabelEncoder()
    ohe = OneHotEncoder(sparse_output=False, drop="first")

    df["jk_name"] = le_jk.fit_transform(df["jk_name"])
    df["geo_1"] = le_g1.fit_transform(df["geo_1"])
    df["geo_2"] = le_g2.fit_transform(df["geo_2"])
    df["geo_3"] = le_g3.fit_transform(df["geo_3"])

    geo = ohe.fit_transform(df[['geo_0']])
    geo_names = ohe.get_feature_names_out(['geo_0'])

    f_geo = pd.DataFrame(geo, columns=geo_names)

    df = pd.concat([df.reset_index(drop=True), f_geo.reset_index(drop=True)], axis=1)

    df = df.drop(columns=['object_title', 'object_subtitle', 'geo_0'])

    # теперь - NLP

    text_embeddings = get_embeddings(df['desc'].tolist())

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(text_embeddings)

    pca = PCA(n_components=0.9, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings_scaled)

    for i in range(reduced_embeddings.shape[1]):
        df[f'text_pca_{i + 1}'] = reduced_embeddings[:, i]

    df = df.drop(columns=['desc'])

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination='auto',
        random_state=42
    )

    outliers = iso_forest.fit_predict(df)

    df = df[outliers == 1]

    with open(SAVE_PATH, "wb") as f:
        try:
            dill.dump(df, f)
        except Exception as e:
            print(f"Ошибка загрузки существующих данных: {e}")
