"""
Скрипт для изучения структуры данных в облаке.

Загружает один файл и выводит его структуру, колонки и примеры данных.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cloud_loader import init_loader
from src.data.data_parser import detect_data_structure, normalize_dataframe
import polars as pl


def explore_marketplace_file():
    """Изучает структуру файла маркетплейса."""
    print("=== Изучение структуры данных маркетплейса ===\n")
    
    # Инициализируем загрузчик
    loader = init_loader(
        public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
    )
    
    # Пробуем загрузить один файл (например, 01082.pq)
    print("Попытка загрузить файл: marketplace/events/01082.pq")
    try:
        # Загружаем БЕЗ нормализации сначала, чтобы увидеть исходную структуру
        df_raw = loader.read_parquet_from_url("marketplace/events/01082.pq", normalize=False)
        
        print(f"\n✅ Файл успешно загружен!")
        print(f"Размер: {df_raw.shape[0]} строк, {df_raw.shape[1]} колонок")
        
        print(f"\n📋 ИСХОДНАЯ СТРУКТУРА:")
        print(f"Колонки ({len(df_raw.columns)}):")
        for col in df_raw.columns:
            print(f"  - {col}: {df_raw[col].dtype}")
        
        print(f"\nСхема данных:")
        print(df_raw.schema)
        
        print(f"\nПервые 3 строки (исходные):")
        print(df_raw.head(3))
        
        # Определяем структуру
        print(f"\n🔍 АНАЛИЗ СТРУКТУРЫ:")
        structure = detect_data_structure(df_raw)
        print(f"Определенный тип: {structure['type']}")
        print(f"Есть user_id: {structure.get('has_user_id', False)}")
        print(f"Есть item_id: {structure.get('has_item_id', False)}")
        print(f"Есть brand_id: {structure.get('has_brand_id', False)}")
        print(f"Есть amount: {structure.get('has_amount', False)}")
        
        # Нормализуем
        print(f"\n🔄 НОРМАЛИЗАЦИЯ:")
        df_normalized = normalize_dataframe(df_raw, "marketplace", "marketplace/events/01082.pq")
        
        print(f"Размер после нормализации: {df_normalized.shape[0]} строк, {df_normalized.shape[1]} колонок")
        print(f"Колонки после нормализации:")
        for col in df_normalized.columns:
            print(f"  - {col}: {df_normalized[col].dtype}")
        
        print(f"\nПервые 3 строки (нормализованные):")
        print(df_normalized.head(3))
        
        # Статистика
        print(f"\n📊 СТАТИСТИКА:")
        numeric_cols = [col for col in df_normalized.columns if df_normalized[col].dtype in [pl.Int64, pl.Float64]]
        if numeric_cols:
            print(df_normalized.select(numeric_cols).describe())
        
        # Уникальные значения
        print(f"\n🔢 Уникальные значения (первые 5):")
        for col in df_normalized.columns[:5]:
            if df_normalized[col].dtype in [pl.Utf8, pl.Categorical]:
                unique_vals = df_normalized[col].unique()[:5].to_list()
                print(f"  {col}: {unique_vals}")
        
        return df_normalized
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        print(f"\nПробуем другой файл...")
        
        # Пробуем другие возможные имена файлов
        for test_file in ["01081.pq", "01080.pq", "00001.pq", "1.pq"]:
            try:
                print(f"\nПробуем: marketplace/events/{test_file}")
                df = loader.read_parquet_from_url(f"marketplace/events/{test_file}")
                print(f"✅ Успешно загружен {test_file}!")
                print(f"Размер: {df.shape}")
                print(f"Колонки: {df.columns}")
                print(f"\nПервые 3 строки:")
                print(df.head(3))
                return df
            except Exception as e2:
                print(f"  ❌ {test_file}: {e2}")
                continue
        
        return None


def explore_brands_file():
    """Изучает структуру файла брендов."""
    print("\n\n=== Изучение структуры данных брендов ===\n")
    
    loader = init_loader(
        public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
    )
    
    try:
        df = loader.read_parquet_from_url("brands.pq")
        print(f"✅ Файл brands.pq загружен!")
        print(f"Размер: {df.shape}")
        print(f"Колонки: {df.columns}")
        print(f"\nСхема:")
        print(df.schema)
        print(f"\nПервые 10 строк:")
        print(df.head(10))
        return df
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def explore_users_file():
    """Изучает структуру файла пользователей."""
    print("\n\n=== Изучение структуры данных пользователей ===\n")
    
    loader = init_loader(
        public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
    )
    
    try:
        df = loader.read_parquet_from_url("users.pq")
        print(f"✅ Файл users.pq загружен!")
        print(f"Размер: {df.shape}")
        print(f"Колонки: {df.columns}")
        print(f"\nСхема:")
        print(df.schema)
        print(f"\nПервые 10 строк:")
        print(df.head(10))
        return df
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


if __name__ == "__main__":
    print("🔍 Изучение структуры данных в облаке...\n")
    
    # Изучаем файлы
    mp_df = explore_marketplace_file()
    brands_df = explore_brands_file()
    users_df = explore_users_file()
    
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ ИНФОРМАЦИЯ")
    print("="*60)
    
    if mp_df is not None:
        print(f"\n✅ Marketplace events:")
        print(f"   Колонки: {', '.join(mp_df.columns)}")
        print(f"   Размер: {mp_df.shape}")
    
    if brands_df is not None:
        print(f"\n✅ Brands:")
        print(f"   Колонки: {', '.join(brands_df.columns)}")
        print(f"   Размер: {brands_df.shape}")
    
    if users_df is not None:
        print(f"\n✅ Users:")
        print(f"   Колонки: {', '.join(users_df.columns)}")
        print(f"   Размер: {users_df.shape}")

