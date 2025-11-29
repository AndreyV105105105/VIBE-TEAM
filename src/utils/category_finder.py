"""
Модуль для агрессивного поиска категорий для брендов пользователя.

Использует несколько уровней fallback для максимально надежного определения категорий.
"""

from typing import Dict, List, Optional, Set
import polars as pl
from pathlib import Path


def find_categories_for_brands_aggressive(
    brand_ids: List[str],
    items_catalog: Optional[Dict[str, pl.DataFrame]] = None,
    loader=None,
    user_item_ids: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Агрессивный поиск категорий для брендов с множественными fallback уровнями.
    
    :param brand_ids: Список brand_id для поиска
    :param items_catalog: Каталоги товаров (marketplace, retail)
    :param loader: Загрузчик данных (для прямого доступа к файлам)
    :param user_item_ids: Item IDs пользователя (для дополнительного поиска)
    :return: Словарь brand_id -> category
    """
    brand_to_category = {}
    brand_ids_normalized = [str(bid).replace(".0", "") if bid else None for bid in brand_ids if bid]
    brand_ids_normalized = [bid for bid in brand_ids_normalized if bid and bid != "unknown"]
    
    if not brand_ids_normalized:
        return brand_to_category
    
    print(f"🔍 Агрессивный поиск категорий для {len(brand_ids_normalized)} брендов...")
    
    # Уровень 1: Поиск в items_catalog (marketplace и retail)
    if items_catalog:
        for catalog_name, catalog_df in items_catalog.items():
            if catalog_df.height == 0 or "brand_id" not in catalog_df.columns:
                continue
            
            # Определяем колонку категории
            category_col = None
            for col in catalog_df.columns:
                if col.lower() in ["category", "category_id"]:
                    category_col = col
                    break
            
            if not category_col:
                continue
            
            print(f"   📦 Уровень 1: Проверка {catalog_name} каталога ({catalog_df.height} товаров)...")
            
            # Нормализуем brand_id для сравнения
            catalog_df_normalized = catalog_df.with_columns(
                pl.col("brand_id").cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias("brand_id_normalized")
            )
            
            # Группируем по brand_id и находим самую частую категорию
            for brand_id in brand_ids_normalized:
                if brand_id in brand_to_category:
                    continue  # Уже нашли категорию
                
                brand_items = catalog_df_normalized.filter(
                    pl.col("brand_id_normalized") == str(brand_id)
                )
                
                if brand_items.height > 0:
                    # Находим категории для этого бренда
                    valid_categories = brand_items.filter(
                        pl.col(category_col).is_not_null() &
                        (pl.col(category_col) != "") &
                        (pl.col(category_col).cast(pl.Utf8) != "nan")
                    )
                    
                    if valid_categories.height > 0:
                        # Берем самую частую категорию
                        category_counts = valid_categories[category_col].value_counts()
                        if category_counts.height > 0:
                            top_category = category_counts["category" if "category" in category_counts.columns else category_col][0]
                            brand_to_category[brand_id] = str(top_category)
                            print(f"      ✅ Brand {brand_id}: найдена категория '{top_category}' из {catalog_name} ({valid_categories.height} товаров)")
    
    # Уровень 2: Прямой поиск в items.pq файлах через loader
    if loader:
        missing_brands = [bid for bid in brand_ids_normalized if bid not in brand_to_category]
        if missing_brands:
            print(f"   📦 Уровень 2: Прямой поиск в items.pq для {len(missing_brands)} брендов...")
            
            try:
                # Пробуем marketplace items
                mp_items_lazy = loader.load_marketplace_items(
                    brand_ids=missing_brands,
                    item_ids=None,
                    use_lazy=True,
                    include_embedding=False
                )
                
                if mp_items_lazy is not None:
                    try:
                        mp_items = mp_items_lazy.limit(5000).collect()  # Увеличиваем лимит для лучшего покрытия
                        if mp_items.height > 0:
                            category_col = None
                            for col in mp_items.columns:
                                if col.lower() in ["category", "category_id"]:
                                    category_col = col
                                    break
                            
                            if category_col and "brand_id" in mp_items.columns:
                                mp_items_normalized = mp_items.with_columns(
                                    pl.col("brand_id").cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias("brand_id_normalized")
                                )
                                
                                for brand_id in missing_brands:
                                    if brand_id in brand_to_category:
                                        continue
                                    
                                    brand_items = mp_items_normalized.filter(
                                        pl.col("brand_id_normalized") == str(brand_id)
                                    ).filter(
                                        pl.col(category_col).is_not_null() &
                                        (pl.col(category_col) != "")
                                    )
                                    
                                    if brand_items.height > 0:
                                        category_counts = brand_items[category_col].value_counts()
                                        if category_counts.height > 0:
                                            top_category = category_counts[category_col][0]
                                            brand_to_category[brand_id] = str(top_category)
                                            print(f"         ✅ Brand {brand_id}: '{top_category}' из marketplace items ({brand_items.height} товаров)")
                    except Exception as e:
                        print(f"         ⚠ Ошибка при загрузке marketplace items: {e}")
                
                # Пробуем retail items
                still_missing = [bid for bid in missing_brands if bid not in brand_to_category]
                if still_missing:
                    retail_items_lazy = loader.load_retail_items(
                        brand_ids=still_missing,
                        item_ids=None,
                        use_lazy=True,
                        include_embedding=False
                    )
                    
                    if retail_items_lazy is not None:
                        try:
                            retail_items = retail_items_lazy.limit(5000).collect()
                            if retail_items.height > 0:
                                category_col = None
                                for col in retail_items.columns:
                                    if col.lower() in ["category", "category_id"]:
                                        category_col = col
                                        break
                                
                                if category_col and "brand_id" in retail_items.columns:
                                    retail_items_normalized = retail_items.with_columns(
                                        pl.col("brand_id").cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias("brand_id_normalized")
                                    )
                                    
                                    for brand_id in still_missing:
                                        if brand_id in brand_to_category:
                                            continue
                                        
                                        brand_items = retail_items_normalized.filter(
                                            pl.col("brand_id_normalized") == str(brand_id)
                                        ).filter(
                                            pl.col(category_col).is_not_null() &
                                            (pl.col(category_col) != "")
                                        )
                                        
                                        if brand_items.height > 0:
                                            category_counts = brand_items[category_col].value_counts()
                                            if category_counts.height > 0:
                                                top_category = category_counts[category_col][0]
                                                brand_to_category[brand_id] = str(top_category)
                                                print(f"         ✅ Brand {brand_id}: '{top_category}' из retail items ({brand_items.height} товаров)")
                        except Exception as e:
                            print(f"         ⚠ Ошибка при загрузке retail items: {e}")
            except Exception as e:
                print(f"      ⚠ Ошибка при прямом поиске в items.pq: {e}")
    
    # Уровень 3: Поиск через item_id пользователя
    if user_item_ids:
        missing_brands = [bid for bid in brand_ids_normalized if bid not in brand_to_category]
        if missing_brands and items_catalog:
            print(f"   📦 Уровень 3: Поиск через item_id пользователя для {len(missing_brands)} брендов...")
            
            for catalog_name, catalog_df in items_catalog.items():
                if catalog_df.height == 0 or "item_id" not in catalog_df.columns:
                    continue
                
                category_col = None
                for col in catalog_df.columns:
                    if col.lower() in ["category", "category_id"]:
                        category_col = col
                        break
                
                if not category_col:
                    continue
                
                # Фильтруем товары пользователя
                user_items_df = catalog_df.filter(
                    pl.col("item_id").cast(pl.Utf8).is_in([str(iid) for iid in user_item_ids])
                )
                
                if user_items_df.height > 0 and "brand_id" in user_items_df.columns:
                    user_items_normalized = user_items_df.with_columns(
                        pl.col("brand_id").cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias("brand_id_normalized")
                    )
                    
                    for brand_id in missing_brands:
                        if brand_id in brand_to_category:
                            continue
                        
                        brand_items = user_items_normalized.filter(
                            pl.col("brand_id_normalized") == str(brand_id)
                        ).filter(
                            pl.col(category_col).is_not_null() &
                            (pl.col(category_col) != "")
                        )
                        
                        if brand_items.height > 0:
                            category_counts = brand_items[category_col].value_counts()
                            if category_counts.height > 0:
                                top_category = category_counts[category_col][0]
                                brand_to_category[brand_id] = str(top_category)
                                print(f"         ✅ Brand {brand_id}: '{top_category}' через item_id ({brand_items.height} товаров)")
    
    # Уровень 4: Попытка найти в brands.pq (если есть категории там)
    missing_brands = [bid for bid in brand_ids_normalized if bid not in brand_to_category]
    if missing_brands and loader:
        print(f"   📦 Уровень 4: Поиск в brands.pq для {len(missing_brands)} брендов...")
        try:
            brands_df = loader.load_brands()
            if brands_df is not None and brands_df.height > 0:
                # Проверяем, есть ли категории в brands
                category_col = None
                for col in brands_df.columns:
                    if col.lower() in ["category", "category_id", "brand_category"]:
                        category_col = col
                        break
                
                if category_col and "brand_id" in brands_df.columns:
                    brands_normalized = brands_df.with_columns(
                        pl.col("brand_id").cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias("brand_id_normalized")
                    )
                    
                    for brand_id in missing_brands:
                        if brand_id in brand_to_category:
                            continue
                        
                        brand_row = brands_normalized.filter(
                            pl.col("brand_id_normalized") == str(brand_id)
                        ).filter(
                            pl.col(category_col).is_not_null() &
                            (pl.col(category_col) != "")
                        )
                        
                        if brand_row.height > 0:
                            category = brand_row[category_col][0]
                            brand_to_category[brand_id] = str(category)
                            print(f"         ✅ Brand {brand_id}: '{category}' из brands.pq")
        except Exception as e:
            print(f"         ⚠ Ошибка при поиске в brands.pq: {e}")
    
    found_count = len(brand_to_category)
    missing_count = len(brand_ids_normalized) - found_count
    print(f"   ✅ Найдено категорий: {found_count}/{len(brand_ids_normalized)}")
    if missing_count > 0:
        print(f"   ⚠ Не найдено категорий для брендов: {[bid for bid in brand_ids_normalized if bid not in brand_to_category][:5]}...")
    
    return brand_to_category

