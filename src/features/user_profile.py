"""
Модуль для создания профилей пользователей.

Формирует профили на основе агрегированных данных и паттернов поведения.
Использует embedding товаров для улучшения профиля (опционально).
"""

from typing import Dict, List, Optional
import polars as pl
import numpy as np


def create_user_profile(
    user_events: Dict[str, pl.DataFrame],
    patterns: Optional[List] = None,
    user_id: Optional[str] = None,
    items_with_embeddings: Optional[Dict[str, pl.DataFrame]] = None,
    item_to_brand_map: Optional[Dict[str, str]] = None,
    brands_categories_map: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Создает профиль пользователя на основе событий и паттернов.
    
    :param user_events: Словарь с событиями по доменам
    :param patterns: Список паттернов поведения
    :param user_id: ID пользователя
    :param items_with_embeddings: Каталоги товаров с эмбеддингами (опционально)
    :param item_to_brand_map: Маппинг item_id -> brand_id для восстановления пропусков
    :param brands_categories_map: Маппинг brand_id -> category для обогащения профиля
    :return: Словарь с профилем пользователя
    """
    profile = {}
    
    if user_id:
        profile["user_id"] = user_id
    
    # Статистики по маркетплейсу (используем category из items если доступна)
    mp_df = user_events.get("marketplace", pl.DataFrame())
    retail_df = user_events.get("retail", pl.DataFrame())
    
    # Объединяем marketplace и retail для общей статистики просмотров
    all_views = []
    if mp_df.height > 0:
        all_views.append(mp_df)
    if retail_df.height > 0:
        all_views.append(retail_df)
    
    if all_views:
        combined_views = pl.concat(all_views)
        profile["num_views"] = combined_views.height
        profile["unique_items"] = combined_views["item_id"].n_unique() if "item_id" in combined_views.columns else 0
        
        print(f"   📊 Всего событий просмотра: {combined_views.height}")
        print(f"   📊 Уникальных товаров: {profile['unique_items']}")
        print(f"   📊 Колонки в событиях: {list(combined_views.columns)}")
        
        # Проверяем наличие категорий в событиях
        has_category = "category" in combined_views.columns
        has_category_id = "category_id" in combined_views.columns
        print(f"   📊 category в событиях: {has_category}, category_id в событиях: {has_category_id}")
        
        if has_category:
            non_null_cats = combined_views.filter(pl.col("category").is_not_null()).height
            print(f"   📊 Событий с category: {non_null_cats} из {combined_views.height}")
        if has_category_id:
            non_null_cat_ids = combined_views.filter(pl.col("category_id").is_not_null()).height
            print(f"   📊 Событий с category_id: {non_null_cat_ids} из {combined_views.height}")
        
        # Простое извлечение категории - берем значение из колонки
        def extract_category_value(row_data, category_cols):
            """
            Извлекает категорию из строки данных.
            Берет первую доступную колонку с непустым значением.
            """
            for col in category_cols:
                if col in row_data:
                    value = row_data[col]
                    if value is not None and value != "":
                        return str(value).strip()
            return None
        
        # Топ категория - улучшенное извлечение с обогащением из items
        # Сначала пробуем обогатить события категориями из items (даже если категории есть, но null)
        item_to_category_map = {}
        if items_with_embeddings and combined_views.height > 0 and "item_id" in combined_views.columns:
            print(f"   🔍 items_with_embeddings доступен: {len(items_with_embeddings)} каталогов")
            print(f"   🔍 Попытка извлечения категорий из {len(items_with_embeddings)} каталогов items...")
            try:
                # Собираем категории из всех каталогов items
                for catalog_name, items_df in items_with_embeddings.items():
                    if items_df.height > 0 and "item_id" in items_df.columns:
                        # Ищем стандартные колонки категорий
                        category_cols = []
                        for col in ["category", "category_id"]:
                            if col in items_df.columns:
                                category_cols.append(col)
                        
                        if category_cols:
                            print(f"   📦 Найдены колонки категорий в {catalog_name}: {category_cols}")
                            
                            # Создаем маппинг item_id -> категория
                            # Нормализуем item_id к строке для надежного маппинга
                            for row in items_df.iter_rows(named=True):
                                item_id_raw = row.get("item_id")
                                if item_id_raw is not None:
                                    # Нормализуем item_id: убираем .0 если это float, приводим к строке
                                    item_id = str(item_id_raw).replace(".0", "").strip()
                                    if item_id and item_id.lower() not in ["none", "null", "nan", ""]:
                                        category = extract_category_value(row, category_cols)
                                        if category:
                                            item_to_category_map[item_id] = category
                        else:
                            print(f"   ⚠ В {catalog_name} не найдено колонок категорий. Доступные колонки: {list(items_df.columns)}")
                
                if item_to_category_map:
                    print(f"   ✅ Создан маппинг категорий для {len(item_to_category_map)} товаров")
                    # Показываем примеры категорий
                    sample_categories = list(item_to_category_map.values())[:5]
                    print(f"   📋 Примеры категорий: {sample_categories}")
                else:
                    print(f"   ⚠ Не найдено категорий в каталогах items")
            except Exception as e:
                print(f"⚠ Ошибка при создании маппинга категорий из items: {e}")
                import traceback
                print(f"   Детали: {traceback.format_exc()}")
        
        # Обогащаем события категориями из items (если категорий нет или они null)
        if item_to_category_map and "item_id" in combined_views.columns:
            try:
                # Нормализуем item_id в событиях для маппинга
                # Приводим к строке и убираем .0 если есть
                def normalize_item_id_for_mapping(item_id):
                    if item_id is None:
                        return None
                    normalized = str(item_id).replace(".0", "").strip()
                    if normalized and normalized.lower() not in ["none", "null", "nan", ""]:
                        return normalized
                    return None
                
                # Добавляем категории из items к событиям
                enriched_views = combined_views
                category_from_items = pl.Series([
                    item_to_category_map.get(normalize_item_id_for_mapping(item_id), None)
                    for item_id in combined_views["item_id"].to_list()
                ])
                
                enriched_views = enriched_views.with_columns(category_from_items.alias("category_from_items"))
                
                # Проверяем, сколько категорий удалось добавить
                matched_count = sum(1 for cat in category_from_items if cat is not None)
                print(f"   📊 Маппинг категорий: {matched_count} из {len(category_from_items)} событий получили категорию из items")
                
                # Диагностика: проверяем типы item_id
                if matched_count == 0 and combined_views.height > 0:
                    sample_item_ids_events = [str(id) for id in combined_views["item_id"].head(5).to_list()]
                    sample_item_ids_map = list(item_to_category_map.keys())[:5]
                    print(f"   🔍 Диагностика маппинга:")
                    print(f"      Примеры item_id в событиях: {sample_item_ids_events}")
                    print(f"      Примеры item_id в маппинге: {sample_item_ids_map}")
                    # Проверяем совпадения
                    normalized_events = [str(id).replace(".0", "").strip() for id in combined_views["item_id"].head(10).to_list()]
                    matches = [id for id in normalized_events if id in item_to_category_map]
                    print(f"      Совпадений: {len(matches)} из {len(normalized_events)} проверенных")
                
                # Используем category_from_items если оригинальная категория null
                category_col = "category" if "category" in enriched_views.columns else "category_id"
                if category_col in enriched_views.columns:
                    enriched_views = enriched_views.with_columns(
                        pl.when(pl.col("category_from_items").is_not_null())
                        .then(pl.col("category_from_items"))
                        .otherwise(pl.col(category_col))
                        .alias("final_category")
                    )
                else:
                    enriched_views = enriched_views.with_columns(
                        pl.col("category_from_items").alias("final_category")
                    )
                
                # Извлекаем топ категорию и список всех категорий из обогащенных данных
                valid_categories = enriched_views.filter(
                    pl.col("final_category").is_not_null() & 
                    (pl.col("final_category") != "") & 
                    (pl.col("final_category").cast(pl.Utf8) != "nan")
                )
                if valid_categories.height > 0:
                    # Список всех уникальных категорий с частотой (используем value_counts вместо mode)
                    category_counts = valid_categories["final_category"].value_counts().sort("count", descending=True)
                    all_categories = category_counts["final_category"].to_list()
                    
                    # Топ категория - самая частая (первая в отсортированном списке)
                    profile["top_category"] = all_categories[0] if all_categories else None
                    profile["all_categories"] = all_categories  # Список всех категорий
                    profile["category_counts"] = dict(zip(
                        category_counts["final_category"].to_list(),
                        category_counts["count"].to_list()
                    ))  # Словарь категория -> частота
                    
                    if profile["top_category"]:
                        top_count = profile["category_counts"].get(profile["top_category"], 0)
                        print(f"✅ Извлечена top_category: {profile['top_category']} ({top_count} раз(а)) из {valid_categories.height} событий с категориями")
                        print(f"   📋 Всего уникальных категорий: {len(all_categories)}")
                        if len(all_categories) > 1:
                            top3_str = ", ".join([f"{cat} ({profile['category_counts'].get(cat, 0)} раз)" for cat in all_categories[:3]])
                            print(f"   📊 Топ-3 категории: {top3_str}")
                    else:
                        print(f"⚠ Не удалось определить top_category из {valid_categories.height} событий")
                else:
                    profile["top_category"] = None
                    profile["all_categories"] = []
                    profile["category_counts"] = {}
                    print(f"⚠ Не найдено событий с валидными категориями после обогащения")
            except Exception as e:
                print(f"⚠ Ошибка при обогащении событий категориями: {e}")
                # Fallback на стандартное извлечение
                category_col = "category" if "category" in combined_views.columns else "category_id"
                if category_col in combined_views.columns:
                    valid_categories = combined_views.filter(
                        pl.col(category_col).is_not_null() & 
                        (pl.col(category_col) != "") & 
                        (pl.col(category_col).cast(pl.Utf8) != "nan")
                    )
                    if valid_categories.height > 0:
                        # Список всех категорий с частотой
                        category_counts = valid_categories[category_col].value_counts().sort("count", descending=True)
                        all_categories_list = category_counts[category_col].to_list()
                        
                        # Топ категория - самая частая
                        profile["top_category"] = all_categories_list[0] if all_categories_list else None
                        profile["all_categories"] = all_categories_list
                        profile["category_counts"] = dict(zip(
                            category_counts[category_col].to_list(),
                            category_counts["count"].to_list()
                        ))
                        
                        if profile["top_category"]:
                            top_count = profile["category_counts"].get(profile["top_category"], 0)
                            print(f"✅ Извлечена top_category (fallback): {profile['top_category']} ({top_count} раз(а))")
                    else:
                        profile["top_category"] = None
                        profile["all_categories"] = []
                        profile["category_counts"] = {}
                        print(f"⚠ Не найдено валидных категорий в колонке {category_col}")
                else:
                    profile["top_category"] = None
                    profile["all_categories"] = []
                    profile["category_counts"] = {}
                    print(f"⚠ Колонка {category_col} отсутствует в событиях")
        else:
            # Стандартное извлечение категорий из событий
            category_col = "category" if "category" in combined_views.columns else "category_id"
            if category_col in combined_views.columns:
                valid_categories = combined_views.filter(
                    pl.col(category_col).is_not_null() & 
                    (pl.col(category_col) != "") & 
                    (pl.col(category_col).cast(pl.Utf8) != "nan")
                )
                if valid_categories.height > 0:
                    # Список всех категорий с частотой
                    category_counts = valid_categories[category_col].value_counts().sort("count", descending=True)
                    all_categories_list = category_counts[category_col].to_list()
                    
                    # Топ категория - самая частая
                    profile["top_category"] = all_categories_list[0] if all_categories_list else None
                    profile["all_categories"] = all_categories_list
                    profile["category_counts"] = dict(zip(
                        category_counts[category_col].to_list(),
                        category_counts["count"].to_list()
                    ))
                    
                    if profile["top_category"]:
                        top_count = profile["category_counts"].get(profile["top_category"], 0)
                        print(f"✅ Извлечена top_category (стандартный метод): {profile['top_category']} ({top_count} раз(а))")
                        print(f"   📋 Всего уникальных категорий: {len(all_categories_list)}")
                else:
                    profile["top_category"] = None
                    profile["all_categories"] = []
                    profile["category_counts"] = {}
                    print(f"⚠ Не найдено валидных категорий в колонке {category_col} (стандартный метод)")
            else:
                profile["top_category"] = None
                print(f"⚠ Колонки category и category_id отсутствуют в событиях")
        
        # Если категория все еще не найдена, пробуем извлечь напрямую из items (fallback)
        if not profile.get("top_category") and item_to_category_map and combined_views.height > 0 and "item_id" in combined_views.columns:
            try:
                # Нормализуем item_id для маппинга
                def normalize_item_id(item_id):
                    if item_id is None:
                        return None
                    normalized = str(item_id).replace(".0", "").strip()
                    if normalized and normalized.lower() not in ["none", "null", "nan", ""]:
                        return normalized
                    return None
                
                user_item_ids = combined_views["item_id"].unique().to_list()
                user_categories = []
                matched_ids = []
                for item_id in user_item_ids:
                    normalized_id = normalize_item_id(item_id)
                    if normalized_id and normalized_id in item_to_category_map:
                        category = item_to_category_map[normalized_id]
                        if category:
                            user_categories.append(category)
                            matched_ids.append(normalized_id)
                
                if user_categories:
                    from collections import Counter
                    top_category_counter = Counter(user_categories)
                    most_common = top_category_counter.most_common(1)
                    if most_common:
                        profile["top_category"] = most_common[0][0]
                        print(f"✅ Обогащена top_category из items (fallback): {profile['top_category']} (из {len(matched_ids)} товаров)")
                else:
                    print(f"⚠ Не удалось найти категории для item_id пользователя. Попробовано {len(user_item_ids)} item_id, совпадений в маппинге: 0")
                    if len(user_item_ids) > 0:
                        sample_ids = [normalize_item_id(uid) for uid in user_item_ids[:5]]
                        sample_in_map = [mid for mid in sample_ids if mid and mid in item_to_category_map]
                        print(f"   Примеры item_id пользователя: {sample_ids}")
                        print(f"   Совпадений в маппинге: {len(sample_in_map)} из {len(sample_ids)}")
                        if item_to_category_map:
                            sample_map_keys = list(item_to_category_map.keys())[:5]
                            print(f"   Примеры item_id в маппинге: {sample_map_keys}")
            except Exception as e:
                print(f"⚠ Ошибка при обогащении категорий из items (fallback): {e}")
                import traceback
                print(f"   Детали: {traceback.format_exc()}")
        
        # Регион (если есть)
        if "region" in combined_views.columns:
            region = combined_views["region"].mode().to_list()
            profile["region"] = region[0] if region else None
        else:
            profile["region"] = None
        
        # Статистика по action_type
        if "action_type" in combined_views.columns:
            action_counts = combined_views["action_type"].value_counts()
            profile["action_types"] = dict(zip(action_counts["action_type"].to_list(), action_counts["count"].to_list()))
        else:
            profile["action_types"] = {}
    else:
        profile["num_views"] = 0
        profile["unique_items"] = 0
        profile["top_category"] = None
        profile["all_categories"] = []
        profile["category_counts"] = {}
        profile["region"] = None
        profile["action_types"] = {}
    
    # Статистики по retail отдельно
    if retail_df.height > 0:
        profile["num_retail_events"] = retail_df.height
        if "action_type" in retail_df.columns:
            orders = retail_df.filter(pl.col("action_type") == "order")
            profile["num_retail_orders"] = orders.height
        else:
            profile["num_retail_orders"] = 0
    else:
        profile["num_retail_events"] = 0
        profile["num_retail_orders"] = 0
    
    # Статистики по платежам (включая receipts)
    pay_df = user_events.get("payments", pl.DataFrame())
    receipts_df = user_events.get("receipts", pl.DataFrame())
    
    # Объединяем payments и receipts для полной статистики
    # Приводим к единой схеме: выбираем только общие колонки
    all_payments = []
    
    # Определяем общие колонки для объединения
    common_cols = ["user_id", "amount", "timestamp", "domain"]
    optional_cols = ["brand_id"]  # Опциональные колонки
    
    if pay_df.height > 0:
        # Выбираем только нужные колонки из pay_df
        pay_cols = [col for col in common_cols + optional_cols if col in pay_df.columns]
        if pay_cols:
            all_payments.append(pay_df.select(pay_cols))
    
    if receipts_df.height > 0:
        # Нормализуем receipts: используем price как amount
        receipts_normalized = receipts_df
        if "price" in receipts_df.columns and "amount" not in receipts_df.columns:
            receipts_normalized = receipts_df.with_columns(pl.col("price").alias("amount"))
        
        # Выбираем только нужные колонки из receipts_normalized
        receipts_cols = [col for col in common_cols + optional_cols if col in receipts_normalized.columns]
        if receipts_cols:
            all_payments.append(receipts_normalized.select(receipts_cols))
    
    if all_payments:
        # Определяем общий набор колонок
        all_cols = set()
        for df in all_payments:
            all_cols.update(df.columns)
        all_cols = list(all_cols)
        
        # Определяем целевые типы для каждой колонки (приводим к единому формату)
        # Сначала проверяем, есть ли timestamp с типом Duration
        has_duration_timestamp = False
        for df in all_payments:
            if "timestamp" in df.columns and df["timestamp"].dtype == pl.Duration:
                has_duration_timestamp = True
                break
        
        target_types = {}
        for col in all_cols:
            if col in ["user_id", "brand_id", "domain"]:
                target_types[col] = pl.Utf8  # Все ID приводим к строке
            elif col == "amount":
                target_types[col] = pl.Float64
            elif col == "timestamp":
                # Если есть Duration timestamp, не приводим к единому типу (оставляем как есть)
                # При объединении используем how="diagonal" для поддержки разных типов
                if has_duration_timestamp:
                    target_types[col] = None  # Не приводим к единому типу
                else:
                    target_types[col] = pl.Datetime
            else:
                target_types[col] = pl.Utf8  # По умолчанию строка
        
        # Приводим все DataFrame к единой схеме
        unified_payments = []
        for df in all_payments:
            cast_exprs = []
            
            # Обрабатываем все колонки (существующие и отсутствующие)
            for col in all_cols:
                if col in df.columns:
                    # Колонка существует - приводим к целевому типу
                    current_type = df[col].dtype
                    target_type = target_types[col]
                    # Специальная обработка для timestamp с Duration
                    if col == "timestamp" and target_type is None:
                        # Не приводим timestamp к единому типу - оставляем как есть
                        cast_exprs.append(pl.col(col))
                    elif current_type != target_type:
                        # Специальная обработка для timestamp
                        if col == "timestamp":
                            if current_type == pl.Duration:
                                # Duration нельзя привести к другому типу напрямую
                                # Оставляем Duration как есть - при объединении используем how="diagonal"
                                cast_exprs.append(pl.col(col))
                            elif target_type == pl.Utf8 and current_type == pl.Datetime:
                                # Datetime -> строка
                                cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
                            else:
                                # Обычное приведение для timestamp
                                cast_exprs.append(pl.col(col).cast(target_type, strict=False).alias(col))
                        elif col == "brand_id" and target_type == pl.Utf8:
                            # Специальная обработка для brand_id: убираем .0
                            cast_exprs.append(
                                pl.col(col).cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias(col)
                            )
                        else:
                            # Для остальных колонок - обычное приведение типов
                            cast_exprs.append(pl.col(col).cast(target_type, strict=False).alias(col))
                    else:
                        # Типы совпадают, но для brand_id все равно проверим нормализацию
                        if col == "brand_id" and current_type == pl.Utf8:
                             cast_exprs.append(
                                pl.col(col).str.replace(r"\.0$", "").alias(col)
                            )
                        else:
                            cast_exprs.append(pl.col(col))
                else:
                    # Колонка отсутствует - добавляем с null значением нужного типа
                    cast_exprs.append(pl.lit(None).cast(target_types[col]).alias(col))
            
            # Применяем все преобразования
            if cast_exprs:
                df = df.with_columns(cast_exprs)
            
            # Если передан item_to_brand_map и есть item_id, пробуем восстановить brand_id
            if item_to_brand_map and "item_id" in df.columns and "brand_id" in df.columns:
                try:
                    # Используем map_dict (replace) для заполнения
                    # Если brand_id null или empty или unknown, пробуем взять из item_id
                    df = df.with_columns(
                        pl.when(
                            pl.col("brand_id").is_null() | (pl.col("brand_id") == "") | (pl.col("brand_id") == "unknown")
                        ).then(
                            pl.col("item_id").cast(pl.Utf8).replace(item_to_brand_map, default=pl.col("brand_id"))
                        ).otherwise(
                            pl.col("brand_id")
                        ).alias("brand_id")
                    )
                except Exception as e:
                    print(f"⚠ Ошибка при восстановлении brand_id из item_id: {e}")

            # Выбираем колонки в правильном порядке
            unified_payments.append(df.select(all_cols))
        
        # Объединяем DataFrames
        # Если есть Duration timestamp, используем how="diagonal" для гибкого объединения
        if has_duration_timestamp:
            # Используем diagonal для объединения с разными типами
            pay_df = pl.concat(unified_payments, how="diagonal")
        else:
            pay_df = pl.concat(unified_payments)
    if pay_df.height > 0:
        profile["num_payments"] = pay_df.height
        
        if "amount" in pay_df.columns:
            # Проверяем тип данных amount
            amount_col = pay_df["amount"]
            amount_dtype = amount_col.dtype
            
            # Преобразуем в числовой тип если нужно
            if amount_dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                try:
                    pay_df = pay_df.with_columns(pl.col("amount").cast(pl.Float64, strict=False))
                except:
                    pass
            
            # Всегда используем абсолютные значения для расчетов (отрицательные = возвраты, но считаем как положительные)
            amount_abs = pay_df["amount"].abs()
            
            # Диагностика: показываем примеры исходных значений и полную статистику
            sample_values = pay_df["amount"].head(10).to_list() if pay_df.height > 0 else []
            negative_count = (pay_df["amount"] < 0).sum() if pay_df.height > 0 else 0
            
            # Получаем полную статистику для диагностики
            amount_stats = pay_df.select([
                pl.col("amount").min().alias("min"),
                pl.col("amount").max().alias("max"),
                pl.col("amount").mean().alias("mean"),
                pl.col("amount").abs().min().alias("min_abs"),
                pl.col("amount").abs().max().alias("max_abs"),
                pl.col("amount").abs().mean().alias("mean_abs"),
                pl.col("amount").abs().quantile(0.95).alias("p95"),  # 95-й перцентиль
                pl.col("amount").abs().quantile(0.99).alias("p99")   # 99-й перцентиль
            ])
            
            if amount_stats.height > 0:
                stats = amount_stats.row(0)
                min_val, max_val, mean_val, min_abs, max_abs, mean_abs_val, p95, p99 = stats
                print(f"📊 Статистика amount (до обработки): min=${min_val:.2f}, max=${max_val:.2f}, mean=${mean_val:.2f}")
                print(f"   Абсолютные значения: min=${min_abs:.2f}, max=${max_abs:.2f}, mean=${mean_abs_val:.2f}")
                if p95 is not None:
                    p99_val = p99 if p99 is not None else 0.0
                    print(f"   Перцентили: P95=${p95:.2f}, P99=${p99_val:.2f}")
                print(f"   Примеры значений: {sample_values[:5]}")
                print(f"   Всего записей: {pay_df.height}")
                
                # Предупреждение, если max кажется слишком маленьким
                if max_abs is not None and max_abs < 50:
                    print(f"⚠ ВНИМАНИЕ: Максимальная сумма (${max_abs:.2f}) кажется слишком маленькой для реальных транзакций!")
                    print(f"   Проверьте, правильно ли данные загружены и не фильтруются ли большие значения.")
            
            if negative_count > 0:
                print(f"⚠ Обнаружено {negative_count} отрицательных значений amount (возвраты). Используем абсолютные значения.")
            
            # Вычисляем статистики на абсолютных значениях
            amount_mean = amount_abs.mean()
            amount_sum = amount_abs.sum()
            amount_max = amount_abs.max()
            amount_min = amount_abs.min()
            
            # Показываем, откуда берется max - это реальное максимальное значение из данных
            if amount_max is not None:
                # Находим строку с максимальным значением для диагностики
                max_row = pay_df.filter(pl.col("amount").abs() == amount_max).head(1)
                if max_row.height > 0:
                    max_info = max_row.select(["amount", "brand_id", "timestamp"]).row(0)
                    print(f"🔍 Максимальная транзакция найдена: amount=${max_info[0]:.2f}, brand_id={max_info[1]}, timestamp={max_info[2]}")
                    print(f"   Это РЕАЛЬНОЕ значение из данных пользователя, не фиксированное!")
            
            # Сохраняем значения (гарантируем, что они не отрицательные и не NaN)
            # Проверка на NaN: value == value возвращает False для NaN
            avg_val = float(amount_mean) if amount_mean is not None and amount_mean == amount_mean else 0.0
            sum_val = float(amount_sum) if amount_sum is not None and amount_sum == amount_sum else 0.0
            max_val = float(amount_max) if amount_max is not None and amount_max == amount_max else 0.0
            min_val = float(amount_min) if amount_min is not None and amount_min == amount_min else 0.0
            
            print(f"📈 Вычисленные значения из {pay_df.height} транзакций:")
            print(f"   - max_val = ${max_val:.2f} (это максимальная сумма из всех транзакций пользователя)")
            print(f"   - min_val = ${min_val:.2f} (это минимальная сумма)")
            print(f"   - avg_val = ${avg_val:.2f} (это средняя сумма)")
            print(f"   - sum_val = ${sum_val:.2f} (это общая сумма всех транзакций)")
            
            # Финальная проверка на валидность (не должно быть отрицательных после abs())
            if avg_val < 0:
                print(f"⚠ ОШИБКА: avg_tx отрицательный ({avg_val}) после abs()! Это невозможно. Устанавливаем 0")
                avg_val = 0.0
            if sum_val < 0:
                print(f"⚠ ОШИБКА: total_tx отрицательный ({sum_val}) после abs()! Это невозможно. Устанавливаем 0")
                sum_val = 0.0
            
            profile["avg_tx"] = avg_val
            profile["total_tx"] = sum_val
            profile["max_tx"] = max_val
            profile["min_tx"] = min_val
            
            print(f"✅ Финальная статистика платежей: avg_tx={profile['avg_tx']:.2f} $, total_tx={profile['total_tx']:.2f} $, записей={pay_df.height}")
            print(f"   Проверка: avg_tx >= 0: {profile['avg_tx'] >= 0}, total_tx >= 0: {profile['total_tx'] >= 0}")
        else:
            profile["avg_tx"] = 0
            profile["total_tx"] = 0
            profile["max_tx"] = 0
            profile["min_tx"] = 0
        
        # Топ бренд (сохраняем и ID и название, если доступно)
        # Ищем brand_id во всех источниках: payments, receipts, marketplace, retail
        # Также собираем категории брендов для анализа
        
        # Объединяем все источники для поиска брендов
        # ВАЖНО: Приводим все brand_id к строковому типу для совместимости при concat
        all_brand_sources = []
        
        # Вспомогательная функция для нормализации brand_id
        def normalize_brand_column(df: pl.DataFrame) -> pl.DataFrame:
            """Приводит brand_id к строковому типу, обрабатывая все возможные варианты."""
            if df.height == 0:
                return df
            try:
                # Пробуем привести к строке через различные методы
                if "brand_id" in df.columns:
                    # Сначала пытаемся через cast
                    try:
                        return df.select([
                            pl.col("brand_id").cast(pl.Utf8, strict=False).alias("brand_id")
                        ])
                    except:
                        # Если не получается, пробуем через with_columns
                        try:
                            return df.with_columns(
                                pl.col("brand_id").cast(pl.Utf8, strict=False).alias("brand_id")
                            ).select(["brand_id"])
                        except:
                            # Последний вариант - через преобразование в Python и обратно
                            brand_series = df["brand_id"].to_list()
                            brand_strings = [str(b) if b is not None else None for b in brand_series]
                            return pl.DataFrame({"brand_id": brand_strings})
                return df
            except Exception as e:
                print(f"⚠ Ошибка при нормализации brand_id: {e}")
                return pl.DataFrame({"brand_id": pl.Series([], dtype=pl.Utf8)})
        
        if pay_df.height > 0 and "brand_id" in pay_df.columns:
            brand_df = normalize_brand_column(pay_df)
            if brand_df.height > 0:
                all_brand_sources.append(brand_df)
                print(f"   📊 Payments: {pay_df.height} транзакций, brand_id присутствует")
        
        if receipts_df.height > 0 and "brand_id" in receipts_df.columns:
            brand_df = normalize_brand_column(receipts_df)
            if brand_df.height > 0:
                all_brand_sources.append(brand_df)
                print(f"   📊 Receipts: {receipts_df.height} чеков, brand_id присутствует")
        
        if mp_df.height > 0 and "brand_id" in mp_df.columns:
            brand_df = normalize_brand_column(mp_df)
            if brand_df.height > 0:
                all_brand_sources.append(brand_df)
                print(f"   📊 Marketplace: {mp_df.height} событий, brand_id присутствует")
        
        if retail_df.height > 0 and "brand_id" in retail_df.columns:
            brand_df = normalize_brand_column(retail_df)
            if brand_df.height > 0:
                all_brand_sources.append(brand_df)
                print(f"   📊 Retail: {retail_df.height} событий, brand_id присутствует")
        
        # Объединяем все источники брендов с явным указанием схемы
        if all_brand_sources:
            # Создаем новый список с нормализованными DataFrame
            normalized_sources = []
            for i, df in enumerate(all_brand_sources):
                if df.height == 0:
                    continue
                try:
                    # Проверяем схему и нормализуем
                    schema = df.schema
                    if "brand_id" not in schema:
                        print(f"   ⚠ DataFrame {i} не содержит brand_id, пропускаем")
                        continue
                    
                    current_type = schema["brand_id"]
                    # Если тип уже Utf8, проверяем что все значения строковые
                    if current_type == pl.Utf8:
                        normalized_sources.append(df)
                    else:
                        # Приводим к Utf8
                        print(f"   ⚠ DataFrame {i}: brand_id имеет тип {current_type}, приводим к Utf8")
                        try:
                            # Сначала пробуем через cast
                            normalized_df = df.with_columns(
                                pl.col("brand_id").cast(pl.Utf8, strict=False).alias("brand_id")
                            )
                            # Проверяем результат
                            if normalized_df.schema["brand_id"] == pl.Utf8:
                                normalized_sources.append(normalized_df)
                            else:
                                # Если cast не сработал, используем Python-конвертацию
                                raise ValueError("Cast не привел к Utf8")
                        except Exception as e:
                            print(f"   ⚠ Cast не сработал для DataFrame {i}: {e}, используем Python-конвертацию")
                            # Fallback: конвертируем через Python
                            brand_values = df["brand_id"].to_list()
                            brand_strings = [str(b) if b is not None else None for b in brand_values]
                            normalized_sources.append(pl.DataFrame({"brand_id": brand_strings}))
                except Exception as e:
                    print(f"   ⚠ Ошибка при обработке DataFrame {i}: {e}, пропускаем")
                    continue
            
            if normalized_sources:
                # Используем how="diagonal" для автоматического приведения типов
                try:
                    combined_brands = pl.concat(normalized_sources, how="diagonal")
                except Exception as e1:
                    print(f"   ⚠ Ошибка при concat с diagonal: {e1}, пробуем обычный concat")
                    try:
                        combined_brands = pl.concat(normalized_sources)
                    except Exception as e2:
                        print(f"   ⚠ Ошибка при обычном concat: {e2}, создаем пустой DataFrame")
                        combined_brands = pl.DataFrame({"brand_id": pl.Series([], dtype=pl.Utf8)})
            else:
                # Нет валидных источников
                combined_brands = pl.DataFrame({"brand_id": pl.Series([], dtype=pl.Utf8)})
            
            # Фильтруем невалидные бренды
            valid_brands = combined_brands.filter(
                pl.col("brand_id").is_not_null() & 
                (pl.col("brand_id") != "unknown") & 
                (pl.col("brand_id") != "") &
                (pl.col("brand_id").cast(pl.Utf8) != "nan")
            )
            
            if valid_brands.height > 0:
                # Приводим brand_id к строке и нормализуем
                valid_brands_normalized = valid_brands.with_columns(
                    pl.col("brand_id").cast(pl.Utf8).str.replace(r"\.0$", "").alias("brand_id_normalized")
                )
                
                top_brand_list = valid_brands_normalized["brand_id_normalized"].mode().to_list()
                if top_brand_list:
                    profile["top_brand"] = top_brand_list[0]
                    profile["top_brand_id"] = top_brand_list[0]
                    # Выводим ID бренда (используем название из brands_map если доступно через импорт, иначе ID)
                    # Выводим ID бренда
                    print(f"✅ Определен топ бренд: Brand {profile['top_brand']} (ID: {profile['top_brand']}) (из {valid_brands.height} валидных записей)")
                else:
                    profile["top_brand"] = None
                    profile["top_brand_id"] = None
                    print(f"⚠ Не удалось определить топ бренд (mode() вернул пустой список)")
            else:
                profile["top_brand"] = None
                profile["top_brand_id"] = None
                print(f"⚠ Не удалось определить топ бренд (нет валидных brand_id в {combined_brands.height} записях)")
                # Показываем примеры для отладки
                if combined_brands.height > 0:
                    sample_brands = combined_brands["brand_id"].head(10).to_list()
                    print(f"   Примеры brand_id в данных: {sample_brands[:5]}")
                    
                    # Считаем сколько None vs других значений
                    null_count = combined_brands.filter(pl.col("brand_id").is_null()).height
                    non_null_count = combined_brands.filter(pl.col("brand_id").is_not_null()).height
                    print(f"   Статистика: None значений = {null_count}, не-None = {non_null_count}")
                    
                    # Проверяем, есть ли другие идентификаторы в исходных данных
                    if pay_df.height > 0:
                        print(f"   🔍 Диагностика: проверяем доступные колонки в payments...")
                        pay_cols = pay_df.columns
                        print(f"   Колонки в payments: {pay_cols}")
                        # Показываем примеры данных из payments для диагностики
                        if pay_df.height > 0:
                            sample_pay = pay_df.head(3)
                            print(f"   Примеры строк payments: {sample_pay}")
                    if receipts_df.height > 0:
                        print(f"   🔍 Диагностика: проверяем доступные колонки в receipts...")
                        receipt_cols = receipts_df.columns
                        print(f"   Колонки в receipts: {receipt_cols}")
                        if receipts_df.height > 0:
                            sample_receipts = receipts_df.head(3)
                            print(f"   Примеры строк receipts: {sample_receipts}")
            
            # Собираем все уникальные бренды пользователя (даже если топ бренд не найден)
            unique_brands = combined_brands["brand_id"].drop_nulls().unique().to_list()
            profile["brand_ids"] = [str(b) for b in unique_brands if b and str(b) != "unknown" and str(b) != "nan" and str(b).lower() != "none"]
            
            # Если не нашли топ бренд, но есть brand_ids, используем первый
            if not profile.get("top_brand") and profile.get("brand_ids"):
                profile["top_brand"] = profile["brand_ids"][0]
                profile["top_brand_id"] = profile["brand_ids"][0]
                # Выводим ID бренда
                print(f"   ℹ Использован первый доступный brand_id: Brand {profile['top_brand']} (ID: {profile['top_brand']})")
            elif not profile.get("top_brand") and not profile.get("brand_ids"):
                # Гарантируем, что brand_ids существует как пустой список
                profile["brand_ids"] = []
                print(f"   ⚠ Невозможно определить топ бренд: все brand_id в данных равны None или пустые")
        elif "brand_id" in pay_df.columns:
            # Fallback: проверяем только payments (старая логика)
            valid_brands = pay_df.filter(
                pl.col("brand_id").is_not_null() & 
                (pl.col("brand_id") != "unknown") & 
                (pl.col("brand_id") != "")
            )
            
            if valid_brands.height > 0:
                top_brand = valid_brands["brand_id"].mode().to_list()
                profile["top_brand"] = top_brand[0] if top_brand else None
                profile["top_brand_id"] = top_brand[0] if top_brand else None
                # Выводим ID бренда
                print(f"✅ Определен топ бренд (fallback): Brand {profile['top_brand']} (ID: {profile['top_brand']})")
            else:
                profile["top_brand"] = None
                profile["top_brand_id"] = None
                print(f"⚠ Не удалось определить топ бренд (нет валидных данных в payments)")
            
            # Собираем все уникальные бренды пользователя
            unique_brands = pay_df["brand_id"].unique().to_list()
            profile["brand_ids"] = [str(b) for b in unique_brands if b and str(b) != "unknown"]
        else:
            # Нет brand_id ни в одном источнике
            profile["top_brand"] = None
            profile["top_brand_id"] = None
            if "brand_ids" not in profile:
                profile["brand_ids"] = []
            print(f"⚠ Колонка brand_id отсутствует во всех источниках данных")
        
        # Обогащаем категориями брендов из маппинга (для всех случаев, когда есть brand_ids)
        # ВАЖНО: Этот блок должен быть вне блока if pay_df.height > 0, чтобы работать всегда
        if brands_categories_map and profile.get("brand_ids"):
            brand_categories = []
            for brand_id in profile["brand_ids"]:
                # Нормализуем brand_id для поиска (удаляем .0)
                brand_id_str = str(brand_id)
                if brand_id_str.endswith(".0"):
                    brand_id_str = brand_id_str[:-2]
                
                # Пробуем найти категорию для бренда
                # Пробуем несколько вариантов ключа
                cat = brands_categories_map.get(brand_id_str)
                if not cat:
                    # Пробуем с .0
                    cat = brands_categories_map.get(f"{brand_id_str}.0")
                if not cat:
                    # Пробуем без нормализации
                    cat = brands_categories_map.get(str(brand_id))
                
                if cat:
                    brand_categories.append(cat)
            
            if brand_categories:
                from collections import Counter
                profile["brand_categories"] = brand_categories
                profile["top_brand_category"] = Counter(brand_categories).most_common(1)[0][0]
                print(f"✅ Обогащено {len(brand_categories)} категорий брендов из маппинга")
                print(f"   ✅ top_brand_category: {profile['top_brand_category']}")
                
                # Fallback: если top_category не найдена, используем top_brand_category
                if not profile.get("top_category") and profile["top_brand_category"]:
                    profile["top_category"] = profile["top_brand_category"]
                    print(f"   ℹ Использована top_brand_category как top_category: {profile['top_category']}")
            else:
                # Если brands_categories_map не содержит категорий, но есть brand_ids, пытаемся извлечь из items
                if profile.get("brand_ids") and items_with_embeddings:
                    print(f"   🔍 Попытка извлечения категорий для {len(profile['brand_ids'])} брендов из items каталогов...")
                    brand_categories_from_items = []
                    for brand_id in profile["brand_ids"]:
                        brand_id_str = str(brand_id)
                        if brand_id_str.endswith(".0"):
                            brand_id_str = brand_id_str[:-2]
                        
                        # Ищем категорию в items каталогах по brand_id
                        for catalog_name, items_df in items_with_embeddings.items():
                            if items_df.height == 0:
                                continue
                            
                            # Проверяем наличие brand_id и category в каталоге
                            if "brand_id" not in items_df.columns or "category" not in items_df.columns:
                                continue
                            
                            # Фильтруем по brand_id
                            brand_items = items_df.filter(
                                pl.col("brand_id").cast(pl.Utf8) == brand_id_str
                            )
                            
                            if brand_items.height > 0:
                                # Берем наиболее частую категорию для этого бренда
                                valid_cats = brand_items.filter(
                                    pl.col("category").is_not_null() & 
                                    (pl.col("category") != "") & 
                                    (pl.col("category").cast(pl.Utf8) != "nan")
                                )
                                
                                if valid_cats.height > 0:
                                    top_cat = valid_cats["category"].mode().to_list()
                                    if top_cat:
                                        brand_categories_from_items.append(top_cat[0])
                                        print(f"      ✅ Найдена категория '{top_cat[0]}' для бренда {brand_id_str} из {catalog_name}")
                                        break  # Нашли, переходим к следующему бренду
                    
                    if brand_categories_from_items:
                        from collections import Counter
                        profile["brand_categories"] = brand_categories_from_items
                        profile["top_brand_category"] = Counter(brand_categories_from_items).most_common(1)[0][0]
                        print(f"   ✅ Извлечено {len(brand_categories_from_items)} категорий брендов из items каталогов")
                        print(f"   ✅ top_brand_category: {profile['top_brand_category']}")
                        
                        # Fallback: если top_category не найдена, используем top_brand_category
                        if not profile.get("top_category"):
                            profile["top_category"] = profile["top_brand_category"]
                            print(f"   ℹ Использована top_brand_category (из items) как top_category: {profile['top_category']}")
                    
                    # Если категории все еще не найдены, используем эвристики
                    if not profile.get("top_category") and not profile.get("top_brand_category"):
                        profile["top_category"] = _determine_category_by_heuristics(profile)
                        if profile["top_category"]:
                            print(f"   ℹ Определена категория по эвристикам: {profile['top_category']}")
                else:
                    print(f"   Нет brand_ids для поиска категорий")
                
                if brands_categories_map:
                    print(f"   Доступные ключи в brands_categories_map: {list(brands_categories_map.keys())[:10]}...")
        else:
            if "brand_categories" not in profile:
                profile["brand_categories"] = []
            profile["top_brand_category"] = None
    else:
        profile["num_payments"] = 0
        profile["avg_tx"] = 0
        profile["total_tx"] = 0
        profile["max_tx"] = 0
        profile["min_tx"] = 0
        profile["top_brand"] = None
    
    # Финальный fallback: если top_category не найдена, используем top_brand_category
    if not profile.get("top_category") and profile.get("top_brand_category"):
        profile["top_category"] = profile["top_brand_category"]
        print(f"   ℹ Финальный fallback: использована top_brand_category как top_category: {profile['top_category']}")
    
    # Финальный fallback 2: если категория все еще не найдена, используем эвристики
    if not profile.get("top_category"):
        category_from_heuristics = _determine_category_by_heuristics(profile)
        if category_from_heuristics:
            profile["top_category"] = category_from_heuristics
            print(f"   ℹ Финальный fallback (эвристики): определена категория '{profile['top_category']}'")
    
    # Временные характеристики
    # Объединяем события для вычисления временных характеристик
    # Выбираем только timestamp, так как у разных доменов разные схемы
    # (marketplace имеет item_id, payments имеет brand_id, но нет item_id)
    normalized_events = []
    for df in user_events.values():
        if df.height > 0 and "timestamp" in df.columns:
            # Выбираем только timestamp для объединения
            # Это гарантирует одинаковую схему для всех доменов
            df_normalized = df.select(["timestamp"])
            normalized_events.append(df_normalized)
    
    if normalized_events:
        try:
            combined = pl.concat(normalized_events)
            timestamps = combined["timestamp"].to_list()
        except Exception as e:
            print(f"⚠ Ошибка при объединении событий для временных характеристик: {e}")
            # Собираем timestamps из каждого DataFrame отдельно
            timestamps = []
            for df in user_events.values():
                if df.height > 0 and "timestamp" in df.columns:
                    timestamps.extend(df["timestamp"].to_list())
        
        if timestamps:
            # Конвертируем в datetime если нужно
            try:
                from datetime import datetime
                dt_timestamps = [
                    t if isinstance(t, datetime) else datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                    for t in timestamps
                ]
                profile["days_active"] = (max(dt_timestamps) - min(dt_timestamps)).days + 1
                profile["events_per_day"] = len(timestamps) / max(profile["days_active"], 1)
            except:
                profile["days_active"] = 1
                profile["events_per_day"] = len(timestamps)
        else:
            profile["days_active"] = 0
            profile["events_per_day"] = 0
    else:
        profile["days_active"] = 0
        profile["events_per_day"] = 0
    
    # Паттерны
    if patterns:
        profile["num_patterns"] = len(patterns)
        
        # Кодируем паттерны как бинарные фичи
        common_patterns = [
            ("V", "P", "V"),  # просмотр → оплата → просмотр
            ("V", "V", "P"),  # два просмотра → оплата
            ("P", "V", "C"),  # оплата → просмотр → клик
            ("V", "P", "P"),  # просмотр → оплата → оплата
        ]
        
        pattern_strings = ["→".join(p) for p in patterns] if patterns else []
        
        for pattern in common_patterns:
            pattern_str = "→".join(pattern)
            profile[f"has_pattern_{pattern_str.replace('→', '_')}"] = 1 if pattern_str in pattern_strings else 0
        
        # Основной паттерн как строка
        if patterns:
            profile["pattern"] = "→".join(patterns[0]) if isinstance(patterns[0], tuple) else str(patterns[0])
        else:
            profile["pattern"] = "unknown"
    else:
        profile["num_patterns"] = 0
        profile["pattern"] = "unknown"
        for pattern in [("V", "P", "V"), ("V", "V", "P"), ("P", "V", "C"), ("V", "P", "P")]:
            pattern_str = "→".join(pattern)
            profile[f"has_pattern_{pattern_str.replace('→', '_')}"] = 0
    
    # Использование embedding для улучшения профиля (опционально)
    # Embedding - это векторное представление товара, которое кодирует его семантические свойства
    # Можно использовать для:
    # 1. Поиска похожих товаров (cosine similarity)
    # 2. Кластеризации интересов пользователя
    # 3. Улучшения рекомендаций через collaborative filtering
    if items_with_embeddings:
        try:
            # Собираем embedding всех товаров пользователя
            user_item_ids = set()
            if user_events.get("marketplace", pl.DataFrame()).height > 0:
                mp_df = user_events["marketplace"]
                if "item_id" in mp_df.columns:
                    user_item_ids.update(mp_df["item_id"].unique().to_list())
            if user_events.get("retail", pl.DataFrame()).height > 0:
                retail_df = user_events["retail"]
                if "item_id" in retail_df.columns:
                    user_item_ids.update(retail_df["item_id"].unique().to_list())
            
            if user_item_ids:
                # Объединяем embedding из всех каталогов
                all_embeddings = []
                for catalog_name, items_df in items_with_embeddings.items():
                    if items_df.height > 0 and "item_id" in items_df.columns and "embedding" in items_df.columns:
                        # Фильтруем только товары пользователя
                        user_items = items_df.filter(pl.col("item_id").is_in(list(user_item_ids)))
                        if user_items.height > 0:
                            # Извлекаем embedding
                            for row in user_items.iter_rows(named=True):
                                emb = row.get("embedding")
                                if emb is not None:
                                    # Embedding может быть списком или numpy массивом
                                    if isinstance(emb, list):
                                        all_embeddings.append(np.array(emb))
                                    elif isinstance(emb, np.ndarray):
                                        all_embeddings.append(emb)
                
                if all_embeddings:
                    # Вычисляем средний embedding (представление интересов пользователя)
                    avg_embedding = np.mean(all_embeddings, axis=0)
                    profile["avg_item_embedding"] = avg_embedding.tolist()  # Сохраняем как список для JSON
                    profile["embedding_dim"] = len(avg_embedding)
                    
                    # Вычисляем дисперсию embedding (разнообразие интересов)
                    if len(all_embeddings) > 1:
                        embedding_variance = np.var(all_embeddings, axis=0).mean()
                        profile["embedding_diversity"] = float(embedding_variance)
                    else:
                        profile["embedding_diversity"] = 0.0
                    
                    print(f"✅ Использованы embedding для {len(all_embeddings)} товаров (размерность: {len(avg_embedding)})")
        except Exception as e:
            print(f"⚠ Ошибка при обработке embedding: {e}")
            profile["embedding_dim"] = 0
            profile["embedding_diversity"] = 0.0
    else:
        profile["embedding_dim"] = 0
        profile["embedding_diversity"] = 0.0
    
    return profile


def _determine_category_by_heuristics(profile: Dict) -> Optional[str]:
    """
    Определяет категорию пользователя по эвристикам, если категория не найдена в данных.
    
    Использует паттерны поведения, суммы транзакций, количество покупок для определения
    наиболее вероятной категории интересов пользователя.
    
    :param profile: Профиль пользователя
    :return: Название категории или None
    """
    # Эвристики на основе статистики пользователя
    avg_tx = profile.get('avg_tx', 0)
    total_tx = profile.get('total_tx', 0)
    num_payments = profile.get('num_payments', 0)
    num_views = profile.get('num_views', 0)
    max_tx = profile.get('max_tx', 0)
    
    # Эвристика 1: Высокие суммы транзакций -> вероятно electronics или real_estate
    if max_tx > 1000 or total_tx > 5000:
        if num_payments < 5:
            return "electronics"  # Крупные, редкие покупки
        else:
            return "real_estate"  # Много крупных покупок
    
    # Эвристика 2: Средние суммы, много покупок -> retail, clothing, food
    if num_payments > 10 and avg_tx < 100:
        if num_views > num_payments * 2:
            return "clothing"  # Много просмотров перед покупкой
        else:
            return "food"  # Частые покупки
    
    # Эвристика 3: Низкие суммы, часто -> food, pharmacy
    if avg_tx < 50 and num_payments > 5:
        return "pharmacy" if num_payments > 20 else "food"
    
    # Эвристика 4: Средние суммы, мало покупок -> entertainment, books
    if 50 <= avg_tx <= 200 and num_payments <= 5:
        return "entertainment"
    
    # Эвристика 5: Много просмотров, мало покупок -> research behavior -> electronics
    if num_views > 20 and num_payments <= 3:
        return "electronics"
    
    # Эвристика 6: По умолчанию - retail для активных пользователей
    if num_payments > 0 or num_views > 0:
        return "retail"
    
    # Если нет активности, возвращаем None
    return None


def profile_to_features(profile: Dict) -> List[float]:
    """
    Преобразует профиль в вектор признаков для модели.
    
    :param profile: Профиль пользователя
    :return: Список числовых признаков
    """
    features = []
    
    # Числовые признаки
    numeric_features = [
        "num_views", "num_payments", "avg_tx", "total_tx",
        "days_active", "events_per_day", "unique_items",
        "num_patterns"
    ]
    
    for feat in numeric_features:
        features.append(float(profile.get(feat, 0)))
    
    # Бинарные признаки паттернов
    pattern_features = [
        "has_pattern_V_P_V",
        "has_pattern_V_V_P",
        "has_pattern_P_V_C",
        "has_pattern_V_P_P"
    ]
    
    for feat in pattern_features:
        features.append(float(profile.get(feat, 0)))
    
    # Категориальные (one-hot encoding через индексы)
    if profile.get("top_category"):
        features.append(float(profile["top_category"]))
    else:
        features.append(0.0)
    
    if profile.get("region"):
        features.append(float(profile["region"]))
    else:
        features.append(0.0)
    
    return features


