"""
Основной модуль для работы системы рекомендаций.

Интегрирует все компоненты: загрузку данных, построение графов,
извлечение паттернов, создание профилей, рекомендации и объяснения.
"""

from typing import Dict, List, Optional
import polars as pl

from src.data.cloud_loader import init_loader, get_loader
from src.features.graph_builder import build_behavior_graph, get_graph_statistics
from src.features.pattern_miner import extract_patterns, pattern_to_string
from src.features.user_profile import create_user_profile
from src.features.graph_analyzer import analyze_graph_with_yandexgpt, generate_rules_from_graph
from src.modeling.nbo_model import recommend as ml_recommend
from src.modeling.rule_engine import RuleEngine
from src.app.explainer import explain_recommendation
from typing import Dict as TypingDict


def process_user(
    user_id: str,
    use_cloud: bool = True,
    use_yandexgpt_for_analysis: bool = True,
    top_k: int = 3
) -> Dict:
    """
    Обрабатывает пользователя и возвращает рекомендации.
    
    :param user_id: ID пользователя
    :param use_cloud: Использовать данные из облака
    :param use_yandexgpt_for_analysis: Использовать YandexGPT для анализа графа
    :param top_k: Количество рекомендаций
    :return: Словарь с рекомендациями и анализом
    """
    # Загрузка данных
    brands_map: Dict[str, str] = {}  # Маппинг brand_id -> brand_name
    brands_categories_map: Dict[str, str] = {}  # Маппинг brand_id -> category
    item_to_brand_map: Dict[str, str] = {}  # Маппинг item_id -> brand_id
    
    if use_cloud:
        loader = get_loader()
        if loader is None:
            loader = init_loader(
                public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
            )
        
        # Загружаем справочник брендов для сопоставления brand_id с названиями и категориями
        print(f"📚 Загрузка справочника брендов...")
        
        try:
            brands_df = loader.load_brands()
            if brands_df.height > 0:
                print(f"✅ Загружено {brands_df.height} брендов")
                print(f"   Колонки в brands.pq: {brands_df.columns}")
                # Безопасный вывод примера данных
                try:
                    sample_row = brands_df.head(1).to_dicts()[0]
                    # Убираем embedding из вывода, так как он огромный
                    if "embedding" in sample_row:
                        sample_row["embedding"] = "[VECTOR]"
                    print(f"   Пример данных (1 строка): {sample_row}")
                except:
                    print("   Не удалось вывести пример данных")
                
                # Определяем колонки для маппинга
                brand_id_col = None
                brand_name_col = None
                brand_category_col = None
                
                # 1. Ищем ID
                for col in brands_df.columns:
                    if col.lower() in ["brand_id", "brandid", "id", "merchant_id"]:
                        brand_id_col = col
                        break
                
                # 2. Ищем Название
                for col in brands_df.columns:
                    if col.lower() in ["name", "brand_name", "title", "brand_title", "brand", "slug", "caption", "merchant_name"]:
                        brand_name_col = col
                        break
                
                # Если название не найдено, ищем любую строковую колонку (кроме ID и Category)
                if not brand_name_col:
                    schema = brands_df.schema
                    for col_name, dtype in schema.items():
                        if col_name == brand_id_col: continue
                        if dtype == pl.Utf8 and col_name.lower() not in ["category", "embedding", "description"]:
                            print(f"   ℹ Используем колонку '{col_name}' как название бренда (эвристика)")
                            brand_name_col = col_name
                            break
                
                # 3. Ищем Категорию
                for col in brands_df.columns:
                    if col.lower() in ["category", "category_id", "categoryid", "cat_id", "cat", 
                                       "merchant_category", "merchant_category_id", "mcc", "mcc_code", "industry"]:
                        brand_category_col = col
                        break
                
                print(f"   Найдены колонки: ID='{brand_id_col}', Name='{brand_name_col}', Category='{brand_category_col}'")
                
                if brand_id_col:
                    # Создаем маппинг brand_id -> brand_name
                    # Если нет колонки с именем, используем ID как имя
                    use_id_as_name = False
                    if not brand_name_col:
                        print("   ⚠ Колонка с названием бренда не найдена. Будем использовать ID как название.")
                        use_id_as_name = True
                        brand_name_col = brand_id_col # Placeholder
                    
                    for row in brands_df.iter_rows(named=True):
                        # Нормализуем ID: удаляем .0 и приводим к строке
                        brand_id_raw = str(row.get(brand_id_col, ""))
                        if brand_id_raw.endswith(".0"):
                            brand_id_raw = brand_id_raw[:-2]
                        brand_id = brand_id_raw
                        
                        if use_id_as_name:
                            brand_name = f"Brand {brand_id}"
                        else:
                            brand_name = str(row.get(brand_name_col, ""))
                        
                        if brand_id and brand_name:
                            brands_map[brand_id] = brand_name
                            
                    print(f"✅ Создан маппинг названий для {len(brands_map)} брендов")
                    
                    # Создаем маппинг brand_id -> category из brands.pq (если есть)
                    if brand_category_col:
                        for row in brands_df.iter_rows(named=True):
                            # Нормализуем ID
                            brand_id_raw = str(row.get(brand_id_col, ""))
                            if brand_id_raw.endswith(".0"):
                                brand_id_raw = brand_id_raw[:-2]
                            brand_id = brand_id_raw
                            
                            category = str(row.get(brand_category_col, ""))
                            if brand_id and category and category.lower() not in ["none", "null", "nan", ""]:
                                brands_categories_map[brand_id] = category
                        print(f"✅ Создан маппинг категорий из brands.pq для {len(brands_categories_map)} брендов")
            else:
                print(f"⚠ Справочник брендов пуст или не найден")
        except Exception as e:
            print(f"⚠ Ошибка при загрузке справочника брендов: {e}")
        
        # Если категории не найдены в brands.pq, извлекаем их из items.pq
        # ОПТИМИЗАЦИЯ: загружаем каталоги только если нужно, и только нужные колонки
        # Согласно T-ECD документации, категории товаров находятся в items.pq
        # Извлекаем категории брендов из каталогов товаров (items.pq)
        # ВАЖНО: Всегда пытаемся извлечь, даже если маппинг уже заполнен из brands.pq
        # Это позволяет дополнить маппинг категориями из items
        print(f"📦 Извлечение категорий брендов из каталогов товаров (items.pq)...")
        if len(brands_categories_map) > 0:
            print(f"   Текущий размер маппинга: {len(brands_categories_map)} брендов (будет дополнен)")
        print(f"   ⚡ Используем оптимизацию: только нужные колонки, без embedding (экономия ~30 ГБ)")
        try:
            # Сначала собираем brand_id из событий пользователя (если уже загружены)
            # Это позволит применить predicate pushdown
            user_brand_ids = set()
            # Пока не загружены события, пропускаем predicate pushdown
            # Но все равно используем projection pushdown (только нужные колонки)
            
            # Загружаем каталоги товаров с оптимизацией (только нужные колонки, без embedding)
            # Используем LazyFrame для отложенной загрузки
            marketplace_items_lazy = loader.load_marketplace_items(
                brand_ids=None,  # Пока не знаем brand_id пользователя
                use_lazy=True
            )
            retail_items_lazy = loader.load_retail_items(
                brand_ids=None,
                use_lazy=True
            )
            
            # Объединяем LazyFrames
            # ВАЖНО: Даже если marketplace items.pq поврежден, используем retail items
            all_items_lazy = []
            if marketplace_items_lazy is not None:
                try:
                    schema = marketplace_items_lazy.collect_schema()
                    if len(schema) > 0:
                        all_items_lazy.append(marketplace_items_lazy)
                        print(f"   ✅ Marketplace items LazyFrame добавлен (схема: {len(schema)} колонок)")
                except Exception as e:
                    print(f"   ⚠ Marketplace items LazyFrame не удалось добавить: {e}")
            
            if retail_items_lazy is not None:
                try:
                    schema = retail_items_lazy.collect_schema()
                    if len(schema) > 0:
                        all_items_lazy.append(retail_items_lazy)
                        print(f"   ✅ Retail items LazyFrame добавлен (схема: {len(schema)} колонок)")
                except Exception as e:
                    print(f"   ⚠ Retail items LazyFrame не удалось добавить: {e}")
            
            if not all_items_lazy:
                print(f"   ⚠ Нет доступных items LazyFrames для извлечения категорий")
            
            if all_items_lazy:
                # Объединяем LazyFrames (еще не загружены в память!)
                # ВАЖНО: Если только один источник, используем его напрямую (без concat)
                if len(all_items_lazy) == 1:
                    combined_lazy = all_items_lazy[0]
                else:
                    # Пробуем объединить с diagonal для автоматического приведения типов
                    try:
                        combined_lazy = pl.concat(all_items_lazy, how="diagonal")
                    except Exception as e1:
                        print(f"   ⚠ Ошибка при concat с diagonal: {e1}, пробуем обычный concat")
                        try:
                            # Перед обычным concat нормализуем типы brand_id в каждом LazyFrame
                            normalized_lazy = []
                            for lazy_frame in all_items_lazy:
                                try:
                                    schema = lazy_frame.collect_schema()
                                    if "brand_id" in schema:
                                        # Приводим brand_id к строке
                                        normalized_frame = lazy_frame.with_columns(
                                            pl.col("brand_id").cast(pl.Utf8, strict=False).alias("brand_id")
                                        )
                                        normalized_lazy.append(normalized_frame)
                                    else:
                                        normalized_lazy.append(lazy_frame)
                                except:
                                    normalized_lazy.append(lazy_frame)
                            combined_lazy = pl.concat(normalized_lazy)
                        except Exception as e2:
                            print(f"   ⚠ Ошибка при обычном concat после нормализации: {e2}")
                            # Если и это не работает, используем только retail items
                            retail_only = [lf for lf in all_items_lazy if "retail" in str(lf) or any("retail" in str(lf) for _ in [1])]
                            if retail_only:
                                combined_lazy = retail_only[0]
                                print(f"   ⚠ Используем только retail items из-за проблем с объединением")
                            else:
                                combined_lazy = all_items_lazy[0]
                                print(f"   ⚠ Используем первый доступный источник")
                
                # Проверяем наличие нужных колонок
                try:
                    schema = combined_lazy.collect_schema()
                    has_brand_id = "brand_id" in schema
                    has_category = any(col.lower() in ["category_id", "category", "categoryid"] for col in schema)
                except Exception as e:
                    print(f"⚠ Ошибка при получении схемы combined_lazy: {e}")
                    has_brand_id = False
                    has_category = False
                
                if has_brand_id and has_category:
                    # Определяем колонку категории
                    category_col = None
                    for col in schema:
                        if col.lower() in ["category_id", "category", "categoryid", "cat_id", "cat"]:
                            category_col = col
                            break
                    
                    if category_col:
                        # Проверяем наличие brand_id перед группировкой
                        if "brand_id" not in schema:
                            print(f"⚠ brand_id не найден в items.pq. Используем item_id для группировки.")
                            # Если нет brand_id, группируем по item_id (но это не даст категории брендов)
                            # В этом случае пропускаем извлечение категорий брендов
                            print(f"⚠ Невозможно извлечь категории брендов без brand_id. Пропускаем.")
                        else:
                            # ОПТИМИЗАЦИЯ: Загружаем категории только для первых N брендов как кэш
                            # Основная загрузка будет после загрузки событий пользователя
                            # Это ускоряет начальную загрузку, но все равно дает базовый набор категорий
                            print(f"   ⚡ Ограниченная загрузка категорий (первые 1000 брендов для кэша)...")
                            try:
                                # Ограничиваем количество брендов для быстрой загрузки
                                brand_categories_lazy = combined_lazy.group_by("brand_id").agg([
                                    pl.col(category_col).mode().alias("top_category"),
                                    pl.count().alias("item_count")
                                ]).filter(
                                    pl.col("top_category").is_not_null()
                                ).head(1000)  # Ограничиваем первыми 1000 брендами
                                
                                brand_categories = brand_categories_lazy.collect()
                                
                                initial_count = len(brands_categories_map)
                                
                                # Создаем маппинг brand_id -> category
                                for row in brand_categories.iter_rows(named=True):
                                    brand_id = str(row.get("brand_id", ""))
                                    # Нормализуем ID (удаляем .0)
                                    if brand_id.endswith(".0"):
                                        brand_id = brand_id[:-2]
                                        
                                    top_categories = row.get("top_category", [])
                                    if brand_id and top_categories and len(top_categories) > 0:
                                        # Берем первую (самую частую) категорию
                                        category = str(top_categories[0])
                                        if category and category.lower() not in ["none", "null", "nan", ""]:
                                            brands_categories_map[brand_id] = category
                                
                                added_count = len(brands_categories_map) - initial_count
                                print(f"✅ Загружено {added_count} категорий брендов в кэш (всего: {len(brands_categories_map)})")
                                print(f"   (Примеры ID: {list(brands_categories_map.keys())[:5]})")
                                print(f"   ℹ Остальные категории будут загружены для конкретных брендов пользователя")
                            except Exception as e:
                                print(f"⚠ Ошибка при ограниченной загрузке категорий: {e}")
                                print(f"   ℹ Продолжаем - категории будут загружены для брендов пользователя")
                    else:
                        print(f"⚠ Не найдена колонка категории в items.pq. Колонки: {list(schema.keys())}")
                else:
                    print(f"⚠ В items.pq отсутствуют нужные колонки. brand_id: {has_brand_id}, category: {has_category}")
            else:
                print(f"⚠ Каталоги товаров (items.pq) не найдены или пусты")
        except Exception as e:
            print(f"⚠ Ошибка при извлечении категорий из items.pq: {e}")
            import traceback
            print(f"   Детали: {traceback.format_exc()}")
        
        if len(brands_categories_map) == 0:
            print(f"⚠ Категории брендов не найдены ни в brands.pq, ни в items.pq")
        
        # Для публичных папок без API токена нужно указывать конкретные файлы
        # Ограничиваем количество файлов для быстрой загрузки
        num_files = 3  # Уменьшено с 10 до 3 для быстрой загрузки
        start_file = 1082  # Начальный номер файла
        
        marketplace_files = [
            f"{i:05d}.pq" for i in range(start_file, start_file + num_files)
        ]
        payments_files = [
            f"{i:05d}.pq" for i in range(start_file, start_file + num_files)
        ]
        
        # Оптимизированная загрузка: сначала фильтруем по user_id, затем загружаем только нужные данные
        print(f"Загрузка данных для пользователя {user_id}...")
        
        try:
            print(f"📊 Фильтрация marketplace events для пользователя {user_id}...")
            # Оптимизация: используем projection pushdown - выбираем только нужные колонки до фильтрации
            marketplace_lazy = loader.load_marketplace_events(file_list=marketplace_files, days=5)
            # Фильтруем по user_id на уровне LazyFrame (эффективно)
            # Используем collect_schema() чтобы избежать PerformanceWarning
            if marketplace_lazy is not None:
                schema = marketplace_lazy.collect_schema()
                if "user_id" in schema:
                    print(f"🔍 Фильтруем по user_id {user_id}...")
                    # Оптимизация: сначала фильтруем, потом выбираем колонки (projection pushdown)
                    # Сначала проверяем, какие колонки доступны в схеме
                    schema = marketplace_lazy.collect_schema()
                    available_cols = list(schema.keys())
                    
                    # Собираем список колонок для select (только те, что есть в данных)
                    select_cols = ["user_id", "item_id", "timestamp", "domain"]
                    
                    # Добавляем опциональные колонки только если они есть
                    if "category_id" in available_cols:
                        select_cols.append(pl.col("category_id").alias("category_id"))
                    if "brand_id" in available_cols:
                        select_cols.append(pl.col("brand_id").alias("brand_id"))
                    if "action_type" in available_cols:
                        select_cols.append(pl.col("action_type").alias("action_type"))
                    if "subdomain" in available_cols:
                        select_cols.append(pl.col("subdomain").alias("subdomain"))
                    if "price" in available_cols:
                        select_cols.append(pl.col("price").alias("price"))
                    if "count" in available_cols:
                        select_cols.append(pl.col("count").alias("count"))
                    if "os" in available_cols:
                        select_cols.append(pl.col("os").alias("os"))
                    
                    user_marketplace_lazy = (
                        marketplace_lazy
                        .filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                        # Выбираем только нужные колонки для ускорения (только те, что есть)
                        .select(select_cols)
                    )
                    
                    # Проверяем тип timestamp перед сортировкой
                    timestamp_dtype = schema.get("timestamp")
                    if timestamp_dtype == pl.Duration:
                        # Если timestamp в формате Duration, пропускаем сортировку
                        # Просто берем первые 100 строк
                        print("⚠ Timestamp в формате Duration, пропускаем сортировку")
                        user_marketplace = user_marketplace_lazy.limit(100).collect()
                    else:
                        # Ограничиваем количество событий для экономии памяти и токенов
                        # Берем только последние 100 событий и агрегируем
                        print("📅 Сортировка по timestamp...")
                        user_marketplace = user_marketplace_lazy.sort("timestamp", descending=True).limit(100).collect()
                    
                    print(f"✅ Найдено {user_marketplace.height} событий marketplace для пользователя {user_id}")
                    
                    # Агрегируем данные: топ категории, топ товары
                    if user_marketplace.height > 0:
                        # Группируем по категориям и товарам для упрощения
                        user_marketplace = user_marketplace.select([
                            "timestamp", "item_id", "category_id", "domain"
                        ]).head(50)  # Ограничиваем до 50 самых свежих событий
                else:
                    user_marketplace = pl.DataFrame()
            else:
                user_marketplace = pl.DataFrame()
        except Exception as e:
            import traceback
            print(f"❌ Ошибка при загрузке marketplace events: {e}")
            print(f"Трассировка: {traceback.format_exc()}")
            user_marketplace = pl.DataFrame()
        
        try:
            print(f"💳 Фильтрация payments events для пользователя {user_id}...")
            # ОПТИМИЗАЦИЯ: передаем user_id для predicate pushdown (фильтрация ДО загрузки)
            payments_lazy = loader.load_payments_events(file_list=payments_files, days=5, user_id=user_id)
            if payments_lazy is not None:
                schema = payments_lazy.collect_schema()
                if "user_id" in schema:
                    # Если user_id уже был передан в load_payments_events, фильтрация уже применена
                    # Но на всякий случай проверяем и применяем еще раз (если не был передан)
                    user_payments_lazy = payments_lazy
                    # Проверяем, применен ли уже фильтр (если нет - применяем)
                    # Это безопасно, т.к. если фильтр уже применен, он просто не найдет лишних строк
                    print(f"🔍 Применяем фильтр по user_id {user_id}...")
                    user_payments_lazy = user_payments_lazy.filter(
                        pl.col("user_id").cast(pl.Utf8) == str(user_id)
                    )
                    
                    # Проверяем тип timestamp перед сортировкой
                    timestamp_dtype = schema.get("timestamp")
                    if timestamp_dtype == pl.Duration:
                        # Если timestamp в формате Duration, пропускаем сортировку
                        print("⚠ Timestamp в формате Duration, пропускаем сортировку")
                        user_payments = user_payments_lazy.limit(50).collect()
                    else:
                        # Ограничиваем и агрегируем платежи
                        print("📅 Сортировка по timestamp...")
                        user_payments = user_payments_lazy.sort("timestamp", descending=True).limit(50).collect()
                    
                    # Применяем нормализацию после collect() только если данные не были нормализованы
                    # (для LazyFrame оптимизации данные могут быть не полностью нормализованы)
                    if user_payments.height > 0 and "domain" not in user_payments.columns:
                        print("📋 Применяем нормализацию данных...")
                        from src.data.data_parser import normalize_payments_events
                        user_payments = normalize_payments_events(user_payments, file_path="payments/events")
                    
                    print(f"✅ Найдено {user_payments.height} платежей для пользователя {user_id}")
                    
                    if user_payments.height > 0:
                        # Агрегируем: сумма по брендам
                        user_payments = user_payments.select([
                            "timestamp", "brand_id", "amount", "domain"
                        ]).head(30)  # Ограничиваем до 30 самых свежих платежей
                else:
                    user_payments = pl.DataFrame()
            else:
                user_payments = pl.DataFrame()
        except Exception as e:
            import traceback
            print(f"❌ Ошибка при загрузке payments events: {e}")
            print(f"Трассировка: {traceback.format_exc()}")
            user_payments = pl.DataFrame()
        
        # Загрузка retail events
        user_retail = pl.DataFrame()
        try:
            print(f"🛒 Фильтрация retail events для пользователя {user_id}...")
            retail_lazy = loader.load_retail_events(file_list=marketplace_files, limit=3)
            if retail_lazy is not None:
                schema = retail_lazy.collect_schema()
                if "user_id" in schema:
                    user_retail_lazy = retail_lazy.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                    timestamp_dtype = schema.get("timestamp")
                    if timestamp_dtype == pl.Duration:
                        user_retail = user_retail_lazy.limit(100).collect()
                    else:
                        user_retail = user_retail_lazy.sort("timestamp", descending=True).limit(100).collect()
                    print(f"✅ Найдено {user_retail.height} событий retail для пользователя {user_id}")
        except Exception as e:
            print(f"⚠ Ошибка при загрузке retail events: {e}")
            user_retail = pl.DataFrame()
        
        # Загрузка payments receipts (чеки с детализацией товаров)
        user_receipts = pl.DataFrame()
        try:
            print(f"🧾 Фильтрация payments receipts для пользователя {user_id}...")
            receipts_lazy = loader.load_payments_receipts(file_list=payments_files, days=5, user_id=user_id)
            if receipts_lazy is not None:
                schema = receipts_lazy.collect_schema()
                if "user_id" in schema:
                    user_receipts_lazy = receipts_lazy.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                    timestamp_dtype = schema.get("timestamp")
                    if timestamp_dtype == pl.Duration:
                        user_receipts = user_receipts_lazy.limit(50).collect()
                    else:
                        user_receipts = user_receipts_lazy.sort("timestamp", descending=True).limit(50).collect()
                    print(f"✅ Найдено {user_receipts.height} чеков для пользователя {user_id}")
        except Exception as e:
            print(f"⚠ Ошибка при загрузке payments receipts: {e}")
            user_receipts = pl.DataFrame()
        
        # Загрузка каталогов товаров для обогащения данных категориями
        # ОПТИМИЗАЦИЯ: загружаем только нужные колонки и только для нужных товаров
        items_catalog = {}
        try:
            print(f"📦 Загрузка каталогов товаров для обогащения данных...")
            print(f"   ⚡ Используем оптимизацию: только нужные колонки (item_id, brand_id, category), без embedding")
            
            # Собираем item_id из событий пользователя для фильтрации
            user_item_ids = set()
            if user_marketplace.height > 0 and "item_id" in user_marketplace.columns:
                user_item_ids.update(user_marketplace["item_id"].unique().to_list())
            if user_retail.height > 0 and "item_id" in user_retail.columns:
                user_item_ids.update(user_retail["item_id"].unique().to_list())
            if user_receipts.height > 0 and "approximate_item_id" in user_receipts.columns:
                user_item_ids.update(user_receipts["approximate_item_id"].unique().to_list())
            
            # Собираем brand_id для дополнительной фильтрации
            user_brand_ids = set()
            if user_payments.height > 0 and "brand_id" in user_payments.columns:
                user_brand_ids.update(user_payments["brand_id"].unique().to_list())
            if user_marketplace.height > 0 and "brand_id" in user_marketplace.columns:
                user_brand_ids.update(user_marketplace["brand_id"].unique().to_list())
            if user_retail.height > 0 and "brand_id" in user_retail.columns:
                user_brand_ids.update(user_retail["brand_id"].unique().to_list())
            
            brand_ids_list = [str(bid) for bid in user_brand_ids] if user_brand_ids else None
            item_ids_list = [str(iid) for iid in user_item_ids] if user_item_ids else None
            
            print(f"   📊 Фильтрация: {len(user_item_ids)} уникальных item_id, {len(user_brand_ids)} уникальных brand_id")
            
            # Загружаем каталоги с оптимизацией (только нужные колонки, фильтрация)
            # Embedding НЕ загружаем по умолчанию (экономия ~30 ГБ)
            # Если нужен embedding, можно загрузить отдельно только для товаров пользователя
            marketplace_items_lazy = loader.load_marketplace_items(
                brand_ids=brand_ids_list,
                item_ids=item_ids_list,
                use_lazy=True,
                include_embedding=False  # Embedding не нужен для обогащения категориями
            )
            retail_items_lazy = loader.load_retail_items(
                brand_ids=brand_ids_list,
                item_ids=item_ids_list,
                use_lazy=True,
                include_embedding=False  # Embedding не нужен для обогащения категориями
            )
            
            # Дополнительная фильтрация по item_id (predicate pushdown)
            if item_ids_list and marketplace_items_lazy is not None:
                try:
                    schema = marketplace_items_lazy.collect_schema()
                    if "item_id" in schema:
                        marketplace_items_lazy = marketplace_items_lazy.filter(
                            pl.col("item_id").cast(pl.Utf8).is_in(item_ids_list)
                        )
                        print(f"   ⚡ Дополнительная фильтрация marketplace по {len(item_ids_list)} item_id")
                except Exception as e:
                    print(f"   ⚠ Ошибка фильтрации marketplace по item_id: {e}")
            
            if item_ids_list and retail_items_lazy is not None:
                try:
                    schema = retail_items_lazy.collect_schema()
                    if "item_id" in schema:
                        retail_items_lazy = retail_items_lazy.filter(
                            pl.col("item_id").cast(pl.Utf8).is_in(item_ids_list)
                        )
                        print(f"   ⚡ Дополнительная фильтрация retail по {len(item_ids_list)} item_id")
                except Exception as e:
                    print(f"   ⚠ Ошибка фильтрации retail по item_id: {e}")
            
            # Загружаем в память только отфильтрованные данные
            if marketplace_items_lazy is not None:
                try:
                    schema = marketplace_items_lazy.collect_schema()
                    if len(schema) > 0:
                        marketplace_items = marketplace_items_lazy.collect()
                        if marketplace_items.height > 0:
                            items_catalog["marketplace"] = marketplace_items
                            print(f"✅ Загружено {marketplace_items.height} товаров из marketplace/items.pq (после фильтрации)")
                            print(f"   💾 Экономия: загружено только {marketplace_items.height} товаров вместо миллионов")
                except Exception as e:
                    print(f"⚠ Ошибка при загрузке marketplace items: {e}")
            
            if retail_items_lazy is not None:
                try:
                    schema = retail_items_lazy.collect_schema()
                    if len(schema) > 0:
                        retail_items = retail_items_lazy.collect()
                        if retail_items.height > 0:
                            items_catalog["retail"] = retail_items
                            print(f"✅ Загружено {retail_items.height} товаров из retail/items.pq (после фильтрации)")
                            print(f"   💾 Экономия: загружено только {retail_items.height} товаров вместо миллионов")
                except Exception as e:
                    print(f"⚠ Ошибка при загрузке retail items: {e}")
                    
        except Exception as e:
            print(f"⚠ Ошибка при загрузке каталогов товаров: {e}")
            import traceback
            print(f"   Детали: {traceback.format_exc()}")
        
        # Обогащаем события категориями из каталогов
        # Определяем тип товаров по префиксу item_id для выбора правильного каталога
        if items_catalog and user_marketplace.height > 0 and "item_id" in user_marketplace.columns:
            try:
                # Определяем префиксы item_id для выбора правильного каталога
                item_ids_list = user_marketplace["item_id"].unique().to_list()
                
                # Пробуем обогатить из обоих каталогов (retail и marketplace)
                # Сначала пробуем retail_items для товаров с префиксом nfmcg_
                retail_enriched = False
                if "retail" in items_catalog:
                    retail_items = items_catalog.get("retail")
                    if retail_items is not None and retail_items.height > 0 and "item_id" in retail_items.columns:
                        category_col = "category" if "category" in retail_items.columns else "category_id"
                        if category_col in retail_items.columns:
                            # Объединяем с retail каталогом
                            user_marketplace = user_marketplace.join(
                                retail_items.select(["item_id", category_col, "subcategory"] if "subcategory" in retail_items.columns else ["item_id", category_col]),
                                on="item_id",
                                how="left"
                            )
                            enriched_count = user_marketplace.filter(pl.col(category_col).is_not_null()).height
                            if enriched_count > 0:
                                print(f"✅ Обогащено {enriched_count} событий marketplace категориями из retail_items")
                                retail_enriched = True
                
                # Затем пробуем marketplace_items для товаров, которые не обогатились
                if "marketplace" in items_catalog:
                    mp_items = items_catalog.get("marketplace")
                    if mp_items is not None and mp_items.height > 0 and "item_id" in mp_items.columns:
                        category_col = "category" if "category" in mp_items.columns else "category_id"
                        if category_col in mp_items.columns:
                            # Объединяем только те события, которые еще не имеют категорий
                            current_category_col = "category" if "category" in user_marketplace.columns else ("category_id" if "category_id" in user_marketplace.columns else None)
                            
                            if current_category_col is None or user_marketplace.filter(pl.col(current_category_col).is_not_null()).height < user_marketplace.height:
                                # Если категорий нет или не все события обогащены, пробуем marketplace
                                user_marketplace = user_marketplace.join(
                                    mp_items.select(["item_id", category_col, "subcategory"] if "subcategory" in mp_items.columns else ["item_id", category_col]),
                                    on="item_id",
                                    how="left",
                                    suffix="_mp"
                                )
                                
                                # Объединяем категории (используем первую не-null)
                                if f"{category_col}_mp" in user_marketplace.columns:
                                    if current_category_col:
                                        user_marketplace = user_marketplace.with_columns(
                                            pl.coalesce([pl.col(current_category_col), pl.col(f"{category_col}_mp")]).alias(category_col)
                                        ).drop(f"{category_col}_mp")
                                    else:
                                        user_marketplace = user_marketplace.rename({f"{category_col}_mp": category_col})
                                
                                final_category_col = category_col if current_category_col is None else current_category_col
                                enriched_count = user_marketplace.filter(pl.col(final_category_col).is_not_null()).height
                                if enriched_count > 0 and not retail_enriched:
                                    print(f"✅ Обогащено {enriched_count} событий marketplace категориями из marketplace_items")
            except Exception as e:
                print(f"⚠ Ошибка при обогащении marketplace категориями: {e}")
                import traceback
                print(f"   Детали: {traceback.format_exc()}")
        
        if items_catalog and user_retail.height > 0:
            try:
                retail_items_cat = items_catalog.get("retail")
                if retail_items_cat is not None and "item_id" in retail_items_cat.columns and "category" in retail_items_cat.columns:
                    user_retail = user_retail.join(
                        retail_items_cat.select(["item_id", "category", "subcategory"]),
                        on="item_id",
                        how="left"
                    )
                    print(f"✅ Обогащено {user_retail.filter(pl.col('category').is_not_null()).height} событий retail категориями")
            except Exception as e:
                print(f"⚠ Ошибка при обогащении retail категориями: {e}")
        
        # Обогащаем receipts категориями (используем approximate_item_id)
        if items_catalog and user_receipts.height > 0:
            try:
                # Пробуем обогатить из обоих каталогов
                for catalog_name, catalog_df in items_catalog.items():
                    if "item_id" in catalog_df.columns and "category" in catalog_df.columns:
                        # Переименовываем approximate_item_id в item_id для join
                        user_receipts = user_receipts.join(
                            catalog_df.select(["item_id", "category", "subcategory"]),
                            left_on="approximate_item_id",
                            right_on="item_id",
                            how="left"
                        )
                print(f"✅ Обогащено {user_receipts.filter(pl.col('category').is_not_null()).height} чеков категориями")
            except Exception as e:
                print(f"⚠ Ошибка при обогащении receipts категориями: {e}")
        
        # После обогащения событий категориями, извлекаем категории брендов из items_catalog
        # для brand_id пользователя (это важно - теперь мы знаем brand_id пользователя!)
        # Сначала собираем brand_id пользователя
        user_brand_ids_set = set()
        
        if user_payments.height > 0 and "brand_id" in user_payments.columns:
            brand_ids = user_payments["brand_id"].drop_nulls().unique().to_list()
            user_brand_ids_set.update([str(bid) for bid in brand_ids if bid])
        
        if user_marketplace.height > 0 and "brand_id" in user_marketplace.columns:
            brand_ids = user_marketplace["brand_id"].drop_nulls().unique().to_list()
            user_brand_ids_set.update([str(bid) for bid in brand_ids if bid])
        
        if user_retail.height > 0 and "brand_id" in user_retail.columns:
            brand_ids = user_retail["brand_id"].drop_nulls().unique().to_list()
            user_brand_ids_set.update([str(bid) for bid in brand_ids if bid])
        
        # Нормализуем brand_id (удаляем .0)
        user_brand_ids_normalized = []
        for bid in user_brand_ids_set:
            if bid and bid != "unknown":
                # Удаляем .0 в конце если есть
                if bid.endswith(".0"):
                    bid = bid[:-2]
                user_brand_ids_normalized.append(bid)
        
        # Если есть бренды пользователя, загружаем дополнительно товары для этих брендов
        # (даже если их нет в событиях пользователя - нужны для извлечения категорий)
        if user_brand_ids_normalized:
            print(f"🔍 Загрузка товаров для {len(user_brand_ids_normalized)} брендов пользователя для извлечения категорий...")
            
            try:
                # Загружаем товары для брендов пользователя (без фильтрации по item_id)
                print(f"   🔍 Попытка загрузить товары для брендов: {user_brand_ids_normalized[:5]}...")
                brand_items_marketplace_lazy = loader.load_marketplace_items(
                    brand_ids=user_brand_ids_normalized,
                    item_ids=None,  # Без фильтрации по item_id - нужны все товары бренда
                    use_lazy=True,
                    include_embedding=False
                )
                brand_items_retail_lazy = loader.load_retail_items(
                    brand_ids=user_brand_ids_normalized,
                    item_ids=None,
                    use_lazy=True,
                    include_embedding=False
                )
                
                # Проверяем, что загрузка прошла успешно
                if brand_items_marketplace_lazy is None:
                    print(f"   ⚠ Marketplace items lazy frame = None (возможно, файл не найден)")
                else:
                    try:
                        schema = brand_items_marketplace_lazy.collect_schema()
                        print(f"   ✅ Marketplace items schema: {list(schema.keys())}")
                    except:
                        print(f"   ⚠ Не удалось получить schema для marketplace items")
                
                if brand_items_retail_lazy is None:
                    print(f"   ⚠ Retail items lazy frame = None (возможно, файл не найден)")
                else:
                    try:
                        schema = brand_items_retail_lazy.collect_schema()
                        print(f"   ✅ Retail items schema: {list(schema.keys())}")
                    except:
                        print(f"   ⚠ Не удалось получить schema для retail items")
                
                # Добавляем в items_catalog или обновляем существующие
                if brand_items_marketplace_lazy is not None:
                    try:
                        brand_marketplace_items = brand_items_marketplace_lazy.limit(1000).collect()  # Ограничиваем для производительности
                        if brand_marketplace_items.height > 0:
                            # Проверяем наличие категорий в загруженных товарах
                            has_category_col = any(col.lower() in ["category", "category_id"] for col in brand_marketplace_items.columns)
                            if has_category_col:
                                category_col = [col for col in brand_marketplace_items.columns if col.lower() in ["category", "category_id"]][0]
                                non_null_categories = brand_marketplace_items.filter(pl.col(category_col).is_not_null()).height
                                print(f"   📊 Marketplace: {brand_marketplace_items.height} товаров, {non_null_categories} с категориями")
                            
                            if "marketplace" in items_catalog:
                                # Объединяем с существующими
                                items_catalog["marketplace"] = pl.concat([items_catalog["marketplace"], brand_marketplace_items]).unique(subset=["item_id"], keep="first")
                                print(f"   ✅ Обновлен marketplace каталог: добавлено товаров для брендов пользователя")
                            else:
                                items_catalog["marketplace"] = brand_marketplace_items
                                print(f"   ✅ Загружен marketplace каталог: {brand_marketplace_items.height} товаров для брендов")
                        else:
                            print(f"   ⚠ Marketplace: не найдено товаров для брендов {user_brand_ids_normalized[:3]}...")
                    except Exception as e:
                        print(f"   ⚠ Ошибка при загрузке marketplace товаров для брендов: {e}")
                        import traceback
                        print(f"   Детали: {traceback.format_exc()}")
                
                if brand_items_retail_lazy is not None:
                    try:
                        brand_retail_items = brand_items_retail_lazy.limit(1000).collect()
                        if brand_retail_items.height > 0:
                            # Проверяем наличие категорий в загруженных товарах
                            has_category_col = any(col.lower() in ["category", "category_id"] for col in brand_retail_items.columns)
                            if has_category_col:
                                category_col = [col for col in brand_retail_items.columns if col.lower() in ["category", "category_id"]][0]
                                non_null_categories = brand_retail_items.filter(pl.col(category_col).is_not_null()).height
                                print(f"   📊 Retail: {brand_retail_items.height} товаров, {non_null_categories} с категориями")
                            
                            # Проверяем, какие brand_id есть в загруженных товарах
                            if "brand_id" in brand_retail_items.columns:
                                # Нормализуем brand_id для сравнения
                                brand_retail_items_normalized = brand_retail_items.with_columns(
                                    pl.col("brand_id").cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias("brand_id_normalized")
                                )
                                unique_brands_in_items = brand_retail_items_normalized["brand_id_normalized"].drop_nulls().unique().to_list()
                                print(f"   📊 Retail: уникальные brand_id в товарах: {unique_brands_in_items[:10]}")
                                print(f"   📊 Retail: бренды пользователя: {user_brand_ids_normalized[:10]}")
                                matching = [b for b in user_brand_ids_normalized if str(b) in [str(ub) for ub in unique_brands_in_items]]
                                print(f"   📊 Retail: найдено совпадений: {len(matching)} из {len(user_brand_ids_normalized)}")
                            
                            if "retail" in items_catalog:
                                # Объединяем с существующими
                                items_catalog["retail"] = pl.concat([items_catalog["retail"], brand_retail_items]).unique(subset=["item_id"], keep="first")
                                print(f"   ✅ Обновлен retail каталог: добавлено товаров для брендов пользователя")
                            else:
                                items_catalog["retail"] = brand_retail_items
                                print(f"   ✅ Загружен retail каталог: {brand_retail_items.height} товаров для брендов")
                        else:
                            print(f"   ⚠ Retail: не найдено товаров для брендов {user_brand_ids_normalized[:3]}...")
                            # Пробуем выяснить почему - проверяем есть ли вообще товары в retail
                            try:
                                all_retail_lazy = loader.load_retail_items(brand_ids=None, item_ids=None, use_lazy=True, include_embedding=False)
                                if all_retail_lazy:
                                    sample_retail = all_retail_lazy.limit(10).collect()
                                    if sample_retail.height > 0 and "brand_id" in sample_retail.columns:
                                        sample_brands = sample_retail["brand_id"].unique().to_list()
                                        print(f"      Примеры brand_id в retail (всего): {sample_brands[:10]}")
                            except:
                                pass
                    except Exception as e:
                        print(f"   ⚠ Ошибка при загрузке retail товаров для брендов: {e}")
                        import traceback
                        print(f"   Детали: {traceback.format_exc()}")
                        
            except Exception as e:
                print(f"   ⚠ Ошибка при дополнительной загрузке товаров для брендов: {e}")
                import traceback
                print(f"   Детали: {traceback.format_exc()}")
        
        # Теперь извлекаем категории брендов из обновленного items_catalog
        # Обогащаем маппинг даже если он уже заполнен из brands.pq
        # ВАЖНО: Это дополнительное извлечение категорий для конкретных брендов пользователя
        if items_catalog and user_brand_ids_normalized:
            try:
                print(f"🔍 Извлечение категорий брендов для {len(user_brand_ids_normalized)} брендов пользователя из items_catalog...")
                print(f"   Brand IDs пользователя: {user_brand_ids_normalized[:5]}...")
                
                # Пробуем извлечь категории из обоих каталогов (retail и marketplace)
                for catalog_name, catalog_df in items_catalog.items():
                    if catalog_df.height > 0 and "brand_id" in catalog_df.columns:
                        # Определяем колонку категории
                        category_col = None
                        for col in catalog_df.columns:
                            if col.lower() in ["category", "category_id"]:
                                category_col = col
                                break
                        
                        if category_col:
                            print(f"   📦 Проверка каталога {catalog_name}: {catalog_df.height} товаров, колонка категории: {category_col}")
                            
                            # Нормализуем brand_id в каталоге для сравнения
                            # Простой подход: приводим к строке и удаляем .0
                            catalog_df_normalized = catalog_df.with_columns(
                                pl.col("brand_id").cast(pl.Utf8, strict=False).str.replace(r"\.0$", "").alias("brand_id_normalized")
                            )
                            
                            # Нормализуем user_brand_ids аналогично
                            user_brand_ids_for_filter = [str(b).replace(".0", "") if b else None for b in user_brand_ids_normalized]
                            user_brand_ids_for_filter = [b for b in user_brand_ids_for_filter if b and b != "nan" and b != "null" and b != ""]
                            
                            # Проверяем наличие брендов пользователя в каталоге
                            unique_brands_in_catalog = catalog_df_normalized["brand_id_normalized"].drop_nulls().unique().to_list()
                            unique_brands_str = [str(b).replace(".0", "") if b else None for b in unique_brands_in_catalog]
                            unique_brands_clean = [b for b in unique_brands_str if b and b != "nan" and b != "null" and b != ""]
                            
                            matching_brands = [b for b in user_brand_ids_for_filter if b in unique_brands_clean]
                            print(f"      Найдено совпадений брендов: {len(matching_brands)} из {len(user_brand_ids_for_filter)}")
                            if matching_brands:
                                print(f"      Совпадающие бренды: {matching_brands[:5]}...")
                            else:
                                print(f"      Бренды пользователя: {user_brand_ids_for_filter[:5]}...")
                                print(f"      Бренды в каталоге (примеры): {unique_brands_clean[:10]}...")
                            
                            # Фильтруем по brand_id пользователя
                            # Используем оригинальные user_brand_ids_normalized, но также пробуем все варианты
                            catalog_filtered = catalog_df_normalized.filter(
                                pl.col("brand_id_normalized").is_in(user_brand_ids_for_filter)
                            )
                            
                            # Если ничего не нашли, пробуем без нормализации (на случай если они уже строки)
                            if catalog_filtered.height == 0:
                                print(f"      ⚠ Фильтрация по нормализованным ID не дала результатов, пробуем оригинальные значения...")
                                catalog_filtered = catalog_df.filter(
                                    pl.col("brand_id").cast(pl.Utf8, strict=False).is_in([str(b) for b in user_brand_ids_normalized if b])
                                )
                            
                            if catalog_filtered.height > 0:
                                print(f"   📦 Найдено {catalog_filtered.height} товаров в {catalog_name} для брендов пользователя")
                                
                                # Проверяем, сколько товаров с непустыми категориями
                                items_with_categories = catalog_filtered.filter(
                                    pl.col(category_col).is_not_null() & 
                                    (pl.col(category_col).cast(pl.Utf8) != "") &
                                    (pl.col(category_col).cast(pl.Utf8) != "null") &
                                    (pl.col(category_col).cast(pl.Utf8) != "nan")
                                )
                                print(f"      Товаров с категориями: {items_with_categories.height} из {catalog_filtered.height}")
                                
                                if items_with_categories.height > 0:
                                    # Группируем по brand_id и находим самую частую категорию
                                    brand_categories = items_with_categories.group_by("brand_id_normalized").agg([
                                        pl.col(category_col).mode().alias("top_category"),
                                        pl.count().alias("item_count")
                                    ]).filter(
                                        pl.col("top_category").is_not_null()
                                    )
                                    
                                    # Добавляем в маппинг
                                    catalog_found_count = 0
                                    for row in brand_categories.iter_rows(named=True):
                                        brand_id = str(row.get("brand_id_normalized", ""))
                                        # Дополнительная нормализация
                                        if brand_id.endswith(".0"):
                                            brand_id = brand_id[:-2]
                                        
                                        top_categories = row.get("top_category", [])
                                        if brand_id and top_categories and len(top_categories) > 0:
                                            category = str(top_categories[0])
                                            if category and category.lower() not in ["none", "null", "nan", ""]:
                                                # Обновляем маппинг (перезаписываем если уже есть)
                                                brands_categories_map[brand_id] = category
                                                catalog_found_count += 1
                                                print(f"      ✅ Бренд {brand_id}: категория '{category}' (найдено {row.get('item_count', 0)} товаров)")
                                    
                                    print(f"   ✅ Извлечено категорий из {catalog_name}: {catalog_found_count} брендов")
                                else:
                                    # Пробуем найти любые категории, даже если они пустые
                                    sample_categories = catalog_filtered.select([category_col, "brand_id_normalized"]).head(10)
                                    print(f"      ⚠ Проблема: все категории пустые или null для этого каталога")
                                    print(f"      Примеры данных: {sample_categories}")
                            else:
                                print(f"   ⚠ В каталоге {catalog_name} не найдено товаров для брендов пользователя")
                                # Диагностика: проверяем общее количество товаров в каталоге
                                if catalog_df.height > 0:
                                    total_brands_in_catalog = catalog_df["brand_id"].n_unique() if "brand_id" in catalog_df.columns else 0
                                    print(f"      Всего товаров в каталоге: {catalog_df.height}, уникальных брендов: {total_brands_in_catalog}")
                                    if "brand_id" in catalog_df.columns:
                                        sample_brands = catalog_df["brand_id"].drop_nulls().unique().head(10).to_list()
                                        print(f"      Примеры brand_id в каталоге: {sample_brands}")
                
                if brands_categories_map:
                    # Подсчитываем сколько из брендов пользователя нашли
                    found_for_user = len([b for b in user_brand_ids_normalized if b in brands_categories_map])
                    print(f"✅ Всего категорий брендов для пользователя: {found_for_user} из {len(user_brand_ids_normalized)}")
                    if found_for_user > 0:
                        print(f"   Примеры: {list(brands_categories_map.items())[:3]}")
                    if found_for_user < len(user_brand_ids_normalized):
                        missing = [b for b in user_brand_ids_normalized if b not in brands_categories_map]
                        print(f"   ⚠ Не найдено категорий для брендов: {missing}")
                        
                        # ПОСЛЕДНЯЯ ПОПЫТКА: Загружаем категории напрямую из items.pq для недостающих брендов
                        if missing:
                            print(f"   🔍 Последняя попытка: загрузка категорий напрямую из items.pq для {len(missing)} брендов...")
                            try:
                                # Загружаем товары напрямую из items.pq для недостающих брендов
                                for missing_brand_id in missing[:10]:  # Ограничиваем до 10 для производительности
                                    try:
                                        # Пробуем загрузить из marketplace
                                        brand_items_mp = loader.load_marketplace_items(
                                            brand_ids=[str(missing_brand_id)],
                                            item_ids=None,
                                            use_lazy=False,
                                            include_embedding=False
                                        )
                                        if brand_items_mp is not None and brand_items_mp.height > 0:
                                            # Ищем колонку категории
                                            category_col_mp = None
                                            for col in brand_items_mp.columns:
                                                if col.lower() in ["category", "category_id"]:
                                                    category_col_mp = col
                                                    break
                                            
                                            if category_col_mp:
                                                # Находим самую частую категорию для этого бренда
                                                brand_cat = brand_items_mp.filter(
                                                    pl.col(category_col_mp).is_not_null()
                                                ).group_by(category_col_mp).agg([
                                                    pl.count().alias("count")
                                                ]).sort("count", descending=True).head(1)
                                                
                                                if brand_cat.height > 0:
                                                    category = str(brand_cat[category_col_mp][0])
                                                    if category and category.lower() not in ["none", "null", "nan", ""]:
                                                        brands_categories_map[str(missing_brand_id)] = category
                                                        print(f"      ✅ Бренд {missing_brand_id}: категория '{category}' (из marketplace)")
                                                        continue
                                        
                                        # Если не нашли в marketplace, пробуем retail
                                        brand_items_rt = loader.load_retail_items(
                                            brand_ids=[str(missing_brand_id)],
                                            item_ids=None,
                                            use_lazy=False,
                                            include_embedding=False
                                        )
                                        if brand_items_rt is not None and brand_items_rt.height > 0:
                                            category_col_rt = None
                                            for col in brand_items_rt.columns:
                                                if col.lower() in ["category", "category_id"]:
                                                    category_col_rt = col
                                                    break
                                            
                                            if category_col_rt:
                                                brand_cat = brand_items_rt.filter(
                                                    pl.col(category_col_rt).is_not_null()
                                                ).group_by(category_col_rt).agg([
                                                    pl.count().alias("count")
                                                ]).sort("count", descending=True).head(1)
                                                
                                                if brand_cat.height > 0:
                                                    category = str(brand_cat[category_col_rt][0])
                                                    if category and category.lower() not in ["none", "null", "nan", ""]:
                                                        brands_categories_map[str(missing_brand_id)] = category
                                                        print(f"      ✅ Бренд {missing_brand_id}: категория '{category}' (из retail)")
                                    except Exception as e:
                                        print(f"      ⚠ Ошибка при загрузке категории для бренда {missing_brand_id}: {e}")
                                        continue
                                
                                # Финальная проверка
                                final_found = len([b for b in user_brand_ids_normalized if b in brands_categories_map])
                                if final_found > found_for_user:
                                    print(f"   ✅ После прямой загрузки: найдено категорий для {final_found} из {len(user_brand_ids_normalized)} брендов")
                            except Exception as e:
                                print(f"   ⚠ Ошибка при прямой загрузке категорий: {e}")
                else:
                    print(f"⚠ Не найдено категорий ни для одного из {len(user_brand_ids_normalized)} брендов пользователя")
            except Exception as e:
                print(f"⚠ Ошибка при извлечении категорий брендов из items_catalog: {e}")
                import traceback
                print(f"   Детали: {traceback.format_exc()}")
        
        user_events = {
            "marketplace": user_marketplace,
            "payments": user_payments,
            "retail": user_retail,
            "receipts": user_receipts
        }
        
        # Проверяем, что есть хотя бы какие-то данные
        total_events = (user_marketplace.height + user_payments.height + 
                       user_events.get("retail", pl.DataFrame()).height + 
                       user_events.get("receipts", pl.DataFrame()).height)
        if total_events == 0:
            print(f"⚠ Предупреждение: для пользователя {user_id} не найдено событий в загруженных файлах")
        else:
            print(f"✅ Всего найдено {total_events} событий для пользователя {user_id}")
            print(f"   - Marketplace: {user_marketplace.height}")
            print(f"   - Payments: {user_payments.height}")
            print(f"   - Retail: {user_events.get('retail', pl.DataFrame()).height}")
            print(f"   - Receipts: {user_events.get('receipts', pl.DataFrame()).height}")
    else:
        # Локальная загрузка (если реализована)
        from src.data.loader import load_user_events
        user_events = load_user_events(data_root="data/", user_id=user_id, days=2)
    
    # Построение графа
    print(f"🕸️ Построение графа поведения для пользователя {user_id}...")
    graph = build_behavior_graph(
        mp_df=user_events["marketplace"],
        pay_df=user_events["payments"],
        retail_df=user_events.get("retail", pl.DataFrame()),
        receipts_df=user_events.get("receipts", pl.DataFrame()),
        user_id=user_id,
        time_window_hours=24
    )
    print(f"✅ Граф построен: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} рёбер")
    
    graph_stats = get_graph_statistics(graph)
    
    # Анализ графа через YandexGPT (опционально)
    graph_analysis = None
    if use_yandexgpt_for_analysis and graph.number_of_nodes() > 0:
        try:
            graph_analysis = analyze_graph_with_yandexgpt(graph, user_id, brands_map=brands_map)
        except Exception as e:
            print(f"Ошибка анализа графа через YandexGPT: {e}")
    
    # Извлечение паттернов
    print(f"🔍 Извлечение паттернов поведения...")
    patterns = extract_patterns(user_events, min_pattern_len=3, min_support=2)
    pattern_strings = [pattern_to_string(p) for p in patterns]
    print(f"✅ Найдено {len(patterns)} паттернов")
    
    # Генерация правил из графа через YandexGPT
    graph_rules = []
    if use_yandexgpt_for_analysis and patterns:
        try:
            print(f"🤖 Генерация правил из графа через YandexGPT...")
            graph_rules = generate_rules_from_graph(graph, user_id)
            print(f"✅ Сгенерировано {len(graph_rules)} правил")
        except Exception as e:
            print(f"❌ Ошибка генерации правил из графа: {e}")
    
    # Опциональная загрузка embedding для улучшения профиля
    # Embedding загружаем ТОЛЬКО для товаров пользователя (экономия памяти)
    items_with_embeddings = None
    if use_cloud:
        # Собираем item_id для опциональной загрузки embedding
        user_item_ids = set()
        if user_events.get("marketplace", pl.DataFrame()).height > 0:
            mp_df = user_events["marketplace"]
            if "item_id" in mp_df.columns:
                user_item_ids.update(mp_df["item_id"].unique().to_list())
        if user_events.get("retail", pl.DataFrame()).height > 0:
            retail_df = user_events["retail"]
            if "item_id" in retail_df.columns:
                user_item_ids.update(retail_df["item_id"].unique().to_list())
        if user_events.get("receipts", pl.DataFrame()).height > 0:
            receipts_df = user_events["receipts"]
            if "approximate_item_id" in receipts_df.columns:
                user_item_ids.update(receipts_df["approximate_item_id"].unique().to_list())
        
        if user_item_ids and len(user_item_ids) > 0:
            try:
                print(f"🔍 Загрузка embedding для {len(user_item_ids)} товаров пользователя (опционально)...")
                # Загружаем embedding ТОЛЬКО для товаров пользователя
                mp_items_emb = loader.load_marketplace_items(
                    item_ids=[str(iid) for iid in user_item_ids],
                    use_lazy=False,
                    include_embedding=True  # Загружаем embedding только для нужных товаров
                )
                retail_items_emb = loader.load_retail_items(
                    item_ids=[str(iid) for iid in user_item_ids],
                    use_lazy=False,
                    include_embedding=True
                )
                
                items_with_embeddings = {}
                if mp_items_emb is not None:
                    try:
                        if hasattr(mp_items_emb, 'collect'):
                            mp_items_emb = mp_items_emb.collect()
                        if mp_items_emb.height > 0 and "embedding" in mp_items_emb.columns:
                            items_with_embeddings["marketplace"] = mp_items_emb
                            print(f"✅ Загружены embedding для {mp_items_emb.height} товаров marketplace")
                    except:
                        pass
                
                if retail_items_emb is not None:
                    try:
                        if hasattr(retail_items_emb, 'collect'):
                            retail_items_emb = retail_items_emb.collect()
                        if retail_items_emb.height > 0 and "embedding" in retail_items_emb.columns:
                            items_with_embeddings["retail"] = retail_items_emb
                            print(f"✅ Загружены embedding для {retail_items_emb.height} товаров retail")
                    except:
                        pass
                
                if not items_with_embeddings:
                    print(f"⚠ Embedding не загружены (не найдены или недоступны)")
            except Exception as e:
                print(f"⚠ Ошибка при загрузке embedding: {e}")
                items_with_embeddings = None
    
    # Создание профиля пользователя
    print(f"👤 Создание профиля пользователя...")
    
    # Объединяем items_catalog с items_with_embeddings для передачи в create_user_profile
    # items_catalog содержит категории, items_with_embeddings - embedding (если загружены)
    all_items_for_profile = {}
    if items_catalog:
        all_items_for_profile.update(items_catalog)
    if items_with_embeddings:
        # Если embedding уже загружены, объединяем с каталогами
        for catalog_name, items_df in items_with_embeddings.items():
            if catalog_name in all_items_for_profile:
                # Объединяем: берем категории из items_catalog, embedding из items_with_embeddings
                catalog_df = all_items_for_profile[catalog_name]
                if "item_id" in catalog_df.columns and "item_id" in items_df.columns:
                    # Объединяем по item_id, добавляя embedding
                    if "embedding" in items_df.columns:
                        all_items_for_profile[catalog_name] = catalog_df.join(
                            items_df.select(["item_id", "embedding"]),
                            on="item_id",
                            how="left"
                        )
            else:
                all_items_for_profile[catalog_name] = items_df
    
    # Если items_catalog пуст, но items_with_embeddings есть, используем их
    if not all_items_for_profile and items_with_embeddings:
        all_items_for_profile = items_with_embeddings
    
    profile = create_user_profile(
        user_events=user_events,
        patterns=patterns,
        user_id=user_id,
        items_with_embeddings=all_items_for_profile if all_items_for_profile else None,
        item_to_brand_map=item_to_brand_map,
        brands_categories_map=brands_categories_map
    )
    
    # Fallback: Если топ-категория по товарам не определена, используем категорию бренда
    # (логика определения топ категории бренда уже внутри create_user_profile)
    if not profile.get("top_category") and profile.get("top_brand_category"):
        profile["top_category"] = profile["top_brand_category"]
        print(f"   ℹ Использована категория бренда как топ-категория профиля")
    
    print(f"✅ Профиль создан")
    
    # Рекомендации через модель ML (с улучшенным fallback)
    print(f"🤖 Генерация ML рекомендаций...")
    ml_recommendations = []
    try:
        # Передаем граф и паттерны для улучшенного fallback
        ml_recommendations = ml_recommend(profile, top_k=top_k, graph=graph, patterns=patterns)
        print(f"✅ Получено {len(ml_recommendations)} ML рекомендаций")
    except Exception as e:
        print(f"⚠ Ошибка ML рекомендаций: {e}")
        # Используем улучшенный fallback напрямую
        from src.modeling.nbo_model import NBOModel
        model = NBOModel()
        ml_recommendations = model._fallback_recommendations(profile, top_k, graph, patterns)
    
    # Рекомендации через правила
    rule_engine = RuleEngine()
    rule_recommendations = []
    if pattern_strings:
        try:
            rule_recommendations = rule_engine.recommend_from_patterns(
                pattern_strings,
                user_context=profile
            )
        except Exception as e:
            print(f"Ошибка рекомендаций по правилам: {e}")
    
    # Объединяем рекомендации
    all_recommendations = []
    
    # Добавляем ML рекомендации
    for rec in ml_recommendations:
        product = rec["product"]
        score = rec["score"]
        
        # Генерируем объяснение (используем YandexGPT только если включен)
        try:
            reason = explain_recommendation(profile, product, use_yandexgpt=use_yandexgpt_for_analysis)
        except Exception as e:
            print(f"Ошибка генерации объяснения: {e}")
            reason = f"Рекомендуется на основе вашего профиля"
        
        all_recommendations.append({
            "product": product,
            "score": score,
            "source": "ML модель",
            "reason": reason
        })
    
    # Добавляем рекомендации по правилам
    for rec in rule_recommendations[:top_k]:
        product = rec["product"]
        score = rec["score"]
        
        # Берем лучшее объяснение из правил
        if rec["reasons"]:
            reason = rec["reasons"][0]["reason"]
        else:
            try:
                reason = explain_recommendation(profile, product, use_yandexgpt=use_yandexgpt_for_analysis)
            except:
                reason = f"Рекомендуется на основе паттернов поведения"
        
        all_recommendations.append({
            "product": product,
            "score": score,
            "source": "Правила",
            "reason": reason
        })
    
    # Сортируем и берем топ-K
    all_recommendations.sort(key=lambda x: x["score"], reverse=True)
    final_recommendations = all_recommendations[:top_k]
    
    print(f"📊 Финальная статистика:")
    print(f"   - Брендов в маппинге названий: {len(brands_map)}")
    if len(brands_map) > 0:
        print(f"     Примеры ключей brands_map: {list(brands_map.keys())[:5]}")
    
    print(f"   - Брендов в маппинге категорий: {len(brands_categories_map)}")
    if len(brands_categories_map) > 0:
        print(f"     Примеры ключей brands_categories_map: {list(brands_categories_map.keys())[:5]}")
        
    if profile.get('top_brand'):
        top_brand_val = profile['top_brand']
        print(f"   - Топ бренд в профиле: '{top_brand_val}' (тип: {type(top_brand_val)})")
        
        # Пробуем разные варианты поиска ключа
        keys_to_try = [
            str(top_brand_val), 
            str(top_brand_val).replace(".0", ""), 
            str(int(float(top_brand_val))) if str(top_brand_val).replace(".", "", 1).isdigit() else str(top_brand_val)
        ]
        
        found = False
        for key in keys_to_try:
            if brands_map.get(key):
                print(f"     -> Название найдено по ключу '{key}': {brands_map.get(key)}")
                # Исправляем в профиле если нашли
                if key != str(top_brand_val):
                    print(f"     -> ⚠ Несовпадение форматов! В профиле '{top_brand_val}', в мапе '{key}'")
                found = True
                break
        
        if not found:
            print(f"     -> Название НЕ найдено в маппинге. Пробовали ключи: {keys_to_try}")
    else:
        print(f"   - Топ бренд в профиле не установлен (None или пустой)")
        if profile.get('brand_ids'):
            print(f"     Но есть brand_ids: {profile.get('brand_ids')[:5]}")

    return {
        "user_id": user_id,
        "profile": profile,
        "graph": graph,  # Добавляем сам граф для визуализации
        "graph_stats": graph_stats,
        "patterns": pattern_strings,
        "graph_analysis": graph_analysis,
        "graph_rules": graph_rules,
        "recommendations": final_recommendations,
        "brands_map": brands_map,  # Маппинг brand_id -> brand_name
        "brands_categories_map": brands_categories_map  # Маппинг brand_id -> category
    }


def main() -> None:
    """
    Основная функция для демонстрации работы системы.
    """
    print("Система рекомендаций Next Best Offer для ПСБ")
    print("=" * 50)
    
    # Пример использования
    user_id = input("Введите ID пользователя (или нажмите Enter для тестового): ").strip()
    if not user_id:
        user_id = "12345"  # Тестовый ID
    
    print(f"\nОбработка пользователя {user_id}...")
    
    try:
        result = process_user(
            user_id=user_id,
            use_cloud=True,
            use_yandexgpt_for_analysis=True,
            top_k=3
        )
        
        print("\n" + "=" * 50)
        print("РЕКОМЕНДАЦИИ:")
        print("=" * 50)
        
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"\n{i}. {rec['product']}")
            print(f"   Оценка: {rec['score']:.2f}")
            print(f"   Источник: {rec['source']}")
            print(f"   Объяснение: {rec['reason']}")
        
        print("\n" + "=" * 50)
        print("СТАТИСТИКА:")
        print("=" * 50)
        print(f"Паттернов найдено: {len(result['patterns'])}")
        print(f"Узлов в графе: {result['graph_stats']['nodes']}")
        print(f"Связей в графе: {result['graph_stats']['edges']}")
        
    except Exception as e:
        print(f"Ошибка при обработке пользователя: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

