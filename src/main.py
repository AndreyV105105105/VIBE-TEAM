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
    
    if use_cloud:
        loader = get_loader()
        if loader is None:
            loader = init_loader(
                public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
            )
        
        # Загружаем справочник брендов для сопоставления brand_id с названиями и категориями
        print(f"📚 Загрузка справочника брендов...")
        brands_categories_map: Dict[str, str] = {}  # Маппинг brand_id -> category
        
        try:
            brands_df = loader.load_brands()
            if brands_df.height > 0:
                print(f"✅ Загружено {brands_df.height} брендов")
                print(f"   Колонки: {brands_df.columns}")
                
                # Определяем колонки для маппинга
                brand_id_col = None
                brand_name_col = None
                brand_category_col = None
                
                for col in brands_df.columns:
                    col_lower = col.lower()
                    if col_lower in ["brand_id", "brandid", "id"]:
                        brand_id_col = col
                    elif col_lower in ["name", "brand_name", "title", "brand_title", "brand"]:
                        brand_name_col = col
                    elif col_lower in ["category", "category_id", "categoryid", "cat_id", "cat", 
                                       "merchant_category", "merchant_category_id", "mcc", "mcc_code"]:
                        brand_category_col = col
                
                if brand_id_col:
                    # Создаем маппинг brand_id -> brand_name
                    if brand_name_col:
                        for row in brands_df.iter_rows(named=True):
                            brand_id = str(row.get(brand_id_col, ""))
                            brand_name = str(row.get(brand_name_col, ""))
                            if brand_id and brand_name:
                                brands_map[brand_id] = brand_name
                        print(f"✅ Создан маппинг названий для {len(brands_map)} брендов")
                    
                    # Создаем маппинг brand_id -> category из brands.pq (если есть)
                    if brand_category_col:
                        for row in brands_df.iter_rows(named=True):
                            brand_id = str(row.get(brand_id_col, ""))
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
        if len(brands_categories_map) == 0:
            print(f"📦 Извлечение категорий брендов из каталогов товаров (items.pq)...")
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
                all_items_lazy = []
                if marketplace_items_lazy is not None:
                    try:
                        schema = marketplace_items_lazy.collect_schema()
                        if len(schema) > 0:
                            all_items_lazy.append(marketplace_items_lazy)
                    except:
                        pass
                
                if retail_items_lazy is not None:
                    try:
                        schema = retail_items_lazy.collect_schema()
                        if len(schema) > 0:
                            all_items_lazy.append(retail_items_lazy)
                    except:
                        pass
                
                if all_items_lazy:
                    # Объединяем LazyFrames (еще не загружены в память!)
                    combined_lazy = pl.concat(all_items_lazy)
                    
                    # Проверяем наличие нужных колонок
                    schema = combined_lazy.collect_schema()
                    has_brand_id = "brand_id" in schema
                    has_category = any(col.lower() in ["category_id", "category", "categoryid"] for col in schema)
                    
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
                                # Группируем по brand_id и находим самую частую категорию для каждого бренда
                                # Это выполняется на LazyFrame - данные еще не загружены в память!
                                brand_categories_lazy = combined_lazy.group_by("brand_id").agg([
                                    pl.col(category_col).mode().alias("top_category"),
                                    pl.count().alias("item_count")
                                ]).filter(
                                    pl.col("top_category").is_not_null()
                                )
                            
                                # ТОЛЬКО СЕЙЧАС загружаем в память (после всех оптимизаций)
                                print(f"   ⚡ Применяем агрегацию на LazyFrame (данные еще не загружены в память)...")
                                try:
                                    brand_categories = brand_categories_lazy.collect()
                                    
                                    print(f"✅ Загружено {brand_categories.height} уникальных брендов (после агрегации)")
                                    
                                    # Создаем маппинг brand_id -> category
                                    for row in brand_categories.iter_rows(named=True):
                                        brand_id = str(row.get("brand_id", ""))
                                        top_categories = row.get("top_category", [])
                                        if brand_id and top_categories and len(top_categories) > 0:
                                            # Берем первую (самую частую) категорию
                                            category = str(top_categories[0])
                                            if category and category.lower() not in ["none", "null", "nan", ""]:
                                                brands_categories_map[brand_id] = category
                                    
                                    print(f"✅ Извлечено категорий для {len(brands_categories_map)} брендов из каталогов товаров")
                                    print(f"   💾 Экономия памяти: загружены только агрегированные данные, не весь каталог (~30 ГБ)")
                                except Exception as e:
                                    print(f"⚠ Ошибка при агрегации категорий брендов: {e}")
                                    import traceback
                                    print(f"   Детали: {traceback.format_exc()}")
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
                    user_marketplace_lazy = (
                        marketplace_lazy
                        .filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                        # Выбираем только нужные колонки для ускорения
                        .select([
                            "user_id", "item_id", "timestamp", "domain",
                            pl.col("category_id").alias("category_id"),
                            pl.col("brand_id").alias("brand_id"),
                            pl.col("action_type").alias("action_type"),
                            pl.col("subdomain").alias("subdomain"),
                            pl.col("price").alias("price"),
                            pl.col("count").alias("count"),
                            pl.col("os").alias("os")
                        ])
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
        if items_catalog and user_marketplace.height > 0:
            try:
                mp_items = items_catalog.get("marketplace")
                if mp_items is not None and "item_id" in mp_items.columns and "category" in mp_items.columns:
                    # Объединяем с каталогом для получения категорий
                    user_marketplace = user_marketplace.join(
                        mp_items.select(["item_id", "category", "subcategory"]),
                        on="item_id",
                        how="left"
                    )
                    print(f"✅ Обогащено {user_marketplace.filter(pl.col('category').is_not_null()).height} событий marketplace категориями")
            except Exception as e:
                print(f"⚠ Ошибка при обогащении marketplace категориями: {e}")
        
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
    profile = create_user_profile(
        user_events=user_events,
        patterns=patterns,
        user_id=user_id,
        items_with_embeddings=items_with_embeddings
    )
    
    # Добавляем категории брендов в профиль на основе маппинга
    if brands_categories_map and profile.get("brand_ids"):
        brand_categories = []
        for brand_id in profile.get("brand_ids", []):
            category = brands_categories_map.get(str(brand_id))
            if category:
                brand_categories.append(category)
        
        if brand_categories:
            # Топ категория брендов (самая частая)
            from collections import Counter
            category_counts = Counter(brand_categories)
            top_brand_category = category_counts.most_common(1)[0][0] if category_counts else None
            profile["top_brand_category"] = top_brand_category
            profile["brand_categories"] = list(set(brand_categories))  # Уникальные категории
            print(f"✅ Найдено {len(set(brand_categories))} категорий брендов, топ: {top_brand_category}")
        else:
            profile["top_brand_category"] = None
            profile["brand_categories"] = []
    else:
        profile["top_brand_category"] = None
        profile["brand_categories"] = []
    
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

