"""
Модуль для парсинга и нормализации данных из различных источников.

Автоматически определяет структуру данных и приводит их к единому формату.
"""

from typing import Dict, Optional, List
import polars as pl
from datetime import datetime


def normalize_marketplace_events(df: pl.DataFrame, file_path: str = "") -> pl.DataFrame:
    """
    Нормализует события маркетплейса к единому формату.
    
    Ожидаемые колонки после нормализации:
    - user_id: ID пользователя
    - item_id: ID товара
    - category_id: ID категории (опционально)
    - timestamp: Временная метка
    - domain: "marketplace"
    - region: Регион (опционально)
    - price: Цена (опционально)
    
    :param df: Исходный DataFrame
    :param file_path: Путь к файлу (для логирования)
    :return: Нормализованный DataFrame
    """
    if df.height == 0:
        return df
    
    # Оптимизация: избегаем клонирования
    result = df
    
    # Добавляем domain если его нет
    if "domain" not in result.columns:
        result = result.with_columns(pl.lit("marketplace").alias("domain"))
    
    # Нормализуем user_id (может быть в разных форматах)
    if "user_id" not in result.columns:
        # Пробуем альтернативные названия
        for alt_name in ["user", "userId", "userid", "uid", "client_id", "User", "UserID", "UserID", "UID"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "user_id"})
                break
        else:
            # Если не найдено, создаем фиктивную колонку (для отладки)
            print(f"Предупреждение: колонка user_id не найдена в файле {file_path}. Доступные колонки: {result.columns}")
            # Если DataFrame пустой, возвращаем как есть
            if result.height == 0:
                return result
            # Если есть данные, но нет user_id, создаем фиктивный
            result = result.with_columns(pl.lit("unknown").alias("user_id"))
    
    # Нормализуем item_id
    if "item_id" not in result.columns:
        for alt_name in ["item", "itemId", "itemid", "product_id", "productId", "product"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "item_id"})
                break
        else:
            # Если нет item_id, создаем фиктивный
            result = result.with_columns(pl.lit("unknown").alias("item_id"))
    
    # Нормализуем category_id
    if "category_id" not in result.columns:
        for alt_name in ["category", "categoryId", "categoryid", "cat_id", "cat"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "category_id"})
                break
        else:
            # Если нет category_id, создаем null
            result = result.with_columns(pl.lit(None).alias("category_id"))
    
    # Нормализуем brand_id
    if "brand_id" not in result.columns:
        for alt_name in ["brand", "Brand", "brandId", "brandid", "merchant_id", "merchantId"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "brand_id"})
                break
        # Если brand_id нет, не создаем фиктивный - оставляем как есть (может быть null)
    
    # Приводим brand_id к строке и удаляем .0 если это float
    if "brand_id" in result.columns:
        try:
            result = result.with_columns(
                pl.col("brand_id").cast(pl.Utf8).str.replace(r"\.0$", "")
            )
        except:
            pass
    
    # Нормализуем timestamp
    if "timestamp" not in result.columns:
        for alt_name in ["time", "Time", "ts", "date", "datetime", "event_time", "eventTime"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "timestamp"})
                break
        else:
            # Если нет timestamp, создаем текущее время
            result = result.with_columns(pl.lit(datetime.now()).alias("timestamp"))
    
    # Конвертируем timestamp в datetime если нужно
    if result["timestamp"].dtype != pl.Datetime:
        try:
            result = result.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            )
        except:
            try:
                result = result.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
                )
            except:
                # Если не удалось распарсить, оставляем как есть
                pass
    
    # Нормализуем region (опционально)
    if "region" not in result.columns:
        for alt_name in ["Region", "REGION", "reg", "Reg", "geo_region"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "region"})
                break
    
    # Нормализуем price (опционально)
    if "price" not in result.columns:
        for alt_name in ["Price", "PRICE", "amount", "Amount", "cost", "Cost"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "price"})
                break
    
    # Выбираем только нужные колонки
    expected_cols = ["user_id", "item_id", "category_id", "timestamp", "domain"]
    optional_cols = ["region", "price", "brand_id", "action_type", "subdomain", "count", "os"]
    
    available_cols = [col for col in expected_cols + optional_cols if col in result.columns]
    
    return result.select(available_cols)


def normalize_payments_events(df: pl.DataFrame, file_path: str = "") -> pl.DataFrame:
    """
    Нормализует события платежей к единому формату.
    
    Ожидаемые колонки после нормализации:
    - user_id: ID пользователя
    - brand_id: ID бренда
    - amount: Сумма платежа (в долларах)
    - timestamp: Временная метка
    - domain: "payments"
    
    :param df: Исходный DataFrame
    :param file_path: Путь к файлу (для логирования)
    :return: Нормализованный DataFrame
    """
    if df.height == 0:
        return df
    
    # Диагностика: показываем исходные колонки
    print(f"📋 Парсинг payments events из {file_path}:")
    print(f"   Исходные колонки: {df.columns}")
    print(f"   Количество строк: {df.height}")
    
    result = df.clone()
    
    # Добавляем domain если его нет
    if "domain" not in result.columns:
        result = result.with_columns(pl.lit("payments").alias("domain"))
    
    # Нормализуем user_id
    if "user_id" not in result.columns:
        for alt_name in ["user", "userId", "userid", "uid", "client_id"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "user_id"})
                break
        else:
            # Если не найдено, создаем фиктивную колонку (для отладки)
            print(f"Предупреждение: колонка user_id не найдена в файле {file_path}. Доступные колонки: {result.columns}")
            # Если DataFrame пустой, возвращаем как есть
            if result.height == 0:
                return result
            # Если есть данные, но нет user_id, создаем фиктивный
            result = result.with_columns(pl.lit("unknown").alias("user_id"))
    
    # Нормализуем brand_id
    if "brand_id" not in result.columns:
        for alt_name in ["brand", "Brand", "brandId", "brandid", "merchant_id", "merchantId"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "brand_id"})
                break
        else:
            # Если нет brand_id, создаем фиктивный
            result = result.with_columns(pl.lit("unknown").alias("brand_id"))
    
    # Приводим brand_id к строке и удаляем .0 если это float
    if "brand_id" in result.columns:
        try:
            # Сначала кастуем к строке, чтобы обработать все типы
            result = result.with_columns(
                pl.col("brand_id").cast(pl.Utf8).str.replace(r"\.0$", "")
            )
        except:
            pass
    
    # Нормализуем amount
    if "amount" not in result.columns:
        for alt_name in ["Amount", "AMOUNT", "sum", "Sum", "value", "Value", "price", "Price"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "amount"})
                break
        else:
            # Если нет amount, создаем 0
            result = result.with_columns(pl.lit(0.0).alias("amount"))
    
    # Проверяем тип amount и приводим к числовому формату (значения уже в долларах, не конвертируем)
    if "amount" in result.columns and result.height > 0:
        try:
            # Проверяем тип и конвертируем в числовой
            if result["amount"].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                result = result.with_columns(pl.col("amount").cast(pl.Float64, strict=False))
            
            # Диагностика: показываем статистику (значения уже в долларах)
            non_zero = result.filter(pl.col("amount").abs() > 0.001)
            if non_zero.height > 0:
                amount_stats = non_zero.select([
                    pl.col("amount").abs().min().alias("min_abs"),
                    pl.col("amount").abs().max().alias("max_abs"),
                    pl.col("amount").abs().mean().alias("mean_abs"),
                    pl.col("amount").abs().quantile(0.5).alias("median_abs")
                ])
                
                if amount_stats.height > 0:
                    stats = amount_stats.row(0)
                    min_abs, max_abs, mean_abs, median_abs = stats
                    print(f"💵 Значения amount (в долларах): min=${min_abs:.2f}, max=${max_abs:.2f}, mean=${mean_abs:.2f}, median=${median_abs:.2f}")
        except Exception as e:
            print(f"⚠ Не удалось обработать amount: {e}")
    
    # Нормализуем timestamp
    if "timestamp" not in result.columns:
        for alt_name in ["time", "Time", "ts", "date", "datetime", "event_time", "eventTime"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "timestamp"})
                break
        else:
            result = result.with_columns(pl.lit(datetime.now()).alias("timestamp"))
    
    # Конвертируем timestamp в datetime если нужно
    if result["timestamp"].dtype != pl.Datetime:
        try:
            result = result.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            )
        except:
            try:
                result = result.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
                )
            except:
                pass
    
    # Выбираем только нужные колонки
    expected_cols = ["user_id", "brand_id", "amount", "timestamp", "domain"]
    available_cols = [col for col in expected_cols if col in result.columns]
    
    # Финальная диагностика
    if result.height > 0:
        print(f"   ✅ После нормализации: колонки {available_cols}, строк: {result.height}")
        if "amount" in available_cols:
            amount_sample = result.select(pl.col("amount")).head(5).to_series().to_list()
            print(f"   💵 Примеры значений amount: {amount_sample}")
    else:
        print(f"   ⚠ После нормализации DataFrame пуст")
    
    return result.select(available_cols)


def normalize_retail_events(df: pl.DataFrame, file_path: str = "") -> pl.DataFrame:
    """
    Нормализует события ритейла к единому формату.
    
    :param df: Исходный DataFrame
    :param file_path: Путь к файлу (для логирования)
    :return: Нормализованный DataFrame
    """
    if df.height == 0:
        return df
    
    # Оптимизация: избегаем клонирования
    result = df
    
    # Добавляем domain если его нет
    if "domain" not in result.columns:
        result = result.with_columns(pl.lit("retail").alias("domain"))
    
    # Нормализуем user_id
    if "user_id" not in result.columns:
        for alt_name in ["user", "userId", "userid", "uid", "client_id"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "user_id"})
                break
        else:
            # Если не найдено, создаем фиктивную колонку (для отладки)
            print(f"Предупреждение: колонка user_id не найдена в файле {file_path}. Доступные колонки: {result.columns}")
            # Если DataFrame пустой, возвращаем как есть
            if result.height == 0:
                return result
            # Если есть данные, но нет user_id, создаем фиктивный
            result = result.with_columns(pl.lit("unknown").alias("user_id"))
    
    # Нормализуем item_id
    if "item_id" not in result.columns:
        for alt_name in ["item", "itemId", "itemid", "product_id", "productId", "product"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "item_id"})
                break
        # Если item_id нет, не создаем фиктивный - оставляем как есть
    
    # Нормализуем brand_id
    if "brand_id" not in result.columns:
        for alt_name in ["brand", "Brand", "brandId", "brandid", "merchant_id", "merchantId"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "brand_id"})
                break
        # Если brand_id нет, не создаем фиктивный - оставляем как есть (может быть null)
    
    # Нормализуем category_id
    if "category_id" not in result.columns:
        for alt_name in ["category", "categoryId", "categoryid", "cat_id", "cat"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "category_id"})
                break
        # Если category_id нет, не создаем - оставляем как есть
    
    # Нормализуем action_type
    if "action_type" not in result.columns:
        for alt_name in ["action", "actionType", "actiontype", "type", "event_type", "eventType"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "action_type"})
                break
    
    # Нормализуем subdomain
    if "subdomain" not in result.columns:
        for alt_name in ["subdomain", "Subdomain", "context", "Context", "source", "Source"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "subdomain"})
                break
    
    # Нормализуем price
    if "price" not in result.columns:
        for alt_name in ["Price", "PRICE", "amount", "Amount", "cost", "Cost"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "price"})
                break
    
    # Нормализуем count
    if "count" not in result.columns:
        for alt_name in ["Count", "COUNT", "quantity", "Quantity", "qty", "Qty"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "count"})
                break
    
    # Нормализуем os
    if "os" not in result.columns:
        for alt_name in ["OS", "os", "operating_system", "OperatingSystem", "platform", "Platform"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "os"})
                break
    
    # Нормализуем timestamp
    if "timestamp" not in result.columns:
        for alt_name in ["time", "Time", "ts", "date", "datetime", "event_time", "eventTime"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "timestamp"})
                break
        else:
            result = result.with_columns(pl.lit(datetime.now()).alias("timestamp"))
    
    # Конвертируем timestamp в datetime если нужно
    if result["timestamp"].dtype != pl.Datetime:
        try:
            result = result.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            )
        except:
            try:
                result = result.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
                )
            except:
                pass
    
    # Выбираем нужные колонки (сохраняем все важные поля из retail events)
    expected_cols = ["user_id", "timestamp", "domain"]
    optional_cols = ["item_id", "brand_id", "category_id", "action_type", "subdomain", "price", "count", "os"]
    
    available_cols = [col for col in expected_cols + optional_cols if col in result.columns]
    
    return result.select(available_cols)


def detect_data_structure(df: pl.DataFrame) -> Dict[str, any]:
    """
    Автоматически определяет структуру данных.
    
    :param df: DataFrame для анализа
    :return: Словарь с информацией о структуре
    """
    if df.height == 0:
        return {"type": "empty", "columns": []}
    
    columns = df.columns
    schema = df.schema
    
    # Определяем тип данных по колонкам
    has_user_id = any(col.lower() in ["user_id", "user", "userid", "uid"] for col in columns)
    has_item_id = any(col.lower() in ["item_id", "item", "itemid", "product_id"] for col in columns)
    has_brand_id = any(col.lower() in ["brand_id", "brand", "brandid", "merchant_id"] for col in columns)
    has_amount = any(col.lower() in ["amount", "sum", "value", "price"] for col in columns)
    has_category = any(col.lower() in ["category_id", "category", "categoryid"] for col in columns)
    
    data_type = "unknown"
    if has_item_id and has_category:
        data_type = "marketplace"
    elif has_brand_id and has_amount:
        data_type = "payments"
    elif has_user_id:
        data_type = "retail"
    
    return {
        "type": data_type,
        "columns": columns,
        "schema": schema,
        "has_user_id": has_user_id,
        "has_item_id": has_item_id,
        "has_brand_id": has_brand_id,
        "has_amount": has_amount,
        "has_category": has_category,
        "num_rows": df.height
    }


def normalize_dataframe(df: pl.DataFrame, domain: str, file_path: str = "") -> pl.DataFrame:
    """
    Нормализует DataFrame в зависимости от домена.
    
    :param df: Исходный DataFrame
    :param domain: Домен данных ("marketplace", "payments", "retail")
    :param file_path: Путь к файлу (для логирования)
    :return: Нормализованный DataFrame
    """
    if domain == "marketplace":
        return normalize_marketplace_events(df, file_path)
    elif domain == "payments":
        return normalize_payments_events(df, file_path)
    elif domain == "retail":
        return normalize_retail_events(df, file_path)
    else:
        # Пытаемся определить автоматически
        structure = detect_data_structure(df)
        detected_type = structure["type"]
        
        if detected_type == "marketplace":
            return normalize_marketplace_events(df, file_path)
        elif detected_type == "payments":
            return normalize_payments_events(df, file_path)
        elif detected_type == "retail":
            return normalize_retail_events(df, file_path)
        else:
            # Если не удалось определить, возвращаем как есть
            return df

