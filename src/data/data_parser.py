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
    
    result = df.clone()
    
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
    optional_cols = ["region", "price"]
    
    available_cols = [col for col in expected_cols + optional_cols if col in result.columns]
    
    return result.select(available_cols)


def normalize_payments_events(df: pl.DataFrame, file_path: str = "") -> pl.DataFrame:
    """
    Нормализует события платежей к единому формату.
    
    Ожидаемые колонки после нормализации:
    - user_id: ID пользователя
    - brand_id: ID бренда
    - amount: Сумма платежа
    - timestamp: Временная метка
    - domain: "payments"
    
    :param df: Исходный DataFrame
    :param file_path: Путь к файлу (для логирования)
    :return: Нормализованный DataFrame
    """
    if df.height == 0:
        return df
    
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
    
    # Нормализуем amount
    if "amount" not in result.columns:
        for alt_name in ["Amount", "AMOUNT", "sum", "Sum", "value", "Value", "price", "Price"]:
            if alt_name in result.columns:
                result = result.rename({alt_name: "amount"})
                break
        else:
            # Если нет amount, создаем 0
            result = result.with_columns(pl.lit(0.0).alias("amount"))
    
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
    
    result = df.clone()
    
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
    
    # Минимальный набор колонок
    expected_cols = ["user_id", "timestamp", "domain"]
    available_cols = [col for col in expected_cols if col in result.columns]
    
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

