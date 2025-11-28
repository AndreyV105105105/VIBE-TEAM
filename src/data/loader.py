"""
Модуль для загрузки данных из локальных файлов.

Альтернатива cloud_loader для работы с локальными Parquet файлами.
"""

import glob
from pathlib import Path
from typing import Dict, Optional
import polars as pl
from datetime import datetime, timedelta


def load_user_events(
    data_root: str,
    user_id: str,
    days: int = 2
) -> Dict[str, pl.DataFrame]:
    """
    Загружает события пользователя из локальных файлов.
    
    :param data_root: Корневая директория с данными
    :param user_id: ID пользователя
    :param days: Количество дней для загрузки
    :return: Словарь с событиями по доменам
    """
    # Находим все файлы событий
    mp_files = sorted(glob.glob(f"{data_root}/marketplace/events/*.pq"))
    pay_files = sorted(glob.glob(f"{data_root}/payments/events/*.pq"))
    retail_files = sorted(glob.glob(f"{data_root}/retail/events/*.pq"))
    
    # Берем последние N файлов (предполагая партиционирование по дням)
    recent_mp = mp_files[-days:] if mp_files else []
    recent_pay = pay_files[-days:] if pay_files else []
    recent_retail = retail_files[-days:] if retail_files else []
    
    # Читаем и фильтруем по user_id
    mp_frames = []
    for f in recent_mp:
        try:
            df = pl.scan_parquet(f)
            # Конвертируем user_id в строку для сравнения
            mp_user = df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id)).collect()
            if mp_user.height > 0:
                mp_frames.append(mp_user)
        except Exception as e:
            print(f"Ошибка при загрузке {f}: {e}")
            continue
    
    pay_frames = []
    for f in recent_pay:
        try:
            df = pl.scan_parquet(f)
            # Конвертируем user_id в строку для сравнения
            pay_user = df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id)).collect()
            if pay_user.height > 0:
                pay_frames.append(pay_user)
        except Exception as e:
            print(f"Ошибка при загрузке {f}: {e}")
            continue
    
    retail_frames = []
    for f in recent_retail:
        try:
            df = pl.scan_parquet(f)
            # Конвертируем user_id в строку для сравнения
            retail_user = df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id)).collect()
            if retail_user.height > 0:
                retail_frames.append(retail_user)
        except Exception as e:
            print(f"Ошибка при загрузке {f}: {e}")
            continue
    
    # Объединяем все фреймы
    marketplace_df = pl.concat(mp_frames) if mp_frames else pl.DataFrame()
    payments_df = pl.concat(pay_frames) if pay_frames else pl.DataFrame()
    retail_df = pl.concat(retail_frames) if retail_frames else pl.DataFrame()
    
    return {
        "marketplace": marketplace_df,
        "payments": payments_df,
        "retail": retail_df
    }


def load_brands(data_root: str) -> pl.DataFrame:
    """Загружает справочник брендов."""
    brands_file = Path(data_root) / "brands.pq"
    if brands_file.exists():
        return pl.read_parquet(brands_file)
    return pl.DataFrame()


def load_users(data_root: str) -> pl.DataFrame:
    """Загружает справочник пользователей."""
    users_file = Path(data_root) / "users.pq"
    if users_file.exists():
        return pl.read_parquet(users_file)
    return pl.DataFrame()

