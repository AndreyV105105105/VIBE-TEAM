"""
Утилита для поиска и получения списка пользователей из данных.
"""

from typing import List, Optional
import polars as pl
from src.data.cloud_loader import get_loader, init_loader


def get_available_users(
    limit: int = 100,
    num_files: int = 1,  # Уменьшено до 1 файла для максимальной скорости
    start_file: int = 1082
) -> List[str]:
    """
    Получает список доступных пользователей из данных в облаке.
    
    Возвращает только тех пользователей, которые реально есть в событиях (marketplace + payments).
    
    :param limit: Максимальное количество пользователей для возврата (по умолчанию 100)
    :param num_files: Количество файлов для проверки (уменьшено до 1 для скорости)
    :param start_file: Начальный номер файла
    :return: Список ID пользователей (строки) - только те, у кого есть события
    """
    loader = get_loader()
    if loader is None:
        loader = init_loader(public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg")
    
    print(f"Загрузка пользователей из событий (marketplace + payments)...")
    all_user_ids = set()
    
    try:
        # 1. Загружаем пользователей из marketplace events
        print(f"Загрузка из marketplace/events/{start_file:05d}.pq...")
        test_file = f"{start_file:05d}.pq"
        marketplace_lazy = loader.load_marketplace_events(file_list=[test_file], days=None)  # Без фильтрации по дате для получения всех пользователей
        
        if marketplace_lazy is not None:
            schema = marketplace_lazy.collect_schema()
            if "user_id" in schema:
                mp_users = (
                    marketplace_lazy
                    .select(pl.col("user_id").cast(pl.Utf8).alias("user_id"))
                    .unique()
                    .limit(limit * 2)  # Берем больше, так как потом объединим с payments
                    .collect()
                )
                mp_user_ids = [str(uid) for uid in mp_users["user_id"].to_list() if uid is not None and str(uid).strip()]
                all_user_ids.update(mp_user_ids)
                print(f"  Найдено {len(mp_user_ids)} пользователей в marketplace")
        
        # 2. Загружаем пользователей из payments events (если нужно больше)
        if len(all_user_ids) < limit:
            print(f"Загрузка из payments/events/{start_file:05d}.pq...")
            try:
                payments_lazy = loader.load_payments_events(file_list=[test_file], days=None)  # Без фильтрации по дате
                
                if payments_lazy is not None:
                    schema = payments_lazy.collect_schema()
                    if "user_id" in schema:
                        pay_users = (
                            payments_lazy
                            .select(pl.col("user_id").cast(pl.Utf8).alias("user_id"))
                            .unique()
                            .limit(limit * 2)
                            .collect()
                        )
                        pay_user_ids = [str(uid) for uid in pay_users["user_id"].to_list() if uid is not None and str(uid).strip()]
                        all_user_ids.update(pay_user_ids)
                        print(f"  Найдено {len(pay_user_ids)} пользователей в payments")
            except Exception as e:
                print(f"  ⚠ Не удалось загрузить payments: {e}")
        
        # Возвращаем только тех, у кого есть события
        result = list(all_user_ids)[:limit]
        print(f"✅ Найдено {len(result)} уникальных пользователей с событиями (из {len(all_user_ids)} всего)")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"❌ Ошибка при загрузке пользователей: {e}")
        traceback.print_exc()
        return []


def get_users_from_users_file(limit: int = 100) -> List[str]:
    """
    Получает список пользователей из файла users.pq в облаке.
    
    :param limit: Максимальное количество пользователей
    :return: Список ID пользователей (строки)
    """
    loader = get_loader()
    if loader is None:
        loader = init_loader(public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg")
    
    try:
        print("Загрузка пользователей из файла users.pq в облаке...")
        users_df = loader.load_users()
        
        if users_df is None or users_df.height == 0:
            print("Файл users.pq не найден или пуст")
            return []
        
        print(f"Загружен users.pq, строк: {users_df.height}, колонки: {users_df.columns}")
        
        # Проверяем наличие user_id или альтернативных названий
        user_id_col = None
        if "user_id" in users_df.columns:
            user_id_col = "user_id"
        else:
            # Ищем альтернативные названия
            for alt_name in ["user", "userId", "userid", "uid", "client_id", "User", "UserID", "UID"]:
                if alt_name in users_df.columns:
                    user_id_col = alt_name
                    print(f"Используем колонку {alt_name} как user_id")
                    break
        
        if user_id_col is None:
            print(f"Колонка user_id не найдена. Доступные колонки: {users_df.columns}")
            return []
        
        # Получаем уникальные user_id
        # load_users() возвращает обычный DataFrame, а не LazyFrame, поэтому .collect() не нужен
        user_ids = users_df.select(
            pl.col(user_id_col).cast(pl.Utf8).alias("user_id")
        ).unique().limit(limit)["user_id"].to_list()
        
        # Конвертируем все в строки и убираем пустые значения
        result = [str(uid) for uid in user_ids if uid is not None and str(uid).strip() != ""]
        
        print(f"Найдено {len(result)} уникальных пользователей в users.pq")
        return result
    except Exception as e:
        import traceback
        print(f"Ошибка при загрузке users.pq из облака: {e}")
        print(f"Трассировка: {traceback.format_exc()}")
        return []


def search_users_by_pattern(pattern: str, limit: int = 50) -> List[str]:
    """
    Ищет пользователей по паттерну в ID.
    
    :param pattern: Паттерн для поиска
    :param limit: Максимальное количество результатов
    :return: Список ID пользователей
    """
    # Загружаем ограниченное количество пользователей для быстрого поиска
    all_users = get_available_users(limit=100, num_files=5)
    
    if not all_users:
        return []
    
    # Фильтруем по паттерну
    matching_users = [
        user_id for user_id in all_users
        if pattern.lower() in str(user_id).lower()
    ]
    
    return matching_users[:limit]


def get_user_statistics(user_id: str) -> dict:
    """
    Получает базовую статистику по пользователю.
    
    :param user_id: ID пользователя
    :return: Словарь со статистикой
    """
    loader = get_loader()
    if loader is None:
        loader = init_loader(public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg")
    
    num_files = 3  # Уменьшено для быстрой загрузки
    start_file = 1082
    
    marketplace_files = [
        f"{i:05d}.pq" for i in range(start_file, start_file + num_files)
    ]
    
    try:
        marketplace_lazy = loader.load_marketplace_events(file_list=marketplace_files)
        # Конвертируем user_id в строку для сравнения
        user_data = marketplace_lazy.filter(
            pl.col("user_id").cast(pl.Utf8) == str(user_id)
        ).collect()
        
        if user_data.height == 0:
            return {
                "exists": False,
                "num_events": 0
            }
        
        return {
            "exists": True,
            "num_events": user_data.height,
            "has_payments": False  # Можно добавить проверку payments
        }
    except:
        return {
            "exists": False,
            "num_events": 0
        }

