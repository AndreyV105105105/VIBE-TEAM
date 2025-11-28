"""
Скрипт для предварительного обучения модели с помощью YandexGPT.

Использует YandexGPT для генерации обучающих данных на основе реальных профилей пользователей.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cloud_loader import init_loader, get_loader
from src.features.user_profile import create_user_profile
from src.modeling.nbo_model import NBOModel
import polars as pl


def collect_user_profiles(limit: int = 50) -> list:
    """
    Собирает профили пользователей из данных для обучения.
    
    :param limit: Количество пользователей для сбора
    :return: Список профилей
    """
    print(f"📊 Сбор профилей {limit} пользователей для обучения...")
    
    loader = get_loader()
    if loader is None:
        loader = init_loader(public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg")
    
    # Загружаем пользователей
    from src.utils.user_finder import get_available_users
    user_ids = get_available_users(limit=limit, num_files=1)
    
    profiles = []
    for i, user_id in enumerate(user_ids[:limit]):
        try:
            print(f"Обработка пользователя {i+1}/{len(user_ids[:limit])}: {user_id}")
            
            # Загружаем данные пользователя
            marketplace_files = ["01082.pq"]  # Один файл для скорости
            payments_files = ["01082.pq"]
            
            marketplace_lazy = loader.load_marketplace_events(file_list=marketplace_files, days=None)
            payments_lazy = loader.load_payments_events(file_list=payments_files, days=None)
            
            if marketplace_lazy is not None:
                schema = marketplace_lazy.collect_schema()
                if "user_id" in schema:
                    user_marketplace = marketplace_lazy.filter(
                        pl.col("user_id").cast(pl.Utf8) == str(user_id)
                    ).limit(50).collect()
                else:
                    user_marketplace = pl.DataFrame()
            else:
                user_marketplace = pl.DataFrame()
            
            if payments_lazy is not None:
                schema = payments_lazy.collect_schema()
                if "user_id" in schema:
                    user_payments = payments_lazy.filter(
                        pl.col("user_id").cast(pl.Utf8) == str(user_id)
                    ).limit(30).collect()
                else:
                    user_payments = pl.DataFrame()
            else:
                user_payments = pl.DataFrame()
            
            if user_marketplace.height == 0 and user_payments.height == 0:
                continue
            
            # Создаем профиль
            user_events = {
                "marketplace": user_marketplace,
                "payments": user_payments
            }
            
            profile = create_user_profile(
                user_events=user_events,
                patterns=[],
                user_id=user_id
            )
            
            profiles.append(profile)
            
        except Exception as e:
            print(f"⚠ Ошибка при обработке пользователя {user_id}: {e}")
            continue
    
    print(f"✅ Собрано {len(profiles)} профилей")
    return profiles


def main():
    """Основная функция для обучения модели."""
    import time
    start_time = time.time()
    
    print("🚀 Начало предварительного обучения модели с YandexGPT")
    print("=" * 60)
    print("⏱️  Оценка времени: ~5-10 минут (зависит от скорости YandexGPT API)")
    print("💾 Модель будет сохранена в ./models/nbo_model.pkl (сохранится после перезапуска контейнера)")
    print("=" * 60)
    
    # Собираем профили пользователей
    profiles = collect_user_profiles(limit=20)  # Уменьшено до 20 для ускорения
    
    if len(profiles) == 0:
        print("❌ Не удалось собрать профили пользователей")
        return
    
    # Создаем и обучаем модель
    model = NBOModel()
    
    print(f"\n🤖 Обучение модели с помощью YandexGPT на {len(profiles)} профилях...")
    print("⏳ Это может занять несколько минут из-за вызовов YandexGPT API...")
    
    model.train_with_yandexgpt(profiles, use_synthetic=True)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Обучение завершено за {elapsed_time:.1f} секунд ({elapsed_time/60:.1f} минут)!")
    print(f"📁 Модель сохранена в: {model.model_path}")
    print(f"💡 Модель сохранится на хосте (./models/) и будет доступна после перезапуска контейнера")


if __name__ == "__main__":
    main()

