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
    
    # Выводим путь модели для диагностики
    print(f"\n📁 Информация о пути модели:")
    print(f"   - Путь модели: {model.model_path}")
    from pathlib import Path
    import os
    model_path_obj = Path(model.model_path)
    print(f"   - Абсолютный путь: {model_path_obj.resolve()}")
    print(f"   - Директория: {model_path_obj.parent}")
    print(f"   - Директория существует: {model_path_obj.parent.exists()}")
    if model_path_obj.parent.exists():
        print(f"   - Права на запись: {os.access(model_path_obj.parent, os.W_OK)}")
    
    print(f"\n🤖 Обучение модели с помощью YandexGPT на {len(profiles)} профилях...")
    print("⏳ Это может занять несколько минут из-за вызовов YandexGPT API...")
    
    model.train_with_yandexgpt(profiles, use_synthetic=True)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Обучение завершено за {elapsed_time:.1f} секунд ({elapsed_time/60:.1f} минут)!")
    
    # Проверяем, что модель действительно сохранена
    from pathlib import Path
    model_file = Path(model.model_path).resolve()
    print(f"\n📁 Проверка сохранения модели:")
    print(f"   - Путь в коде: {model.model_path}")
    print(f"   - Абсолютный путь: {model_file}")
    print(f"   - Файл существует: {model_file.exists()}")
    
    if model_file.exists():
        file_size = model_file.stat().st_size
        print(f"   - ✅ Размер файла: {file_size / 1024:.2f} KB")
        print(f"\n💡 Модель успешно сохранена и будет доступна на хосте в папке ./models/")
        print(f"   Volume mount: ./models -> /app/models в контейнере")
    else:
        print(f"   - ❌ ФАЙЛ НЕ НАЙДЕН!")
        print(f"\n⚠️  ПРОБЛЕМА: Модель не сохранилась. Возможные причины:")
        print(f"   1. Проблема с правами доступа к папке models")
        print(f"   2. Ошибка при сохранении (проверьте логи выше)")
        print(f"   3. Volume mount не работает (проверьте docker-compose.yml)")
        print(f"\n💡 Попробуйте:")
        print(f"   - Проверить, существует ли папка ./models на хосте")
        print(f"   - Проверить права доступа к папке")
        print(f"   - Запустить: docker-compose exec vibe-team python scripts/check_model.py")


if __name__ == "__main__":
    main()

