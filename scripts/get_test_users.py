"""
Скрипт для получения списка тестовых пользователей из данных.

Выводит список доступных пользователей для тестирования системы.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.user_finder import get_available_users, get_users_from_users_file
from src.data.cloud_loader import init_loader


def main():
    """Получает и выводит список тестовых пользователей."""
    print("=" * 60)
    print("Получение списка тестовых пользователей")
    print("=" * 60)
    
    # Инициализируем загрузчик
    try:
        loader = init_loader(public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg")
        print("✅ Загрузчик инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации загрузчика: {e}")
        return
    
    print("\n1. Попытка загрузить пользователей из файла users.pq...")
    try:
        users_from_file = get_users_from_users_file(limit=50)
        if users_from_file:
            print(f"✅ Найдено {len(users_from_file)} пользователей в users.pq")
            print("\nПервые 20 пользователей:")
            for i, user_id in enumerate(users_from_file[:20], 1):
                print(f"  {i}. {user_id}")
            if len(users_from_file) > 20:
                print(f"  ... и еще {len(users_from_file) - 20} пользователей")
            return
        else:
            print("⚠️ Файл users.pq пуст или не найден")
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке users.pq: {e}")
    
    print("\n2. Попытка загрузить пользователей из событий...")
    try:
        users_from_events = get_available_users(limit=50, num_files=10, start_file=1082)
        if users_from_events:
            print(f"✅ Найдено {len(users_from_events)} пользователей в событиях")
            print("\nПервые 20 пользователей:")
            for i, user_id in enumerate(users_from_events[:20], 1):
                print(f"  {i}. {user_id}")
            if len(users_from_events) > 20:
                print(f"  ... и еще {len(users_from_events) - 20} пользователей")
            
            print("\n" + "=" * 60)
            print("💡 Для использования скопируйте любой ID из списка выше")
            print("=" * 60)
        else:
            print("❌ Пользователи не найдены в событиях")
            print("\nВозможные причины:")
            print("  - Файлы не загружаются из облака")
            print("  - Указанные файлы не существуют")
            print("  - В файлах нет данных")
    except Exception as e:
        print(f"❌ Ошибка при загрузке пользователей из событий: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

