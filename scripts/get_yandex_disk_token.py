"""
Скрипт для получения OAuth токена Яндекс Диска.

Использование:
1. Запустите скрипт: python scripts/get_yandex_disk_token.py
2. Перейдите по выведенной ссылке
3. Авторизуйтесь в Яндекс
4. Скопируйте код из URL (параметр 'code')
5. Введите код в скрипт
6. Получите токен и добавьте его в переменную окружения YANDEX_DISK_TOKEN
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.yandex_cloud import get_yandex_disk_oauth_url, exchange_code_for_token


def main():
    print("=" * 60)
    print("Получение OAuth токена для Яндекс Диска")
    print("=" * 60)
    print()
    
    # Генерируем URL для авторизации
    oauth_url = get_yandex_disk_oauth_url()
    
    print("Шаг 1: Перейдите по следующей ссылке и авторизуйтесь:")
    print()
    print(oauth_url)
    print()
    print("Шаг 2: После авторизации вы будете перенаправлены на страницу с кодом.")
    print("        Скопируйте значение параметра 'code' из URL.")
    print()
    print("Шаг 3: Введите код авторизации ниже:")
    print()
    
    # Получаем код от пользователя
    authorization_code = input("Код авторизации: ").strip()
    
    if not authorization_code:
        print("Ошибка: код не введен")
        return
    
    print()
    print("Обмен кода на токен...")
    
    # Обмениваем код на токен
    token = exchange_code_for_token(authorization_code)
    
    if token:
        print()
        print("=" * 60)
        print("✅ Токен успешно получен!")
        print("=" * 60)
        print()
        print("Добавьте этот токен в переменную окружения:")
        print()
        print(f"export YANDEX_DISK_TOKEN={token}")
        print()
        print("Или для Windows PowerShell:")
        print()
        print(f'$env:YANDEX_DISK_TOKEN="{token}"')
        print()
        print("Или добавьте в docker-compose.yml:")
        print()
        print(f'  environment:')
        print(f'    - YANDEX_DISK_TOKEN={token}')
        print()
    else:
        print()
        print("❌ Ошибка при получении токена.")
        print("Проверьте правильность кода авторизации и попробуйте снова.")


if __name__ == "__main__":
    main()

