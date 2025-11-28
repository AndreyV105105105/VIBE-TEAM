"""
Быстрый скрипт для обмена кода авторизации на токен Яндекс Диска.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.yandex_cloud import exchange_code_for_token

# Код авторизации
AUTHORIZATION_CODE = "2aivby57q7rabyum"

def main():
    print("Обмен кода авторизации на токен...")
    print(f"Код: {AUTHORIZATION_CODE}")
    print()
    
    token = exchange_code_for_token(AUTHORIZATION_CODE)
    
    if token:
        print("=" * 60)
        print("✅ ТОКЕН УСПЕШНО ПОЛУЧЕН!")
        print("=" * 60)
        print()
        print("Токен:")
        print(token)
        print()
        print("=" * 60)
        print("Добавьте этот токен в переменную окружения:")
        print()
        print(f"export YANDEX_DISK_TOKEN={token}")
        print()
        print("Или для Windows PowerShell:")
        print()
        print(f'$env:YANDEX_DISK_TOKEN="{token}"')
        print()
        print("Или добавьте в docker-compose.yml в секцию environment:")
        print()
        print(f'    - YANDEX_DISK_TOKEN={token}')
        print()
        print("=" * 60)
    else:
        print("❌ Ошибка при получении токена.")
        print("Проверьте правильность кода авторизации.")

if __name__ == "__main__":
    main()

