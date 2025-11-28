"""
Простой скрипт для обмена кода авторизации на токен Яндекс Диска.
Не требует зависимостей проекта.
"""

import requests
import base64
import urllib.parse

# OAuth credentials
CLIENT_ID = "c9e5ffe9548744f4bae6767417d6f5ce"
CLIENT_SECRET = "dd033946a6d14099a18c1a40498628e8"
REDIRECT_URI = "https://oauth.yandex.ru/verification_code"

# Код авторизации
AUTHORIZATION_CODE = "2aivby57q7rabyum"

def exchange_code_for_token(authorization_code: str):
    """Обменивает код авторизации на OAuth токен."""
    url = "https://oauth.yandex.ru/token"
    
    # Используем Basic Auth с ClientID и ClientSecret
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": REDIRECT_URI
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        return result.get("access_token")
    except Exception as e:
        print(f"Ошибка при получении токена: {e}")
        if hasattr(response, 'text'):
            print(f"Ответ сервера: {response.text}")
        return None

def main():
    print("=" * 60)
    print("Обмен кода авторизации на токен Яндекс Диска")
    print("=" * 60)
    print()
    print(f"Код авторизации: {AUTHORIZATION_CODE}")
    print()
    print("Отправка запроса...")
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

