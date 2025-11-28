"""
Утилита для получения и работы с Yandex Cloud API ключами.

Читает ключи из переменных окружения (.env файл).
ВАЖНО: Создайте .env файл на основе .env.example и добавьте туда свои ключи.
"""

from typing import Dict, Optional
import os
import requests
import base64
from pathlib import Path

# Пытаемся загрузить переменные окружения из .env файла
try:
    from dotenv import load_dotenv
    # Загружаем .env из корня проекта
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Пробуем загрузить из текущей директории
        load_dotenv()
except ImportError:
    # python-dotenv не установлен, используем только переменные окружения системы
    pass

# Yandex Cloud API ключи (читаем из переменных окружения)
# ВАЖНО: Ключи должны быть в .env файле (см. .env.example)
YANDEX_CLOUD_FOLDER_ID = os.getenv("YANDEX_CLOUD_FOLDER_ID")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")

YANDEX_CLOUD_CONSOLE_URL = "https://console.cloud.yandex.ru/"

# Яндекс Диск OAuth credentials (читаем из переменных окружения)
# ВАЖНО: Ключи должны быть в .env файле (см. .env.example)
YANDEX_DISK_CLIENT_ID = os.getenv("YANDEX_DISK_CLIENT_ID")
YANDEX_DISK_CLIENT_SECRET = os.getenv("YANDEX_DISK_CLIENT_SECRET")
YANDEX_DISK_REDIRECT_URI = os.getenv("YANDEX_DISK_REDIRECT_URI", "https://oauth.yandex.ru/verification_code")


def get_yandex_cloud_config(
    env_path: Optional[str] = None,
    auto_download: bool = False
) -> Dict[str, str]:
    """
    Получает конфигурацию Yandex Cloud (folder_id и api_key).
    
    Читает из переменных окружения (.env файл).
    
    :param env_path: Не используется (оставлен для совместимости)
    :param auto_download: Не используется (оставлен для совместимости)
    :return: Словарь с конфигурацией
    :raises ValueError: Если ключи не найдены в переменных окружения
    """
    if not YANDEX_CLOUD_FOLDER_ID or not YANDEX_CLOUD_API_KEY:
        raise ValueError(
            "YANDEX_CLOUD_FOLDER_ID и YANDEX_CLOUD_API_KEY должны быть установлены в переменных окружения. "
            "Создайте .env файл на основе .env.example"
        )
    return {
        "folder_id": YANDEX_CLOUD_FOLDER_ID,
        "api_key": YANDEX_CLOUD_API_KEY
    }


# Глобальный кэш конфигурации (инициализируется при первом вызове)
_cached_config: Optional[Dict[str, str]] = None


def get_cached_config(
    env_path: Optional[str] = None,
    auto_download: bool = False
) -> Dict[str, str]:
    """
    Получает конфигурацию с кэшированием.
    
    Читает из переменных окружения (.env файл) и кэширует результат.
    
    :param env_path: Не используется (оставлен для совместимости)
    :param auto_download: Не используется (оставлен для совместимости)
    :return: Словарь с конфигурацией
    :raises ValueError: Если ключи не найдены в переменных окружения
    """
    global _cached_config
    if _cached_config is None:
        _cached_config = get_yandex_cloud_config()
    return _cached_config


def get_yandex_disk_oauth_url() -> str:
    """
    Генерирует URL для авторизации Яндекс Диска через OAuth.
    
    Пользователь должен перейти по этой ссылке, авторизоваться,
    и получить код авторизации, который затем можно обменять на токен.
    
    :return: URL для авторизации
    """
    params = {
        "response_type": "code",
        "client_id": YANDEX_DISK_CLIENT_ID,
        "redirect_uri": YANDEX_DISK_REDIRECT_URI
    }
    
    import urllib.parse
    query_string = urllib.parse.urlencode(params)
    return f"https://oauth.yandex.ru/authorize?{query_string}"


def exchange_code_for_token(authorization_code: str) -> Optional[str]:
    """
    Обменивает код авторизации на OAuth токен для Яндекс Диска.
    
    :param authorization_code: Код авторизации, полученный после перехода по OAuth URL
    :return: OAuth токен или None в случае ошибки
    :raises ValueError: Если YANDEX_DISK_CLIENT_ID или YANDEX_DISK_CLIENT_SECRET не установлены
    """
    if not YANDEX_DISK_CLIENT_ID or not YANDEX_DISK_CLIENT_SECRET:
        raise ValueError(
            "YANDEX_DISK_CLIENT_ID и YANDEX_DISK_CLIENT_SECRET должны быть установлены в переменных окружения. "
            "Создайте .env файл на основе .env.example"
        )
    url = "https://oauth.yandex.ru/token"
    
    # Используем Basic Auth с ClientID и ClientSecret
    credentials = f"{YANDEX_DISK_CLIENT_ID}:{YANDEX_DISK_CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": YANDEX_DISK_REDIRECT_URI
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        return result.get("access_token")
    except Exception as e:
        print(f"Ошибка при получении токена: {e}")
        return None

