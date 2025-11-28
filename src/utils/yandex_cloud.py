"""
Утилита для получения и работы с Yandex Cloud API ключами.

Использует прямые значения folder_id и api_key.
"""

from typing import Dict, Optional
import requests
import base64

# Yandex Cloud API ключи
YANDEX_CLOUD_FOLDER_ID = "b1gst3c7cskk2big5fqn"
YANDEX_CLOUD_API_KEY = "AQVNxQ_-mwN1bNst5oDEaWiRvm5cSFOvq_MzLoIz"

YANDEX_CLOUD_CONSOLE_URL = "https://console.cloud.yandex.ru/"

# Яндекс Диск OAuth credentials
YANDEX_DISK_CLIENT_ID = "c9e5ffe9548744f4bae6767417d6f5ce"
YANDEX_DISK_CLIENT_SECRET = "dd033946a6d14099a18c1a40498628e8"
YANDEX_DISK_REDIRECT_URI = "https://oauth.yandex.ru/verification_code"


def get_yandex_cloud_config(
    env_path: Optional[str] = None,
    auto_download: bool = False
) -> Dict[str, str]:
    """
    Получает конфигурацию Yandex Cloud (folder_id и api_key).
    
    Использует прямые значения из констант.
    
    :param env_path: Не используется (оставлен для совместимости)
    :param auto_download: Не используется (оставлен для совместимости)
    :return: Словарь с конфигурацией
    """
    return {
        "folder_id": YANDEX_CLOUD_FOLDER_ID,
        "api_key": YANDEX_CLOUD_API_KEY
    }


# Глобальный кэш конфигурации
_cached_config: Dict[str, str] = {
    "folder_id": YANDEX_CLOUD_FOLDER_ID,
    "api_key": YANDEX_CLOUD_API_KEY
}


def get_cached_config(
    env_path: Optional[str] = None,
    auto_download: bool = False
) -> Dict[str, str]:
    """
    Получает конфигурацию с кэшированием.
    
    Использует прямые значения из констант.
    
    :param env_path: Не используется (оставлен для совместимости)
    :param auto_download: Не используется (оставлен для совместимости)
    :return: Словарь с конфигурацией
    """
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
    """
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

