"""
Константы проекта.
"""

# Продукты ПСБ
PSB_PRODUCTS = [
    "Ипотека",
    "Кредитная карта",
    "Вклад",
    "Кредит",
    "Дебетовая карта"
]

# Типы событий
EVENT_TYPES = {
    "VIEW": "V",
    "PAY": "P",
    "CLICK": "C"
}

# Домены данных
DATA_DOMAINS = [
    "marketplace",
    "payments",
    "retail",
    "offers"
]

# Параметры по умолчанию
DEFAULT_TIME_WINDOW_HOURS = 24
DEFAULT_MIN_PATTERN_LEN = 3
DEFAULT_MIN_SUPPORT = 2
DEFAULT_TOP_K = 3

# YandexGPT параметры
YANDEXGPT_TEMPERATURE = 0.3
YANDEXGPT_MAX_TOKENS = 200

# Пути к файлам
MODELS_DIR = "models"
DATA_DIR = "data"
CACHE_DIR = ".cache"

