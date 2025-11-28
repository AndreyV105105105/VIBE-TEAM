"""
Модули для работы с данными.
"""

from .cloud_loader import YandexDiskLoader, init_loader, get_loader
from .loader import load_user_events, load_brands, load_users
from .data_parser import (
    normalize_marketplace_events,
    normalize_payments_events,
    normalize_retail_events,
    normalize_dataframe,
    detect_data_structure
)

__all__ = [
    "YandexDiskLoader",
    "init_loader",
    "get_loader",
    "load_user_events",
    "load_brands",
    "load_users",
    "normalize_marketplace_events",
    "normalize_payments_events",
    "normalize_retail_events",
    "normalize_dataframe",
    "detect_data_structure"
]

