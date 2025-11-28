"""
Скрипт для скачивания API ключей Yandex Cloud.

Использование:
    python scripts/download_yandex_keys.py
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.yandex_cloud import download_yandex_cloud_keys


def main() -> None:
    """Основная функция для скачивания ключей."""
    try:
        env_path = download_yandex_cloud_keys(force_download=False)
        print(f"\nКлючи успешно скачаны в файл: {env_path}")
        print("Теперь можно использовать YandexGPT API в проекте.")
    except Exception as e:
        print(f"Ошибка при скачивании ключей: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

