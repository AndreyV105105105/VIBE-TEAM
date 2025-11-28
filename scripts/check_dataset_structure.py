"""
Скрипт для проверки структуры папки с dataset в Яндекс Диске.
Помогает определить, где находится папка с данными.
"""

import os
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests

# Токен из docker-compose.yml
YANDEX_DISK_TOKEN = "y0__xDu39DhAxjR9TsgsK6AtRUwxqr-5geCNXrvPMewIJ4UjCRvWoVs8z_7KQ"

def list_folder(path: str = "/", max_depth: int = 3, current_depth: int = 0) -> None:
    """
    Рекурсивно выводит структуру папки.
    
    :param path: Путь к папке (начинается с /)
    :param max_depth: Максимальная глубина рекурсии
    :param current_depth: Текущая глубина
    """
    if current_depth >= max_depth:
        return
    
    url = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f"OAuth {YANDEX_DISK_TOKEN}"}
    params = {"path": path, "limit": 1000}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("_embedded", {}).get("items", [])
        
        indent = "  " * current_depth
        
        if current_depth == 0:
            print("=" * 60)
            print(f"Структура Яндекс Диска (путь: {path})")
            print("=" * 60)
        else:
            print(f"{indent}📁 {path}")
        
        # Сначала выводим файлы
        files = [item for item in items if item.get("type") == "file"]
        folders = [item for item in items if item.get("type") == "dir"]
        
        for file_item in files:
            name = file_item.get("name", "")
            size = file_item.get("size", 0)
            size_mb = size / (1024 * 1024) if size > 0 else 0
            print(f"{indent}  📄 {name} ({size_mb:.2f} MB)")
        
        # Затем папки
        for folder_item in folders:
            folder_name = folder_item.get("name", "")
            folder_path = folder_item.get("path", "")
            
            # Проверяем, содержит ли папка интересующие нас файлы
            if any(keyword in folder_name.lower() for keyword in ["marketplace", "payments", "retail", "dataset", "data", "users", "brands"]):
                print(f"{indent}  📁 {folder_name} ⭐ (возможно, здесь dataset)")
                # Рекурсивно просматриваем эту папку
                list_folder(folder_path, max_depth, current_depth + 1)
            else:
                print(f"{indent}  📁 {folder_name}")
                # Для других папок тоже можно просмотреть, но ограничим глубину
                if current_depth < 2:
                    list_folder(folder_path, max_depth, current_depth + 1)
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"{indent}❌ Папка {path} не найдена")
        else:
            print(f"{indent}❌ Ошибка при доступе к {path}: {e}")
    except Exception as e:
        print(f"{indent}❌ Ошибка: {e}")

def find_dataset_folder(base_path: str = "/", keywords: list = None) -> list:
    """
    Ищет папки, которые могут содержать dataset.
    
    :param base_path: Базовый путь для поиска
    :param keywords: Ключевые слова для поиска
    :return: Список путей к потенциальным папкам с dataset
    """
    if keywords is None:
        keywords = ["marketplace", "payments", "retail", "dataset", "data"]
    
    url = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f"OAuth {YANDEX_DISK_TOKEN}"}
    params = {"path": base_path, "limit": 1000}
    
    found_paths = []
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("_embedded", {}).get("items", [])
        
        for item in items:
            if item.get("type") == "dir":
                folder_name = item.get("name", "").lower()
                folder_path = item.get("path", "")
                
                # Проверяем, содержит ли название папки ключевые слова
                if any(keyword in folder_name for keyword in keywords):
                    found_paths.append(folder_path)
                    print(f"✅ Найдена потенциальная папка с dataset: {folder_path}")
                    
                    # Проверяем содержимое этой папки
                    check_folder_contents(folder_path)
        
        return found_paths
    
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        return []

def check_folder_contents(folder_path: str) -> None:
    """Проверяет содержимое папки на наличие файлов dataset."""
    url = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f"OAuth {YANDEX_DISK_TOKEN}"}
    params = {"path": folder_path, "limit": 1000}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("_embedded", {}).get("items", [])
        
        # Ищем известные файлы и папки
        found_items = []
        for item in items:
            name = item.get("name", "").lower()
            if any(keyword in name for keyword in ["marketplace", "payments", "retail", "users", "brands", ".pq", ".parquet"]):
                found_items.append(item.get("name"))
        
        if found_items:
            print(f"   📋 Найдены файлы/папки: {', '.join(found_items[:5])}")
            if len(found_items) > 5:
                print(f"   ... и еще {len(found_items) - 5} элементов")
    
    except Exception as e:
        print(f"   ❌ Ошибка при проверке содержимого: {e}")

def main():
    print("Проверка структуры Яндекс Диска для поиска dataset")
    print("=" * 60)
    print()
    
    # Сначала ищем папки с dataset
    print("🔍 Поиск папок с dataset...")
    print()
    found_paths = find_dataset_folder()
    
    print()
    print("=" * 60)
    print("📂 Полная структура корневой папки:")
    print("=" * 60)
    print()
    
    # Выводим структуру корневой папки
    list_folder("/", max_depth=3)
    
    print()
    print("=" * 60)
    if found_paths:
        print("✅ Найдены потенциальные папки с dataset:")
        for path in found_paths:
            print(f"   - {path}")
        print()
        print("💡 Если dataset находится в одной из этих папок, нужно указать")
        print("   базовый путь при инициализации загрузчика.")
    else:
        print("⚠️  Не найдено явных папок с dataset в корне.")
        print("   Возможно, файлы находятся прямо в корневой папке.")
        print("   Или dataset находится в другой папке.")
    print("=" * 60)

if __name__ == "__main__":
    main()

