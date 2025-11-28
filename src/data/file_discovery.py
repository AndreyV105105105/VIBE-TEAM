"""
Утилита для обнаружения файлов в публичных папках Яндекс Диска.

Для публичных папок без API токена можно использовать парсинг HTML или
передавать список файлов напрямую.
"""

import requests
from typing import List, Optional
import re
from bs4 import BeautifulSoup


def discover_files_from_public_folder(
    folder_url: str,
    folder_path: str = ""
) -> List[str]:
    """
    Пытается обнаружить файлы в публичной папке через парсинг HTML.
    
    ВНИМАНИЕ: Это не всегда работает, так как Яндекс Диск может использовать
    динамическую загрузку контента через JavaScript.
    
    :param folder_url: URL публичной папки
    :param folder_path: Путь к подпапке (например, "marketplace/events")
    :return: Список имен файлов
    """
    try:
        # Формируем URL для просмотра папки
        if folder_path:
            view_url = f"{folder_url}?path={folder_path}"
        else:
            view_url = folder_url
        
        response = requests.get(view_url, timeout=10)
        response.raise_for_status()
        
        # Парсим HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ищем ссылки на файлы .pq
        files = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Ищем файлы .pq
            if '.pq' in text or '.pq' in href:
                # Извлекаем имя файла
                match = re.search(r'([0-9]+\.pq|[a-zA-Z_]+\.pq)', text or href)
                if match:
                    files.append(match.group(1))
        
        return list(set(files))  # Убираем дубликаты
        
    except Exception as e:
        print(f"Ошибка при обнаружении файлов: {e}")
        return []


def get_known_marketplace_files() -> List[str]:
    """
    Возвращает список известных файлов маркетплейса для тестирования.
    
    На основе структуры датасета файлы называются типа 01082.pq, 01081.pq и т.д.
    Это временное решение - в реальности нужно использовать API или знать структуру.
    
    :return: Список примерных имен файлов
    """
    # Генерируем возможные имена файлов (примерный диапазон)
    # В реальности нужно знать точные имена или использовать API
    files = []
    for i in range(1000, 1200):  # Примерный диапазон
        files.append(f"{i:05d}.pq")
    return files

