"""
Модуль для загрузки данных напрямую из Яндекс Диска без скачивания.

Поддерживает:
- Прямое чтение Parquet файлов через HTTP/HTTPS (polars)
- Работу с Яндекс Диск API
- Кэширование для ускорения повторных запросов
"""

import os
import re
from typing import Optional, List, Dict
from pathlib import Path
import polars as pl
import requests
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.yandex_cloud import (
    YANDEX_DISK_CLIENT_ID,
    YANDEX_DISK_CLIENT_SECRET,
    YANDEX_DISK_REDIRECT_URI
)


class YandexDiskLoader:
    """
    Загрузчик данных с Яндекс Диска.
    
    Поддерживает два режима:
    1. Публичные ссылки (если папка публичная)
    2. API с токеном (для приватных папок)
    """
    
    def __init__(
        self,
        public_link: Optional[str] = None,
        api_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        base_path: Optional[str] = None,
        prefer_cache: bool = False
    ):
        """
        Инициализация загрузчика.
        
        :param public_link: Публичная ссылка на папку Яндекс Диска
                           Пример: "https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
        :param api_token: Токен Яндекс Диск API (опционально)
        :param cache_dir: Директория для кэширования (опционально)
        :param base_path: Базовый путь к папке с dataset (для API с токеном)
                         Пример: "/Загрузки/Dataset_case_1" или "Загрузки/Dataset_case_1"
                         Если не указан, используется корень диска
        :param prefer_cache: Если True, система будет использовать кэш как основной источник
                           и загружать из облака только если файла нет в кэше
        """
        self.public_link = public_link
        self.api_token = api_token or os.getenv("YANDEX_DISK_TOKEN")
        self.cache_dir = cache_dir or ".cache"
        self.prefer_cache = prefer_cache
        
        # Базовый путь к dataset (нормализуем: убираем disk:, добавляем / в начало если нужно)
        if base_path:
            # Убираем префикс disk: если есть
            base_path = base_path.replace("disk:", "").strip()
            # Убираем начальный / если есть (API работает и с ним, и без)
            if base_path.startswith("/"):
                base_path = base_path[1:]
            self.base_path = base_path
        else:
            # Пробуем получить из переменной окружения
            env_base_path = os.getenv("YANDEX_DISK_BASE_PATH")
            if env_base_path:
                env_base_path = env_base_path.replace("disk:", "").strip()
                if env_base_path.startswith("/"):
                    env_base_path = env_base_path[1:]
                self.base_path = env_base_path
            else:
                self.base_path = None
        
        # Базовый URL для публичных ссылок
        if public_link:
            # Извлекаем ID папки из ссылки
            match = re.search(r'/d/([a-zA-Z0-9_-]+)', public_link)
            if match:
                self.folder_id = match.group(1)
                self.base_url = f"https://disk.yandex.ru/d/{self.folder_id}"
            else:
                raise ValueError("Неверный формат публичной ссылки")
        
        # Если токен не указан, пытаемся получить его через OAuth
        if not self.api_token:
            # Пробуем получить токен из переменной окружения или использовать OAuth
            # Для автоматического получения токена нужна авторизация пользователя
            # Пока используем публичный доступ, но с предупреждением
            print("⚠ ВНИМАНИЕ: API токен Яндекс Диска не указан.")
            print("   Для публичных папок Яндекс Диск может блокировать автоматические запросы")
            print("   и возвращать HTML-страницы вместо файлов (капча, ограничения доступа).")
            print("   OAuth credentials настроены, но для получения токена требуется авторизация пользователя.")
            print("   Для автоматической работы рекомендуется получить токен вручную на https://oauth.yandex.ru/")
            print("   и добавить его через переменную окружения YANDEX_DISK_TOKEN.")
    
    def _get_download_link(self, file_path: str) -> str:
        """
        Получает прямую ссылку на скачивание файла.
        
        :param file_path: Путь к файлу относительно корня папки
        :return: Прямая ссылка на скачивание
        """
        if self.api_token:
            # Используем API для получения прямой ссылки
            return self._get_api_download_link(file_path)
        else:
            # Для публичных папок Яндекс Диска используем правильный формат
            import urllib.parse
            
            # Правильный формат для публичных папок: нужно использовать /download с правильным кодированием
            # Путь должен быть закодирован как один параметр
            if "/" in file_path:
                # Для путей с / кодируем весь путь целиком
                encoded_path = urllib.parse.quote(file_path, safe='')
            else:
                encoded_path = urllib.parse.quote(file_path, safe='')
            
            # Используем формат для публичных папок
            # Вариант 1: стандартный формат (может требовать авторизацию)
            download_url = f"https://disk.yandex.ru/d/{self.folder_id}/download?path={encoded_path}"
            
            # Альтернативный вариант: прямой доступ через публичную ссылку
            # Но это не всегда работает для подпапок
            
            return download_url
    
    def _get_api_download_link(self, file_path: str) -> str:
        """
        Получает прямую ссылку через Яндекс Диск API.
        
        :param file_path: Путь к файлу (относительно базового пути или корня)
        :return: Прямая ссылка на скачивание
        """
        # Формируем полный путь с учетом базового пути
        full_path = self._get_full_path(file_path)
        
        url = "https://cloud-api.yandex.net/v1/disk/resources/download"
        headers = {"Authorization": f"OAuth {self.api_token}"}
        params = {"path": full_path}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()["href"]
    
    def _get_full_path(self, relative_path: str) -> str:
        """
        Формирует полный путь к файлу с учетом базового пути.
        
        :param relative_path: Относительный путь (например, "marketplace/events/01082.pq")
        :return: Полный путь (например, "/Загрузки/Dataset_case_1/marketplace/events/01082.pq")
        """
        if self.base_path:
            # Убираем начальный / из relative_path если есть
            if relative_path.startswith("/"):
                relative_path = relative_path[1:]
            # Объединяем базовый путь и относительный путь
            full_path = f"/{self.base_path}/{relative_path}"
            # Убираем двойные слеши
            full_path = full_path.replace("//", "/")
            return full_path
        else:
            # Если базовый путь не указан, используем relative_path как есть
            if not relative_path.startswith("/"):
                relative_path = f"/{relative_path}"
            return relative_path
    
    def list_files(self, folder_path: str = "") -> List[Dict[str, str]]:
        """
        Получает список файлов в папке.
        
        :param folder_path: Путь к папке (относительно корня)
        :return: Список файлов с метаданными
        """
        if self.api_token:
            return self._list_files_api(folder_path)
        else:
            # Для публичных папок без API токена мы не можем получить список файлов
            # Возвращаем пустой список - нужно использовать file_list параметр
            return []
    
    def _list_files_api(self, folder_path: str) -> List[Dict[str, str]]:
        """
        Список файлов через API.
        
        :param folder_path: Путь к папке (относительно базового пути)
        """
        # Формируем полный путь с учетом базового пути
        full_path = self._get_full_path(folder_path)
        
        url = "https://cloud-api.yandex.net/v1/disk/resources"
        headers = {"Authorization": f"OAuth {self.api_token}"}
        params = {"path": full_path, "limit": 1000}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        items = response.json().get("_embedded", {}).get("items", [])
        return [
            {
                "name": item["name"],
                "path": item["path"],
                "type": item["type"],
                "size": item.get("size", 0)
            }
            for item in items
        ]
    
    def read_parquet_from_url(
        self,
        file_path: str,
        use_cache: bool = True,
        normalize: bool = True
    ) -> pl.DataFrame:
        """
        Читает Parquet файл напрямую из Яндекс Диска.
        
        Для публичных папок Яндекс Диска используется формат:
        https://disk.yandex.ru/d/{folder_id}/download?path={encoded_path}
        
        :param file_path: Путь к файлу относительно корня папки
        :param use_cache: Использовать ли кэш
        :param normalize: Нормализовать ли данные (привести к единому формату)
        :return: DataFrame с данными
        """
        # Получаем прямую ссылку на скачивание
        download_url = self._get_download_link(file_path)
        
        # Яндекс Диск не поддерживает range requests, поэтому всегда скачиваем через временный файл
        # Это гарантирует полную загрузку файла
        try:
            df = self._read_with_temp_file(download_url, file_path, use_cache)
        except Exception as e:
            # Если не удалось загрузить, возвращаем пустой DataFrame
            print(f"Не удалось загрузить файл {file_path}: {e}")
            return pl.DataFrame()
        
        # Нормализуем данные если нужно
        if normalize:
            from src.data.data_parser import normalize_dataframe, detect_data_structure
            
            # Определяем домен из пути
            domain = "unknown"
            if "marketplace" in file_path:
                domain = "marketplace"
            elif "payments" in file_path:
                domain = "payments"
            elif "retail" in file_path:
                domain = "retail"
            else:
                # Пытаемся определить автоматически
                structure = detect_data_structure(df)
                domain = structure.get("type", "unknown")
            
            # Нормализуем
            if domain != "unknown":
                df = normalize_dataframe(df, domain, file_path)
        
        return df
    
    def _read_with_temp_file(
        self,
        download_url: str,
        file_path: str,
        use_cache: bool
    ) -> pl.DataFrame:
        """
        Читает файл через временный файл с кэшированием.
        
        Оптимизированная версия для быстрой загрузки из кэша.
        
        ЛОГИКА РАБОТЫ С ДАТАМИ:
        - Файлы называются типа 01082.pq, 01083.pq - это номера дней (day numbers)
        - Каждый файл может содержать данные за один или несколько дней
        - Фильтрация по датам происходит ПОСЛЕ загрузки по колонке timestamp
        - Параметр days=5 означает: загрузить файлы, затем отфильтровать события за последние 5 дней
        """
        cache_path = Path(self.cache_dir) / file_path.replace("/", "_")
        
        # Если prefer_cache=True, используем кэш как основной источник
        # и не загружаем из облака, если файл есть в кэше
        if self.prefer_cache and cache_path.exists():
            try:
                file_size = cache_path.stat().st_size
                if file_size >= 8:
                    with open(cache_path, "rb") as f:
                        first_4_bytes = f.read(4)
                    if first_4_bytes == b"PAR1":
                        # Файл валиден, используем из кэша без обращения к облаку
                        df = pl.read_parquet(cache_path)
                        return df
            except Exception as e:
                print(f"⚠ Ошибка при чтении из кэша {file_path}: {e}, пробуем загрузить из облака")
        
        # Проверяем кэш (оптимизированная версия)
        if use_cache and cache_path.exists():
            try:
                file_size = cache_path.stat().st_size
                # Быстрая проверка: только первые и последние байты для кэшированных файлов
                if file_size >= 8:
                    with open(cache_path, "rb") as f:
                        first_4_bytes = f.read(4)
                        f.seek(-4, 2)
                        last_4_bytes = f.read(4)
                    
                    if first_4_bytes == b"PAR1" and last_4_bytes == b"PAR1":
                        # Файл валиден, читаем
                        try:
                            df = pl.read_parquet(cache_path)
                            return df
                        except Exception as e:
                            print(f"⚠ Ошибка при чтении parquet {file_path} (несмотря на PAR1): {e}")
                            # Файл поврежден, удаляем
                            cache_path.unlink()
                    else:
                        # Файл поврежден (неполный), удаляем и перезагружаем
                        print(f"⚠ Файл {file_path} поврежден (нет PAR1 в начале или конце), удаляем...")
                        cache_path.unlink()
                else:
                    # Файл слишком маленький
                    cache_path.unlink()
            except Exception as e:
                # Если ошибка при чтении кэша, удаляем поврежденный файл
                try:
                    cache_path.unlink()
                except:
                    pass
                # Продолжаем скачивание
        
        # Скачиваем файл полностью
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Скачивание файла {file_path} из {download_url}...")
        
        # Используем сессию для лучшего контроля
        import time
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': f'https://disk.yandex.ru/d/{self.folder_id}'
        })
        
        try:
            # Добавляем небольшую задержку, чтобы избежать капчи
            time.sleep(0.5)
            
            # Увеличиваем таймаут для больших файлов (users.pq может быть ~100MB)
            response = session.get(download_url, stream=True, timeout=300, allow_redirects=True)
            
            # Проверяем, что это не HTML страница (капча или ошибка)
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type or 'application/xhtml' in content_type:
                # Читаем первые байты для проверки
                first_chunk = next(response.iter_content(chunk_size=1024), b'')
                first_chunk_lower = first_chunk.lower()
                if b'<html' in first_chunk_lower or b'captcha' in first_chunk_lower or b'forbidden' in first_chunk_lower or b'<!doctype' in first_chunk_lower:
                    # Пробуем альтернативный формат URL
                    print(f"⚠ Яндекс Диск вернул HTML. Пробуем альтернативный формат URL для {file_path}...")
                    # Альтернативный формат: используем прямой доступ через публичную ссылку
                    # Для файлов в подпапках это может не работать без API токена
                    raise ValueError(f"Яндекс Диск вернул HTML вместо файла (возможно, требуется капча или файл недоступен). Content-Type: {content_type}. Для публичных папок рекомендуется использовать Яндекс Диск API с токеном.")
            
            response.raise_for_status()
            
            # Проверяем размер файла из заголовков и URL
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                # Пробуем получить размер из URL параметров (если есть)
                import re
                size_match = re.search(r'fsize=(\d+)', download_url)
                if size_match:
                    total_size = int(size_match.group(1))
                    print(f"Размер файла из URL: {total_size} байт ({total_size / 1024 / 1024:.2f} MB)")
                else:
                    print(f"Предупреждение: размер файла {file_path} неизвестен")
            
            # Скачиваем файл полностью с проверкой прогресса и повторными попытками
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    downloaded_size = 0
                    # Увеличиваем размер чанка для больших файлов
                    chunk_size = 65536 if total_size > 10 * 1024 * 1024 else 8192  # 64KB для больших файлов
                    
                    with open(cache_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                # Показываем прогресс для больших файлов
                                if total_size > 0 and downloaded_size % (10 * 1024 * 1024) == 0:
                                    progress = (downloaded_size / total_size) * 100
                                    print(f"  Прогресс: {downloaded_size / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({progress:.1f}%)")
                    
                    # Проверяем размер скачанного файла
                    file_size = cache_path.stat().st_size
                    
                    # Проверяем, что файл скачан полностью
                    if total_size > 0:
                        if file_size < total_size:
                            print(f"⚠ Файл скачан не полностью: {file_size}/{total_size} байт ({file_size / total_size * 100:.1f}%). Попытка {retry_count + 1}/{max_retries}")
                            if retry_count < max_retries - 1:
                                # Удаляем неполный файл и пробуем снова
                                cache_path.unlink()
                                retry_count += 1
                                time.sleep(3)  # Задержка перед повтором
                                # Переоткрываем соединение с увеличенным таймаутом
                                response = session.get(download_url, stream=True, timeout=300, allow_redirects=True)
                                response.raise_for_status()
                                continue
                            else:
                                raise ValueError(f"Файл скачан не полностью после {max_retries} попыток: {file_size}/{total_size} байт")
                    
                    # Проверяем минимальный размер
                    if file_size < 4:
                        raise ValueError(f"Файл слишком маленький: {file_size} байт")
                    
                    # Проверяем, что это не HTML файл
                    with open(cache_path, "rb") as f:
                        first_bytes = f.read(min(1024, file_size))
                        if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                            raise ValueError(f"Скачанный файл является HTML страницей, а не Parquet файлом")
                    
                    # Проверяем сигнатуру Parquet (должен начинаться И заканчиваться на PAR1)
                    with open(cache_path, "rb") as f:
                        first_4_bytes = f.read(4)
                        if file_size >= 8:
                            f.seek(-4, 2)  # Переходим к концу файла
                            last_4_bytes = f.read(4)
                        else:
                            last_4_bytes = b""
                    
                    # Parquet файл должен начинаться И заканчиваться на PAR1
                    if first_4_bytes != b"PAR1":
                        raise ValueError(f"Файл не является валидным Parquet файлом (не начинается с PAR1). Первые байты: {first_4_bytes.hex()}")
                    
                    if file_size >= 8 and last_4_bytes != b"PAR1":
                        raise ValueError(f"Файл не является валидным Parquet файлом (не заканчивается на PAR1). Последние байты: {last_4_bytes.hex()}, размер: {file_size} байт. Возможно, файл скачан не полностью.")
                    
                    print(f"✅ Файл {file_path} успешно скачан ({file_size} байт, {file_size / 1024 / 1024:.2f} MB, проверка PAR1 пройдена)")
                    
                    # Читаем из кэша
                    return pl.read_parquet(cache_path)
                    
                except Exception as e:
                    if retry_count < max_retries - 1:
                        print(f"⚠ Ошибка при скачивании, повторная попытка {retry_count + 1}/{max_retries}: {e}")
                        if cache_path.exists():
                            cache_path.unlink()
                        retry_count += 1
                        time.sleep(3)
                        # Переоткрываем соединение
                        response = session.get(download_url, stream=True, timeout=300, allow_redirects=True)
                        response.raise_for_status()
                        continue
                    else:
                        raise
            
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if '403' in error_msg or 'captcha' in error_msg.lower():
                print(f"Ошибка 403 (капча) при скачивании {file_path}. Попробуйте позже или используйте API токен.")
            else:
                print(f"Ошибка при скачивании {file_path}: {e}")
            if cache_path.exists():
                cache_path.unlink()  # Удаляем неполный файл
            raise
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
            if cache_path.exists():
                cache_path.unlink()  # Удаляем поврежденный файл
            raise
    
    def load_marketplace_events(
        self,
        file_list: Optional[List[str]] = None,
        limit: Optional[int] = None,
        days: Optional[int] = None
    ) -> pl.LazyFrame:
        """
        Загружает события маркетплейса.
        
        ВАЖНО: Для публичных папок без API токена необходимо передавать
        конкретный список файлов через параметр file_list.
        
        :param file_list: Список конкретных имен файлов для загрузки
                         Например: ["01082.pq", "01081.pq", "01080.pq"]
                         Если не указан и нет API токена, будет ошибка
        :param limit: Ограничение количества файлов (если file_list не указан)
        :param days: Фильтровать данные за последние N дней (опционально, по умолчанию все данные)
        :return: LazyFrame со всеми событиями
        """
        # Если передан список файлов, используем его
        if file_list:
            events_files = [{"name": f, "type": "file"} for f in file_list]
        else:
            # Получаем список файлов через API (только если есть токен)
            if not self.api_token:
                raise ValueError(
                    "Для публичных папок без API токена необходимо указать file_list "
                    "с конкретными именами файлов. Например:\n"
                    "loader.load_marketplace_events(file_list=['01082.pq', '01081.pq'])"
                )
            
            events_files = self.list_files("marketplace/events")
            
            # Ограничиваем количество (для тестирования)
            if limit:
                events_files = events_files[:limit]
        
        # Загружаем файлы с оптимизацией для кэша
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        frames = []
        cache_path = Path(self.cache_dir)
        
        # Проверяем, какие файлы уже в кэше
        cached_files = {}
        for file_info in events_files:
            file_path = f"marketplace/events/{file_info['name']}"
            cache_file_path = cache_path / file_path.replace("/", "_")
            if cache_file_path.exists():
                cached_files[file_info['name']] = cache_file_path
        
        # Если все файлы в кэше, загружаем параллельно
        if len(cached_files) == len(events_files) and len(events_files) > 1:
            # Параллельная загрузка из кэша
            def load_cached_file(file_info):
                file_path = f"marketplace/events/{file_info['name']}"
                try:
                    df = self.read_parquet_from_url(file_path, normalize=True, use_cache=True)
                    if df.height > 0 and "user_id" in df.columns:
                        return (file_info['name'], df)
                except Exception as e:
                    print(f"⚠ Ошибка при загрузке {file_info['name']} из кэша: {e}")
                return None
            
            # Оптимизация: увеличиваем количество потоков для параллельной загрузки
            max_workers = min(8, len(events_files), os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(load_cached_file, file_info): file_info for file_info in events_files}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        frames.append(result[1])
                        print(f"✅ Загружен из кэша {result[0]}: {frames[-1].height} строк")
        else:
            # Последовательная загрузка (если есть файлы не в кэше)
            for idx, file_info in enumerate(events_files):
                file_path = f"marketplace/events/{file_info['name']}"
                try:
                    # Задержка только для файлов не из кэша
                    if file_info['name'] not in cached_files and idx > 0:
                        time.sleep(0.5)
                    
                    df = self.read_parquet_from_url(file_path, normalize=True)
                    # Проверяем, что DataFrame не пустой и содержит данные
                    if df.height > 0:
                        # Проверяем наличие колонки user_id
                        if "user_id" in df.columns:
                            frames.append(df)
                            print(f"✅ Загружен {file_info['name']}: {df.height} строк")
                        else:
                            print(f"⚠ Файл {file_info['name']} не содержит колонку 'user_id'")
                    else:
                        print(f"⚠ Файл {file_info['name']} пустой")
                        
                    # Если загрузили достаточно данных, можно остановиться
                    if len(frames) >= 1 and limit and limit <= 1:
                        break
                        
                except Exception as e:
                    error_str = str(e)
                    if '403' in error_str or 'captcha' in error_str.lower() or 'forbidden' in error_str.lower():
                        print(f"⚠ Ошибка 403 при загрузке {file_info['name']}. Пропускаем.")
                        time.sleep(2.0)
                    elif 'HTML' in error_str or 'html' in error_str:
                        print(f"⚠ HTML вместо файла {file_info['name']}. Пропускаем.")
                        time.sleep(1.0)
                    else:
                        print(f"❌ Ошибка при загрузке {file_info['name']}: {e}")
                    continue
        
        if not frames:
            # Возвращаем пустой LazyFrame с правильной схемой вместо ошибки
            print("⚠ Не удалось загрузить ни один файл. Возвращаем пустой DataFrame.")
            return pl.DataFrame({
                "user_id": pl.Utf8,
                "item_id": pl.Utf8,
                "category_id": pl.Utf8,
                "timestamp": pl.Datetime,
                "domain": pl.Utf8
            }).lazy()
        
        # Объединяем в LazyFrame
        combined = pl.concat(frames).lazy()
        
        # Фильтруем по дате, если указано количество дней
        if days and days > 0:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            # Проверяем наличие колонки timestamp
            schema = combined.collect_schema()
            if "timestamp" in schema:
                timestamp_dtype = schema["timestamp"]
                
                # Duration нельзя сравнивать с Datetime напрямую - пропускаем фильтрацию
                if timestamp_dtype == pl.Duration:
                    print(f"⚠ Timestamp в формате Duration, пропускаем фильтрацию по дате (Duration нельзя сравнить с Datetime)")
                    return combined
                
                # Пробуем преобразовать в Datetime только если это не Duration
                if timestamp_dtype != pl.Datetime:
                    try:
                        # Пробуем преобразовать через строку, если это строка
                        combined = combined.with_columns(
                            pl.col("timestamp").cast(pl.Datetime, strict=False)
                        )
                        # Проверяем, что преобразование прошло успешно
                        new_schema = combined.collect_schema()
                        if new_schema.get("timestamp") != pl.Datetime:
                            print(f"⚠ Не удалось преобразовать timestamp в Datetime (тип: {timestamp_dtype}), пропускаем фильтрацию")
                            return combined
                    except Exception as e:
                        # Если не удалось преобразовать, пропускаем фильтрацию
                        print(f"⚠ Не удалось преобразовать timestamp в Datetime: {e}, пропускаем фильтрацию по дате")
                        return combined
                
                # Используем pl.lit для правильного сравнения
                try:
                    combined = combined.filter(pl.col("timestamp") >= pl.lit(cutoff_date))
                    print(f"📅 Фильтрация marketplace: загружены данные за последние {days} дней (с {cutoff_date.date()})")
                except Exception as e:
                    print(f"⚠ Ошибка при фильтрации по дате: {e}, пропускаем фильтрацию")
                    return combined
        
        return combined
    
    def load_payments_events(
        self,
        file_list: Optional[List[str]] = None,
        limit: Optional[int] = None,
        days: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> pl.LazyFrame:
        """
        Загружает события платежей.
        
        ОПТИМИЗАЦИЯ: Если указан user_id и файлы в кэше, использует predicate pushdown
        для фильтрации ДО загрузки всех данных в память.
        
        :param file_list: Список конкретных имен файлов для загрузки
        :param limit: Ограничение количества файлов
        :param days: Фильтровать данные за последние N дней (опционально)
        :param user_id: ID пользователя для фильтрации (опционально, для оптимизации)
        :return: LazyFrame со всеми событиями
        """
        # Если передан список файлов, используем его
        if file_list:
            events_files = [{"name": f, "type": "file"} for f in file_list]
        else:
            # Получаем список файлов через API (только если есть токен)
            if not self.api_token:
                # Для публичных папок без API возвращаем пустой DataFrame
                return pl.DataFrame().lazy()
            
            events_files = self.list_files("payments/events")
            
            if limit:
                events_files = events_files[:limit]
        
        # Загружаем файлы с оптимизацией для кэша
        import time
        from src.data.data_parser import normalize_dataframe
        frames = []
        lazy_frames = []
        cache_path = Path(self.cache_dir)
        
        # Проверяем, какие файлы уже в кэше
        cached_files = {}
        for file_info in events_files:
            file_path = f"payments/events/{file_info['name']}"
            cache_file_path = cache_path / file_path.replace("/", "_")
            if cache_file_path.exists():
                cached_files[file_info['name']] = cache_file_path
        
        # ОПТИМИЗАЦИЯ: Если указан user_id и все файлы в кэше, используем LazyFrame с predicate pushdown
        use_lazy_optimization = user_id and len(cached_files) == len(events_files) and len(events_files) > 0
        
        if use_lazy_optimization:
            # Используем LazyFrame для predicate pushdown - фильтруем ДО загрузки
            print(f"⚡ Используем predicate pushdown для user_id={user_id} (фильтрация ДО загрузки)")
            for file_info in events_files:
                file_path = f"payments/events/{file_info['name']}"
                cache_file_path = cached_files.get(file_info['name'])
                if cache_file_path and cache_file_path.exists():
                    try:
                        # Читаем как LazyFrame для predicate pushdown
                        lazy_df = pl.scan_parquet(str(cache_file_path))
                        
                        # Нормализуем колонки на уровне LazyFrame (базовая нормализация)
                        schema = lazy_df.collect_schema()
                        
                        # Переименовываем колонки если нужно
                        rename_dict = {}
                        if "price" in schema and "amount" not in schema:
                            rename_dict["price"] = "amount"
                        if "user_id" not in schema:
                            # Пропускаем файлы без user_id
                            continue
                        
                        if rename_dict:
                            lazy_df = lazy_df.rename(rename_dict)
                        
                        # Добавляем domain если его нет
                        if "domain" not in lazy_df.collect_schema():
                            lazy_df = lazy_df.with_columns(pl.lit("payments").alias("domain"))
                        
                        # ПРИМЕНЯЕМ ФИЛЬТР ДО collect() - это и есть predicate pushdown!
                        # Polars оптимизирует это и читает только нужные строки из Parquet
                        lazy_df = lazy_df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                        
                        lazy_frames.append(lazy_df)
                        print(f"   ✅ Добавлен LazyFrame для {file_info['name']} с фильтром по user_id (predicate pushdown)")
                    except Exception as e:
                        print(f"⚠ Ошибка при создании LazyFrame для {file_info['name']}: {e}")
                        # Fallback: загружаем как обычно
                        try:
                            df = self.read_parquet_from_url(file_path, normalize=True, use_cache=True)
                            if df.height > 0 and "user_id" in df.columns:
                                # Фильтруем после загрузки (медленнее, но работает)
                                df = df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                                if df.height > 0:
                                    frames.append(df)
                        except Exception as e2:
                            print(f"⚠ Ошибка при fallback загрузке {file_info['name']}: {e2}")
            
            # Объединяем LazyFrames
            if lazy_frames:
                combined = pl.concat(lazy_frames)
                # Применяем нормализацию на уровне LazyFrame
                # (нормализация будет применена при collect())
                if days and days > 0:
                    from datetime import datetime, timedelta
                    cutoff_date = datetime.now() - timedelta(days=days)
                    schema = combined.collect_schema()
                    if "timestamp" in schema and schema["timestamp"] == pl.Datetime:
                        combined = combined.filter(pl.col("timestamp") >= pl.lit(cutoff_date))
                        print(f"📅 Фильтрация payments: загружены данные за последние {days} дней (с {cutoff_date.date()})")
                return combined
        else:
            # Стандартная загрузка (если нет user_id или файлы не все в кэше)
            # Если все файлы в кэше, загружаем параллельно
            if len(cached_files) == len(events_files) and len(events_files) > 1:
                # Параллельная загрузка из кэша
                def load_cached_file(file_info):
                    file_path = f"payments/events/{file_info['name']}"
                    try:
                        df = self.read_parquet_from_url(file_path, normalize=True, use_cache=True)
                        if df.height > 0 and "user_id" in df.columns:
                            return (file_info['name'], df)
                    except Exception as e:
                        print(f"⚠ Ошибка при загрузке {file_info['name']} из кэша: {e}")
                    return None
                
                # Оптимизация: увеличиваем количество потоков для параллельной загрузки
                max_workers = min(8, len(events_files), os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(load_cached_file, file_info): file_info for file_info in events_files}
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            frames.append(result[1])
            else:
                # Последовательная загрузка (если есть файлы не в кэше)
                for idx, file_info in enumerate(events_files):
                    file_path = f"payments/events/{file_info['name']}"
                    try:
                        # Задержка только для файлов не из кэша
                        if file_info['name'] not in cached_files and idx > 0:
                            time.sleep(0.5)
                        
                        df = self.read_parquet_from_url(file_path, normalize=True)
                        # Проверяем, что DataFrame не пустой
                        if df.height > 0:
                            # Проверяем наличие колонки user_id
                            if "user_id" in df.columns:
                                # Диагностика: проверяем наличие amount/price
                                if "amount" in df.columns:
                                    amount_sample = df.select(pl.col("amount")).head(3).to_series().to_list()
                                    print(f"   ✅ Загружено {df.height} строк, amount: {amount_sample}")
                                elif "price" in df.columns:
                                    print(f"   ⚠ Файл содержит 'price' вместо 'amount'. Колонки: {df.columns}")
                                frames.append(df)
                            else:
                                print(f"   ⚠ Файл {file_path} не содержит колонку 'user_id'. Колонки: {df.columns}")
                        else:
                            print(f"   ⚠ Файл {file_path} пустой")
                    except Exception as e:
                        print(f"Ошибка при загрузке {file_path}: {e}")
                        continue
        
        if not frames and not lazy_frames:
            # Возвращаем пустой LazyFrame с правильной схемой
            return pl.DataFrame({
                "user_id": pl.Utf8,
                "brand_id": pl.Utf8,
                "amount": pl.Float64,
                "timestamp": pl.Datetime,
                "domain": pl.Utf8
            }).lazy()
        
        # Объединяем в LazyFrame
        if frames:
            combined = pl.concat(frames).lazy()
        else:
            combined = pl.concat(lazy_frames)
        
        # Фильтруем по дате, если указано количество дней
        if days and days > 0:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            # Проверяем наличие колонки timestamp
            schema = combined.collect_schema()
            if "timestamp" in schema:
                timestamp_dtype = schema["timestamp"]
                
                # Duration нельзя сравнивать с Datetime напрямую - пропускаем фильтрацию
                if timestamp_dtype == pl.Duration:
                    print(f"⚠ Timestamp в формате Duration, пропускаем фильтрацию по дате (Duration нельзя сравнить с Datetime)")
                    return combined
                
                # Пробуем преобразовать в Datetime только если это не Duration
                if timestamp_dtype != pl.Datetime:
                    try:
                        # Пробуем преобразовать через строку, если это строка
                        combined = combined.with_columns(
                            pl.col("timestamp").cast(pl.Datetime, strict=False)
                        )
                        # Проверяем, что преобразование прошло успешно
                        new_schema = combined.collect_schema()
                        if new_schema.get("timestamp") != pl.Datetime:
                            print(f"⚠ Не удалось преобразовать timestamp в Datetime (тип: {timestamp_dtype}), пропускаем фильтрацию")
                            return combined
                    except Exception as e:
                        # Если не удалось преобразовать, пропускаем фильтрацию
                        print(f"⚠ Не удалось преобразовать timestamp в Datetime: {e}, пропускаем фильтрацию по дате")
                        return combined
                
                # Используем pl.lit для правильного сравнения
                try:
                    combined = combined.filter(pl.col("timestamp") >= pl.lit(cutoff_date))
                    print(f"📅 Фильтрация payments: загружены данные за последние {days} дней (с {cutoff_date.date()})")
                except Exception as e:
                    print(f"⚠ Ошибка при фильтрации по дате: {e}, пропускаем фильтрацию")
                    return combined
        
        return combined
    
    def load_brands(self) -> pl.DataFrame:
        """Загружает справочник брендов."""
        return self.read_parquet_from_url("brands.pq")
    
    def load_marketplace_items(
        self,
        brand_ids: Optional[List[str]] = None,
        item_ids: Optional[List[str]] = None,
        use_lazy: bool = True,
        include_embedding: bool = False
    ) -> pl.LazyFrame:
        """
        Загружает каталог товаров маркетплейса с оптимизацией.
        
        :param brand_ids: Список brand_id для фильтрации (predicate pushdown) - экономит память
        :param item_ids: Список item_id для фильтрации (predicate pushdown) - экономит память
        :param use_lazy: Использовать LazyFrame для отложенной загрузки
        :param include_embedding: Загружать ли embedding (только если нужен, т.к. занимает много места)
        :return: LazyFrame или DataFrame с товарами
        """
        try:
            # Используем projection pushdown - загружаем только нужные колонки
            # ВАЖНО: используем только стандартные названия колонок из спецификации Yandex Cloud Data Set
            # Согласно спецификации Yandex Cloud Data Set:
            # - item_id: str (обязательно)
            # - brand_id: u64 (опционально)
            # - category: str (название категории, опционально, может быть null)
            # - category_id: ID категории (опционально)
            # - subcategory: str (подкатегория, опционально, может быть null)
            # - price: f64 (цена как число с плавающей точкой, опционально, может быть null)
            needed_cols = ["item_id"]  # item_id обязателен
            optional_cols = ["brand_id", "category", "category_id", "subcategory", "price"]  # Стандартные колонки из спецификации
            if include_embedding:
                optional_cols.append("embedding")  # Добавляем embedding только если нужен
            
            # Пробуем загрузить как LazyFrame для оптимизации
            cache_path = Path(self.cache_dir)
            cache_file = cache_path / "marketplace_items.pq"
            
            if cache_file.exists():
                # Загружаем из кэша с projection pushdown
                lazy_df = pl.scan_parquet(str(cache_file))
                
                # Проверяем, какие колонки доступны
                schema = lazy_df.collect_schema()
                
                # Проверяем обязательные колонки
                if "item_id" not in schema:
                    print(f"⚠ В marketplace/items.pq нет обязательной колонки item_id")
                    print(f"   Доступные колонки: {list(schema.keys())}")
                    return pl.DataFrame().lazy()
                
                # Собираем доступные колонки (обязательные + опциональные)
                available_cols = ["item_id"]  # item_id всегда есть
                for col in optional_cols:
                    if col in schema:
                        available_cols.append(col)
                
                # Projection pushdown: выбираем только нужные колонки
                lazy_df = lazy_df.select(available_cols)
                
                # Predicate pushdown: фильтруем по brand_id и item_id ДО загрузки
                # ВАЖНО: проверяем наличие колонки в available_cols (после select)
                if brand_ids and "brand_id" in available_cols:
                    try:
                        brand_ids_str = [str(bid) for bid in brand_ids]
                        lazy_df = lazy_df.filter(pl.col("brand_id").cast(pl.Utf8).is_in(brand_ids_str))
                        print(f"⚡ Применен predicate pushdown: фильтрация по {len(brand_ids)} брендам ДО загрузки")
                    except Exception as e:
                        print(f"⚠ Ошибка фильтрации по brand_id: {e}. Пропускаем фильтрацию по brand_id.")
                elif brand_ids:
                    print(f"⚠ brand_id не найден в marketplace/items.pq. Доступные колонки: {available_cols}. Пропускаем фильтрацию по brand_id.")
                
                if item_ids and "item_id" in available_cols:
                    try:
                        item_ids_str = [str(iid) for iid in item_ids]
                        lazy_df = lazy_df.filter(pl.col("item_id").cast(pl.Utf8).is_in(item_ids_str))
                        print(f"⚡ Применен predicate pushdown: фильтрация по {len(item_ids)} товарам ДО загрузки")
                    except Exception as e:
                        print(f"⚠ Ошибка фильтрации по item_id: {e}. Пропускаем фильтрацию по item_id.")
                elif item_ids:
                    print(f"⚠ item_id не найден в marketplace/items.pq. Доступные колонки: {available_cols}. Пропускаем фильтрацию по item_id.")
                
                if use_lazy:
                    return lazy_df
                else:
                    return lazy_df.collect()
            else:
                # Загружаем из облака (только если нет в кэше)
                print(f"⚠ marketplace/items.pq не в кэше. Рекомендуется закэшировать файл для оптимизации.")
                df = self.read_parquet_from_url("marketplace/items.pq", normalize=False)
                
                # Проверяем обязательные колонки
                if "item_id" not in df.columns:
                    print(f"⚠ В marketplace/items.pq нет обязательной колонки item_id")
                    print(f"   Доступные колонки: {list(df.columns)}")
                    return pl.DataFrame().lazy() if use_lazy else pl.DataFrame()
                
                # Собираем доступные колонки (обязательные + опциональные)
                available_cols = ["item_id"]  # item_id всегда есть
                optional_cols = ["brand_id", "category", "category_id", "subcategory", "price"]
                if include_embedding:
                    optional_cols.append("embedding")
                for col in optional_cols:
                    if col in df.columns:
                        available_cols.append(col)
                
                if available_cols:
                    df = df.select(available_cols)
                    
                    # Фильтруем по brand_id и item_id если указаны
                    # ВАЖНО: проверяем наличие колонки в df.columns (после select)
                    if brand_ids and "brand_id" in df.columns:
                        try:
                            brand_ids_str = [str(bid) for bid in brand_ids]
                            df = df.filter(pl.col("brand_id").cast(pl.Utf8).is_in(brand_ids_str))
                            print(f"⚡ Отфильтровано по {len(brand_ids)} брендам")
                        except Exception as e:
                            print(f"⚠ Ошибка фильтрации по brand_id: {e}. Пропускаем фильтрацию по brand_id.")
                    elif brand_ids:
                        print(f"⚠ brand_id не найден в marketplace/items.pq. Доступные колонки: {list(df.columns)}. Пропускаем фильтрацию по brand_id.")
                    
                    if item_ids and "item_id" in df.columns:
                        try:
                            item_ids_str = [str(iid) for iid in item_ids]
                            df = df.filter(pl.col("item_id").cast(pl.Utf8).is_in(item_ids_str))
                            print(f"⚡ Отфильтровано по {len(item_ids)} товарам")
                        except Exception as e:
                            print(f"⚠ Ошибка фильтрации по item_id: {e}. Пропускаем фильтрацию по item_id.")
                    elif item_ids:
                        print(f"⚠ item_id не найден в marketplace/items.pq. Доступные колонки: {list(df.columns)}. Пропускаем фильтрацию по item_id.")
                
                return df.lazy() if use_lazy else df
                
        except Exception as e:
            print(f"⚠ Ошибка при загрузке marketplace/items.pq: {e}")
            return pl.DataFrame().lazy()
    
    def load_retail_items(
        self,
        brand_ids: Optional[List[str]] = None,
        item_ids: Optional[List[str]] = None,
        use_lazy: bool = True,
        include_embedding: bool = False
    ) -> pl.LazyFrame:
        """
        Загружает каталог товаров ритейла с оптимизацией.
        
        :param brand_ids: Список brand_id для фильтрации (predicate pushdown) - экономит память
        :param item_ids: Список item_id для фильтрации (predicate pushdown) - экономит память
        :param use_lazy: Использовать LazyFrame для отложенной загрузки
        :param include_embedding: Загружать ли embedding (только если нужен, т.к. занимает много места)
        :return: LazyFrame или DataFrame с товарами
        """
        try:
            # Используем projection pushdown - загружаем только нужные колонки
            # Согласно спецификации Yandex Cloud Data Set для retail/items.pq:
            # - item_id: str (обязательно)
            # - brand_id: u64 (опционально)
            # - category: str (название категории, опционально, может быть null)
            # - subcategory: str (подкатегория, опционально, может быть null)
            # - price: f64 (цена как число с плавающей точкой, опционально, может быть null или отрицательным)
            # - embedding: array[f32, 300] (опционально)
            # ПРИМЕЧАНИЕ: В retail/items.pq НЕТ category_id (только в marketplace/items.pq)
            needed_cols = ["item_id"]  # item_id обязателен
            optional_cols = ["brand_id", "category", "subcategory", "price"]  # Опциональные колонки
            if include_embedding:
                optional_cols.append("embedding")  # Добавляем embedding только если нужен
            
            # Пробуем загрузить как LazyFrame для оптимизации
            cache_path = Path(self.cache_dir)
            cache_file = cache_path / "retail_items.pq"
            
            if cache_file.exists():
                # Загружаем из кэша с projection pushdown
                lazy_df = pl.scan_parquet(str(cache_file))
                
                # Проверяем, какие колонки доступны
                schema = lazy_df.collect_schema()
                
                # Собираем доступные колонки (обязательные + опциональные)
                available_cols = ["item_id"]  # item_id всегда есть
                for col in optional_cols:
                    if col in schema:
                        available_cols.append(col)
                
                if "item_id" not in schema:
                    print(f"⚠ В retail/items.pq нет обязательной колонки item_id")
                    print(f"   Доступные колонки: {list(schema.keys())}")
                    return pl.DataFrame().lazy()
                
                # Projection pushdown: выбираем только нужные колонки
                lazy_df = lazy_df.select(available_cols)
                
                # Predicate pushdown: фильтруем по brand_id и item_id ДО загрузки
                # ВАЖНО: проверяем наличие колонки в available_cols (после select)
                if brand_ids and "brand_id" in available_cols:
                    try:
                        brand_ids_str = [str(bid) for bid in brand_ids]
                        lazy_df = lazy_df.filter(pl.col("brand_id").cast(pl.Utf8).is_in(brand_ids_str))
                        print(f"⚡ Применен predicate pushdown: фильтрация по {len(brand_ids)} брендам ДО загрузки")
                    except Exception as e:
                        print(f"⚠ Ошибка фильтрации по brand_id: {e}. Пропускаем фильтрацию по brand_id.")
                elif brand_ids:
                    print(f"⚠ brand_id не найден в retail/items.pq. Доступные колонки: {available_cols}. Пропускаем фильтрацию по brand_id.")
                
                if item_ids and "item_id" in available_cols:
                    try:
                        item_ids_str = [str(iid) for iid in item_ids]
                        lazy_df = lazy_df.filter(pl.col("item_id").cast(pl.Utf8).is_in(item_ids_str))
                        print(f"⚡ Применен predicate pushdown: фильтрация по {len(item_ids)} товарам ДО загрузки")
                    except Exception as e:
                        print(f"⚠ Ошибка фильтрации по item_id: {e}. Пропускаем фильтрацию по item_id.")
                elif item_ids:
                    print(f"⚠ item_id не найден в retail/items.pq. Доступные колонки: {available_cols}. Пропускаем фильтрацию по item_id.")
                
                if use_lazy:
                    return lazy_df
                else:
                    return lazy_df.collect()
            else:
                # Загружаем из облака (только если нет в кэше)
                print(f"⚠ retail/items.pq не в кэше. Рекомендуется закэшировать файл для оптимизации.")
                df = self.read_parquet_from_url("retail/items.pq", normalize=False)
                
                # Проверяем обязательные колонки
                if "item_id" not in df.columns:
                    print(f"⚠ В retail/items.pq нет обязательной колонки item_id")
                    print(f"   Доступные колонки: {list(df.columns)}")
                    return pl.DataFrame().lazy() if use_lazy else pl.DataFrame()
                
                # Собираем доступные колонки (обязательные + опциональные)
                # Согласно спецификации Yandex Cloud Data Set для retail/items.pq:
                # - category: str (название категории как строка)
                # - subcategory: str (подкатегория как строка)
                # - price: f64 (цена как число с плавающей точкой, может быть null или отрицательным)
                # ПРИМЕЧАНИЕ: В retail/items.pq НЕТ category_id (только в marketplace/items.pq)
                available_cols = ["item_id"]  # item_id всегда есть
                optional_cols = ["brand_id", "category", "subcategory", "price"]
                if include_embedding:
                    optional_cols.append("embedding")
                for col in optional_cols:
                    if col in df.columns:
                        available_cols.append(col)
                
                if available_cols:
                    df = df.select(available_cols)
                    
                    # Фильтруем по brand_id и item_id если указаны
                    # ВАЖНО: проверяем наличие колонки в df.columns (после select)
                    if brand_ids and "brand_id" in df.columns:
                        try:
                            brand_ids_str = [str(bid) for bid in brand_ids]
                            df = df.filter(pl.col("brand_id").cast(pl.Utf8).is_in(brand_ids_str))
                            print(f"⚡ Отфильтровано по {len(brand_ids)} брендам")
                        except Exception as e:
                            print(f"⚠ Ошибка фильтрации по brand_id: {e}. Пропускаем фильтрацию по brand_id.")
                    elif brand_ids:
                        print(f"⚠ brand_id не найден в retail/items.pq. Доступные колонки: {list(df.columns)}. Пропускаем фильтрацию по brand_id.")
                    
                    if item_ids and "item_id" in df.columns:
                        try:
                            item_ids_str = [str(iid) for iid in item_ids]
                            df = df.filter(pl.col("item_id").cast(pl.Utf8).is_in(item_ids_str))
                            print(f"⚡ Отфильтровано по {len(item_ids)} товарам")
                        except Exception as e:
                            print(f"⚠ Ошибка фильтрации по item_id: {e}. Пропускаем фильтрацию по item_id.")
                    elif item_ids:
                        print(f"⚠ item_id не найден в retail/items.pq. Доступные колонки: {list(df.columns)}. Пропускаем фильтрацию по item_id.")
                
                return df.lazy() if use_lazy else df
                
        except Exception as e:
            print(f"⚠ Ошибка при загрузке retail/items.pq: {e}")
            return pl.DataFrame().lazy()
    
    def load_payments_items(
        self,
        brand_ids: Optional[List[str]] = None,
        item_ids: Optional[List[str]] = None,
        use_lazy: bool = True,
        include_embedding: bool = False
    ) -> pl.LazyFrame:
        """
        Загружает каталог товаров payments с оптимизацией.
        
        :param brand_ids: Список brand_id для фильтрации (predicate pushdown) - экономит память
        :param item_ids: Список item_id для фильтрации (predicate pushdown) - экономит память
        :param use_lazy: Использовать LazyFrame для отложенной загрузки
        :param include_embedding: Загружать ли embedding (только если нужен, т.к. занимает много места)
        :return: LazyFrame или DataFrame с товарами
        """
        try:
            # Используем projection pushdown - загружаем только нужные колонки
            # Согласно спецификации Yandex Cloud Data Set для payments/items.pq:
            # - item_id: str (обязательно) - может быть approximate_item_id
            # - brand_id: u64 (опционально)
            # - category: str (название категории, опционально, может быть null)
            # - category_id: ID категории (опционально)
            # - subcategory: str (подкатегория, опционально, может быть null)
            # - price: f64 (цена как число с плавающей точкой, опционально, может быть null)
            needed_cols = ["item_id"]  # item_id обязателен
            optional_cols = ["brand_id", "category", "category_id", "subcategory", "price"]  # Опциональные колонки
            if include_embedding:
                optional_cols.append("embedding")  # Добавляем embedding только если нужен
            
            # Пробуем загрузить как LazyFrame для оптимизации
            cache_path = Path(self.cache_dir)
            cache_file = cache_path / "payments_items.pq"
            
            if cache_file.exists():
                # Загружаем из кэша с projection pushdown
                lazy_df = pl.scan_parquet(str(cache_file))
                
                # Проверяем, какие колонки доступны
                schema = lazy_df.collect_schema()
                
                # Собираем доступные колонки (обязательные + опциональные)
                available_cols = ["item_id"]  # item_id всегда есть
                for col in optional_cols:
                    if col in schema:
                        available_cols.append(col)
                
                if "item_id" not in schema:
                    print(f"⚠ В payments/items.pq нет обязательной колонки item_id")
                    print(f"   Доступные колонки: {list(schema.keys())}")
                    return pl.DataFrame().lazy()
                
                # Projection pushdown: выбираем только нужные колонки
                lazy_df = lazy_df.select(available_cols)
                
                # Predicate pushdown: фильтруем по brand_id и item_id ДО загрузки
                # ВАЖНО: проверяем наличие колонки в available_cols (после select)
                if brand_ids and "brand_id" in available_cols:
                    try:
                        brand_ids_str = [str(bid) for bid in brand_ids]
                        lazy_df = lazy_df.filter(pl.col("brand_id").cast(pl.Utf8).is_in(brand_ids_str))
                        print(f"⚡ Применен predicate pushdown: фильтрация по {len(brand_ids)} брендам ДО загрузки")
                    except Exception as e:
                        print(f"⚠ Ошибка фильтрации по brand_id: {e}. Пропускаем фильтрацию по brand_id.")
                elif brand_ids:
                    print(f"⚠ brand_id не найден в payments/items.pq. Доступные колонки: {available_cols}. Пропускаем фильтрацию по brand_id.")
                
                if item_ids and "item_id" in available_cols:
                    try:
                        item_ids_str = [str(iid) for iid in item_ids]
                        lazy_df = lazy_df.filter(pl.col("item_id").cast(pl.Utf8).is_in(item_ids_str))
                        print(f"⚡ Применен predicate pushdown: фильтрация по {len(item_ids)} товарам ДО загрузки")
                    except Exception as e:
                        print(f"⚠ Ошибка фильтрации по item_id: {e}. Пропускаем фильтрацию по item_id.")
                elif item_ids:
                    print(f"⚠ item_id не найден в payments/items.pq. Доступные колонки: {available_cols}. Пропускаем фильтрацию по item_id.")
                
                if use_lazy:
                    return lazy_df
                else:
                    return lazy_df.collect()
            else:
                # Загружаем из облака (только если нет в кэше)
                print(f"⚠ payments/items.pq не в кэше. Рекомендуется закэшировать файл для оптимизации.")
                df = self.read_parquet_from_url("payments/items.pq", normalize=False)
                
                # Проверяем обязательные колонки
                if "item_id" not in df.columns:
                    print(f"⚠ В payments/items.pq нет обязательной колонки item_id")
                    print(f"   Доступные колонки: {list(df.columns)}")
                    return pl.DataFrame().lazy() if use_lazy else pl.DataFrame()
                
                # Собираем доступные колонки (обязательные + опциональные)
                available_cols = ["item_id"]  # item_id всегда есть
                optional_cols = ["brand_id", "category", "category_id", "subcategory", "price"]
                if include_embedding:
                    optional_cols.append("embedding")
                for col in optional_cols:
                    if col in df.columns:
                        available_cols.append(col)
                
                if available_cols:
                    df = df.select(available_cols)
                    
                    # Фильтруем по brand_id и item_id если указаны
                    # ВАЖНО: проверяем наличие колонки в df.columns (после select)
                    if brand_ids and "brand_id" in df.columns:
                        try:
                            brand_ids_str = [str(bid) for bid in brand_ids]
                            df = df.filter(pl.col("brand_id").cast(pl.Utf8).is_in(brand_ids_str))
                            print(f"⚡ Отфильтровано по {len(brand_ids)} брендам")
                        except Exception as e:
                            print(f"⚠ Ошибка фильтрации по brand_id: {e}. Пропускаем фильтрацию по brand_id.")
                    elif brand_ids:
                        print(f"⚠ brand_id не найден в payments/items.pq. Доступные колонки: {list(df.columns)}. Пропускаем фильтрацию по brand_id.")
                    
                    if item_ids and "item_id" in df.columns:
                        try:
                            item_ids_str = [str(iid) for iid in item_ids]
                            df = df.filter(pl.col("item_id").cast(pl.Utf8).is_in(item_ids_str))
                            print(f"⚡ Отфильтровано по {len(item_ids)} товарам")
                        except Exception as e:
                            print(f"⚠ Ошибка фильтрации по item_id: {e}. Пропускаем фильтрацию по item_id.")
                    elif item_ids:
                        print(f"⚠ item_id не найден в payments/items.pq. Доступные колонки: {list(df.columns)}. Пропускаем фильтрацию по item_id.")
                
                return df.lazy() if use_lazy else df
                
        except Exception as e:
            print(f"⚠ Ошибка при загрузке payments/items.pq: {e}")
            return pl.DataFrame().lazy()
    
    def load_payments_receipts(
        self,
        file_list: Optional[List[str]] = None,
        limit: Optional[int] = None,
        days: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> pl.LazyFrame:
        """
        Загружает чеки из payments/receipts с детализацией товаров.
        
        :param file_list: Список конкретных имен файлов для загрузки
        :param limit: Ограничение количества файлов
        :param days: Количество дней для загрузки
        :param user_id: ID пользователя для фильтрации (predicate pushdown)
        :return: LazyFrame с чеками
        """
        # Если передан список файлов, используем его
        if file_list:
            events_files = [{"name": f, "type": "file"} for f in file_list]
        else:
            # Получаем список файлов через API (только если есть токен)
            if not self.api_token:
                return pl.DataFrame().lazy()
            
            events_files = self.list_files("payments/receipts")
            
            if limit:
                events_files = events_files[:limit]
        
        # Если все файлы в кэше И передан user_id, используем predicate pushdown
        cache_path = Path(self.cache_dir)
        cached_files = [f for f in events_files if (cache_path / f"payments_receipts_{f['name']}").exists()]
        
        if user_id and len(cached_files) == len(events_files) and len(events_files) > 0:
            print(f"⚡ Используем predicate pushdown для receipts user_id={user_id}")
            lazy_frames = []
            for file_info in events_files:
                file_path = f"payments/receipts/{file_info['name']}"
                cache_file_path = cache_path / file_path.replace("/", "_")
                try:
                    lazy_df = pl.scan_parquet(str(cache_file_path))
                    schema = lazy_df.collect_schema()
                    if "user_id" in schema:
                        lazy_df = lazy_df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                        lazy_frames.append(lazy_df)
                except Exception as e:
                    print(f"⚠ Ошибка при создании LazyFrame для receipts {file_info['name']}: {e}")
            
            if lazy_frames:
                combined = pl.concat(lazy_frames)
                if days and days > 0:
                    from datetime import datetime, timedelta
                    cutoff_date = datetime.now() - timedelta(days=days)
                    schema = combined.collect_schema()
                    if "timestamp" in schema and schema["timestamp"] == pl.Datetime:
                        combined = combined.filter(pl.col("timestamp") >= pl.lit(cutoff_date))
                return combined
        
        # Стандартная загрузка
        frames = []
        for file_info in events_files:
            file_path = f"payments/receipts/{file_info['name']}"
            try:
                df = self.read_parquet_from_url(file_path, normalize=False)
                if df.height > 0 and "user_id" in df.columns:
                    if user_id:
                        df = df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                    if df.height > 0:
                        frames.append(df)
            except Exception as e:
                print(f"⚠ Ошибка при загрузке {file_path}: {e}")
                continue
        
        if not frames:
            return pl.DataFrame().lazy()
        
        combined = pl.concat(frames).lazy()
        
        if days and days > 0:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            schema = combined.collect_schema()
            if "timestamp" in schema:
                if schema["timestamp"] == pl.Datetime:
                    combined = combined.filter(pl.col("timestamp") >= pl.lit(cutoff_date))
        
        return combined
    
    def load_users(self) -> pl.DataFrame:
        """
        Загружает справочник пользователей.
        
        Пробует загрузить из разных возможных путей:
        - users.pq (в корне)
        - users/users.pq
        - data/users.pq
        """
        # Пробуем разные возможные пути к файлу users.pq
        possible_paths = [
            "users.pq",  # В корне папки
            "users/users.pq",  # В подпапке users
            "data/users.pq",  # В подпапке data
            "users.pq",  # Еще раз для надежности
        ]
        
        for path in possible_paths:
            try:
                print(f"Попытка загрузить users.pq из пути: {path}")
                df = self.read_parquet_from_url(path, normalize=False)
                
                # Проверяем, что файл не пустой
                if df.height > 0:
                    print(f"Успешно загружен users.pq из {path}, строк: {df.height}")
                    # Проверяем наличие колонки user_id
                    if "user_id" in df.columns:
                        return df
                    else:
                        # Пробуем найти альтернативные названия
                        for alt_name in ["user", "userId", "userid", "uid", "client_id"]:
                            if alt_name in df.columns:
                                print(f"Переименовываем {alt_name} в user_id")
                                df = df.rename({alt_name: "user_id"})
                                return df
                        print(f"Предупреждение: файл {path} не содержит колонку user_id. Колонки: {df.columns}")
                else:
                    print(f"Файл {path} пустой")
            except Exception as e:
                print(f"Не удалось загрузить {path}: {e}")
                continue
        
        # Если ничего не получилось, возвращаем пустой DataFrame
        print("Не удалось загрузить users.pq ни из одного пути")
        return pl.DataFrame()
    
    def load_retail_events(
        self,
        file_list: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pl.LazyFrame:
        """
        Загружает события ритейла.
        
        :param file_list: Список конкретных имен файлов для загрузки
        :param limit: Ограничение количества файлов
        :return: LazyFrame со всеми событиями
        """
        # Если передан список файлов, используем его
        if file_list:
            events_files = [{"name": f, "type": "file"} for f in file_list]
        else:
            # Получаем список файлов через API (только если есть токен)
            if not self.api_token:
                return pl.DataFrame().lazy()
            
            events_files = self.list_files("retail/events")
            
            if limit:
                events_files = events_files[:limit]
        
        frames = []
        for file_info in events_files:
            file_path = f"retail/events/{file_info['name']}"
            try:
                # Нормализуем данные автоматически
                df = self.read_parquet_from_url(file_path, normalize=True)
                frames.append(df)
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {e}")
                continue
        
        if not frames:
            return pl.DataFrame().lazy()
        
        return pl.concat(frames).lazy()


# Глобальный экземпляр загрузчика (можно переопределить)
_loader: Optional[YandexDiskLoader] = None


def init_loader(
    public_link: Optional[str] = None,
    api_token: Optional[str] = None,
    base_path: Optional[str] = None,
    prefer_cache: bool = False
) -> YandexDiskLoader:
    """
    Инициализирует глобальный загрузчик.
    
    :param public_link: Публичная ссылка на Яндекс Диск
    :param api_token: Токен API (опционально)
    :param base_path: Базовый путь к папке с dataset (для API с токеном)
                     Пример: "/Загрузки/Dataset_case_1"
    :param prefer_cache: Если True, система будет использовать кэш как основной источник
                        и загружать из облака только если файла нет в кэше
    :return: Экземпляр загрузчика
    """
    global _loader
    _loader = YandexDiskLoader(
        public_link=public_link or os.getenv("YANDEX_DISK_PUBLIC_LINK"),
        api_token=api_token,
        base_path=base_path,
        prefer_cache=prefer_cache or os.getenv("PREFER_CACHE", "false").lower() == "true"
    )
    return _loader


def get_loader() -> Optional[YandexDiskLoader]:
    """Получает глобальный загрузчик."""
    return _loader
