"""
Модуль для извлечения паттернов поведения из событий.

Использует простые алгоритмы без внешних библиотек для поиска частых последовательностей.
"""

from collections import Counter
from typing import List, Tuple, Dict
import polars as pl


def extract_sequences(events: pl.DataFrame) -> List[str]:
    """
    Извлекает последовательность событий из DataFrame.
    
    :param events: DataFrame с событиями пользователя
    :return: Список кодированных событий (V=view, P=pay, C=click)
    """
    seq = []
    
    # Сортируем по времени
    if "timestamp" in events.columns:
        events_sorted = events.sort("timestamp")
    else:
        events_sorted = events
    
    for row in events_sorted.iter_rows(named=True):
        # Получаем domain из колонки или определяем по структуре данных
        domain = row.get("domain", "")
        
        # Если domain нет в колонках, пытаемся определить по наличию других колонок
        if not domain:
            if "item_id" in events.columns or "category_id" in events.columns:
                domain = "marketplace"
            elif "brand_id" in events.columns or "amount" in events.columns:
                domain = "payments"
            elif "event_type" in events.columns:
                event_type = row.get("event_type", "")
                if event_type == "click":
                    domain = "offers"
        
        # Кодируем события с учетом action_type и domain
        action_type = row.get("action_type", "")
        
        if domain == "marketplace" or domain == "retail":
            # Детальная кодировка для marketplace и retail
            if action_type == "view":
                seq.append("V")  # View
            elif action_type == "click":
                seq.append("C")  # Click
            elif action_type == "add_to_cart":
                seq.append("A")  # Add to cart
            elif action_type == "order":
                seq.append("O")  # Order
            else:
                seq.append("V")  # По умолчанию view
        elif domain == "payments":
            seq.append("P")  # Pay/Transaction
        elif domain == "receipts":
            seq.append("R")  # Receipt (детализированная покупка)
        elif domain == "offers":
            if action_type == "click":
                seq.append("C")  # Click
            elif action_type == "impression":
                seq.append("I")  # Impression
            else:
                seq.append("V")  # По умолчанию view
        else:
            seq.append("?")  # Unknown domain
    
    return seq


def find_frequent_patterns(
    sequences: List[List[str]],
    min_pattern_len: int = 3,
    min_support: int = 2
) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Находит частые паттерны в последовательностях.
    
    :param sequences: Список последовательностей событий
    :param min_pattern_len: Минимальная длина паттерна
    :param min_support: Минимальная поддержка (количество вхождений)
    :return: Список кортежей (паттерн, частота)
    """
    all_patterns = []
    
    for seq in sequences:
        # Генерируем все подпоследовательности заданной длины
        for length in range(min_pattern_len, min(len(seq) + 1, 6)):  # Максимум 5
            for i in range(len(seq) - length + 1):
                pattern = tuple(seq[i:i+length])
                all_patterns.append(pattern)
    
    # Подсчитываем частоту
    pattern_counts = Counter(all_patterns)
    
    # Возвращаем частые паттерны
    frequent = [
        (pattern, count)
        for pattern, count in pattern_counts.items()
        if count >= min_support
    ]
    
    # Сортируем по частоте
    frequent.sort(key=lambda x: x[1], reverse=True)
    
    return frequent


def extract_patterns(
    user_events: Dict[str, pl.DataFrame],
    min_pattern_len: int = 3,
    min_support: int = 2
) -> List[Tuple[str, ...]]:
    """
    Извлекает паттерны из событий пользователя.
    
    :param user_events: Словарь с событиями по доменам
    :param min_pattern_len: Минимальная длина паттерна
    :param min_support: Минимальная поддержка
    :return: Список паттернов
    """
    # Объединяем все события, приводя к общей схеме
    # Для извлечения паттернов нужны только timestamp и domain
    normalized_events = []
    
    for domain_name, df in user_events.items():
        if df.height == 0:
            continue
        
        # Проверяем наличие timestamp (обязательно)
        if "timestamp" not in df.columns:
            print(f"⚠ Предупреждение: домен {domain_name} не содержит timestamp, пропускаем")
            continue
        
        # Добавляем domain если его нет
        if "domain" not in df.columns:
            df = df.with_columns(pl.lit(domain_name).alias("domain"))
        
        # Выбираем только timestamp и domain для объединения
        # Это гарантирует одинаковую схему для всех доменов
        df_normalized = df.select(["timestamp", "domain"])
        normalized_events.append(df_normalized)
    
    if not normalized_events:
        return []
    
    # Объединяем в один DataFrame (теперь все имеют одинаковую схему)
    try:
        combined = pl.concat(normalized_events)
    except Exception as e:
        print(f"❌ Ошибка при объединении событий: {e}")
        return []
    
    # Извлекаем последовательность
    sequence = extract_sequences(combined)
    
    if len(sequence) < min_pattern_len:
        return []
    
    # Находим частые паттерны
    frequent_patterns = find_frequent_patterns(
        [sequence],
        min_pattern_len=min_pattern_len,
        min_support=min_support
    )
    
    # Возвращаем только паттерны (без частоты)
    return [pattern for pattern, _ in frequent_patterns]


def pattern_to_string(pattern: Tuple[str, ...]) -> str:
    """
    Преобразует паттерн в строку для отображения.
    
    :param pattern: Паттерн как кортеж
    :return: Строковое представление
    """
    return "→".join(pattern)


def get_pattern_statistics(patterns: List[Tuple[str, ...]]) -> Dict:
    """
    Получает статистику по паттернам.
    
    :param patterns: Список паттернов
    :return: Словарь со статистикой
    """
    if not patterns:
        return {
            "total_patterns": 0,
            "avg_length": 0,
            "most_common": []
        }
    
    lengths = [len(p) for p in patterns]
    pattern_strings = [pattern_to_string(p) for p in patterns]
    
    return {
        "total_patterns": len(patterns),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "most_common": Counter(pattern_strings).most_common(5)
    }

