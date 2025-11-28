"""
Модуль для генерации объяснений рекомендаций через YandexGPT Responses API.

Поддерживает:
- Генерацию объяснений через YandexGPT Responses API (OpenAI compatible)
- Кэширование объяснений для ускорения
- Fallback на шаблоны, если API недоступен
"""

import os
import hashlib
import json
from typing import Dict, Optional

from src.utils.yandex_cloud import get_cached_config


# Кэш для объяснений (в памяти)
_explanation_cache: Dict[str, str] = {}


def build_prompt(profile: Dict, product: str) -> str:
    """
    Строит промпт для YandexGPT на основе профиля пользователя.
    
    :param profile: Профиль пользователя (регион, avg_tx, паттерны и т.д.)
    :param product: Название рекомендуемого продукта
    :return: Текст промпта
    """
    # Извлекаем ключевые данные из профиля (без сырых событий)
    region = profile.get("region", "неизвестен")
    avg_tx = profile.get("avg_tx", 0)
    num_views = profile.get("num_views", 0)
    num_payments = profile.get("num_payments", 0)
    pattern = profile.get("pattern", "неизвестен")
    
    prompt = f"""Клиент из региона {region}, 
средний чек ${avg_tx:.2f}, 
количество просмотров: {num_views},
количество платежей: {num_payments},
поведенческий паттерн: {pattern}.

Почему ему подходит продукт «{product}»? 
Ответь коротко, по-русски, как консультант ПСБ."""
    
    return prompt




def explain_with_yandexgpt(
    profile: Dict,
    product: str,
    use_cache: bool = True
) -> str:
    """
    Генерирует объяснение рекомендации через YandexGPT Responses API.
    
    :param profile: Профиль пользователя
    :param product: Название продукта
    :param use_cache: Использовать кэш для одинаковых запросов
    :return: Текст объяснения
    """
    # Создаем ключ кэша
    cache_key = hashlib.md5(
        json.dumps({**profile, "product": product}, sort_keys=True).encode()
    ).hexdigest()
    
    # Проверяем кэш
    if use_cache and cache_key in _explanation_cache:
        return _explanation_cache[cache_key]
    
    # Получаем конфигурацию Yandex Cloud
    try:
        from src.utils.yandex_gpt_client import call_yandex_gpt
        
        # Строим промпт
        prompt = build_prompt(profile, product)
        
        # Вызываем YandexGPT Responses API
        explanation = call_yandex_gpt(
            input_text=prompt,
            instructions="Ты опытный консультант ПСБ, который объясняет клиентам, почему им подходит тот или иной финансовый продукт. Отвечай коротко, понятно и по-русски.",
            temperature=0.3
        )
        
        # Кэшируем результат
        if use_cache:
            _explanation_cache[cache_key] = explanation
        
        return explanation
        
    except Exception as e:
        print(f"Ошибка при вызове YandexGPT API: {e}")
        return _get_fallback_explanation(profile, product, use_cache, cache_key)


def _get_fallback_explanation(
    profile: Dict,
    product: str,
    use_cache: bool,
    cache_key: str
) -> str:
    """
    Возвращает fallback объяснение на основе шаблонов.
    
    :param profile: Профиль пользователя
    :param product: Название продукта
    :param use_cache: Использовать кэш
    :param cache_key: Ключ кэша
    :return: Текст объяснения
    """
    # Шаблоны для fallback
    fallback_templates = {
        "Ипотека": [
            "Вам подходит ипотека, потому что вы ищете квартиры после ремонта",
            "Исходя из вашего поведения, ипотека поможет вам приобрести недвижимость",
            "Ваш паттерн покупок указывает на интерес к недвижимости - ипотека идеально подходит",
        ],
        "Кредитная карта": [
            "Кредитная карта подходит вам, так как вы часто совершаете покупки",
            "Ваш профиль расходов идеально подходит для кредитной карты",
            "Исходя из вашей активности, кредитная карта упростит ваши платежи",
        ],
        "Вклад": [
            "Вклад подходит вам для накопления средств",
            "Исходя из вашего финансового поведения, вклад поможет сохранить и приумножить средства",
        ],
        "Кредит": [
            "Кредит подходит для ваших финансовых целей",
            "Исходя из вашего профиля, кредит поможет реализовать ваши планы",
        ],
    }
    
    # Выбираем шаблон
    templates = fallback_templates.get(product, [
        f"Вам подходит {product}, потому что ваш профиль совпадает с типичным клиентом"
    ])
    
    import random
    explanation = random.choice(templates)
    
    # Кэшируем fallback
    if use_cache:
        _explanation_cache[cache_key] = explanation
    
    return explanation


def explain_recommendation(
    profile: Dict,
    product: str,
    use_cache: bool = True,
    use_yandexgpt: bool = True
) -> str:
    """
    Генерирует объяснение для рекомендации продукта.
    
    Основная функция для использования в коде.
    
    :param profile: Профиль пользователя
    :param product: Название рекомендуемого продукта
    :param use_cache: Использовать кэш для ускорения
    :param use_yandexgpt: Использовать YandexGPT (если False, используется fallback)
    :return: Текст объяснения
    """
    if not use_yandexgpt:
        # Fallback: простое объяснение без YandexGPT
        print(f"📝 Используется fallback объяснение (без YandexGPT)")
        return _get_fallback_explanation(profile, product, use_cache, None)
    
    try:
        return explain_with_yandexgpt(profile, product, use_cache=use_cache)
    except Exception as e:
        print(f"Ошибка при генерации объяснения через YandexGPT: {e}")
        # Fallback на простое объяснение
        return _get_fallback_explanation(profile, product, use_cache, None)


def clear_cache() -> None:
    """Очищает кэш объяснений."""
    global _explanation_cache
    _explanation_cache.clear()

