"""
Модуль для генерации правил рекомендаций через YandexGPT.

Использует YandexGPT для генерации правил паттерн → продукт.
"""

from typing import Dict, List, Optional
import json
import re

from src.utils.yandex_gpt_client import call_yandex_gpt


def generate_rule_from_pattern(
    pattern: str,
    user_context: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Генерирует правило рекомендации на основе паттерна поведения.
    
    :param pattern: Паттерн поведения (например, "V→P→V")
    :param user_context: Дополнительный контекст пользователя (регион, avg_tx и т.д.)
    :return: Правило в формате {"pattern": "...", "product": "...", "reason": "..."}
    """
    # Сжатый контекст (только ключевые метрики)
    context_parts = []
    if user_context:
        region = user_context.get('region', '?')
        avg_tx = user_context.get('avg_tx', 0)
        if region != '?':
            context_parts.append(f"Р:{region}")
        if avg_tx > 0:
            context_parts.append(f"Чек:${avg_tx:.0f}")
    
    context_text = "|" + "|".join(context_parts) if context_parts else ""
    
    # Максимально сжатый промпт с списком конкретных продуктов ПСБ
    psb_products_list = (
        "Семейная ипотека, Ипотека «Вторичное жилье», Ипотека «Новостройка», "
        "Госпрограмма «Новые субъекты», Семейная военная ипотека, Военная ипотека, "
        "Кредит на любые цели, Рефинансирование кредитов, Экспресс-кредит «Турбоденьги», "
        "Кредитная карта «100+», Кредитная карта «180 дней без %», "
        "Вклад «Сильная ставка», Вклад «Ставка на будущее», Вклад «Драгоценный», "
        "Дебетовая карта «Твой кэшбэк», Дебетовая карта «Только вперед», "
        "Накопительный счет «Акцент», ПСБ Инвестиции"
    )
    
    prompt = f"Паттерн:{pattern}{context_text}|Продукт из списка ПСБ: {psb_products_list} JSON:{{product,confidence,reason}}"
    
    instructions = (
        "Эксперт банковских рекомендаций ПСБ. Паттерн → конкретный продукт ПСБ. "
        "Используй ТОЛЬКО конкретные названия продуктов из списка (например: 'Кредит на любые цели', 'Кредитная карта «100+»', 'Вклад «Сильная ставка»'). "
        "НЕ используй общие категории типа 'Кредит', 'Ипотека', 'Кредитка'."
    )
    
    try:
        response = call_yandex_gpt(
            input_text=prompt,
            instructions=instructions,
            temperature=0.2
        )
        
        # Извлекаем JSON из ответа
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            rule_data = json.loads(json_match.group())
            product_name = rule_data.get("product", "")
            # Проверяем, что это конкретный продукт ПСБ, а не общая категория
            generic_products = ["Кредит", "Ипотека", "Кредитка", "Вклад", "Дебет", "Кредитная карта", "Дебетовая карта"]
            if product_name in generic_products or not product_name:
                # Fallback на конкретный продукт по паттерну
                if "P" in pattern or "R" in pattern:  # Payments/Requests
                    product_name = "Кредит на любые цели"
                elif "V" in pattern:  # Views
                    product_name = "Кредитная карта «100+»"
                else:
                    product_name = "Дебетовая карта «Твой кэшбэк»"
            
            return {
                "pattern": pattern,
                "product": product_name,
                "confidence": rule_data.get("confidence", "средняя"),
                "reason": rule_data.get("reason", "На основе анализа паттерна")
            }
        
        # Fallback на конкретный продукт
        if "P" in pattern or "R" in pattern:
            fallback_product = "Кредит на любые цели"
        elif "V" in pattern:
            fallback_product = "Кредитная карта «100+»"
        else:
            fallback_product = "Дебетовая карта «Твой кэшбэк»"
        
        return {
            "pattern": pattern,
            "product": fallback_product,
            "confidence": "средняя",
            "reason": "Общий паттерн поведения"
        }
        
    except Exception as e:
        print(f"Ошибка при генерации правила: {e}")
        # Fallback на конкретный продукт
        if "P" in pattern or "R" in pattern:
            fallback_product = "Кредит на любые цели"
        elif "V" in pattern:
            fallback_product = "Кредитная карта «100+»"
        else:
            fallback_product = "Дебетовая карта «Твой кэшбэк»"
        
        return {
            "pattern": pattern,
            "product": fallback_product,
            "confidence": "низкая",
            "reason": "Ошибка при генерации"
        }



