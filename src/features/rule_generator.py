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
    
    # Максимально сжатый промпт
    prompt = f"Паттерн:{pattern}{context_text}|Продукт? JSON:{{product,confidence,reason}}"
    
    instructions = "Эксперт банковских рекомендаций. Паттерн → продукт (Ипотека/Кредитка/Вклад/Кредит/Дебет)."
    
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
            return {
                "pattern": pattern,
                "product": rule_data.get("product", "Кредитная карта"),
                "confidence": rule_data.get("confidence", "средняя"),
                "reason": rule_data.get("reason", "На основе анализа паттерна")
            }
        
        # Fallback
        return {
            "pattern": pattern,
            "product": "Кредитная карта",
            "confidence": "средняя",
            "reason": "Общий паттерн поведения"
        }
        
    except Exception as e:
        print(f"Ошибка при генерации правила: {e}")
        return {
            "pattern": pattern,
            "product": "Кредитная карта",
            "confidence": "низкая",
            "reason": "Ошибка при генерации"
        }



