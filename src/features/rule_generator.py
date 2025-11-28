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
    context_text = ""
    if user_context:
        context_text = f"\nКонтекст пользователя:\n"
        context_text += f"- Регион: {user_context.get('region', 'неизвестен')}\n"
        context_text += f"- Средний чек: {user_context.get('avg_tx', 0):.0f} ₽\n"
        context_text += f"- Активность: {user_context.get('days_active', 0)} дней\n"
    
    prompt = f"""На основе паттерна поведения пользователя определи, какой финансовый продукт ПСБ ему подходит.

Паттерн: {pattern}
{context_text}

Доступные продукты ПСБ:
- Ипотека
- Кредитная карта
- Вклад
- Кредит
- Дебетовая карта

Ответь в формате JSON:
{{
  "product": "название продукта",
  "confidence": "высокая/средняя/низкая",
  "reason": "краткое объяснение, почему этот продукт подходит"
}}"""
    
    instructions = """Ты эксперт по рекомендательным системам для банков. 
Анализируй паттерны поведения клиентов и рекомендуй подходящие финансовые продукты."""
    
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



