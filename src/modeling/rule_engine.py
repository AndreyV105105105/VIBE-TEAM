"""
Модуль для работы с правилами рекомендаций.

Использует правила для сопоставления паттернов поведения с продуктами.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from src.features.rule_generator import generate_rule_from_pattern


class RuleEngine:
    """
    Движок правил для рекомендаций на основе паттернов.
    """
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        Инициализация движка правил.
        
        :param rules_path: Путь к файлу с правилами (JSON)
        """
        self.rules_path = rules_path or "models/pattern_to_product.json"
        self.rules: Dict[str, Dict] = {}
        self.load_rules()
    
    def load_rules(self) -> None:
        """Загружает правила из файла."""
        rules_file = Path(self.rules_path)
        if rules_file.exists():
            try:
                with open(rules_file, "r", encoding="utf-8") as f:
                    self.rules = json.load(f)
            except Exception as e:
                print(f"Ошибка загрузки правил: {e}")
                self.rules = {}
        else:
            self.rules = {}
    
    def save_rules(self) -> None:
        """Сохраняет правила в файл."""
        rules_file = Path(self.rules_path)
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)
    
    def add_rule(
        self,
        pattern: str,
        product: str,
        reason: str,
        confidence: str = "средняя"
    ) -> None:
        """
        Добавляет правило.
        
        :param pattern: Паттерн поведения
        :param product: Рекомендуемый продукт
        :param reason: Объяснение правила
        :param confidence: Уверенность (высокая/средняя/низкая)
        """
        self.rules[pattern] = {
            "product": product,
            "reason": reason,
            "confidence": confidence
        }
    
    def match_pattern(
        self,
        pattern: str,
        user_context: Optional[Dict] = None
    ) -> Optional[Dict[str, str]]:
        """
        Находит продукт для паттерна.
        
        :param pattern: Паттерн поведения
        :param user_context: Контекст пользователя
        :return: Правило или None
        """
        # Прямое совпадение
        if pattern in self.rules:
            return {
                "pattern": pattern,
                **self.rules[pattern]
            }
        
        # Генерация нового правила через YandexGPT
        try:
            rule = generate_rule_from_pattern(pattern, user_context)
            # Сохраняем новое правило
            self.add_rule(
                pattern=rule["pattern"],
                product=rule["product"],
                reason=rule["reason"],
                confidence=rule["confidence"]
            )
            return rule
        except Exception as e:
            print(f"Ошибка при генерации правила: {e}")
            return None
    
    def recommend_from_patterns(
        self,
        patterns: List[str],
        user_context: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Рекомендует продукты на основе списка паттернов.
        
        :param patterns: Список паттернов
        :param user_context: Контекст пользователя
        :return: Список рекомендаций с продуктами и оценками
        """
        recommendations = defaultdict(lambda: {"score": 0, "reasons": []})
        
        for pattern in patterns:
            rule = self.match_pattern(pattern, user_context)
            
            if rule:
                product = rule["product"]
                confidence = rule["confidence"]
                
                # Оценка на основе уверенности
                score_map = {"высокая": 3, "средняя": 2, "низкая": 1}
                score = score_map.get(confidence, 1)
                
                recommendations[product]["score"] += score
                recommendations[product]["reasons"].append({
                    "pattern": pattern,
                    "reason": rule["reason"],
                    "confidence": confidence
                })
        
        # Сортируем по оценке
        result = [
            {
                "product": product,
                "score": data["score"],
                "reasons": data["reasons"]
            }
            for product, data in recommendations.items()
        ]
        
        result.sort(key=lambda x: x["score"], reverse=True)
        
        return result

