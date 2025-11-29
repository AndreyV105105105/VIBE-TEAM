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
        :param user_context: Контекст пользователя (profile)
        :return: Правило или None
        """
        # 1. Прямое совпадение по сохраненным правилам
        if pattern in self.rules:
            return {
                "pattern": pattern,
                **self.rules[pattern]
            }
            
        # 2. Эвристические правила (бизнес-логика) без использования GPT
        # Это экономит токены и дает быстрый результат для очевидных случаев
        if user_context:
            # Получаем категории из контекста
            top_cat = user_context.get("top_category")
            brand_cat = user_context.get("top_brand_category")
            
            # Эвристика: Категория "Foodstuffs" -> Карта с кэшбэком на супермаркеты
            if top_cat == "Foodstuffs and Beverages" or brand_cat == "Foodstuffs and Beverages":
                rule = {
                    "pattern": pattern,
                    "product": "Дебетовая карта «Твой кэшбэк»",
                    "reason": "Вы часто покупаете продукты питания. Эта карта дает повышенный кэшбэк в супермаркетах.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule
                
            # Эвристика: Категория "Renovation" -> Кредит на ремонт
            if (top_cat and "Renovation" in top_cat) or (brand_cat and "Renovation" in brand_cat):
                rule = {
                    "pattern": pattern,
                    "product": "Кредит на любые цели",
                    "reason": "Мы заметили интерес к товарам для ремонта. Кредит поможет реализовать ваши планы быстрее.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule
                
            # Эвристика: Категория "Electronics" -> Рассрочка или Страховка гаджетов
            if (top_cat and "Electronics" in top_cat) or (brand_cat and "Electronics" in brand_cat):
                rule = {
                    "pattern": pattern,
                    "product": "Кредитная карта «180 дней без %»",
                    "reason": "Для покупок электроники отлично подойдет карта с длинным льготным периодом 180 дней.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Auto" / "Fuel" -> Автокарта
            if (top_cat and any(x in top_cat for x in ["Auto", "Fuel", "Gas"])) or \
               (brand_cat and any(x in brand_cat for x in ["Auto", "Fuel", "Gas"])):
                rule = {
                    "pattern": pattern,
                    "product": "Кредитная карта «Двойной кэшбэк»",
                    "reason": "Получайте кэшбэк за любые покупки, включая АЗС и автоуслуги.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Travel" / "Hotels" -> Мильная карта
            if (top_cat and any(x in top_cat for x in ["Travel", "Hotel", "Airline"])) or \
               (brand_cat and any(x in brand_cat for x in ["Travel", "Hotel", "Airline"])):
                rule = {
                    "pattern": pattern,
                    "product": "Пакет «Orange Premium Club»",
                    "reason": "Для путешественников: доступ в бизнес-залы, страховка и особые привилегии.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Sport" -> Карта ЦСКА
            if (top_cat and any(x in top_cat for x in ["Sport", "Football", "Soccer", "Tickets"])) or \
               (brand_cat and any(x in brand_cat for x in ["Sport", "Football", "Soccer", "Tickets"])):
                rule = {
                    "pattern": pattern,
                    "product": "Клубная карта ПФК ЦСКА",
                    "reason": "Специально для болельщиков: скидки на билеты и атрибутику, уникальный дизайн.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Спорт + Медицина -> Карта "Только вперед"
            if (top_cat and any(x in top_cat for x in ["Health", "Medical", "Pharmacy", "Doctor"])) or \
               (brand_cat and any(x in brand_cat for x in ["Health", "Medical", "Pharmacy", "Doctor"])):
                rule = {
                    "pattern": pattern,
                    "product": "Дебетовая карта «Только вперед»",
                    "reason": "Кэшбэк 7% в категориях «Спорт» и «Аптеки». Заботьтесь о здоровье с выгодой.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Крупные покупки -> Кредитная карта 180 дней
            if "P→P→P" in pattern and "high_value" in pattern: # Hypothetical, relying on logic mainly
                rule = {
                    "pattern": pattern,
                    "product": "Кредитная карта «180 дней без %»",
                    "reason": "Длинный льготный период 180 дней идеален для крупных покупок и ремонта.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Кредит наличными (general)
            if "high_value" in pattern:
                 rule = {
                    "pattern": pattern,
                    "product": "Кредит на любые цели",
                    "reason": "Для реализации больших планов. Выгодная ставка и удобное оформление.",
                    "confidence": "средняя"
                }
                 self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                 return rule

            # Эвристика: Рефинансирование (payment to other banks - hypothetical pattern)
            if "pay_bank" in pattern or (top_cat and "Bank" in top_cat):
                rule = {
                    "pattern": pattern,
                    "product": "Рефинансирование кредитов",
                    "reason": "Объедините кредиты в один с выгодной ставкой. Платите меньше, живите лучше.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Турбоденьги (Small rapid needs)
            if "low_balance" in pattern: # Hypothetical
                rule = {
                    "pattern": pattern,
                    "product": "Экспресс-кредит «Турбоденьги»",
                    "reason": "Деньги здесь и сейчас. Быстрое оформление без лишних документов.",
                    "confidence": "низкая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Паттерн "View -> View -> View" (много просмотров) -> Кредитная карта
            if "V_V_V" in pattern or "V→V→V" in pattern:
                rule = {
                    "pattern": pattern,
                    "product": "Кредитная карта «100+»",
                    "reason": "Вы активно выбираете товары. Кредитная карта позволит купить всё сразу без переплаты.",
                    "confidence": "низкая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

        # 3. Генерация нового правила через YandexGPT (если эвристики не сработали)
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

