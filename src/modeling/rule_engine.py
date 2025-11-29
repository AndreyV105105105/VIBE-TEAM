"""
Модуль для работы с правилами рекомендаций.

Использует правила для сопоставления паттернов поведения с продуктами.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from src.features.rule_generator import generate_rule_from_pattern
from src.utils.category_normalizer import normalize_category, check_category_match, get_category_keywords


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
            
            # Нормализуем категории для проверки
            top_cat_normalized = normalize_category(top_cat) if top_cat else ""
            brand_cat_normalized = normalize_category(brand_cat) if brand_cat else ""
            
            # Эвристика: Категория "Food" -> Карта с кэшбэком на супермаркеты
            if check_category_match(top_cat, ["food"]) or check_category_match(brand_cat, ["food"]):
                rule = {
                    "pattern": pattern,
                    "product": "Дебетовая карта «Твой кэшбэк»",
                    "reason": "Вы часто покупаете продукты питания. Эта карта дает повышенный кэшбэк в супермаркетах.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule
                
            # Эвристика: Категория "Real Estate/Renovation" -> Кредит на ремонт
            if check_category_match(top_cat, ["real_estate"]) or check_category_match(brand_cat, ["real_estate"]):
                rule = {
                    "pattern": pattern,
                    "product": "Кредит на любые цели",
                    "reason": "Мы заметили интерес к товарам для ремонта. Кредит поможет реализовать ваши планы быстрее.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule
                
            # Эвристика: Категория "Electronics" -> Рассрочка или Страховка гаджетов
            if check_category_match(top_cat, ["electronics"]) or check_category_match(brand_cat, ["electronics"]):
                rule = {
                    "pattern": pattern,
                    "product": "Кредитная карта «180 дней без %»",
                    "reason": "Для покупок электроники отлично подойдет карта с длинным льготным периодом 180 дней.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Auto" / "Fuel" -> Автокарта
            if check_category_match(top_cat, ["auto"]) or check_category_match(brand_cat, ["auto"]):
                rule = {
                    "pattern": pattern,
                    "product": "Кредитная карта «Двойной кэшбэк»",
                    "reason": "Получайте кэшбэк за любые покупки, включая АЗС и автоуслуги.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Travel" / "Hotels" -> Мильная карта
            if check_category_match(top_cat, ["travel"]) or check_category_match(brand_cat, ["travel"]):
                rule = {
                    "pattern": pattern,
                    "product": "Пакет «Orange Premium Club»",
                    "reason": "Для путешественников: доступ в бизнес-залы, страховка и особые привилегии.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Sport" -> Карта ЦСКА
            if check_category_match(top_cat, ["sport"]) or check_category_match(brand_cat, ["sport"]):
                rule = {
                    "pattern": pattern,
                    "product": "Клубная карта ПФК ЦСКА",
                    "reason": "Специально для болельщиков: скидки на билеты и атрибутику, уникальный дизайн.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Спорт + Медицина -> Карта "Только вперед"
            if check_category_match(top_cat, ["health", "sport"]) or check_category_match(brand_cat, ["health", "sport"]):
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

            # Эвристика: Категория "Clothing/Fashion" -> Карта с кэшбэком
            if check_category_match(top_cat, ["clothing"]) or check_category_match(brand_cat, ["clothing"]):
                rule = {
                    "pattern": pattern,
                    "product": "Дебетовая карта «Твой кэшбэк»",
                    "reason": "Повышенный кэшбэк на покупки одежды и аксессуаров. Зарабатывайте на каждом шопинге.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Children/Kids" -> Семейные продукты
            if check_category_match(top_cat, ["children"]) or check_category_match(brand_cat, ["children"]):
                rule = {
                    "pattern": pattern,
                    "product": "Семейная ипотека",
                    "reason": "Для семей с детьми предусмотрены специальные условия по ипотеке и льготные ставки.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Beauty/Cosmetics" -> Карта с кэшбэком
            if check_category_match(top_cat, ["beauty"]) or check_category_match(brand_cat, ["beauty"]):
                rule = {
                    "pattern": pattern,
                    "product": "Дебетовая карта «Твой кэшбэк»",
                    "reason": "Кэшбэк на косметику и средства ухода. Красота с выгодой.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Books/Education" -> Инвестиции
            if check_category_match(top_cat, ["books", "education"]) or check_category_match(brand_cat, ["books", "education"]):
                rule = {
                    "pattern": pattern,
                    "product": "ПСБ Инвестиции",
                    "reason": "Инвестируйте в свое будущее. Долгосрочные накопления с профессиональным управлением.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Home Appliances/Furniture" -> Кредит на любые цели
            if check_category_match(top_cat, ["home_appliances", "furniture", "kitchen"]) or check_category_match(brand_cat, ["home_appliances", "furniture", "kitchen"]):
                rule = {
                    "pattern": pattern,
                    "product": "Кредит на любые цели",
                    "reason": "Для обустройства дома - выгодный кредит на покупку техники и мебели.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Construction/Renovation" -> Кредит на ремонт
            if check_category_match(top_cat, ["construction", "tools"]) or check_category_match(brand_cat, ["construction", "tools"]):
                rule = {
                    "pattern": pattern,
                    "product": "Кредит на любые цели",
                    "reason": "Кредит на ремонт и строительные материалы. Реализуйте планы по обустройству жилья.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Pets" -> Накопительный счет
            if check_category_match(top_cat, ["pets"]) or check_category_match(brand_cat, ["pets"]):
                rule = {
                    "pattern": pattern,
                    "product": "Накопительный счет «Про запас»",
                    "reason": "Накопительный счет поможет отложить средства на регулярные расходы, включая уход за питомцами.",
                    "confidence": "низкая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Entertainment/Games" -> Кредитная карта
            if check_category_match(top_cat, ["entertainment", "movies", "music"]) or check_category_match(brand_cat, ["entertainment", "movies", "music"]):
                rule = {
                    "pattern": pattern,
                    "product": "Кредитная карта «180 дней без %»",
                    "reason": "Для крупных покупок техники и развлечений - карта с длинным льготным периодом.",
                    "confidence": "средняя"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Pharmacy" -> Карта "Только вперед"
            if check_category_match(top_cat, ["pharmacy"]) or check_category_match(brand_cat, ["pharmacy"]):
                rule = {
                    "pattern": pattern,
                    "product": "Дебетовая карта «Только вперед»",
                    "reason": "Кэшбэк 7% в аптеках. Заботьтесь о здоровье с выгодой.",
                    "confidence": "высокая"
                }
                self.add_rule(pattern, rule["product"], rule["reason"], rule["confidence"])
                return rule

            # Эвристика: Категория "Jewelry" -> Вклад "Драгоценный"
            if check_category_match(top_cat, ["jewelry"]) or check_category_match(brand_cat, ["jewelry"]):
                rule = {
                    "pattern": pattern,
                    "product": "Вклад «Драгоценный»",
                    "reason": "Специальный вклад для покупки драгоценных металлов с повышенной ставкой.",
                    "confidence": "средняя"
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

