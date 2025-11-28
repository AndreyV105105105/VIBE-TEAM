"""
Модуль для модели рекомендаций Next Best Offer.

Использует машинное обучение для ранжирования продуктов.
"""

import joblib
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.features.user_profile import profile_to_features
from src.utils.yandex_gpt_client import call_yandex_gpt


class NBOModel:
    """
    Модель для рекомендации Next Best Offer.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация модели.
        
        :param model_path: Путь к сохраненной модели
        """
        self.model_path = model_path or "models/nbo_model.pkl"
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.products: List[str] = []
        
        if Path(self.model_path).exists():
            self.load_model()
        else:
            # Инициализируем новую модель
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.products = ["Ипотека", "Кредитная карта", "Вклад", "Кредит", "Дебетовая карта"]
    
    def load_model(self) -> None:
        """Загружает модель из файла."""
        try:
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.scaler = data.get("scaler")
            self.products = data.get("products", ["Ипотека", "Кредитная карта", "Вклад", "Кредит"])
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.scaler = StandardScaler()
            self.products = ["Ипотека", "Кредитная карта", "Вклад", "Кредит"]
    
    def save_model(self) -> None:
        """Сохраняет модель в файл."""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "products": self.products
        }
        
        joblib.dump(data, self.model_path)
    
    def train(
        self,
        X: List[List[float]],
        y: List[str],
        products: Optional[List[str]] = None
    ) -> None:
        """
        Обучает модель.
        
        :param X: Признаки пользователей
        :param y: Рекомендуемые продукты
        :param products: Список всех продуктов
        """
        if products:
            self.products = products
        
        # Преобразуем метки в числовые
        product_to_idx = {p: i for i, p in enumerate(self.products)}
        y_numeric = [product_to_idx.get(label, 0) for label in y]
        
        # Масштабируем признаки
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем модель
        self.model.fit(X_scaled, y_numeric)
        
        # Сохраняем модель
        self.save_model()
    
    def train_with_yandexgpt(
        self,
        user_profiles: List[Dict],
        use_synthetic: bool = True
    ) -> None:
        """
        Предварительное обучение модели с помощью YandexGPT.
        
        Генерирует обучающие данные на основе профилей пользователей,
        используя YandexGPT для определения правильных рекомендаций.
        
        :param user_profiles: Список профилей пользователей
        :param use_synthetic: Использовать ли синтетические данные от YandexGPT
        """
        print("🤖 Предварительное обучение модели с YandexGPT...")
        
        X_train = []
        y_train = []
        
        # Обучаем на реальных профилях
        import time
        total_profiles = len(user_profiles)
        print(f"📊 Обработка {total_profiles} профилей...")
        
        for i, profile in enumerate(user_profiles[:50], 1):  # Ограничиваем для экономии токенов
            try:
                features = profile_to_features(profile)
                X_train.append(features)
                
                # Используем YandexGPT для определения правильного продукта
                print(f"  [{i}/{min(50, total_profiles)}] Запрос к YandexGPT для профиля...")
                product = self._get_recommendation_from_yandexgpt(profile)
                y_train.append(product)
                
                # Небольшая задержка между запросами для избежания rate limiting
                if i < min(50, total_profiles):
                    time.sleep(0.5)
            except Exception as e:
                print(f"⚠ Ошибка при обработке профиля: {e}")
                continue
        
        # Если включен синтетический режим, генерируем дополнительные данные
        if use_synthetic and len(user_profiles) > 0:
            print("📊 Генерация синтетических обучающих данных через YandexGPT...")
            synthetic_data = self._generate_synthetic_training_data(user_profiles[:10])
            X_train.extend(synthetic_data["X"])
            y_train.extend(synthetic_data["y"])
        
        if len(X_train) > 0:
            print(f"✅ Сгенерировано {len(X_train)} обучающих примеров")
            self.train(X_train, y_train)
            print("✅ Модель успешно обучена с помощью YandexGPT")
        else:
            print("⚠ Не удалось сгенерировать обучающие данные")
    
    def _get_recommendation_from_yandexgpt(self, profile: Dict) -> str:
        """
        Получает рекомендацию продукта от YandexGPT на основе профиля.
        
        :param profile: Профиль пользователя
        :return: Название рекомендованного продукта
        """
        # Формируем описание профиля
        profile_text = f"""
Профиль пользователя:
- Просмотров: {profile.get('num_views', 0)}
- Платежей: {profile.get('num_payments', 0)}
- Сумма транзакций: {profile.get('total_tx', 0)}
- Средний платеж: {profile.get('avg_tx', 0)}
- Дней активности: {profile.get('days_active', 0)}
- Уникальных товаров: {profile.get('unique_items', 0)}
- Топ категория: {profile.get('top_category', 'неизвестно')}
- Топ бренд: {profile.get('top_brand', 'неизвестно')}
"""
        
        prompt = f"""{profile_text}

Определи один финансовый продукт ПСБ, который лучше всего подходит этому пользователю.
Доступные продукты: Ипотека, Кредитная карта, Вклад, Кредит, Дебетовая карта.

Ответь только названием продукта, без дополнительных объяснений."""
        
        try:
            response = call_yandex_gpt(
                prompt,
                instructions="Ты эксперт по банковским продуктам. Анализируй профиль пользователя и рекомендуй наиболее подходящий продукт.",
                temperature=0.3
            )
            
            # Извлекаем название продукта из ответа
            response = response.strip()
            for product in self.products:
                if product.lower() in response.lower():
                    return product
            
            # Если не нашли точное совпадение, возвращаем первый продукт
            return self.products[0]
        except Exception as e:
            print(f"⚠ Ошибка при получении рекомендации от YandexGPT: {e}")
            # Fallback на простую эвристику
            if profile.get('total_tx', 0) > 100000:
                return "Вклад"
            elif profile.get('num_payments', 0) > 5:
                return "Кредитная карта"
            else:
                return "Дебетовая карта"
    
    def _generate_synthetic_training_data(self, sample_profiles: List[Dict]) -> Dict:
        """
        Генерирует синтетические обучающие данные через YandexGPT.
        
        :param sample_profiles: Примеры реальных профилей
        :return: Словарь с X и y для обучения
        """
        X_synthetic = []
        y_synthetic = []
        
        # Генерируем вариации профилей
        for profile in sample_profiles[:5]:  # Ограничиваем для экономии токенов
            try:
                # Создаем вариации профиля
                variations = [
                    {**profile, "num_payments": int(profile.get("num_payments", 0) * 1.5)},
                    {**profile, "total_tx": profile.get("total_tx", 0) * 2},
                    {**profile, "num_views": int(profile.get("num_views", 0) * 1.2)},
                ]
                
                for var_profile in variations:
                    features = profile_to_features(var_profile)
                    X_synthetic.append(features)
                    
                    # Получаем рекомендацию от YandexGPT
                    product = self._get_recommendation_from_yandexgpt(var_profile)
                    y_synthetic.append(product)
            except Exception as e:
                print(f"⚠ Ошибка при генерации синтетических данных: {e}")
                continue
        
        return {"X": X_synthetic, "y": y_synthetic}
    
    def predict(
        self,
        user_profile: Dict,
        top_k: int = 3,
        graph: Optional[any] = None,
        patterns: Optional[List] = None
    ) -> List[Dict[str, any]]:
        """
        Предсказывает топ-K продуктов для пользователя.
        
        :param user_profile: Профиль пользователя
        :param top_k: Количество топ рекомендаций
        :param graph: Граф поведения (опционально, для улучшенного fallback)
        :param patterns: Список паттернов поведения (опционально, для улучшенного fallback)
        :return: Список рекомендаций
        """
        # Проверяем, обучена ли модель
        if self.model is None:
            return self._fallback_recommendations(user_profile, top_k, graph, patterns)
        
        # Проверяем, обучена ли модель (есть ли атрибут n_estimators_ после fit)
        if not hasattr(self.model, 'estimators_') or len(self.model.estimators_) == 0:
            print("⚠ Модель не обучена, используем улучшенный fallback с анализом графа и паттернов")
            return self._fallback_recommendations(user_profile, top_k, graph, patterns)
        
        try:
            # Преобразуем профиль в признаки
            features = profile_to_features(user_profile)
            X = np.array([features])
            
            # Масштабируем только если scaler обучен
            if self.scaler and hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                # Scaler обучен (mean_ установлен после fit)
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    # Если transform не работает, используем без масштабирования
                    print(f"⚠ Предупреждение: не удалось применить scaler: {e}, используем без масштабирования")
                    X_scaled = X
            else:
                # Scaler не обучен, используем без масштабирования
                X_scaled = X
            
            # Предсказываем для каждого продукта
            scores = []
            for i, product in enumerate(self.products):
                # Создаем бинарную метку для каждого продукта
                y_binary = np.zeros(len(self.products))
                y_binary[i] = 1
                
                # Предсказываем вероятность
                score = self.model.predict(X_scaled)[0]
                scores.append((product, float(score)))
            
            # Сортируем по оценке
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Возвращаем топ-K
            recommendations = [
                {
                    "product": product,
                    "score": score
                }
                for product, score in scores[:top_k]
            ]
            
            return recommendations
        except Exception as e:
            print(f"⚠ Ошибка при предсказании модели: {e}, используем улучшенный fallback")
            return self._fallback_recommendations(user_profile, top_k, graph, patterns)
    
    def _fallback_recommendations(
        self,
        user_profile: Dict,
        top_k: int = 3,
        graph: Optional[any] = None,
        patterns: Optional[List] = None
    ) -> List[Dict[str, any]]:
        """
        Улучшенный fallback алгоритм на основе профиля, графа и паттернов.
        
        :param user_profile: Профиль пользователя
        :param top_k: Количество рекомендаций
        :param graph: Граф поведения (опционально)
        :param patterns: Список паттернов поведения (опционально)
        :return: Список рекомендаций
        """
        recommendations = []
        
        # Базовые метрики профиля
        num_payments = user_profile.get("num_payments", 0)
        total_tx = user_profile.get("total_tx", 0)
        avg_tx = user_profile.get("avg_tx", 0)
        num_views = user_profile.get("num_views", 0)
        days_active = user_profile.get("days_active", 0)
        unique_items = user_profile.get("unique_items", 0)
        top_category = user_profile.get("top_category")
        top_brand = user_profile.get("top_brand")
        
        # Улучшенный анализ графа (если доступен)
        graph_scores = {}
        if graph is not None:
            try:
                import networkx as nx
                
                if graph.number_of_nodes() > 0:
                    # 1. PageRank для определения важности узлов
                    try:
                        pagerank = nx.pagerank(graph, max_iter=100, weight='weight')
                        
                        # Анализируем типы важных узлов и их категории/бренды
                        item_nodes = []
                        brand_nodes = []
                        category_weights = {}
                        brand_weights = {}
                        
                        for node, data in graph.nodes(data=True):
                            node_importance = pagerank.get(node, 0)
                            if node_importance > 0.01:  # Только важные узлы
                                node_type = data.get("type", "unknown")
                                
                                if node_type == "item":
                                    item_nodes.append((node, node_importance))
                                    category_id = data.get("category_id")
                                    if category_id:
                                        category_weights[category_id] = category_weights.get(category_id, 0) + node_importance
                                
                                elif node_type == "brand":
                                    brand_nodes.append((node, node_importance))
                                    brand_id = data.get("brand_id")
                                    if brand_id:
                                        brand_weights[brand_id] = brand_weights.get(brand_id, 0) + node_importance
                        
                        # Анализ по типам узлов
                        total_item_importance = sum(imp for _, imp in item_nodes)
                        total_brand_importance = sum(imp for _, imp in brand_nodes)
                        
                        # Если доминируют бренды - активные платежи
                        if total_brand_importance > total_item_importance * 1.5:
                            graph_scores["Кредитная карта"] = 0.4
                            graph_scores["Вклад"] = 0.25
                            if len(brand_nodes) > 5:
                                graph_scores["Вклад"] = 0.35  # Много разных брендов = накопления
                        
                        # Если доминируют товары - активный просмотр/исследование
                        elif total_item_importance > total_brand_importance * 2:
                            graph_scores["Ипотека"] = 0.3
                            graph_scores["Кредит"] = 0.25
                            if len(item_nodes) > 10:
                                graph_scores["Ипотека"] = 0.4  # Много просмотров = крупная покупка
                        
                        # Анализ категорий (если есть информация)
                        if category_weights:
                            top_categories = sorted(category_weights.items(), key=lambda x: x[1], reverse=True)[:3]
                            # Категории недвижимости/ремонта указывают на ипотеку
                            for cat_id, weight in top_categories:
                                cat_str = str(cat_id).lower()
                                if any(keyword in cat_str for keyword in ["недвижимость", "ремонт", "дом", "квартира"]):
                                    graph_scores["Ипотека"] = graph_scores.get("Ипотека", 0) + 0.2 * weight
                                    break
                    except Exception as e:
                        print(f"⚠ Ошибка PageRank анализа: {e}")
                    
                    # 2. Анализ структуры графа (пути и связность)
                    try:
                        # Находим все пути от START
                        if "START" in graph:
                            reachable = list(nx.descendants(graph, "START"))
                            if reachable:
                                # Анализируем длину путей
                                path_lengths = []
                                for target in reachable[:20]:  # Ограничиваем для производительности
                                    try:
                                        paths = list(nx.all_simple_paths(graph, "START", target, cutoff=6))
                                        if paths:
                                            path_lengths.extend([len(p) for p in paths[:3]])
                                    except:
                                        continue
                                
                                if path_lengths:
                                    avg_path_length = sum(path_lengths) / len(path_lengths)
                                    max_path_length = max(path_lengths)
                                    
                                    # Длинные пути = сложное поведение = крупные покупки
                                    if avg_path_length > 4 or max_path_length > 5:
                                        graph_scores["Ипотека"] = graph_scores.get("Ипотека", 0) + 0.25
                                        graph_scores["Кредит"] = graph_scores.get("Кредит", 0) + 0.2
                                    
                                    # Много путей = активное исследование
                                    if len(path_lengths) > 15:
                                        graph_scores["Ипотека"] = graph_scores.get("Ипотека", 0) + 0.15
                    except Exception as e:
                        print(f"⚠ Ошибка анализа путей: {e}")
                    
                    # 3. Анализ плотности и кластеризации
                    try:
                        density = nx.density(graph) if graph.number_of_nodes() > 1 else 0
                        
                        # Высокая плотность = активное взаимодействие
                        if density > 0.3:
                            graph_scores["Кредитная карта"] = graph_scores.get("Кредитная карта", 0) + 0.25
                            graph_scores["Дебетовая карта"] = graph_scores.get("Дебетовая карта", 0) + 0.2
                        
                        # Низкая плотность, но много узлов = исследование разных вариантов
                        elif density < 0.2 and graph.number_of_nodes() > 10:
                            graph_scores["Ипотека"] = graph_scores.get("Ипотека", 0) + 0.2
                            graph_scores["Кредит"] = graph_scores.get("Кредит", 0) + 0.15
                        
                        # Анализ степени узлов (средняя степень)
                        degrees = dict(graph.degree())
                        if degrees:
                            avg_degree = sum(degrees.values()) / len(degrees)
                            # Высокая средняя степень = активное взаимодействие
                            if avg_degree > 3:
                                graph_scores["Кредитная карта"] = graph_scores.get("Кредитная карта", 0) + 0.2
                    except Exception as e:
                        print(f"⚠ Ошибка анализа плотности: {e}")
                    
                    # 4. Анализ весов рёбер (частоты взаимодействий)
                    try:
                        edge_weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)]
                        if edge_weights:
                            avg_weight = sum(edge_weights) / len(edge_weights)
                            max_weight = max(edge_weights)
                            
                            # Высокие веса = частые повторяющиеся действия
                            if avg_weight > 2 or max_weight > 5:
                                graph_scores["Кредитная карта"] = graph_scores.get("Кредитная карта", 0) + 0.15
                                graph_scores["Вклад"] = graph_scores.get("Вклад", 0) + 0.1
                    except Exception as e:
                        print(f"⚠ Ошибка анализа весов: {e}")
                        
            except Exception as e:
                print(f"⚠ Ошибка анализа графа в fallback: {e}")
        
        # Улучшенный анализ паттернов (если доступны)
        pattern_scores = {}
        if patterns:
            # Обрабатываем паттерны (могут быть строками или кортежами)
            pattern_strings = []
            for p in patterns[:10]:  # Анализируем больше паттернов
                if isinstance(p, tuple):
                    pattern_strings.append("→".join([str(x) for x in p]))
                elif isinstance(p, str):
                    pattern_strings.append(p)
            
            combined_pattern = " | ".join(pattern_strings)
            
            # 1. Анализ частоты типов событий в паттернах
            view_count = combined_pattern.count("V") + combined_pattern.count("view")
            pay_count = combined_pattern.count("P") + combined_pattern.count("pay")
            click_count = combined_pattern.count("C") + combined_pattern.count("click")
            total_events = view_count + pay_count + click_count
            
            if total_events > 0:
                view_ratio = view_count / total_events
                pay_ratio = pay_count / total_events
                
                # Доминирование платежей
                if pay_ratio > 0.5:
                    pattern_scores["Кредитная карта"] = 0.4
                    pattern_scores["Вклад"] = 0.3
                    if pay_count > 5:
                        pattern_scores["Вклад"] = 0.4  # Много платежей = накопления
                
                # Доминирование просмотров
                elif view_ratio > 0.6:
                    pattern_scores["Ипотека"] = 0.35
                    pattern_scores["Кредит"] = 0.25
                    if view_count > 10:
                        pattern_scores["Ипотека"] = 0.45  # Много просмотров = исследование
                
                # Сбалансированное поведение
                elif 0.3 < view_ratio < 0.6 and 0.2 < pay_ratio < 0.5:
                    pattern_scores["Кредитная карта"] = 0.3
                    pattern_scores["Ипотека"] = 0.25
            
            # 2. Анализ последовательностей (сложные паттерны)
            for pattern_str in pattern_strings:
                # Паттерны исследования: V→V→V или V→V→P
                if "V→V→V" in pattern_str or pattern_str.count("V") >= 3:
                    pattern_scores["Ипотека"] = pattern_scores.get("Ипотека", 0) + 0.15
                    pattern_scores["Кредит"] = pattern_scores.get("Кредит", 0) + 0.1
                
                # Паттерны активных покупок: P→P→P или P→P→V
                if "P→P→P" in pattern_str or (pattern_str.count("P") >= 3 and pay_ratio > 0.5):
                    pattern_scores["Кредитная карта"] = pattern_scores.get("Кредитная карта", 0) + 0.2
                    pattern_scores["Вклад"] = pattern_scores.get("Вклад", 0) + 0.15
                
                # Сложные паттерны принятия решений: V→P→V или P→V→P
                if "V→P→V" in pattern_str or "P→V→P" in pattern_str:
                    pattern_scores["Ипотека"] = pattern_scores.get("Ипотека", 0) + 0.2
                    pattern_scores["Кредит"] = pattern_scores.get("Кредит", 0) + 0.15
                
                # Паттерны быстрых решений: V→P (короткие паттерны)
                if len(pattern_str.split("→")) <= 3 and "V" in pattern_str and "P" in pattern_str:
                    pattern_scores["Кредитная карта"] = pattern_scores.get("Кредитная карта", 0) + 0.15
                    pattern_scores["Дебетовая карта"] = pattern_scores.get("Дебетовая карта", 0) + 0.1
            
            # 3. Анализ разнообразия паттернов
            unique_patterns = len(set(pattern_strings))
            if unique_patterns > 5:
                # Много разных паттернов = сложное поведение = крупные покупки
                pattern_scores["Ипотека"] = pattern_scores.get("Ипотека", 0) + 0.1
                pattern_scores["Кредит"] = pattern_scores.get("Кредит", 0) + 0.1
        
        # Базовые оценки на основе профиля
        base_scores = {}
        
        # Ипотека - если есть активность, платежи и просмотры недвижимости/ремонта
        mortgage_score = 0.0
        if total_tx > 50000:  # Крупные платежи
            mortgage_score += 0.3
        if num_views > 10 and unique_items > 5:  # Активный просмотр
            mortgage_score += 0.25
        if days_active > 7:  # Долгая активность
            mortgage_score += 0.2
        if top_category and ("недвижимость" in str(top_category).lower() or "ремонт" in str(top_category).lower()):
            mortgage_score += 0.25
        base_scores["Ипотека"] = mortgage_score if mortgage_score > 0 else 0.1
        
        # Кредитная карта - если есть регулярные платежи
        card_score = 0.0
        if num_payments > 5:  # Регулярные платежи
            card_score += 0.4
        if avg_tx > 1000 and avg_tx < 50000:  # Средние платежи
            card_score += 0.3
        if days_active > 3:  # Активность
            card_score += 0.2
        base_scores["Кредитная карта"] = card_score if card_score > 0 else 0.2
        
        # Вклад - если есть крупные платежи
        deposit_score = 0.0
        if total_tx > 100000:  # Крупные суммы
            deposit_score += 0.5
        if num_payments > 10:  # Много транзакций
            deposit_score += 0.2
        if avg_tx > 10000:  # Крупные средние платежи
            deposit_score += 0.2
        base_scores["Вклад"] = deposit_score if deposit_score > 0 else 0.15
        
        # Кредит - если есть активность и просмотры
        loan_score = 0.0
        if num_views > 15:  # Много просмотров
            loan_score += 0.4
        if unique_items > 10:  # Разнообразие интересов
            loan_score += 0.3
        if days_active > 5:  # Долгая активность
            loan_score += 0.2
        base_scores["Кредит"] = loan_score if loan_score > 0 else 0.1
        
        # Дебетовая карта - базовая рекомендация
        base_scores["Дебетовая карта"] = 0.25 if (num_payments > 0 or num_views > 0) else 0.3
        
        # Улучшенное объединение оценок с адаптивными весами
        final_scores = {}
        
        # Определяем, какие источники данных доступны
        has_graph = graph is not None and graph.number_of_nodes() > 0
        has_patterns = patterns and len(patterns) > 0
        
        # Адаптивные веса в зависимости от доступности данных
        if has_graph and has_patterns:
            # Все данные доступны - сбалансированные веса
            base_weight, graph_weight, pattern_weight = 0.4, 0.35, 0.25
        elif has_graph:
            # Только граф - увеличиваем его вес
            base_weight, graph_weight, pattern_weight = 0.5, 0.5, 0.0
        elif has_patterns:
            # Только паттерны - увеличиваем их вес
            base_weight, graph_weight, pattern_weight = 0.6, 0.0, 0.4
        else:
            # Только базовые метрики
            base_weight, graph_weight, pattern_weight = 1.0, 0.0, 0.0
        
        for product in self.products:
            final_scores[product] = (
                base_scores.get(product, 0) * base_weight +
                graph_scores.get(product, 0) * graph_weight +
                pattern_scores.get(product, 0) * pattern_weight
            )
            
            # Нормализуем, чтобы максимальная оценка была 1.0
            if final_scores[product] > 1.0:
                final_scores[product] = 1.0
        
        # Сортируем по оценке
        sorted_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-K
        for product, score in sorted_products[:top_k]:
            recommendations.append({
                "product": product,
                "score": float(score)
            })
        
        return recommendations


def recommend(
    user_profile: Dict,
    model_path: Optional[str] = None,
    top_k: int = 3,
    graph: Optional[any] = None,
    patterns: Optional[List] = None
) -> List[Dict[str, any]]:
    """
    Рекомендует продукты для пользователя.
    
    :param user_profile: Профиль пользователя
    :param model_path: Путь к модели
    :param top_k: Количество рекомендаций
    :param graph: Граф поведения (опционально, для улучшенного fallback)
    :param patterns: Список паттернов поведения (опционально, для улучшенного fallback)
    :return: Список рекомендаций
    """
    model = NBOModel(model_path=model_path)
    return model.predict(user_profile, top_k=top_k, graph=graph, patterns=patterns)

