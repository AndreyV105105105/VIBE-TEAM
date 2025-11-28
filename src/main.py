"""
Основной модуль для работы системы рекомендаций.

Интегрирует все компоненты: загрузку данных, построение графов,
извлечение паттернов, создание профилей, рекомендации и объяснения.
"""

from typing import Dict, List, Optional
import polars as pl

from src.data.cloud_loader import init_loader, get_loader
from src.features.graph_builder import build_behavior_graph, get_graph_statistics
from src.features.pattern_miner import extract_patterns, pattern_to_string
from src.features.user_profile import create_user_profile
from src.features.graph_analyzer import analyze_graph_with_yandexgpt, generate_rules_from_graph
from src.modeling.nbo_model import recommend as ml_recommend
from src.modeling.rule_engine import RuleEngine
from src.app.explainer import explain_recommendation


def process_user(
    user_id: str,
    use_cloud: bool = True,
    use_yandexgpt_for_analysis: bool = True,
    top_k: int = 3
) -> Dict:
    """
    Обрабатывает пользователя и возвращает рекомендации.
    
    :param user_id: ID пользователя
    :param use_cloud: Использовать данные из облака
    :param use_yandexgpt_for_analysis: Использовать YandexGPT для анализа графа
    :param top_k: Количество рекомендаций
    :return: Словарь с рекомендациями и анализом
    """
    # Загрузка данных
    if use_cloud:
        loader = get_loader()
        if loader is None:
            loader = init_loader(
                public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
            )
        
        # Для публичных папок без API токена нужно указывать конкретные файлы
        # Ограничиваем количество файлов для быстрой загрузки
        num_files = 3  # Уменьшено с 10 до 3 для быстрой загрузки
        start_file = 1082  # Начальный номер файла
        
        marketplace_files = [
            f"{i:05d}.pq" for i in range(start_file, start_file + num_files)
        ]
        payments_files = [
            f"{i:05d}.pq" for i in range(start_file, start_file + num_files)
        ]
        
        # Оптимизированная загрузка: сначала фильтруем по user_id, затем загружаем только нужные данные
        print(f"Загрузка данных для пользователя {user_id}...")
        
        try:
            print(f"📊 Фильтрация marketplace events для пользователя {user_id}...")
            # Оптимизация: используем projection pushdown - выбираем только нужные колонки до фильтрации
            marketplace_lazy = loader.load_marketplace_events(file_list=marketplace_files, days=5)
            # Фильтруем по user_id на уровне LazyFrame (эффективно)
            # Используем collect_schema() чтобы избежать PerformanceWarning
            if marketplace_lazy is not None:
                schema = marketplace_lazy.collect_schema()
                if "user_id" in schema:
                    print(f"🔍 Фильтруем по user_id {user_id}...")
                    # Оптимизация: сначала фильтруем, потом выбираем колонки (projection pushdown)
                    user_marketplace_lazy = (
                        marketplace_lazy
                        .filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))
                        # Выбираем только нужные колонки для ускорения
                        .select([
                            "user_id", "item_id", "timestamp", "domain",
                            pl.col("category_id").alias("category_id"),
                            pl.col("brand_id").alias("brand_id"),
                            pl.col("action_type").alias("action_type"),
                            pl.col("subdomain").alias("subdomain"),
                            pl.col("price").alias("price"),
                            pl.col("count").alias("count"),
                            pl.col("os").alias("os")
                        ])
                    )
                    
                    # Проверяем тип timestamp перед сортировкой
                    timestamp_dtype = schema.get("timestamp")
                    if timestamp_dtype == pl.Duration:
                        # Если timestamp в формате Duration, пропускаем сортировку
                        # Просто берем первые 100 строк
                        print("⚠ Timestamp в формате Duration, пропускаем сортировку")
                        user_marketplace = user_marketplace_lazy.limit(100).collect()
                    else:
                        # Ограничиваем количество событий для экономии памяти и токенов
                        # Берем только последние 100 событий и агрегируем
                        print("📅 Сортировка по timestamp...")
                        user_marketplace = user_marketplace_lazy.sort("timestamp", descending=True).limit(100).collect()
                    
                    print(f"✅ Найдено {user_marketplace.height} событий marketplace для пользователя {user_id}")
                    
                    # Агрегируем данные: топ категории, топ товары
                    if user_marketplace.height > 0:
                        # Группируем по категориям и товарам для упрощения
                        user_marketplace = user_marketplace.select([
                            "timestamp", "item_id", "category_id", "domain"
                        ]).head(50)  # Ограничиваем до 50 самых свежих событий
                else:
                    user_marketplace = pl.DataFrame()
            else:
                user_marketplace = pl.DataFrame()
        except Exception as e:
            import traceback
            print(f"❌ Ошибка при загрузке marketplace events: {e}")
            print(f"Трассировка: {traceback.format_exc()}")
            user_marketplace = pl.DataFrame()
        
        try:
            print(f"💳 Фильтрация payments events для пользователя {user_id}...")
            payments_lazy = loader.load_payments_events(file_list=payments_files, days=5)
            if payments_lazy is not None:
                schema = payments_lazy.collect_schema()
                if "user_id" in schema:
                    print(f"🔍 Фильтруем по user_id {user_id}...")
                    user_payments_lazy = payments_lazy.filter(
                        pl.col("user_id").cast(pl.Utf8) == str(user_id)
                    )
                    
                    # Проверяем тип timestamp перед сортировкой
                    timestamp_dtype = schema.get("timestamp")
                    if timestamp_dtype == pl.Duration:
                        # Если timestamp в формате Duration, пропускаем сортировку
                        print("⚠ Timestamp в формате Duration, пропускаем сортировку")
                        user_payments = user_payments_lazy.limit(50).collect()
                    else:
                        # Ограничиваем и агрегируем платежи
                        print("📅 Сортировка по timestamp...")
                        user_payments = user_payments_lazy.sort("timestamp", descending=True).limit(50).collect()
                    
                    print(f"✅ Найдено {user_payments.height} платежей для пользователя {user_id}")
                    
                    if user_payments.height > 0:
                        # Агрегируем: сумма по брендам
                        user_payments = user_payments.select([
                            "timestamp", "brand_id", "amount", "domain"
                        ]).head(30)  # Ограничиваем до 30 самых свежих платежей
                else:
                    user_payments = pl.DataFrame()
            else:
                user_payments = pl.DataFrame()
        except Exception as e:
            import traceback
            print(f"❌ Ошибка при загрузке payments events: {e}")
            print(f"Трассировка: {traceback.format_exc()}")
            user_payments = pl.DataFrame()
        
        user_events = {
            "marketplace": user_marketplace,
            "payments": user_payments
        }
        
        # Проверяем, что есть хотя бы какие-то данные
        total_events = user_marketplace.height + user_payments.height
        if total_events == 0:
            print(f"⚠ Предупреждение: для пользователя {user_id} не найдено событий в загруженных файлах")
        else:
            print(f"✅ Всего найдено {total_events} событий для пользователя {user_id} (marketplace: {user_marketplace.height}, payments: {user_payments.height})")
    else:
        # Локальная загрузка (если реализована)
        from src.data.loader import load_user_events
        user_events = load_user_events(data_root="data/", user_id=user_id, days=2)
    
    # Построение графа
    print(f"🕸️ Построение графа поведения для пользователя {user_id}...")
    graph = build_behavior_graph(
        mp_df=user_events["marketplace"],
        pay_df=user_events["payments"],
        user_id=user_id,
        time_window_hours=24
    )
    print(f"✅ Граф построен: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} рёбер")
    
    graph_stats = get_graph_statistics(graph)
    
    # Анализ графа через YandexGPT (опционально)
    graph_analysis = None
    if use_yandexgpt_for_analysis and graph.number_of_nodes() > 0:
        try:
            graph_analysis = analyze_graph_with_yandexgpt(graph, user_id)
        except Exception as e:
            print(f"Ошибка анализа графа через YandexGPT: {e}")
    
    # Извлечение паттернов
    print(f"🔍 Извлечение паттернов поведения...")
    patterns = extract_patterns(user_events, min_pattern_len=3, min_support=2)
    pattern_strings = [pattern_to_string(p) for p in patterns]
    print(f"✅ Найдено {len(patterns)} паттернов")
    
    # Генерация правил из графа через YandexGPT
    graph_rules = []
    if use_yandexgpt_for_analysis and patterns:
        try:
            print(f"🤖 Генерация правил из графа через YandexGPT...")
            graph_rules = generate_rules_from_graph(graph, user_id)
            print(f"✅ Сгенерировано {len(graph_rules)} правил")
        except Exception as e:
            print(f"❌ Ошибка генерации правил из графа: {e}")
    
    # Создание профиля пользователя
    print(f"👤 Создание профиля пользователя...")
    profile = create_user_profile(
        user_events=user_events,
        patterns=patterns,
        user_id=user_id
    )
    print(f"✅ Профиль создан")
    
    # Рекомендации через модель ML (с улучшенным fallback)
    print(f"🤖 Генерация ML рекомендаций...")
    ml_recommendations = []
    try:
        # Передаем граф и паттерны для улучшенного fallback
        ml_recommendations = ml_recommend(profile, top_k=top_k, graph=graph, patterns=patterns)
        print(f"✅ Получено {len(ml_recommendations)} ML рекомендаций")
    except Exception as e:
        print(f"⚠ Ошибка ML рекомендаций: {e}")
        # Используем улучшенный fallback напрямую
        from src.modeling.nbo_model import NBOModel
        model = NBOModel()
        ml_recommendations = model._fallback_recommendations(profile, top_k, graph, patterns)
    
    # Рекомендации через правила
    rule_engine = RuleEngine()
    rule_recommendations = []
    if pattern_strings:
        try:
            rule_recommendations = rule_engine.recommend_from_patterns(
                pattern_strings,
                user_context=profile
            )
        except Exception as e:
            print(f"Ошибка рекомендаций по правилам: {e}")
    
    # Объединяем рекомендации
    all_recommendations = []
    
    # Добавляем ML рекомендации
    for rec in ml_recommendations:
        product = rec["product"]
        score = rec["score"]
        
        # Генерируем объяснение (используем YandexGPT только если включен)
        try:
            reason = explain_recommendation(profile, product, use_yandexgpt=use_yandexgpt_for_analysis)
        except Exception as e:
            print(f"Ошибка генерации объяснения: {e}")
            reason = f"Рекомендуется на основе вашего профиля"
        
        all_recommendations.append({
            "product": product,
            "score": score,
            "source": "ML модель",
            "reason": reason
        })
    
    # Добавляем рекомендации по правилам
    for rec in rule_recommendations[:top_k]:
        product = rec["product"]
        score = rec["score"]
        
        # Берем лучшее объяснение из правил
        if rec["reasons"]:
            reason = rec["reasons"][0]["reason"]
        else:
            try:
                reason = explain_recommendation(profile, product, use_yandexgpt=use_yandexgpt_for_analysis)
            except:
                reason = f"Рекомендуется на основе паттернов поведения"
        
        all_recommendations.append({
            "product": product,
            "score": score,
            "source": "Правила",
            "reason": reason
        })
    
    # Сортируем и берем топ-K
    all_recommendations.sort(key=lambda x: x["score"], reverse=True)
    final_recommendations = all_recommendations[:top_k]
    
    return {
        "user_id": user_id,
        "profile": profile,
        "graph": graph,  # Добавляем сам граф для визуализации
        "graph_stats": graph_stats,
        "patterns": pattern_strings,
        "graph_analysis": graph_analysis,
        "graph_rules": graph_rules,
        "recommendations": final_recommendations
    }


def main() -> None:
    """
    Основная функция для демонстрации работы системы.
    """
    print("Система рекомендаций Next Best Offer для ПСБ")
    print("=" * 50)
    
    # Пример использования
    user_id = input("Введите ID пользователя (или нажмите Enter для тестового): ").strip()
    if not user_id:
        user_id = "12345"  # Тестовый ID
    
    print(f"\nОбработка пользователя {user_id}...")
    
    try:
        result = process_user(
            user_id=user_id,
            use_cloud=True,
            use_yandexgpt_for_analysis=True,
            top_k=3
        )
        
        print("\n" + "=" * 50)
        print("РЕКОМЕНДАЦИИ:")
        print("=" * 50)
        
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"\n{i}. {rec['product']}")
            print(f"   Оценка: {rec['score']:.2f}")
            print(f"   Источник: {rec['source']}")
            print(f"   Объяснение: {rec['reason']}")
        
        print("\n" + "=" * 50)
        print("СТАТИСТИКА:")
        print("=" * 50)
        print(f"Паттернов найдено: {len(result['patterns'])}")
        print(f"Узлов в графе: {result['graph_stats']['nodes']}")
        print(f"Связей в графе: {result['graph_stats']['edges']}")
        
    except Exception as e:
        print(f"Ошибка при обработке пользователя: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

