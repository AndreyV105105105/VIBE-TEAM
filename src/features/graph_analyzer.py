"""
Модуль для анализа графов поведения через YandexGPT.

Использует YandexGPT для:
- Анализа структуры графа поведения
- Выявления значимых паттернов в графе
- Генерации правил на основе графа
"""

import networkx as nx
from typing import Dict, List, Optional
import json

from src.utils.yandex_gpt_client import call_yandex_gpt


def graph_to_text_description(
    graph: nx.DiGraph, 
    max_nodes: int = 10,  # Уменьшено с 15 до 10 для экономии токенов
    brands_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Преобразует граф в компактное текстовое описание для анализа YandexGPT.
    Максимально оптимизировано для экономии токенов.
    
    :param graph: Граф поведения
    :param max_nodes: Максимальное количество узлов для описания (уменьшено до 15)
    :param brands_map: Маппинг brand_id -> brand_name (опционально)
    :return: Компактное текстовое описание графа
    """
    if graph.number_of_nodes() == 0:
        return "Граф пуст"
    
    # Берем только топ узлы по степени (самые важные)
    degrees = dict(graph.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    
    # Агрегируем узлы по типу и категориям/брендам
    node_types = {}
    categories = {}
    brands = {}
    total_brand_amount = 0
    
    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Обрабатываем все типы узлов: item, category, brand, brand_category
        if node_type in ["item", "category"]:
            category = data.get("category") or data.get("category_id")
            if category:
                categories[category] = categories.get(category, 0) + 1
        elif node_type in ["brand", "brand_category"]:
            brand_id = data.get("brand_id")
            amount = abs(data.get("amount", 0))
            if brand_id:
                brands[brand_id] = brands.get(brand_id, 0) + amount
                total_brand_amount += amount
    
    # Берем только топ связи по весу (еще более агрессивно)
    edges_with_weights = [(u, v, data.get("weight", 1)) for u, v, data in graph.edges(data=True)]
    top_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)[:8]  # Только топ 8
    
    # Максимально компактное описание (минимум токенов)
    # Формат: только ключевые данные, без лишних слов
    
    parts = []
    
    # Типы узлов (сокращенно)
    if node_types:
        type_parts = [f"{k[:2]}:{v}" for k, v in sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:3]]
        parts.append("Т:" + ",".join(type_parts))
    
    # Топ категории (только топ 3, сокращенные названия)
    if categories:
        top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        cat_parts = []
        for cat, count in top_cats:
            # Сокращаем длинные названия категорий
            cat_short = cat[:15] if len(cat) > 15 else cat
            cat_parts.append(f"{cat_short}:{count}")
        parts.append("К:" + ",".join(cat_parts))
    
    # Топ бренды (только топ 3, без названий - только ID)
    if brands:
        top_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:3]
        brand_parts = [f"Б{bid}:${amt:.0f}" for bid, amt in top_brands]
        parts.append("Б:" + ",".join(brand_parts))
    
    # Топ связи (только топ 3, очень компактно)
    if top_edges:
        edge_parts = []
        for u, v, weight in top_edges[:3]:
            u_type = graph.nodes[u].get("type", "?")[0]  # Первая буква
            v_type = graph.nodes[v].get("type", "?")[0]
            edge_parts.append(f"{u_type}→{v_type}:{weight}")
        parts.append("С:" + ",".join(edge_parts))
    
    # Статистика (одна строка)
    if graph.number_of_nodes() > 0:
        avg_degree = sum(degrees.values()) / graph.number_of_nodes()
        parts.append(f"Ст:{graph.number_of_nodes()}н,{graph.number_of_edges()}с,{avg_degree:.1f}ср")
    
    return "|".join(parts)  # Разделитель | вместо переносов строк


def analyze_graph_with_yandexgpt(
    graph: nx.DiGraph,
    user_id: str,
    brands_map: Optional[Dict[str, str]] = None,
    max_nodes: int = 10  # Уменьшено до 10 для экономии токенов
) -> Dict[str, str]:
    """
    Анализирует граф поведения через YandexGPT.
    
    :param graph: Граф поведения пользователя
    :param user_id: ID пользователя
    :param brands_map: Маппинг brand_id -> brand_name (опционально)
    :param max_nodes: Максимальное количество узлов для анализа
    :return: Словарь с результатами анализа
    """
    graph_description = graph_to_text_description(graph, max_nodes, brands_map=brands_map)
    
    # Максимально сжатый промпт
    prompt = f"П:{user_id}|{graph_description}|Продукт? (Ипотека/Кредитка/Вклад/Кредит) Обоснование: 1-2 предл."
    
    instructions = "Эксперт по поведенческим данным. Анализ графа → продукт + краткое обоснование."
    
    try:
        analysis = call_yandex_gpt(
            input_text=prompt,
            instructions=instructions,
            temperature=0.3
        )
        
        return {
            "user_id": user_id,
            "analysis": analysis,
            "graph_stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0
            }
        }
    except Exception as e:
        print(f"Ошибка при анализе графа через YandexGPT: {e}")
        return {
            "user_id": user_id,
            "analysis": "Не удалось проанализировать граф",
            "error": str(e)
        }


def extract_patterns_from_graph(
    graph: nx.DiGraph,
    max_path_length: int = 2  # Уменьшено до 2 для экономии токенов
) -> List[str]:
    """
    Извлекает паттерны из графа через анализ путей.
    Оптимизировано для экономии токенов - возвращает только топ паттерны.
    
    :param graph: Граф поведения
    :param max_path_length: Максимальная длина пути для анализа
    :return: Список найденных паттернов (топ 10)
    """
    # Находим все простые пути от START
    if "START" not in graph:
        return []
    
    patterns = []
    
    # Находим узлы, достижимые из START
    try:
        reachable = nx.descendants(graph, "START")
        
        # Ограничиваем количество узлов для анализа (еще более агрессивно)
        for target in list(reachable)[:5]:  # Уменьшено до 5 для экономии токенов
            try:
                paths = list(nx.all_simple_paths(
                    graph, "START", target, cutoff=max_path_length
                ))[:2]  # Берем только первые 2 пути для экономии токенов
                
                for path in paths:
                    if len(path) >= 3:  # Минимальная длина паттерна
                        # Упрощаем названия узлов для компактности
                        simplified_path = []
                        for node in path:
                            if node == "START":
                                simplified_path.append("START")
                            else:
                                node_data = graph.nodes[node]
                                node_type = node_data.get("type", "unknown")
                                if node_type in ["category", "item"]:
                                    # Используем category если доступна, иначе category_id
                                    category = node_data.get("category") or node_data.get("category_id", "?")
                                    simplified_path.append(f"Кат_{category}")
                                elif node_type in ["brand", "brand_category"]:
                                    brand_id = node_data.get("brand_id", "?")
                                    category = node_data.get("category", "")
                                    if category:
                                        simplified_path.append(f"Бр_{brand_id}_{category[:10]}")
                                    else:
                                        simplified_path.append(f"Бренд_{brand_id}")
                                else:
                                    simplified_path.append(str(node)[:20])  # Обрезаем длинные названия
                        pattern_str = " → ".join(simplified_path)
                        patterns.append(pattern_str)
            except:
                continue
    except:
        pass
    
    return patterns[:5]  # Ограничиваем до 5 топ паттернов для экономии токенов


def generate_rules_from_graph(
    graph: nx.DiGraph,
    user_id: str
) -> List[Dict[str, str]]:
    """
    Генерирует правила рекомендаций на основе графа через YandexGPT.
    
    :param graph: Граф поведения
    :param user_id: ID пользователя
    :return: Список правил в формате {"pattern": "...", "product": "...", "reason": "..."}
    """
    patterns = extract_patterns_from_graph(graph)
    
    if not patterns:
        return []
    
    # Только топ 3 паттерна, сжатый формат
    patterns_text = ",".join(patterns[:3])
    
    # Максимально сжатый промпт
    prompt = f"П:{user_id}|Пат:{patterns_text}|Продукт+обоснование JSON:{{product,reason}}"
    
    instructions = "Эксперт банковских рекомендаций. Паттерны → продукт (Ипотека/Кредитка/Вклад/Кредит)."
    
    try:
        response = call_yandex_gpt(
            input_text=prompt,
            instructions=instructions,
            temperature=0.2
        )
        
        # Пытаемся извлечь JSON из ответа
        import re
        # Пробуем найти объект или массив
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                rule = json.loads(json_match.group())
                # Если это один объект, оборачиваем в список
                if isinstance(rule, dict):
                    return [rule]
                return rule if isinstance(rule, list) else [rule]
            except:
                pass
        
        # Пробуем массив
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                rules = json.loads(json_match.group())
                return rules if isinstance(rules, list) else [rules]
            except:
                pass
        
        # Если не удалось распарсить JSON, создаем простое правило
        return [{
            "pattern": patterns[0] if patterns else "unknown",
            "product": "Кредитная карта",
            "reason": "На основе анализа паттернов поведения"
        }]
        
    except Exception as e:
        print(f"Ошибка при генерации правил: {e}")
        return []

