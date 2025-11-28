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


def graph_to_text_description(graph: nx.DiGraph, max_nodes: int = 15) -> str:
    """
    Преобразует граф в компактное текстовое описание для анализа YandexGPT.
    Максимально оптимизировано для экономии токенов.
    
    :param graph: Граф поведения
    :param max_nodes: Максимальное количество узлов для описания (уменьшено до 15)
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
        
        if node_type == "item":
            category_id = data.get("category_id")
            if category_id:
                categories[category_id] = categories.get(category_id, 0) + 1
        elif node_type == "brand":
            brand_id = data.get("brand_id")
            amount = abs(data.get("amount", 0))
            if brand_id:
                brands[brand_id] = brands.get(brand_id, 0) + amount
                total_brand_amount += amount
    
    # Берем только топ связи по весу (еще более агрессивно)
    edges_with_weights = [(u, v, data.get("weight", 1)) for u, v, data in graph.edges(data=True)]
    top_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)[:8]  # Только топ 8
    
    # Компактное описание (минимум токенов)
    description = "Поведение:\n"
    
    # Статистика по типам (кратко)
    type_summary = ", ".join([f"{k}:{v}" for k, v in sorted(node_types.items(), key=lambda x: x[1], reverse=True)])
    description += f"Типы: {type_summary}\n"
    
    # Топ категории (только топ 5)
    if categories:
        top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        cat_summary = ", ".join([f"Кат{cat}:{count}" for cat, count in top_cats])
        description += f"Топ категории: {cat_summary}\n"
    
    # Топ бренды (только топ 5, агрегированные суммы)
    if brands:
        top_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:5]
        brand_summary = ", ".join([f"Бр{brand}:${amount:.0f}" for brand, amount in top_brands])
        description += f"Топ бренды: {brand_summary}\n"
        description += f"Всего потрачено: ${total_brand_amount:.0f}\n"
    
    # Топ связи (только топ 5, упрощенные)
    if top_edges:
        description += "Последовательности:\n"
        for i, (u, v, weight) in enumerate(top_edges[:5], 1):
            u_type = graph.nodes[u].get("type", "?")[:1]  # Только первая буква типа
            v_type = graph.nodes[v].get("type", "?")[:1]
            description += f"{i}. {u_type}→{v_type}(w:{weight})\n"
    
    # Общая статистика (кратко)
    description += f"Стат: узлов={graph.number_of_nodes()}, связей={graph.number_of_edges()}"
    if graph.number_of_nodes() > 0:
        avg_degree = sum(degrees.values()) / graph.number_of_nodes()
        description += f", степень={avg_degree:.1f}"
    
    return description


def analyze_graph_with_yandexgpt(
    graph: nx.DiGraph,
    user_id: str,
    max_nodes: int = 15  # Уменьшено до 15 для экономии токенов
) -> Dict[str, str]:
    """
    Анализирует граф поведения через YandexGPT.
    
    :param graph: Граф поведения пользователя
    :param user_id: ID пользователя
    :param max_nodes: Максимальное количество узлов для анализа
    :return: Словарь с результатами анализа
    """
    graph_description = graph_to_text_description(graph, max_nodes)
    
    # Оптимизированный промпт (минимум токенов)
    prompt = f"""Анализ пользователя {user_id}:

{graph_description}

Ответ (3 предложения): паттерн, продукт (Ипотека/Кредитная карта/Вклад/Кредит), обоснование."""
    
    instructions = """Ты эксперт по анализу поведенческих данных и рекомендательным системам. 
Анализируй графы поведения пользователей и выявляй значимые паттерны для рекомендации финансовых продуктов."""
    
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
    max_path_length: int = 3  # Уменьшено до 3 для экономии токенов
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
                                if node_type == "item":
                                    simplified_path.append(f"Кат_{node_data.get('category_id', '?')}")
                                elif node_type == "brand":
                                    simplified_path.append(f"Бренд_{node_data.get('brand_id', '?')}")
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
    
    # Берем только топ 3 паттерна для экономии токенов
    patterns_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(patterns[:3])])
    
    # Оптимизированный промпт (минимум токенов)
    prompt = f"""Паттерны пользователя {user_id}:
{patterns_text}

Определи продукт (Ипотека/Кредитная карта/Вклад/Кредит) и обоснование (1 предложение).
JSON: {{"product": "...", "reason": "..."}}"""
    
    instructions = """Ты эксперт по рекомендательным системам для банков. 
Анализируй паттерны поведения и генерируй правила для рекомендации финансовых продуктов."""
    
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

