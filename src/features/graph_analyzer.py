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
    max_nodes: int = 15, 
    brands_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Преобразует граф в текстовое описание для анализа YandexGPT.
    Оптимизировано для баланса между экономией токенов и понятностью.
    """
    if graph.number_of_nodes() == 0:
        return "Graph is empty."
    
    # Сортируем узлы по важности (степени)
    degrees = dict(graph.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    subgraph = graph.subgraph([n[0] for n in top_nodes])
    
    # Группируем узлы по типам для структурированного описания
    description_parts = []
    
    # 1. Категории (Categories)
    categories = []
    for n, d in subgraph.nodes(data=True):
        if d.get("type") in ["category", "brand_category"]:
            cat = d.get("category", str(n))
            if cat and cat != "unknown":
                categories.append(cat)
    
    if categories:
        unique_cats = sorted(list(set(categories)))[:5]
        description_parts.append(f"Top Categories: {', '.join(unique_cats)}")
    
    # 2. Бренды (Brands)
    brands = []
    for n, d in subgraph.nodes(data=True):
        if d.get("type") in ["brand", "brand_category"]:
            brand_id = str(d.get("brand_id", ""))
            # Пытаемся найти имя бренда
            brand_name = brands_map.get(brand_id) if brands_map else None
            if not brand_name:
                brand_name = f"Brand_{brand_id}"
            
            amount = d.get("amount", 0)
            brands.append(f"{brand_name}" + (f"(${amount:.0f})" if amount > 0 else ""))
    
    if brands:
        description_parts.append(f"Top Brands: {', '.join(brands[:5])}")
        
    # 3. Ключевые действия (Edges)
    # Сортируем ребра по весу
    edges = sorted(subgraph.edges(data=True), key=lambda x: x[2].get("weight", 1), reverse=True)[:8]
    actions = []
    
    for u, v, data in edges:
        weight = data.get("weight", 1)
        # Определяем тип действия по весу
        action = "interacted"
        if weight >= 5: action = "bought"
        elif weight >= 3: action = "cart"
        elif weight >= 2: action = "viewed"
        
        # Получаем понятные имена
        u_data = subgraph.nodes[u]
        v_data = subgraph.nodes[v]
        
        u_name = u_data.get("category") or u_data.get("brand_name") or u
        v_name = v_data.get("category") or v_data.get("brand_name") or v
        
        # Улучшаем имена брендов
        if brands_map:
            if u_data.get("type") == "brand":
                bid = str(u_data.get("brand_id", ""))
                u_name = brands_map.get(bid, u_name)
            if v_data.get("type") == "brand":
                bid = str(v_data.get("brand_id", ""))
                v_name = brands_map.get(bid, v_name)
        
        actions.append(f"{u_name} -> {action} -> {v_name}")
        
    if actions:
        description_parts.append("Actions:\n" + "\n".join(actions))
        
    return "\n".join(description_parts)


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

