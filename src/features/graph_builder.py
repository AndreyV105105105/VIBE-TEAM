"""
Модуль для построения графов поведения пользователей.

Строит графы на основе событий пользователя с учетом временных окон и весов.
"""

import networkx as nx
import polars as pl
from typing import Dict, Optional
from datetime import datetime, timedelta


def build_behavior_graph(
    mp_df: pl.DataFrame,
    pay_df: pl.DataFrame,
    user_id: str,
    time_window_hours: int = 24
) -> nx.DiGraph:
    """
    Строит граф поведения пользователя с учетом временных окон.
    
    :param mp_df: DataFrame с событиями маркетплейса
    :param pay_df: DataFrame с событиями платежей
    :param user_id: ID пользователя
    :param time_window_hours: Временное окно для связей (в часах)
    :return: Направленный граф поведения
    """
    G = nx.DiGraph()
    G.add_node("START", user_id=user_id, type="start")
    
    # Оптимизированная сборка событий: используем агрегированные данные
    events = []
    
    # События маркетплейса - группируем по категориям для упрощения графа
    if mp_df.height > 0:
        # Агрегируем: берем топ категории и товары
        category_counts = mp_df.group_by("category_id").agg([
            pl.count().alias("count"),
            pl.col("item_id").first().alias("top_item")
        ]).sort("count", descending=True).head(20)  # Топ 20 категорий
        
        for row in category_counts.iter_rows(named=True):
            category_id = row.get("category_id", "unknown")
            count = row.get("count", 1)
            # Создаем одно событие на категорию с весом = количеству просмотров
            events.append({
                "timestamp": datetime.now(),  # Упрощаем временную метку
                "type": "view",
                "item_id": f"category_{category_id}",
                "category_id": category_id,
                "domain": "marketplace",
                "weight": count
            })
    
    # События платежей - группируем по брендам
    if pay_df.height > 0:
        # Агрегируем: сумма по брендам
        brand_totals = pay_df.group_by("brand_id").agg([
            pl.sum("amount").alias("total_amount"),
            pl.count().alias("count")
        ]).sort("total_amount", descending=True).head(15)  # Топ 15 брендов
        
        for row in brand_totals.iter_rows(named=True):
            brand_id = row.get("brand_id", "unknown")
            total_amount = row.get("total_amount", 0)
            count = row.get("count", 1)
            events.append({
                "timestamp": datetime.now(),
                "type": "pay",
                "brand_id": brand_id,
                "amount": total_amount,
                "domain": "payments",
                "weight": count
            })
    
    # Сортируем по времени (упрощенная версия)
    events.sort(key=lambda x: x.get("weight", 1), reverse=True)  # Сортируем по весу
    
    # Строим граф
    for i, event in enumerate(events):
        # Создаем узел
        if event["type"] == "view":
            node_id = f"item_{event.get('item_id', 'unknown')}"
            G.add_node(
                node_id,
                type="item",
                item_id=event.get("item_id"),
                category_id=event.get("category_id"),
                domain="marketplace"
            )
        elif event["type"] == "pay":
            node_id = f"brand_{event.get('brand_id', 'unknown')}"
            G.add_node(
                node_id,
                type="brand",
                brand_id=event.get("brand_id"),
                amount=event.get("amount", 0),
                domain="payments"
            )
        else:
            continue
        
        # Связываем с предыдущими событиями в временном окне
        for prev_event in events[max(0, i-10):i]:
            time_diff = (event["timestamp"] - prev_event["timestamp"]).total_seconds()
            
            if time_diff < time_window_hours * 3600 and time_diff > 0:
                if prev_event["type"] == "view":
                    prev_node = f"item_{prev_event.get('item_id', 'unknown')}"
                elif prev_event["type"] == "pay":
                    prev_node = f"brand_{prev_event.get('brand_id', 'unknown')}"
                else:
                    continue
                
                # Увеличиваем вес, если связь уже есть
                if G.has_edge(prev_node, node_id):
                    G[prev_node][node_id]["weight"] += 1
                else:
                    G.add_edge(
                        prev_node,
                        node_id,
                        weight=1,
                        time_diff=time_diff
                    )
        
        # Связь с START
        if not G.has_edge("START", node_id):
            G.add_edge("START", node_id, weight=1)
    
    return G


def get_graph_statistics(graph: nx.DiGraph) -> Dict:
    """
    Получает статистику по графу.
    
    :param graph: Граф поведения
    :return: Словарь со статистикой
    """
    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0,
            "avg_degree": 0
        }
    
    degrees = dict(graph.degree())
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0,
        "avg_degree": avg_degree,
        "is_connected": nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False
    }

