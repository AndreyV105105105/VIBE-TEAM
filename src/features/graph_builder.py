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
    retail_df: Optional[pl.DataFrame] = None,
    receipts_df: Optional[pl.DataFrame] = None,
    user_id: str = "",
    time_window_hours: int = 24
) -> nx.DiGraph:
    """
    Строит граф поведения пользователя с учетом всех доменов и категорий.
    
    :param mp_df: DataFrame с событиями маркетплейса (с категориями из items)
    :param pay_df: DataFrame с событиями платежей
    :param retail_df: DataFrame с событиями ритейла (с категориями из items)
    :param receipts_df: DataFrame с чеками (с категориями товаров)
    :param user_id: ID пользователя
    :param time_window_hours: Временное окно для связей (в часах)
    :return: Направленный граф поведения
    """
    G = nx.DiGraph()
    G.add_node("START", user_id=user_id, type="start")
    
    # Собираем все события с реальными временными метками и категориями
    events = []
    
    # События маркетплейса - используем категории из items если доступны
    if mp_df.height > 0:
        # Используем category из items если есть, иначе category_id
        category_col = "category" if "category" in mp_df.columns else "category_id"
        subcategory_col = "subcategory" if "subcategory" in mp_df.columns else None
        
        # Группируем по категориям и action_type
        if "action_type" in mp_df.columns:
            category_agg = mp_df.group_by([category_col, "action_type"]).agg([
                pl.count().alias("count"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("item_id").first().alias("top_item"),
                pl.col("brand_id").first().alias("brand_id")
            ]).sort("count", descending=True).head(30)
        else:
            category_agg = mp_df.group_by(category_col).agg([
                pl.count().alias("count"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("item_id").first().alias("top_item"),
                pl.col("brand_id").first().alias("brand_id")
            ]).sort("count", descending=True).head(30)
        
        for row in category_agg.iter_rows(named=True):
            category = row.get(category_col) or row.get("category_id") or "unknown"
            action_type = row.get("action_type", "view")
            count = row.get("count", 1)
            timestamp = row.get("first_timestamp")
            if isinstance(timestamp, datetime):
                event_time = timestamp
            else:
                event_time = datetime.now()
            
            events.append({
                "timestamp": event_time,
                "type": action_type,
                "item_id": row.get("top_item", "unknown"),
                "category": str(category),
                "brand_id": row.get("brand_id"),
                "domain": "marketplace",
                "weight": count
            })
    
    # События ритейла - используем категории из items
    if retail_df is not None and retail_df.height > 0:
        category_col = "category" if "category" in retail_df.columns else "category_id"
        if "action_type" in retail_df.columns:
            retail_agg = retail_df.group_by([category_col, "action_type"]).agg([
                pl.count().alias("count"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("item_id").first().alias("top_item"),
                pl.col("brand_id").first().alias("brand_id")
            ]).sort("count", descending=True).head(20)
        else:
            retail_agg = retail_df.group_by(category_col).agg([
                pl.count().alias("count"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("item_id").first().alias("top_item"),
                pl.col("brand_id").first().alias("brand_id")
            ]).sort("count", descending=True).head(20)
        
        for row in retail_agg.iter_rows(named=True):
            category = row.get(category_col) or "unknown"
            action_type = row.get("action_type", "view")
            count = row.get("count", 1)
            timestamp = row.get("first_timestamp")
            if isinstance(timestamp, datetime):
                event_time = timestamp
            else:
                event_time = datetime.now()
            
            events.append({
                "timestamp": event_time,
                "type": action_type,
                "item_id": row.get("top_item", "unknown"),
                "category": str(category),
                "brand_id": row.get("brand_id"),
                "domain": "retail",
                "weight": count
            })
    
    # События платежей - группируем по брендам
    if pay_df.height > 0:
        brand_totals = pay_df.group_by("brand_id").agg([
            pl.sum("amount").alias("total_amount"),
            pl.count().alias("count"),
            pl.col("timestamp").min().alias("first_timestamp")
        ]).sort("total_amount", descending=True).head(20)
        
        for row in brand_totals.iter_rows(named=True):
            brand_id = row.get("brand_id", "unknown")
            total_amount = row.get("total_amount", 0)
            count = row.get("count", 1)
            timestamp = row.get("first_timestamp")
            if isinstance(timestamp, datetime):
                event_time = timestamp
            else:
                event_time = datetime.now()
            
            events.append({
                "timestamp": event_time,
                "type": "transaction",
                "brand_id": brand_id,
                "amount": total_amount,
                "domain": "payments",
                "weight": count
            })
    
    # Чеки (receipts) - детализация покупок с категориями товаров
    if receipts_df is not None and receipts_df.height > 0:
        category_col = "category" if "category" in receipts_df.columns else None
        if category_col:
            receipt_agg = receipts_df.group_by([category_col, "brand_id"]).agg([
                pl.sum("count").alias("total_count"),
                pl.sum("price").alias("total_price"),
                pl.count().alias("transaction_count"),
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("approximate_item_id").first().alias("item_id")
            ]).sort("total_price", descending=True).head(15)
            
            for row in receipt_agg.iter_rows(named=True):
                category = row.get(category_col) or "unknown"
                brand_id = row.get("brand_id")
                count = row.get("transaction_count", 1)
                timestamp = row.get("first_timestamp")
                if isinstance(timestamp, datetime):
                    event_time = timestamp
                else:
                    event_time = datetime.now()
                
                events.append({
                    "timestamp": event_time,
                    "type": "purchase",
                    "item_id": row.get("item_id", "unknown"),
                    "category": str(category),
                    "brand_id": brand_id,
                    "amount": row.get("total_price", 0),
                    "domain": "receipts",
                    "weight": count
                })
    
    # Сортируем по времени для правильного построения графа
    events.sort(key=lambda x: x.get("timestamp", datetime.now()))
    
    # Строим граф с учетом всех типов событий и категорий
    for i, event in enumerate(events):
        domain = event.get("domain", "unknown")
        event_type = event.get("type", "unknown")
        category = event.get("category", "unknown")
        
        # Создаем узел в зависимости от типа события
        if event_type in ["view", "click", "add_to_cart", "order"]:
            # События просмотра/взаимодействия с товарами
            if category and category != "unknown":
                # Используем категорию как узел (более информативно)
                node_id = f"cat_{category}_{domain}"
                G.add_node(
                    node_id,
                    type="category",
                    category=category,
                    domain=domain,
                    item_id=event.get("item_id"),
                    brand_id=event.get("brand_id"),
                    action_type=event_type
                )
            else:
                # Fallback на item_id
                node_id = f"item_{event.get('item_id', 'unknown')}_{domain}"
                G.add_node(
                    node_id,
                    type="item",
                    item_id=event.get("item_id"),
                    domain=domain,
                    action_type=event_type
                )
        elif event_type in ["transaction", "purchase"]:
            # События платежей/покупок
            brand_id = event.get("brand_id", "unknown")
            if category and category != "unknown":
                # Узел категории бренда
                node_id = f"brand_cat_{brand_id}_{category}"
                G.add_node(
                    node_id,
                    type="brand_category",
                    brand_id=brand_id,
                    category=category,
                    amount=event.get("amount", 0),
                    domain=domain
                )
            else:
                # Fallback на brand_id
                node_id = f"brand_{brand_id}"
                G.add_node(
                    node_id,
                    type="brand",
                    brand_id=brand_id,
                    amount=event.get("amount", 0),
                    domain=domain
                )
        else:
            continue
        
        # Связываем с предыдущими событиями в временном окне
        for j in range(max(0, i-20), i):  # Увеличиваем окно до 20 событий
            prev_event = events[j]
            time_diff = (event["timestamp"] - prev_event["timestamp"]).total_seconds()
            
            if 0 < time_diff < time_window_hours * 3600:
                # Определяем предыдущий узел
                prev_domain = prev_event.get("domain", "unknown")
                prev_type = prev_event.get("type", "unknown")
                prev_category = prev_event.get("category", "unknown")
                
                if prev_type in ["view", "click", "add_to_cart", "order"]:
                    if prev_category and prev_category != "unknown":
                        prev_node = f"cat_{prev_category}_{prev_domain}"
                    else:
                        prev_node = f"item_{prev_event.get('item_id', 'unknown')}_{prev_domain}"
                elif prev_type in ["transaction", "purchase"]:
                    prev_brand_id = prev_event.get("brand_id", "unknown")
                    if prev_category and prev_category != "unknown":
                        prev_node = f"brand_cat_{prev_brand_id}_{prev_category}"
                    else:
                        prev_node = f"brand_{prev_brand_id}"
                else:
                    continue
                
                # Увеличиваем вес, если связь уже есть
                if G.has_edge(prev_node, node_id):
                    G[prev_node][node_id]["weight"] += event.get("weight", 1)
                else:
                    G.add_edge(
                        prev_node,
                        node_id,
                        weight=event.get("weight", 1),
                        time_diff=time_diff,
                        domain_transition=f"{prev_domain}→{domain}"
                    )
        
        # Связь с START
        if not G.has_edge("START", node_id):
            G.add_edge("START", node_id, weight=event.get("weight", 1))
    
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

