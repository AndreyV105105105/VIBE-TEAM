"""
Модуль для создания профилей пользователей.

Формирует профили на основе агрегированных данных и паттернов поведения.
"""

from typing import Dict, List, Optional
import polars as pl


def create_user_profile(
    user_events: Dict[str, pl.DataFrame],
    patterns: Optional[List] = None,
    user_id: Optional[str] = None
) -> Dict:
    """
    Создает профиль пользователя на основе событий и паттернов.
    
    :param user_events: Словарь с событиями по доменам
    :param patterns: Список паттернов поведения
    :param user_id: ID пользователя
    :return: Словарь с профилем пользователя
    """
    profile = {}
    
    if user_id:
        profile["user_id"] = user_id
    
    # Статистики по маркетплейсу
    mp_df = user_events.get("marketplace", pl.DataFrame())
    if mp_df.height > 0:
        profile["num_views"] = mp_df.height
        profile["unique_items"] = mp_df["item_id"].n_unique() if "item_id" in mp_df.columns else 0
        
        # Топ категория
        if "category_id" in mp_df.columns:
            top_category = mp_df["category_id"].mode().to_list()
            profile["top_category"] = top_category[0] if top_category else None
        else:
            profile["top_category"] = None
        
        # Регион (если есть)
        if "region" in mp_df.columns:
            region = mp_df["region"].mode().to_list()
            profile["region"] = region[0] if region else None
        else:
            profile["region"] = None
    else:
        profile["num_views"] = 0
        profile["unique_items"] = 0
        profile["top_category"] = None
        profile["region"] = None
    
    # Статистики по платежам
    pay_df = user_events.get("payments", pl.DataFrame())
    if pay_df.height > 0:
        profile["num_payments"] = pay_df.height
        
        if "amount" in pay_df.columns:
            # Проверяем тип данных amount
            amount_col = pay_df["amount"]
            amount_dtype = amount_col.dtype
            
            # Преобразуем в числовой тип если нужно
            if amount_dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                try:
                    pay_df = pay_df.with_columns(pl.col("amount").cast(pl.Float64, strict=False))
                except:
                    pass
            
            # Всегда используем абсолютные значения для расчетов (отрицательные = возвраты, но считаем как положительные)
            amount_abs = pay_df["amount"].abs()
            
            # Диагностика: показываем примеры исходных значений и полную статистику
            sample_values = pay_df["amount"].head(10).to_list() if pay_df.height > 0 else []
            negative_count = (pay_df["amount"] < 0).sum() if pay_df.height > 0 else 0
            
            # Получаем полную статистику для диагностики
            amount_stats = pay_df.select([
                pl.col("amount").min().alias("min"),
                pl.col("amount").max().alias("max"),
                pl.col("amount").mean().alias("mean"),
                pl.col("amount").abs().min().alias("min_abs"),
                pl.col("amount").abs().max().alias("max_abs"),
                pl.col("amount").abs().mean().alias("mean_abs"),
                pl.col("amount").abs().quantile(0.95).alias("p95"),  # 95-й перцентиль
                pl.col("amount").abs().quantile(0.99).alias("p99")   # 99-й перцентиль
            ])
            
            if amount_stats.height > 0:
                stats = amount_stats.row(0)
                min_val, max_val, mean_val, min_abs, max_abs, mean_abs_val, p95, p99 = stats
                print(f"📊 Статистика amount (до обработки): min=${min_val:.2f}, max=${max_val:.2f}, mean=${mean_val:.2f}")
                print(f"   Абсолютные значения: min=${min_abs:.2f}, max=${max_abs:.2f}, mean=${mean_abs_val:.2f}")
                if p95 is not None:
                    print(f"   Перцентили: P95=${p95:.2f}, P99=${p99:.2f if p99 else 0:.2f}")
                print(f"   Примеры значений: {sample_values[:5]}")
                print(f"   Всего записей: {pay_df.height}")
                
                # Предупреждение, если max кажется слишком маленьким
                if max_abs is not None and max_abs < 50:
                    print(f"⚠ ВНИМАНИЕ: Максимальная сумма (${max_abs:.2f}) кажется слишком маленькой для реальных транзакций!")
                    print(f"   Проверьте, правильно ли данные загружены и не фильтруются ли большие значения.")
            
            if negative_count > 0:
                print(f"⚠ Обнаружено {negative_count} отрицательных значений amount (возвраты). Используем абсолютные значения.")
            
            # Вычисляем статистики на абсолютных значениях
            amount_mean = amount_abs.mean()
            amount_sum = amount_abs.sum()
            amount_max = amount_abs.max()
            amount_min = amount_abs.min()
            
            # Показываем, откуда берется max - это реальное максимальное значение из данных
            if amount_max is not None:
                # Находим строку с максимальным значением для диагностики
                max_row = pay_df.filter(pl.col("amount").abs() == amount_max).head(1)
                if max_row.height > 0:
                    max_info = max_row.select(["amount", "brand_id", "timestamp"]).row(0)
                    print(f"🔍 Максимальная транзакция найдена: amount=${max_info[0]:.2f}, brand_id={max_info[1]}, timestamp={max_info[2]}")
                    print(f"   Это РЕАЛЬНОЕ значение из данных пользователя, не фиксированное!")
            
            # Сохраняем значения (гарантируем, что они не отрицательные и не NaN)
            # Проверка на NaN: value == value возвращает False для NaN
            avg_val = float(amount_mean) if amount_mean is not None and amount_mean == amount_mean else 0.0
            sum_val = float(amount_sum) if amount_sum is not None and amount_sum == amount_sum else 0.0
            max_val = float(amount_max) if amount_max is not None and amount_max == amount_max else 0.0
            min_val = float(amount_min) if amount_min is not None and amount_min == amount_min else 0.0
            
            print(f"📈 Вычисленные значения из {pay_df.height} транзакций:")
            print(f"   - max_val = ${max_val:.2f} (это максимальная сумма из всех транзакций пользователя)")
            print(f"   - min_val = ${min_val:.2f} (это минимальная сумма)")
            print(f"   - avg_val = ${avg_val:.2f} (это средняя сумма)")
            print(f"   - sum_val = ${sum_val:.2f} (это общая сумма всех транзакций)")
            
            # Финальная проверка на валидность (не должно быть отрицательных после abs())
            if avg_val < 0:
                print(f"⚠ ОШИБКА: avg_tx отрицательный ({avg_val}) после abs()! Это невозможно. Устанавливаем 0")
                avg_val = 0.0
            if sum_val < 0:
                print(f"⚠ ОШИБКА: total_tx отрицательный ({sum_val}) после abs()! Это невозможно. Устанавливаем 0")
                sum_val = 0.0
            
            profile["avg_tx"] = avg_val
            profile["total_tx"] = sum_val
            profile["max_tx"] = max_val
            profile["min_tx"] = min_val
            
            print(f"✅ Финальная статистика платежей: avg_tx={profile['avg_tx']:.2f} $, total_tx={profile['total_tx']:.2f} $, записей={pay_df.height}")
            print(f"   Проверка: avg_tx >= 0: {profile['avg_tx'] >= 0}, total_tx >= 0: {profile['total_tx'] >= 0}")
        else:
            profile["avg_tx"] = 0
            profile["total_tx"] = 0
            profile["max_tx"] = 0
            profile["min_tx"] = 0
        
        # Топ бренд
        if "brand_id" in pay_df.columns:
            top_brand = pay_df["brand_id"].mode().to_list()
            profile["top_brand"] = top_brand[0] if top_brand else None
        else:
            profile["top_brand"] = None
    else:
        profile["num_payments"] = 0
        profile["avg_tx"] = 0
        profile["total_tx"] = 0
        profile["max_tx"] = 0
        profile["min_tx"] = 0
        profile["top_brand"] = None
    
    # Временные характеристики
    # Объединяем события для вычисления временных характеристик
    # Выбираем только timestamp, так как у разных доменов разные схемы
    # (marketplace имеет item_id, payments имеет brand_id, но нет item_id)
    normalized_events = []
    for df in user_events.values():
        if df.height > 0 and "timestamp" in df.columns:
            # Выбираем только timestamp для объединения
            # Это гарантирует одинаковую схему для всех доменов
            df_normalized = df.select(["timestamp"])
            normalized_events.append(df_normalized)
    
    if normalized_events:
        try:
            combined = pl.concat(normalized_events)
            timestamps = combined["timestamp"].to_list()
        except Exception as e:
            print(f"⚠ Ошибка при объединении событий для временных характеристик: {e}")
            # Собираем timestamps из каждого DataFrame отдельно
            timestamps = []
            for df in user_events.values():
                if df.height > 0 and "timestamp" in df.columns:
                    timestamps.extend(df["timestamp"].to_list())
        
        if timestamps:
            # Конвертируем в datetime если нужно
            try:
                from datetime import datetime
                dt_timestamps = [
                    t if isinstance(t, datetime) else datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                    for t in timestamps
                ]
                profile["days_active"] = (max(dt_timestamps) - min(dt_timestamps)).days + 1
                profile["events_per_day"] = len(timestamps) / max(profile["days_active"], 1)
            except:
                profile["days_active"] = 1
                profile["events_per_day"] = len(timestamps)
        else:
            profile["days_active"] = 0
            profile["events_per_day"] = 0
    else:
        profile["days_active"] = 0
        profile["events_per_day"] = 0
    
    # Паттерны
    if patterns:
        profile["num_patterns"] = len(patterns)
        
        # Кодируем паттерны как бинарные фичи
        common_patterns = [
            ("V", "P", "V"),  # просмотр → оплата → просмотр
            ("V", "V", "P"),  # два просмотра → оплата
            ("P", "V", "C"),  # оплата → просмотр → клик
            ("V", "P", "P"),  # просмотр → оплата → оплата
        ]
        
        pattern_strings = ["→".join(p) for p in patterns] if patterns else []
        
        for pattern in common_patterns:
            pattern_str = "→".join(pattern)
            profile[f"has_pattern_{pattern_str.replace('→', '_')}"] = 1 if pattern_str in pattern_strings else 0
        
        # Основной паттерн как строка
        if patterns:
            profile["pattern"] = "→".join(patterns[0]) if isinstance(patterns[0], tuple) else str(patterns[0])
        else:
            profile["pattern"] = "unknown"
    else:
        profile["num_patterns"] = 0
        profile["pattern"] = "unknown"
        for pattern in [("V", "P", "V"), ("V", "V", "P"), ("P", "V", "C"), ("V", "P", "P")]:
            pattern_str = "→".join(pattern)
            profile[f"has_pattern_{pattern_str.replace('→', '_')}"] = 0
    
    return profile


def profile_to_features(profile: Dict) -> List[float]:
    """
    Преобразует профиль в вектор признаков для модели.
    
    :param profile: Профиль пользователя
    :return: Список числовых признаков
    """
    features = []
    
    # Числовые признаки
    numeric_features = [
        "num_views", "num_payments", "avg_tx", "total_tx",
        "days_active", "events_per_day", "unique_items",
        "num_patterns"
    ]
    
    for feat in numeric_features:
        features.append(float(profile.get(feat, 0)))
    
    # Бинарные признаки паттернов
    pattern_features = [
        "has_pattern_V_P_V",
        "has_pattern_V_V_P",
        "has_pattern_P_V_C",
        "has_pattern_V_P_P"
    ]
    
    for feat in pattern_features:
        features.append(float(profile.get(feat, 0)))
    
    # Категориальные (one-hot encoding через индексы)
    if profile.get("top_category"):
        features.append(float(profile["top_category"]))
    else:
        features.append(0.0)
    
    if profile.get("region"):
        features.append(float(profile["region"]))
    else:
        features.append(0.0)
    
    return features

