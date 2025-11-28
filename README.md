# Система рекомендаций для ПСБ

> **Рекомендательная система на основе поведенческих паттернов пользователей**  
> Рекомендует финансовые продукты (ипотека, кредитные карты и др.) на основе анализа графов поведения, а не только демографических данных.

---

## Описание проекта

Система анализирует поведенческие паттерны пользователей в экосистеме ПСБ (маркетплейс, платежи, офферы) и рекомендует наиболее подходящие финансовые продукты. 

**Ключевая идея:** 
> *«Просмотр дрели → оплата сантехнику → поиск квартир» → **рекомендация ипотеки***

Система выявляет сложные последовательности действий пользователя и на их основе формирует персонализированные рекомендации с объяснениями.

---

## Цель проекта

Разработать систему рекомендаций, которая:

1. **Выявляет поведенческие паттерны**  — игнорирование сложных паттернов → их выявление через графы и pattern mining
2. **Создает профили пользователей**  — агрегация данных + embeddings паттернов
3. **Прогнозирует потребности**  — нейросеть для ранжирования продуктов
4. **Обеспечивает интерпретируемость**  — объяснения через YandexGPT API
5. **Готова к интеграции**  — локальная работа, без external calls в runtime

---

## Технологический стек

| Компонент | Технология | Назначение |
|-----------|------------|------------|
| **Обработка данных** | `polars` | Быстрое чтение Parquet, работа с 135B+ событий, чтение из HTTP/HTTPS |
| **Облачное хранилище** | `requests` + `polars` | Прямое чтение Parquet из Яндекс Диска без скачивания |
| **Валидация** | `pydantic` | Проверка целостности данных (`user_id`, `event_type`, `amount ≥ 0`) |
| **Граф поведения** | `networkx` | Построение графов взаимодействий пользователя |
| **Pattern Mining** | Встроенный Python (`Counter`, `itertools`) | Выделение последовательностей: `V→P→V`, `view→pay→view` |
| **Feature Engineering** | `polars.groupby().agg()` | Профили: `avg_tx`, `days_active`, `top_categories` |
| **Модель** | `LightFM` или `RandomForestRegressor` | Ранжирование продуктов по профилю пользователя |
| **Объяснения (XAI)** | `YandexGPT API` + fallback шаблоны | Генерация объяснений: *«Вам ипотека, потому что вы ищете квартиры»* |
| **Демо-интерфейс** | `streamlit` | Веб-интерфейс для демонстрации |
| **Визуализация** | `pyvis` | Динамические графы поведения |

> **YandexGPT используется ТОЛЬКО для объяснений**, на вход подаются только агрегаты (регион, категории, суммы) — без сырых событий.

---

## Структура проекта

```
VIBE-TEAM/
├── src/                              # Исходный код
│   ├── __init__.py
│   ├── constants.py                  # Константы проекта
│   ├── main.py                       # Точка входа в приложение
│   │
│   ├── data/                         # Работа с данными
│   │   ├── loader.py                 # Загрузка данных из локальных файлов
│   │   ├── cloud_loader.py           # Загрузка данных из Яндекс Диска (без скачивания)
│   │   └── validator.py              # Валидация через pydantic
│   │
│   ├── features/                     # Feature Engineering
│   │   ├── graph_builder.py          # Построение графа поведения (networkx)
│   │   ├── pattern_miner.py          # Извлечение паттернов (без prefixspan)
│   │   └── user_profile.py           # Создание профиля пользователя
│   │
│   ├── modeling/                     # Модели машинного обучения
│   │   ├── nbo_model.py              # Обучение и инференс модели (LightFM/RandomForest)
│   │   └── rule_engine.py            # Fallback: правила паттерн → продукт
│   │
│   └── app/                          # Приложение и интерфейс
│       ├── main.py                   # Streamlit приложение
│       └── explainer.py              # YandexGPT + кэширование объяснений
│
├── tests/                            # Unit тесты
│   └── __init__.py
│
├── models/                           # Обученные модели (создается при обучении)
│   ├── nbo_model.pkl
│   └── pattern_to_product.json       # Правила: "V_D→P_S→V_Q" → "Ипотека"
│
├── notebooks/                        # Jupyter notebooks для EDA
│   └── eda_patterns.ipynb            # EDA: поиск цепочек, статистика
│
├── data/                             # Данные (Parquet файлы) - не в репозитории
│   ├── marketplace/
│   │   └── events/                   # 575 файлов событий маркетплейса
│   ├── payments/
│   │   └── events/                   # События платежей (если есть)
│   └── offers/
│       └── events/                   # События офферов (если есть)
│
├── pyproject.toml                    # Конфигурация проекта (uv)
├── uv.lock                           # Зависимости проекта (uv)
├── requirements.txt                  # Зависимости (альтернатива uv)
├── .gitignore                        # Git ignore файл
├── .pre-commit-config.yaml           # Средства автоматизации проверки кодстайла
└── README.md                          # Этот файл
```

---

## Как это работает

### Шаг 0: Загрузка данных из облака

```python
from src.data.cloud_loader import init_loader

# Инициализируем загрузчик с публичной ссылкой на Яндекс Диск
loader = init_loader(
    public_link="https://disk.yandex.ru/d/H0ZTzS55GSz1Wg"
)

# Загружаем данные напрямую из облака (без скачивания)
# Polars читает Parquet файлы напрямую через HTTP/HTTPS
marketplace_lazy = loader.load_marketplace_events(limit=100)  # Первые 100 файлов
payments_lazy = loader.load_payments_events(limit=100)
brands_df = loader.load_brands()
users_df = loader.load_users()

# Фильтруем по конкретному пользователю (ленивая загрузка)
user_marketplace = marketplace_lazy.filter(
    pl.col("user_id") == "12345"
).collect()
```

### Шаг 1: Построение графа поведения

```python
from src.features.graph_builder import build_behavior_graph
from src.data.loader import load_user_events

# Вариант A: Из облака (рекомендуется)
user_events = {
    "marketplace": user_marketplace,
    "payments": payments_lazy.filter(pl.col("user_id") == "12345").collect()
}

# Вариант B: Из локальных файлов
# user_events = load_user_events(data_root="data/", user_id="12345", days=2)

# Строим граф с временными окнами
G = build_behavior_graph(
    mp_df=user_events["marketplace"],
    pay_df=user_events["payments"],
    user_id="12345",
    time_window_hours=24
)
```

**Результат:** Граф, где можно искать пути — например, `START → item_drill → brand_plumber → item_flat`.

### Шаг 2: Извлечение паттернов

```python
from src.features.pattern_miner import extract_patterns

# Извлекаем последовательности событий
patterns = extract_patterns(user_events, min_pattern_len=3, min_support=2)

# Пример результата:
# [("V", "P", "V"), ("V", "V", "C"), ...]
# где V=view, P=pay, C=click
```

**Результат:** Паттерны вроде `["V", "P", "V"]` = *«просмотр → оплата → просмотр»*.

### Шаг 3: Создание профиля пользователя

```python
from src.features.user_profile import create_user_profile

# Формируем профиль с фичами
profile = create_user_profile(
    user_events=user_events,
    patterns=patterns
)

# Пример результата:
# {
#     "user_id": "12345",
#     "region": 77,
#     "avg_tx": 45000,
#     "num_views": 15,
#     "num_payments": 3,
#     "has_pattern_VPV": 1,
#     "days_active": 2,
#     ...
# }
```

**Результат:** Вектор признаков для модели.

### Шаг 4: Ранжирование продуктов

```python
from src.modeling.nbo_model import recommend

# Получаем топ-3 рекомендации
recommendations = recommend(
    user_profile=profile,
    model_path="models/nbo_model.pkl"
)

# Пример результата:
# [
#     {"product": "Ипотека", "score": 0.85, "reason": "..."},
#     {"product": "Кредитная карта", "score": 0.72, "reason": "..."},
#     {"product": "Вклад", "score": 0.58, "reason": "..."}
# ]
```

**Результат:** Top-3 продуктов с вероятностями и объяснениями.

### Шаг 5: Генерация объяснений

```python
from src.app.explainer import explain_recommendation

# Генерируем объяснение через YandexGPT (с fallback)
explanation = explain_recommendation(
    profile=profile,
    product="Ипотека"
)

# Пример результата:
# "Вам подходит ипотека, потому что вы ищете квартиры после ремонта"
```

---

## Ключевые особенности

### Офлайн-работа
- Все вычисления локальные
- YandexGPT вызывается только для объяснений (не в runtime инференса)
- Модель обучена заранее

### Интерпретируемость
- Графы поведения визуализируются
- Объяснения генерируются для каждой рекомендации
- Fallback на шаблоны, если YandexGPT недоступен

### Масштабируемость
- `polars` эффективно работает с большими данными
- Партиционирование по датам
- Batch-обработка пользователей
- **Работа с данными из облака** — не нужно скачивать 1+ ТБ данных локально
- Прямое чтение Parquet через HTTP/HTTPS — Polars читает файлы напрямую из Яндекс Диска


---

## Дополнительные материалы

- [Документация Polars](https://pola-rs.github.io/polars-book/)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)
- [Streamlit Docs](https://docs.streamlit.io/)
- [LightFM Guide](https://making.lyst.com/lightfm/docs/home.html)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [PEP 257 Docstring Conventions](https://peps.python.org/pep-0257/)
- [Публичная ссылка на датасет](https://disk.yandex.ru/d/H0ZTzS55GSz1Wg)

---

## Команда

Проект разработан командой **VIBE-TEAM** (3 человека) для хакатона ПСБ.

---

## Лицензия

Внутренний проект для ПСБ.

---

## Известные ограничения

- Модель обучается на выборке (не real-time обучение)
- YandexGPT требует API ключ (есть fallback)
- Визуализация графов работает только в браузере (Streamlit)

