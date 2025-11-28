"""
Streamlit приложение для демонстрации системы рекомендаций Next Best Offer.

Веб-интерфейс для работы с системой рекомендаций ПСБ.
"""

import streamlit as st
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.main import process_user
from src.utils.user_finder import (
    get_available_users,
    get_users_from_users_file,
    search_users_by_pattern,
    get_user_statistics
)
import networkx as nx
from pyvis.network import Network
import tempfile
import os


# Настройка страницы
st.set_page_config(
    page_title="ПСБ - Система рекомендаций Next Best Offer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🏦 Система рекомендаций Next Best Offer")
st.markdown("---")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Вкладки для выбора пользователя
    user_tab1, user_tab2 = st.tabs(["🔍 Поиск", "📋 Список"])
    
    user_id = None
    
    with user_tab1:
        st.write("**Введите или найдите ID пользователя**")
        
        user_id_input = st.text_input(
            "ID пользователя",
            value=st.session_state.get('selected_user_id', ''),
            help="Введите ID пользователя для анализа",
            key="user_id_input"
        )
        
        if user_id_input:
            # Проверяем существование пользователя
            with st.spinner("Проверка пользователя..."):
                stats = get_user_statistics(user_id_input)
                if stats.get("exists"):
                    st.success(f"✅ Пользователь найден ({stats.get('num_events', 0)} событий)")
                    user_id = user_id_input
                else:
                    st.warning("⚠️ Пользователь не найден в данных. Попробуйте другой ID.")
        
        st.divider()
        
        # Поиск по паттерну
        search_pattern = st.text_input(
            "Поиск по паттерну",
            value="",
            help="Найти пользователей, ID которых содержит этот текст",
            key="search_pattern"
        )
        
        if search_pattern:
            with st.spinner("Поиск пользователей..."):
                matching_users = search_users_by_pattern(search_pattern, limit=20)
                if matching_users:
                    st.write(f"Найдено пользователей: {len(matching_users)}")
                    selected_user = st.selectbox(
                        "Выберите пользователя:",
                        options=[""] + matching_users,
                        key="selected_user_search"
                    )
                    if selected_user:
                        user_id = selected_user
                        st.session_state['selected_user_id'] = selected_user
                else:
                    st.info("Пользователи не найдены")
    
    with user_tab2:
        st.write("**Выберите пользователя из списка**")
        
        if st.button("🔄 Обновить список", key="refresh_users"):
            st.session_state['users_list'] = None
            st.session_state['users_loaded'] = False
        
        if 'users_list' not in st.session_state or st.session_state.get('users_loaded', False) == False:
            with st.spinner("Загрузка пользователей из данных (только те, у кого есть события)..."):
                try:
                    # Загружаем только тех пользователей, у кого есть реальные события
                    # Это гарантирует, что для каждого ID будут данные
                    users_from_events = get_available_users(limit=100, num_files=1)
                    if users_from_events and len(users_from_events) > 0:
                        st.session_state['users_list'] = users_from_events
                        st.success(f"✅ Загружено {len(users_from_events)} пользователей с событиями (marketplace + payments)")
                    else:
                        # Fallback: пробуем загрузить из users.pq, но это менее надежно
                        st.info("Загрузка из users.pq (fallback)...")
                        users_from_file = get_users_from_users_file(limit=100)
                        if users_from_file and len(users_from_file) > 0:
                            st.session_state['users_list'] = users_from_file
                            st.warning(f"⚠ Загружено {len(users_from_file)} пользователей из users.pq (некоторые могут не иметь событий)")
                        else:
                            st.warning("Не удалось загрузить пользователей. Попробуйте обновить список позже.")
                            st.session_state['users_list'] = []
                    st.session_state['users_loaded'] = True
                    st.success("✅ Список загружен")
                except Exception as e:
                    st.error(f"Ошибка при загрузке: {e}")
                    st.session_state['users_list'] = []
                    st.session_state['users_loaded'] = True
        
        users_list = st.session_state.get('users_list', [])
        
        if users_list:
            st.write(f"Доступно пользователей: {len(users_list)}")
            selected_user = st.selectbox(
                "Выберите пользователя:",
                options=[""] + users_list[:100],  # Показываем первые 100
                key="selected_user_list"
            )
            if selected_user:
                user_id = selected_user
                st.session_state['selected_user_id'] = selected_user
        else:
            st.info("Список пользователей пуст. Нажмите 'Обновить список'.")
    
    # Используем выбранный user_id или значение по умолчанию
    if user_id is None:
        user_id = st.session_state.get('selected_user_id', '12345')
    
    # Показываем текущий выбранный ID
    if user_id:
        st.info(f"📌 Выбранный пользователь: **{user_id}**")
    
    use_cloud = st.checkbox(
        "Использовать данные из облака",
        value=True,
        help="Загружать данные из Яндекс Диска"
    )
    
    use_yandexgpt = st.checkbox(
        "Использовать YandexGPT для анализа",
        value=True,
        help="Если выключено, будет использоваться fallback решение на основе эвристик, графов и паттернов (без YandexGPT)"
    )
    
    # Показываем информацию о режиме работы
    if not use_yandexgpt:
        st.info("ℹ️ **Fallback режим:** Рекомендации будут генерироваться на основе:\n"
                "- Анализа графов поведения\n"
                "- Извлеченных паттернов\n"
                "- Статистики пользователя\n"
                "- Эвристических правил\n\n"
                "Без использования YandexGPT (экономия токенов)")
    
    top_k = st.slider(
        "Количество рекомендаций",
        min_value=1,
        max_value=10,
        value=3,
        help="Сколько рекомендаций показать"
    )
    
    analyze_button = st.button(
        "🔍 Анализировать",
        type="primary",
        use_container_width=True
    )

# Основной контент
if analyze_button:
    with st.spinner("Обработка пользователя..."):
        try:
            result = process_user(
                user_id=user_id,
                use_cloud=use_cloud,
                use_yandexgpt_for_analysis=use_yandexgpt,
                top_k=top_k
            )
            
            # Сохраняем результат в session state
            st.session_state['result'] = result
            st.session_state['user_id'] = user_id
            
        except Exception as e:
            st.error(f"Ошибка при обработке пользователя: {e}")
            st.exception(e)
            st.stop()

# Отображение результатов
if 'result' in st.session_state:
    result = st.session_state['result']
    user_id = st.session_state.get('user_id', 'unknown')
    
    st.success(f"✅ Анализ пользователя {user_id} завершен!")
    
    # Вкладки для разных разделов
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Рекомендации",
        "📈 Статистика",
        "🕸️ Граф поведения",
        "🔍 Паттерны"
    ])
    
    with tab1:
        st.header("Рекомендации продуктов")
        
        if result['recommendations']:
            for i, rec in enumerate(result['recommendations'], 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader(f"{i}. {rec['product']}")
                        st.write(f"**Объяснение:** {rec['reason']}")
                        st.caption(f"Источник: {rec['source']}")
                    
                    with col2:
                        score = rec['score']
                        st.metric("Оценка", f"{score:.2f}")
                        
                        # Прогресс-бар для визуализации оценки
                        st.progress(score)
                    
                    st.divider()
        else:
            st.info("Рекомендации не найдены. Попробуйте другого пользователя.")
    
    with tab2:
        st.header("Статистика пользователя")
        
        col1, col2, col3, col4 = st.columns(4)
        
        profile = result.get('profile', {})
        graph_stats = result.get('graph_stats', {})
        
        with col1:
            st.metric("Просмотров", profile.get('num_views', 0))
        
        with col2:
            st.metric("Платежей", profile.get('num_payments', 0))
        
        with col3:
            st.metric("Узлов в графе", graph_stats.get('nodes', 0))
        
        with col4:
            st.metric("Связей в графе", graph_stats.get('edges', 0))
        
        st.divider()
        
        # Детальная статистика
        st.subheader("Детальная информация")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Профиль пользователя:**")
            # Форматируем значения для отображения
            avg_tx = profile.get('avg_tx', 0)
            total_tx = profile.get('total_tx', 0)
            
            # Проверяем на отрицательные или некорректные значения
            if avg_tx < 0:
                avg_tx = 0
            if total_tx < 0:
                total_tx = 0
            
            # Форматируем как денежные значения (в долларах)
            avg_tx_display = f"${avg_tx:,.2f}" if avg_tx > 0 else "$0.00"
            total_tx_display = f"${total_tx:,.2f}" if total_tx > 0 else "$0.00"
            
            # Дополнительная диагностика: показываем исходные значения
            st.json({
                "Средний чек": avg_tx_display,
                "Общая сумма": total_tx_display,
                "Дней активности": profile.get('days_active', 0),
                "Уникальных товаров": profile.get('unique_items', 0),
                "Регион": profile.get('region') if profile.get('region') else 'Не указан',
                "Топ категория": profile.get('top_category') if profile.get('top_category') else 'Не указана'
            })
            
            # Показываем диагностику если значения были отрицательными
            if profile.get('avg_tx', 0) < 0 or profile.get('total_tx', 0) < 0:
                st.warning(f"⚠ Обнаружены отрицательные значения в профиле! avg_tx={profile.get('avg_tx')}, total_tx={profile.get('total_tx')}")
                st.info("Это может быть связано с возвратами или ошибкой в данных. Используются абсолютные значения.")
        
        with col2:
            st.write("**Статистика графа:**")
            st.json({
                "Узлов": graph_stats.get('nodes', 0),
                "Связей": graph_stats.get('edges', 0),
                "Плотность": f"{graph_stats.get('density', 0):.4f}",
                "Средняя степень": f"{graph_stats.get('avg_degree', 0):.2f}",
                "Связность": "Да" if graph_stats.get('is_connected', False) else "Нет"
            })
        
        # Анализ графа через YandexGPT (если есть)
        if result.get('graph_analysis'):
            st.divider()
            st.subheader("🤖 Анализ графа (YandexGPT)")
            st.write(result['graph_analysis'].get('analysis', 'Анализ недоступен'))
    
    with tab3:
        st.header("Граф поведения пользователя")
        
        graph = result.get('graph')
        graph_stats = result.get('graph_stats', {})
        
        if graph and graph.number_of_nodes() > 0:
            # Визуализация графа через pyvis
            try:
                # Ограничиваем количество узлов для визуализации (если граф слишком большой)
                max_nodes = 50
                if graph.number_of_nodes() > max_nodes:
                    st.warning(f"⚠️ Граф содержит {graph.number_of_nodes()} узлов. Показываем топ {max_nodes} узлов по степени.")
                    # Берем топ узлов по степени
                    degrees = dict(graph.degree())
                    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                    top_node_ids = [node for node, _ in top_nodes]
                    # Создаем подграф
                    subgraph = graph.subgraph(top_node_ids).copy()
                    graph_to_visualize = subgraph
                else:
                    graph_to_visualize = graph
                
                # Создаем сеть для визуализации
                net = Network(
                    height="600px",
                    width="100%",
                    bgcolor="#222222",
                    font_color="white",
                    directed=True
                )
                
                # Настройки физики для красивого отображения
                net.set_options("""
                {
                    "physics": {
                        "enabled": true,
                        "stabilization": {"iterations": 100},
                        "barnesHut": {
                            "gravitationalConstant": -2000,
                            "centralGravity": 0.1,
                            "springLength": 200,
                            "springConstant": 0.05
                        }
                    },
                    "nodes": {
                        "font": {"size": 14, "color": "white"},
                        "borderWidth": 2
                    },
                    "edges": {
                        "arrows": {"to": {"enabled": true}},
                        "font": {"size": 12, "color": "white"},
                        "smooth": {"type": "continuous"}
                    }
                }
                """)
                
                # Добавляем узлы с цветами по типу
                node_colors = {
                    'item': '#FF6B6B',      # Красный для товаров
                    'brand': '#4ECDC4',    # Бирюзовый для брендов
                    'start': '#95E1D3',    # Светло-бирюзовый для старта
                    'category': '#F38181', # Розовый для категорий
                    'unknown': '#95A5A6'    # Серый для неизвестных
                }
                
                # Добавляем узлы
                for node, data in graph_to_visualize.nodes(data=True):
                    node_type = data.get('type', 'unknown')
                    color = node_colors.get(node_type, '#95A5A6')  # Серый по умолчанию
                    
                    # Улучшаем метку узла (убираем префиксы)
                    node_label = str(node)
                    if node_label.startswith('item_'):
                        node_label = node_label.replace('item_', '')
                        # Показываем category_id если есть
                        if 'category_id' in data:
                            node_label = f"Кат: {data['category_id']}"
                    elif node_label.startswith('brand_'):
                        node_label = node_label.replace('brand_', '')
                        # Показываем brand_id
                        if 'brand_id' in data:
                            node_label = f"Бренд: {data['brand_id']}"
                    elif node_label == 'START':
                        node_label = 'СТАРТ'
                    
                    # Размер узла зависит от степени (количества связей)
                    degree = graph_to_visualize.degree(node)
                    size = 20 + min(degree * 5, 50)  # От 20 до 70
                    
                    # Формируем подсказку
                    tooltip = f"Тип: {node_type}\nСвязей: {degree}"
                    if 'amount' in data:
                        tooltip += f"\nСумма: {data['amount']:.2f} ₽"
                    
                    net.add_node(
                        str(node),
                        label=node_label,
                        color=color,
                        size=size,
                        title=tooltip
                    )
                
                # Добавляем рёбра с весами
                for u, v, data in graph_to_visualize.edges(data=True):
                    weight = data.get('weight', 1)
                    # Толщина ребра зависит от веса
                    width = 1 + min(weight * 2, 5)
                    
                    net.add_edge(
                        str(u),
                        str(v),
                        value=weight,
                        width=width,
                        title=f"Вес: {weight}"
                    )
                
                # Сохраняем граф во временный HTML файл
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp_file:
                    net.save_graph(tmp_file.name)
                    tmp_path = tmp_file.name
                
                # Отображаем граф в Streamlit
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                st.components.v1.html(html_content, height=650)
                
                # Удаляем временный файл после отображения
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Легенда
                st.markdown("---")
                st.subheader("Легенда")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🔴 **Товары** (item)")
                with col2:
                    st.markdown("🔵 **Бренды** (brand)")
                with col3:
                    st.markdown("🟢 **Старт** (start)")
                with col4:
                    st.markdown("🟣 **Категории** (category)")
                
                # Статистика графа
                st.markdown("---")
                st.subheader("Статистика графа")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Узлов", graph_stats.get('nodes', 0))
                with col2:
                    st.metric("Рёбер", graph_stats.get('edges', 0))
                with col3:
                    st.metric("Плотность", f"{graph_stats.get('density', 0):.4f}")
                with col4:
                    st.metric("Средняя степень", f"{graph_stats.get('avg_degree', 0):.2f}")
                
            except Exception as e:
                st.error(f"Ошибка при визуализации графа: {e}")
                st.info("Показываем текстовое представление графа")
                
                # Текстовое представление как fallback
                st.code(f"Узлов: {graph.number_of_nodes()}\nРёбер: {graph.number_of_edges()}")
                if graph.number_of_nodes() <= 50:
                    st.write("**Узлы:**")
                    for node in list(graph.nodes())[:20]:
                        neighbors = list(graph.neighbors(node))
                        st.write(f"- {node} → {neighbors[:5]}")
        else:
            st.info("Граф пуст или не построен. Недостаточно данных для визуализации.")
        
        # Правила из графа
        if result.get('graph_rules'):
            st.markdown("---")
            st.subheader("Правила из графа:")
            for rule in result['graph_rules'][:5]:
                st.write(f"- {rule}")
    
    with tab4:
        st.header("Паттерны поведения")
        
        patterns = result.get('patterns', [])
        
        if patterns:
            st.write(f"Найдено паттернов: **{len(patterns)}**")
            
            for i, pattern in enumerate(patterns, 1):
                st.code(pattern, language=None)
        else:
            st.info("Паттерны не найдены")

else:
    # Начальный экран
    st.info("👈 Введите ID пользователя и нажмите 'Анализировать' для начала работы")
    
    st.markdown("""
    ### Как использовать:
    
    1. **Введите ID пользователя** в боковой панели
    2. **Настройте параметры** (источник данных, использование YandexGPT)
    3. **Нажмите "Анализировать"**
    4. **Просмотрите результаты** во вкладках:
       - 📊 Рекомендации - топ продуктов с объяснениями
       - 📈 Статистика - детальная информация о пользователе
       - 🕸️ Граф поведения - визуализация поведения
       - 🔍 Паттерны - найденные паттерны поведения
    
    ### Особенности:
    - Автоматическая загрузка данных из облака
    - Построение графов поведения
    - Извлечение паттернов
    - Генерация объяснений через YandexGPT
    - Рекомендации на основе ML модели и правил
    """)

