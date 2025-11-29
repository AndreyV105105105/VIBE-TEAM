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

# Загружаем CSS стили ПСБ
def load_psb_styles():
    """Загружает CSS стили для оформления в стиле ПСБ"""
    css_path = Path(__file__).parent / "static" / "styles.css"
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Дополнительные inline стили для Streamlit
    st.markdown("""
    <style>
    /* Скрываем стандартный заголовок Streamlit */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #0A2540 0%, #1A3A5A 100%);
        padding: 1rem;
    }
    
    /* Основной контент - темный фон в стиле ПСБ */
    .main .block-container {
        padding-top: 2rem;
        background-color: #0F1B2E;
    }
    
    /* Основной фон страницы */
    .main {
        background-color: #0F1B2E;
    }
    
    /* Заголовки на темном фоне - белые */
    .main h1, .main h2, .main h3 {
        color: #FFFFFF !important;
    }
    
    /* Обычный текст на темном фоне - светло-серый */
    .main p {
        color: #E0E0E0 !important;
    }
    
    /* Боковая панель */
    section[data-testid="stSidebar"] {
        background-color: #0A2540;
    }
    
    /* Все заголовки и текст в боковой панели - белый */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown strong,
    section[data-testid="stSidebar"] .stText,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }
    
    /* Инпуты в боковой панели */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background-color: #1A3A5A !important;
        color: #FFFFFF !important;
        border-color: #2A4A6A !important;
    }
    
    /* Селектбоксы и слайдеры в боковой панели */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #FFFFFF !important;
    }
    
    /* Информационные блоки в боковой панели */
    section[data-testid="stSidebar"] .stSuccess,
    section[data-testid="stSidebar"] .stWarning,
    section[data-testid="stSidebar"] .stInfo,
    section[data-testid="stSidebar"] .stError {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
        border-left: 4px solid #FF6B00;
    }
    
    /* Кнопки в боковой панели */
    section[data-testid="stSidebar"] button {
        background-color: #FF6B00 !important;
        color: #FFFFFF !important;
    }
    
    /* Убираем темный текст на темном фоне в основном контенте */
    .main .stMarkdown, .main .stText {
        color: #E0E0E0 !important;
    }
    
    /* Метрики Streamlit - белый текст на синем фоне */
    .main [data-testid="stMetricContainer"],
    .main [data-testid="stMetricContainer"] *,
    .main div[data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, #1A3A5A 0%, #0A2540 100%) !important;
        border: 1px solid #2A4A6A !important;
        border-radius: 10px !important;
        padding: 20px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .main [data-testid="stMetricContainer"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        transform: translateY(-2px) !important;
        border-color: #FF6B00 !important;
    }
    
    /* Значения метрик - белые */
    .main [data-testid="stMetricValue"],
    .main [data-testid="stMetricValue"] *,
    .main [data-testid="stMetricValue"] div,
    .main [data-testid="stMetricValue"] span,
    .main [data-testid="stMetricContainer"] [data-testid="stMetricValue"],
    .main div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Метки метрик - белые */
    .main [data-testid="stMetricLabel"],
    .main [data-testid="stMetricLabel"] *,
    .main [data-testid="stMetricLabel"] div,
    .main [data-testid="stMetricLabel"] span,
    .main [data-testid="stMetricContainer"] [data-testid="stMetricLabel"],
    .main div[data-testid="stMetricLabel"],
    .main [data-testid="stMetricContainer"] label {
        color: #FFFFFF !important;
        opacity: 0.95 !important;
        font-weight: 500 !important;
    }
    
    /* Все дочерние элементы в контейнере метрик - белые */
    .main [data-testid="stMetricContainer"] p,
    .main [data-testid="stMetricContainer"] div,
    .main [data-testid="stMetricContainer"] span,
    .main [data-testid="stMetricContainer"] label {
        color: #FFFFFF !important;
    }
    
    /* Улучшенные вкладки */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0A2540;
        padding: 6px;
        border-radius: 10px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 1.05em;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B00 0%, #FF8C42 100%);
        color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(255, 107, 0, 0.35);
        border: 2px solid #FF6B00;
    }
    
    .stTabs [aria-selected="false"] {
        background-color: #1A3A5A;
        color: #FFFFFF;
    }
    
    .stTabs [aria-selected="false"]:hover {
        background-color: #2A4A6A;
    }
    
    /* Улучшенные информационные блоки - темные с белым текстом */
    .main .stSuccess {
        background: linear-gradient(135deg, rgba(10, 37, 64, 0.4) 0%, rgba(26, 58, 90, 0.5) 100%);
        border-left: 5px solid #0A2540;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        color: #FFFFFF !important;
    }
    
    .main .stSuccess p, .main .stSuccess div, .main .stSuccess span {
        color: #FFFFFF !important;
    }
    
    .main .stInfo {
        background: linear-gradient(135deg, rgba(26, 58, 90, 0.4) 0%, rgba(10, 37, 64, 0.5) 100%);
        border-left: 5px solid #FF6B00;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        color: #FFFFFF !important;
    }
    
    .main .stInfo p, .main .stInfo div, .main .stInfo span {
        color: #FFFFFF !important;
    }
    
    .main .stWarning {
        background: linear-gradient(135deg, rgba(26, 58, 90, 0.4) 0%, rgba(10, 37, 64, 0.5) 100%);
        border-left: 5px solid #FFC107;
        border-radius: 8px;
        padding: 18px;
        color: #FFFFFF !important;
    }
    
    .main .stWarning p, .main .stWarning div, .main .stWarning span {
        color: #FFFFFF !important;
    }
    
    /* Улучшенные кнопки */
    .main .stButton > button {
        background: linear-gradient(135deg, #FF6B00 0%, #FF8C42 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 3px 8px rgba(255, 107, 0, 0.3);
        font-size: 1.05em;
    }
    
    .main .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(255, 107, 0, 0.4);
    }
    
    /* Улучшенные разделители */
    .main hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2) 50%, transparent);
        margin: 35px 0;
    }
    
    /* Divider от Streamlit */
    .main [data-testid="stDivider"] {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    .main [data-testid="stDivider"] div {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Улучшенные JSON блоки - темные */
    .main [data-testid="stJson"] {
        background-color: #1A3A5A;
        border: 1px solid #2A4A6A;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        color: #FFFFFF !important;
    }
    
    /* Улучшенные карточки рекомендаций */
    .recommendation-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FAFAFA 100%);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid #E8E8E8;
    }
    
    .recommendation-card:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
        transform: translateY(-3px);
    }
    
    /* Футер в боковой панели - белый */
    section[data-testid="stSidebar"] footer {
        color: #FFFFFF !important;
    }
    
    /* Улучшенные заголовки на темном фоне */
    .dark-bg h1, .dark-bg h2, .dark-bg h3, .dark-bg p, .dark-bg span {
        color: #FFFFFF !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Настройка страницы
st.set_page_config(
    page_title="ПСБ - Система рекомендаций Next Best Offer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загружаем стили
load_psb_styles()

# Логотип и заголовок на темно-синем фоне
static_dir = Path(__file__).parent / "static"
logo_path = static_dir / "logo.jpg"

st.markdown("""
<div style="background: linear-gradient(135deg, #0A2540 0%, #1A3A5A 100%); padding: 30px 35px; border-radius: 12px; margin-bottom: 30px; display: flex; align-items: center; gap: 25px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
""", unsafe_allow_html=True)

col1, col2 = st.columns([2.2, 5.8])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=220)
    else:
        st.markdown('<div style="color: #FFFFFF; font-weight: bold; font-size: 32px; padding: 10px;">ПСБ</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="padding-top: 8px;">
        <h1 style="color: #FFFFFF !important; margin: 0; font-size: 34px; font-weight: 700; letter-spacing: 0.3px; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">Система рекомендаций Next Best Offer</h1>
        <p style="color: #FFFFFF !important; font-size: 1.15em; margin-top: 12px; opacity: 0.95; font-weight: 400; letter-spacing: 0.2px;">Персонализированные рекомендации банковских продуктов</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Вкладки для выбора пользователя
    user_tab1, user_tab2 = st.tabs(["🔍 Поиск", "📋 Список"])
    
    user_id = None
    
    with user_tab1:
        st.markdown('<p style="color: #FFFFFF;"><strong>Введите или найдите ID пользователя</strong></p>', unsafe_allow_html=True)
        
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
                    st.markdown(f'<p style="color: #FFFFFF;">Найдено пользователей: {len(matching_users)}</p>', unsafe_allow_html=True)
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
        st.markdown('<p style="color: #FFFFFF;"><strong>Выберите пользователя из списка</strong></p>', unsafe_allow_html=True)
        
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
            st.markdown(f'<p style="color: #FFFFFF;">Доступно пользователей: {len(users_list)}</p>', unsafe_allow_html=True)
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
    
    # Футер в боковой панели
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #FFFFFF; padding: 10px;">
        <p style="font-size: 0.8em; opacity: 0.8;">© ПСБ</p>
        <p style="font-size: 0.7em; opacity: 0.6;">Система рекомендаций</p>
    </div>
    """, unsafe_allow_html=True)

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
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(10, 37, 64, 0.3) 0%, rgba(26, 58, 90, 0.4) 100%); 
                border-left: 5px solid #FF6B00; border-radius: 10px; padding: 20px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);">
        <p style="color: #FFFFFF !important; font-size: 1.15em; font-weight: 600; margin: 0; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">✅ Анализ пользователя <strong style="color: #FFFFFF !important;">{user_id}</strong> завершен!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Вкладки для разных разделов
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Рекомендации",
        "📈 Статистика",
        "🕸️ Граф поведения",
        "🔍 Паттерны"
    ])
    
    with tab1:
        st.markdown('<h2 style="color: #FFFFFF !important; margin-bottom: 20px;">📊 Рекомендации продуктов</h2>', unsafe_allow_html=True)
        
        if result['recommendations']:
            # Нормализация scores для прогресс-бара
            # 
            # КАК ВЫЧИСЛЯЮТСЯ SCORES:
            # 1. ML модель (RandomForestRegressor): предсказывает релевантность продукта (обычно 0-1, но может быть больше)
            # 2. Fallback алгоритм: комбинация базовых метрик (40%) + граф (35%) + паттерны (25%)
            #    - Базовые метрики: num_payments, total_tx, avg_tx, num_views и т.д.
            #    - Граф: PageRank, центральность узлов, плотность графа
            #    - Паттерны: частота событий, сложность последовательностей
            #    Scores могут быть > 1 (например, 9.00 для "Дебетовая карта")
            # 3. Правила: score = сумма confidence для каждого паттерна (высокая=3, средняя=2, низкая=1)
            #    Может суммироваться, если несколько паттернов указывают на один продукт
            #
            # st.progress() принимает значения от 0 до 1, поэтому нормализуем scores
            max_score = max(rec['score'] for rec in result['recommendations'])
            min_score = min(rec['score'] for rec in result['recommendations'])
            
            # Если все scores одинаковые или max_score = 0, используем относительную нормализацию
            if max_score == min_score or max_score == 0:
                # Используем относительную нормализацию: score / max_score
                # Если max_score = 0, все будут 0
                normalize_score = lambda s: s / max_score if max_score > 0 else 0.0
            else:
                # Min-max нормализация: (score - min_score) / (max_score - min_score)
                normalize_score = lambda s: (s - min_score) / (max_score - min_score)
            
            for i, rec in enumerate(result['recommendations'], 1):
                # Определяем цвет рамки в зависимости от источника
                border_color = "#FF6B00" if "ML" in rec['source'] else "#1A3A5A"
                badge_bg = "#FF6B00" if "ML" in rec['source'] else "#0A2540"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1A3A5A 0%, #0A2540 100%); border-left: 6px solid {border_color}; border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); transition: all 0.3s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 18px;">
                        <h3 style="color: #FFFFFF !important; margin: 0; font-size: 26px; font-weight: 700; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">#{i}. {rec['product']}</h3>
                        <span style="background: {badge_bg}; color: #FFFFFF; padding: 8px 16px; border-radius: 20px; font-size: 0.9em; font-weight: 600; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);">{rec['source']}</span>
                    </div>
                    <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 18px; margin-bottom: 15px; border: 1px solid rgba(255, 255, 255, 0.2);">
                        <p style="color: #FFFFFF !important; margin: 0; line-height: 1.7; font-size: 1.08em; font-weight: 400; opacity: 0.95;">{rec['reason']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Метрики в отдельной строке на синем фоне
                normalized_progress = normalize_score(rec['score'])
                ml_score_val = rec.get('ml_score', 0) if 'ml_score' in rec else 0
                rule_score_val = rec.get('rule_score', 0)
                
                metric_label = "ML модель" if ml_score_val > 0 else ("Правила" if rule_score_val > 0 else "Комбо")
                metric_value = f"{ml_score_val:.2f}" if ml_score_val > 0 else (f"{rule_score_val:.2f}" if rule_score_val > 0 else "✓")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2A4A6A 0%, #1A3A5A 100%); border-radius: 10px; padding: 20px; margin-top: 15px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);">
                    <div style="display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 20px; align-items: center;">
                        <div>
                            <p style="color: #FFFFFF !important; font-weight: 600; margin-bottom: 8px; font-size: 0.95em; opacity: 0.9;">Релевантность: <strong style="color: #FFFFFF !important;">{normalized_progress*100:.1f}%</strong></p>
                        </div>
                        <div style="text-align: center;">
                            <p style="color: #FFFFFF !important; font-size: 0.85em; margin-bottom: 5px; opacity: 0.8;">Общая оценка</p>
                            <p style="color: #FFFFFF !important; font-size: 1.5em; font-weight: 700; margin: 0;">{rec['score']:.2f}</p>
                        </div>
                        <div style="text-align: center;">
                            <p style="color: #FFFFFF !important; font-size: 0.85em; margin-bottom: 5px; opacity: 0.8;">{metric_label}</p>
                            <p style="color: #FFFFFF !important; font-size: 1.5em; font-weight: 700; margin: 0;">{metric_value}</p>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div style="background-color: rgba(0, 0, 0, 0.2); border-radius: 10px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #FF6B00 0%, #FF8C42 100%); height: 100%; width: {normalized_progress*100}%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if i < len(result['recommendations']):
                    st.markdown('<div style="margin: 30px 0; border-bottom: 2px dashed rgba(255, 255, 255, 0.15);"></div>', unsafe_allow_html=True)
        else:
            st.info("Рекомендации не найдены. Попробуйте другого пользователя.")
    
    with tab2:
        st.markdown('<h2 style="color: #FFFFFF !important; margin-bottom: 20px;">📈 Статистика пользователя</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        profile = result.get('profile', {})
        graph_stats = result.get('graph_stats', {})
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1A3A5A 0%, #0A2540 100%); 
                        border: 1px solid #2A4A6A; border-radius: 10px; padding: 20px; 
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); text-align: center;">
                <div style="color: #FFFFFF !important; font-size: 0.9em; font-weight: 500; margin-bottom: 8px; opacity: 0.95;">Просмотров</div>
                <div style="color: #FFFFFF !important; font-size: 2em; font-weight: 700;">""" + str(profile.get('num_views', 0)) + """</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1A3A5A 0%, #0A2540 100%); 
                        border: 1px solid #2A4A6A; border-radius: 10px; padding: 20px; 
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); text-align: center;">
                <div style="color: #FFFFFF !important; font-size: 0.9em; font-weight: 500; margin-bottom: 8px; opacity: 0.95;">Платежей</div>
                <div style="color: #FFFFFF !important; font-size: 2em; font-weight: 700;">""" + str(profile.get('num_payments', 0)) + """</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1A3A5A 0%, #0A2540 100%); 
                        border: 1px solid #2A4A6A; border-radius: 10px; padding: 20px; 
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); text-align: center;">
                <div style="color: #FFFFFF !important; font-size: 0.9em; font-weight: 500; margin-bottom: 8px; opacity: 0.95;">Узлов в графе</div>
                <div style="color: #FFFFFF !important; font-size: 2em; font-weight: 700;">""" + str(graph_stats.get('nodes', 0)) + """</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1A3A5A 0%, #0A2540 100%); 
                        border: 1px solid #2A4A6A; border-radius: 10px; padding: 20px; 
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); text-align: center;">
                <div style="color: #FFFFFF !important; font-size: 0.9em; font-weight: 500; margin-bottom: 8px; opacity: 0.95;">Связей в графе</div>
                <div style="color: #FFFFFF !important; font-size: 2em; font-weight: 700;">""" + str(graph_stats.get('edges', 0)) + """</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Детальная статистика
        st.markdown('<h3 style="color: #FFFFFF !important; margin-top: 30px; margin-bottom: 20px;">📊 Детальная информация</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p style="color: #FFFFFF !important; font-weight: 600; font-size: 1.1em; margin-bottom: 10px;">👤 Профиль пользователя:</p>', unsafe_allow_html=True)
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
            # Получаем название топ бренда из маппинга
            top_brand_id = profile.get('top_brand_id') or profile.get('top_brand')
            top_brand_display = 'Не указан'
            if top_brand_id:
                brands_map = result.get('brands_map', {})
                brand_name = brands_map.get(str(top_brand_id), None)
                if brand_name:
                    top_brand_display = f"{brand_name} (ID: {top_brand_id})"
                else:
                    # Всегда показываем ID, даже если нет названия
                    top_brand_display = f"Brand {top_brand_id} (ID: {top_brand_id})"
            
            st.json({
                "Средний чек": avg_tx_display,
                "Общая сумма": total_tx_display,
                "Дней активности": profile.get('days_active', 0),
                "Уникальных товаров": profile.get('unique_items', 0),
                "Регион": profile.get('region') if profile.get('region') else 'Не указан',
                "Топ категория": profile.get('top_category') if profile.get('top_category') else 'Не указана',
                "Топ бренд": top_brand_display,
                "Категория брендов": profile.get('top_brand_category') if profile.get('top_brand_category') else 'Не указана'
            })
            
            # Показываем диагностику если значения были отрицательными
            if profile.get('avg_tx', 0) < 0 or profile.get('total_tx', 0) < 0:
                st.warning(f"⚠ Обнаружены отрицательные значения в профиле! avg_tx={profile.get('avg_tx')}, total_tx={profile.get('total_tx')}")
                st.info("Это может быть связано с возвратами или ошибкой в данных. Используются абсолютные значения.")
        
        with col2:
            st.markdown('<p style="color: #FFFFFF !important; font-weight: 600; font-size: 1.1em; margin-bottom: 10px;">🕸️ Статистика графа:</p>', unsafe_allow_html=True)
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
        st.markdown('<h2 style="color: #FFFFFF !important; margin-bottom: 20px;">🕸️ Граф поведения пользователя</h2>', unsafe_allow_html=True)
        
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
                
                # Создаем сеть для визуализации в стиле ПСБ
                net = Network(
                    height="600px",
                    width="100%",
                    bgcolor="#0A2540",
                    font_color="#FFFFFF",
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
                
                # Добавляем узлы с цветами в стиле ПСБ
                node_colors = {
                    'item': '#FF6B00',      # Оранжевый ПСБ для товаров
                    'brand': '#1A3A5A',    # Синий ПСБ для брендов
                    'start': '#FFFFFF',    # Белый для старта
                    'category': '#FF8C42', # Светло-оранжевый для категорий
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
                        # Показываем название бренда, если доступно, иначе brand_id
                        if 'brand_id' in data:
                            brand_id = str(data['brand_id'])
                            # Пробуем найти название в маппинге
                            brands_map = result.get('brands_map', {})
                            brand_name = brands_map.get(brand_id, None)
                            if brand_name:
                                node_label = f"Бренд: {brand_name}"
                            else:
                                node_label = f"Бренд: {brand_id}"
                    elif node_label == 'START':
                        node_label = 'СТАРТ'
                    
                    # Размер узла зависит от степени (количества связей)
                    degree = graph_to_visualize.degree(node)
                    size = 20 + min(degree * 5, 50)  # От 20 до 70
                    
                    # Формируем подсказку
                    tooltip = f"Тип: {node_type}\nСвязей: {degree}"
                    if 'amount' in data:
                        tooltip += f"\nСумма: ${data['amount']:.2f}"
                    if 'brand_id' in data:
                        brand_id = str(data['brand_id'])
                        brands_map = result.get('brands_map', {})
                        brand_name = brands_map.get(brand_id, None)
                        if brand_name:
                            tooltip += f"\nБренд: {brand_name} (ID: {brand_id})"
                        else:
                            tooltip += f"\nБренд ID: {brand_id}"
                    
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
                
                # Легенда в стиле ПСБ
                st.markdown("---")
                st.subheader("Легенда")
                legend_html = """
                <div style="display: flex; gap: 30px; flex-wrap: wrap; padding: 15px; background-color: #F5F5F5; border-radius: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #FF6B00; border-radius: 50%;"></div>
                        <span style="color: #0A2540; font-weight: 500;"><strong>Товары</strong></span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #1A3A5A; border-radius: 50%;"></div>
                        <span style="color: #0A2540; font-weight: 500;"><strong>Бренды</strong></span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #FFFFFF; border: 2px solid #0A2540; border-radius: 50%;"></div>
                        <span style="color: #0A2540; font-weight: 500;"><strong>Старт</strong></span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 20px; height: 20px; background-color: #FF8C42; border-radius: 50%;"></div>
                        <span style="color: #0A2540; font-weight: 500;"><strong>Категории</strong></span>
                    </div>
                </div>
                """
                st.markdown(legend_html, unsafe_allow_html=True)
                
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
                    st.markdown('<p style="color: #FFFFFF !important; font-weight: 600; margin-bottom: 10px;">**Узлы:**</p>', unsafe_allow_html=True)
                    for node in list(graph.nodes())[:20]:
                        neighbors = list(graph.neighbors(node))
                        st.markdown(f'<p style="color: #E0E0E0 !important;">- {node} → {neighbors[:5]}</p>', unsafe_allow_html=True)
        else:
            st.info("Граф пуст или не построен. Недостаточно данных для визуализации.")
        
        # Правила из графа
        if result.get('graph_rules'):
            st.markdown("---")
            st.markdown('<h3 style="color: #FFFFFF !important; margin-bottom: 15px;">📋 Правила из графа:</h3>', unsafe_allow_html=True)
            for rule in result['graph_rules'][:5]:
                st.markdown(f'<p style="color: #E0E0E0 !important;">- {rule}</p>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 style="color: #FFFFFF !important; margin-bottom: 20px;">🔍 Паттерны поведения</h2>', unsafe_allow_html=True)
        
        patterns = result.get('patterns', [])
        
        if patterns:
            st.markdown(f'<p style="color: #FFFFFF !important; font-weight: 600; margin-bottom: 15px;">Найдено паттернов: <strong>{len(patterns)}</strong></p>', unsafe_allow_html=True)
            
            for i, pattern in enumerate(patterns, 1):
                st.code(pattern, language=None)
        else:
            st.info("Паттерны не найдены")

else:
    # Начальный экран в стиле ПСБ
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0A2540 0%, #1A3A5A 100%); padding: 30px; border-radius: 10px; color: #FFFFFF; margin-bottom: 30px;">
        <h2 style="color: #FFFFFF; margin-top: 0;">👈 Добро пожаловать в систему рекомендаций ПСБ</h2>
        <p style="font-size: 1.1em; margin-bottom: 0;">Введите ID пользователя в боковой панели и нажмите 'Анализировать' для начала работы</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 25px; border-radius: 8px; border-left: 5px solid #FF6B00; box-shadow: 0 2px 4px rgba(10, 37, 64, 0.1);">
            <h3 style="color: #0A2540; margin-top: 0;">📋 Как использовать:</h3>
            <ol style="color: #1A3A5A; line-height: 1.8;">
                <li><strong>Введите ID пользователя</strong> в боковой панели</li>
                <li><strong>Настройте параметры</strong> (источник данных, использование YandexGPT)</li>
                <li><strong>Нажмите "Анализировать"</strong></li>
                <li><strong>Просмотрите результаты</strong> во вкладках</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 25px; border-radius: 8px; border-left: 5px solid #1A3A5A; box-shadow: 0 2px 4px rgba(10, 37, 64, 0.1);">
            <h3 style="color: #0A2540; margin-top: 0;">✨ Возможности:</h3>
            <ul style="color: #1A3A5A; line-height: 1.8;">
                <li>📊 <strong>Рекомендации</strong> - топ продуктов с объяснениями</li>
                <li>📈 <strong>Статистика</strong> - детальная информация о пользователе</li>
                <li>🕸️ <strong>Граф поведения</strong> - визуализация поведения</li>
                <li>🔍 <strong>Паттерны</strong> - найденные паттерны поведения</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #F5F5F5; padding: 20px; border-radius: 8px; margin-top: 20px;">
        <h3 style="color: #0A2540; margin-top: 0;">🚀 Особенности системы:</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">
            <div style="padding: 15px; background-color: #FFFFFF; border-radius: 5px;">
                <strong style="color: #FF6B00;">☁️ Облако</strong>
                <p style="color: #1A3A5A; margin: 5px 0 0 0; font-size: 0.9em;">Автоматическая загрузка данных из облака</p>
            </div>
            <div style="padding: 15px; background-color: #FFFFFF; border-radius: 5px;">
                <strong style="color: #FF6B00;">📊 Графы</strong>
                <p style="color: #1A3A5A; margin: 5px 0 0 0; font-size: 0.9em;">Построение графов поведения пользователей</p>
            </div>
            <div style="padding: 15px; background-color: #FFFFFF; border-radius: 5px;">
                <strong style="color: #FF6B00;">🤖 ML & AI</strong>
                <p style="color: #1A3A5A; margin: 5px 0 0 0; font-size: 0.9em;">Рекомендации на основе ML модели и YandexGPT</p>
            </div>
            <div style="padding: 15px; background-color: #FFFFFF; border-radius: 5px;">
                <strong style="color: #FF6B00;">🔍 Паттерны</strong>
                <p style="color: #1A3A5A; margin: 5px 0 0 0; font-size: 0.9em;">Автоматическое извлечение паттернов поведения</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

