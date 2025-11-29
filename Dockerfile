# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем системные зависимости
# Этот слой кэшируется, если системные зависимости не меняются
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем pip, setuptools и wheel для лучшей совместимости
# Этот слой кэшируется отдельно
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Копируем ТОЛЬКО файлы зависимостей (для кэширования слоя)
# ⚡ КРИТИЧНО: Этот слой кэшируется Docker, если requirements.txt не изменился
# При изменении кода, но без изменения requirements.txt - зависимости не переустанавливаются!
COPY requirements.txt .

# Устанавливаем Python зависимости
# ⚡ КРИТИЧНО: Этот слой кэшируется вместе с предыдущим
# Если requirements.txt не изменился - pip install выполняется из кэша (мгновенно)
# Если requirements.txt изменился - только тогда переустанавливаются зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения (отдельный слой, пересобирается только при изменении кода)
COPY . .

# Создаем директории для данных и кэша
RUN mkdir -p /app/data /app/models /app/.cache

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose порт для Streamlit
EXPOSE 8501

# Команда по умолчанию - запуск Streamlit приложения
# Используем python -u для небуферизованного вывода
CMD ["python", "-u", "-m", "streamlit", "run", "src/app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--logger.level=debug"]

