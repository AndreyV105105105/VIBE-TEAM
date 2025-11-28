# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем ТОЛЬКО файлы зависимостей (для кэширования слоя)
COPY requirements.txt .

# Устанавливаем Python зависимости
# Docker автоматически кэширует этот слой, если requirements.txt не изменился
# При повторных сборках с тем же requirements.txt этот слой будет использован из кэша
RUN pip install -r requirements.txt

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

