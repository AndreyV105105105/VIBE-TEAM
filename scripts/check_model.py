"""
Скрипт для проверки сохраненной модели.
Показывает, где находится модель и её статус.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling.nbo_model import NBOModel

def main():
    print("=" * 60)
    print("🔍 Проверка модели ML")
    print("=" * 60)
    
    # Создаем экземпляр модели
    model = NBOModel()
    
    print(f"\n📁 Путь модели: {model.model_path}")
    print(f"📂 Абсолютный путь: {Path(model.model_path).resolve()}")
    print(f"📂 Текущая рабочая директория: {Path.cwd()}")
    
    model_path = Path(model.model_path).resolve()
    
    if model_path.exists():
        file_size = model_path.stat().st_size
        print(f"\n✅ Модель найдена!")
        print(f"   - Размер: {file_size / 1024:.2f} KB ({file_size} байт)")
        print(f"   - Путь: {model_path}")
        
        # Пробуем загрузить модель
        try:
            import joblib
            data = joblib.load(str(model_path))
            print(f"\n✅ Модель успешно загружена!")
            print(f"   - Содержит модель: {'model' in data}")
            print(f"   - Содержит scaler: {'scaler' in data}")
            print(f"   - Содержит products: {'products' in data}")
            if 'products' in data:
                print(f"   - Количество продуктов: {len(data['products'])}")
                print(f"   - Примеры: {data['products'][:5]}")
        except Exception as e:
            print(f"\n❌ Ошибка при загрузке модели: {e}")
    else:
        print(f"\n❌ Модель НЕ найдена по пути: {model_path}")
        print(f"\n💡 Возможные причины:")
        print(f"   1. Модель еще не была обучена")
        print(f"   2. Модель была сохранена в другое место")
        print(f"   3. Проблема с volume mount в Docker")
        
        # Проверяем директорию models
        models_dir = model_path.parent
        print(f"\n📂 Проверка директории models:")
        print(f"   - Путь: {models_dir}")
        print(f"   - Существует: {models_dir.exists()}")
        if models_dir.exists():
            files = list(models_dir.iterdir())
            print(f"   - Содержимое ({len(files)} файлов):")
            for f in files:
                if f.is_file():
                    size = f.stat().st_size
                    print(f"     • {f.name} ({size / 1024:.2f} KB)")
                else:
                    print(f"     • {f.name}/ (директория)")
        else:
            print(f"   - Директория не существует, создание...")
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"   - ✅ Директория создана")

if __name__ == "__main__":
    main()

