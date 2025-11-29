"""
Модуль для нормализации категорий товаров.
Преобразует категории в единый формат для использования в правилах и ML-модели.

Поддерживает 26 категорий, соответствующих реальным категориям из датасетов:
- electronics, travel, auto, real_estate, sport, health, food, finance, retail
- clothing, children, beauty, books, home_appliances, furniture, garden, pets
- entertainment, education, music, movies, home_textiles, kitchen, tools
- construction, pharmacy, jewelry, sports_goods

Каждая категория содержит ключевые слова на английском и русском языках
для надежного сопоставления с категориями из различных источников данных.
"""

# Маппинг категорий: английские названия -> ключевые слова на разных языках
CATEGORY_MAPPINGS = {
    "electronics": {
        "en": ["electronics", "electronic", "tech", "technology", "gadget", "device"],
        "ru": ["электроника", "техника", "гаджет", "устройство", "смартфон", "телефон", "компьютер"],
        "keywords": ["electronics", "electronic", "tech", "technology", "gadget", "device", 
                    "электроника", "техника", "гаджет", "устройство", "смартфон"]
    },
    "travel": {
        "en": ["travel", "trip", "tour", "tourism", "hotel", "airline", "flight", "vacation"],
        "ru": ["путешествие", "туризм", "отель", "авиабилет", "отпуск", "поездка"],
        "keywords": ["travel", "trip", "tour", "hotel", "airline", "flight", 
                    "путешествие", "туризм", "отель", "авиабилет", "отпуск"]
    },
    "auto": {
        "en": ["auto", "car", "vehicle", "fuel", "gas", "gasoline", "petrol"],
        "ru": ["авто", "машина", "автомобиль", "бензин", "топливо", "азс"],
        "keywords": ["auto", "car", "vehicle", "fuel", "gas", 
                    "авто", "машина", "автомобиль", "бензин", "азс"]
    },
    "real_estate": {
        "en": ["real estate", "property", "house", "home", "apartment", "renovation", "repair", "furniture"],
        "ru": ["недвижимость", "дом", "квартира", "ремонт", "мебель", "строительство", "интерьер"],
        "keywords": ["real estate", "property", "house", "renovation", "repair", "furniture",
                    "недвижимость", "дом", "квартира", "ремонт", "мебель", "строительство"]
    },
    "sport": {
        "en": ["sport", "sports", "fitness", "gym", "football", "soccer", "tickets"],
        "ru": ["спорт", "фитнес", "футбол", "билет", "тренировка"],
        "keywords": ["sport", "sports", "fitness", "gym", "football", "soccer",
                    "спорт", "фитнес", "футбол", "билет"]
    },
    "health": {
        "en": ["health", "medical", "medicine", "pharmacy", "doctor", "hospital"],
        "ru": ["здоровье", "медицина", "аптека", "врач", "больница", "лекарство"],
        "keywords": ["health", "medical", "medicine", "pharmacy", "doctor",
                    "здоровье", "медицина", "аптека", "врач", "лекарство"]
    },
    "food": {
        "en": ["food", "foodstuff", "beverage", "grocery", "supermarket", "restaurant", "cafe"],
        "ru": ["еда", "продукты", "напитки", "супермаркет", "ресторан", "кафе", "продукты питания"],
        "keywords": ["food", "foodstuff", "beverage", "grocery", "supermarket",
                    "еда", "продукты", "напитки", "супермаркет", "продукты питания"]
    },
    "finance": {
        "en": ["finance", "financial", "bank", "investment", "insurance", "broker"],
        "ru": ["финансы", "банк", "инвестиции", "страхование", "брокер"],
        "keywords": ["finance", "financial", "bank", "investment",
                    "финансы", "банк", "инвестиции", "страхование"]
    },
    "retail": {
        "en": ["retail", "shop", "store", "market", "shopping"],
        "ru": ["розничная", "торговля", "магазин", "покупка", "шопинг"],
        "keywords": ["retail", "shop", "store", "market",
                    "розничная", "торговля", "магазин", "покупка"]
    },
    "clothing": {
        "en": ["clothing", "clothes", "fashion", "apparel", "wear", "garment", "outfit"],
        "ru": ["одежда", "мода", "наряды", "гардероб", "вещи", "платье", "рубашка", "брюки"],
        "keywords": ["clothing", "clothes", "fashion", "apparel", "wear",
                    "одежда", "мода", "наряды", "гардероб", "платье"]
    },
    "children": {
        "en": ["children", "kids", "child", "baby", "infant", "toddler", "toy", "toys"],
        "ru": ["дети", "детский", "ребенок", "малыш", "младенец", "игрушки", "игрушка"],
        "keywords": ["children", "kids", "child", "baby", "toy", "toys",
                    "дети", "детский", "ребенок", "малыш", "игрушки"]
    },
    "beauty": {
        "en": ["beauty", "cosmetics", "makeup", "perfume", "skincare", "cosmetic"],
        "ru": ["красота", "косметика", "макияж", "парфюмерия", "уход", "крем"],
        "keywords": ["beauty", "cosmetics", "makeup", "perfume", "skincare",
                    "красота", "косметика", "макияж", "парфюмерия", "уход"]
    },
    "books": {
        "en": ["books", "book", "literature", "reading", "ebook", "audiobook"],
        "ru": ["книги", "книга", "литература", "чтение", "электронная книга"],
        "keywords": ["books", "book", "literature", "reading",
                    "книги", "книга", "литература", "чтение"]
    },
    "home_appliances": {
        "en": ["home appliances", "appliance", "household", "white goods", "refrigerator", "washing machine"],
        "ru": ["бытовая техника", "техника", "холодильник", "стиральная машина", "микроволновка"],
        "keywords": ["appliance", "household", "refrigerator", "washing machine",
                    "бытовая техника", "техника", "холодильник", "стиральная"]
    },
    "furniture": {
        "en": ["furniture", "sofa", "chair", "table", "bed", "cabinet", "wardrobe"],
        "ru": ["мебель", "диван", "стул", "стол", "кровать", "шкаф", "гардероб"],
        "keywords": ["furniture", "sofa", "chair", "table", "bed",
                    "мебель", "диван", "стул", "стол", "кровать"]
    },
    "garden": {
        "en": ["garden", "gardening", "plants", "flowers", "seeds", "tools", "lawn"],
        "ru": ["сад", "огород", "растения", "цветы", "семена", "инструменты", "газон"],
        "keywords": ["garden", "gardening", "plants", "flowers",
                    "сад", "огород", "растения", "цветы"]
    },
    "pets": {
        "en": ["pets", "pet", "dog", "cat", "animal", "pet food", "pet supplies"],
        "ru": ["питомцы", "животные", "собака", "кошка", "корм", "зоотовары"],
        "keywords": ["pets", "pet", "dog", "cat", "animal",
                    "питомцы", "животные", "собака", "кошка", "зоотовары"]
    },
    "entertainment": {
        "en": ["entertainment", "games", "gaming", "video games", "console", "game"],
        "ru": ["развлечения", "игры", "видеоигры", "консоль", "игра", "приставка"],
        "keywords": ["entertainment", "games", "gaming", "console",
                    "развлечения", "игры", "видеоигры", "консоль"]
    },
    "education": {
        "en": ["education", "learning", "course", "training", "tutoring", "school"],
        "ru": ["образование", "обучение", "курсы", "тренинг", "школа", "учеба"],
        "keywords": ["education", "learning", "course", "training",
                    "образование", "обучение", "курсы", "тренинг"]
    },
    "music": {
        "en": ["music", "musical", "instrument", "guitar", "piano", "album", "song"],
        "ru": ["музыка", "музыкальный", "инструмент", "гитара", "пианино", "альбом"],
        "keywords": ["music", "musical", "instrument", "guitar",
                    "музыка", "музыкальный", "инструмент", "гитара"]
    },
    "movies": {
        "en": ["movies", "cinema", "film", "dvd", "bluray", "streaming", "video"],
        "ru": ["фильмы", "кино", "фильм", "двд", "видео", "стриминг"],
        "keywords": ["movies", "cinema", "film", "dvd", "video",
                    "фильмы", "кино", "фильм", "видео"]
    },
    "home_textiles": {
        "en": ["home textiles", "textile", "bedding", "linen", "towels", "curtains"],
        "ru": ["текстиль", "постельное белье", "белье", "полотенца", "шторы"],
        "keywords": ["textile", "bedding", "linen", "towels",
                    "текстиль", "постельное белье", "белье", "полотенца"]
    },
    "kitchen": {
        "en": ["kitchen", "cookware", "tableware", "utensils", "dishes", "cutlery"],
        "ru": ["кухня", "посуда", "кухонная утварь", "тарелки", "приборы"],
        "keywords": ["kitchen", "cookware", "tableware", "dishes",
                    "кухня", "посуда", "кухонная утварь", "тарелки"]
    },
    "tools": {
        "en": ["tools", "tool", "equipment", "hardware", "drill", "screwdriver"],
        "ru": ["инструменты", "инструмент", "оборудование", "дрель", "отвертка"],
        "keywords": ["tools", "tool", "equipment", "hardware",
                    "инструменты", "инструмент", "оборудование"]
    },
    "construction": {
        "en": ["construction", "building materials", "paint", "wallpaper", "tiles"],
        "ru": ["строительство", "стройматериалы", "краска", "обои", "плитка"],
        "keywords": ["construction", "building materials", "paint",
                    "строительство", "стройматериалы", "краска", "обои"]
    },
    "pharmacy": {
        "en": ["pharmacy", "medicines", "drugs", "vitamins", "supplements", "healthcare"],
        "ru": ["аптека", "лекарства", "медикаменты", "витамины", "добавки", "здоровье"],
        "keywords": ["pharmacy", "medicines", "drugs", "vitamins",
                    "аптека", "лекарства", "медикаменты", "витамины"]
    },
    "jewelry": {
        "en": ["jewelry", "jewellery", "watch", "ring", "necklace", "bracelet"],
        "ru": ["украшения", "ювелирные", "часы", "кольцо", "ожерелье", "браслет"],
        "keywords": ["jewelry", "jewellery", "watch", "ring",
                    "украшения", "ювелирные", "часы", "кольцо"]
    },
    "sports_goods": {
        "en": ["sports goods", "sportswear", "sports equipment", "fitness equipment"],
        "ru": ["спорттовары", "спортивная одежда", "спортивное оборудование", "тренажеры"],
        "keywords": ["sports goods", "sportswear", "sports equipment",
                    "спорттовары", "спортивная одежда", "спортивное оборудование"]
    }
}


def normalize_category(category: str) -> str:
    """
    Нормализует категорию к стандартному формату (английское название).
    
    :param category: Исходная категория (может быть на любом языке)
    :return: Нормализованная категория или исходная, если не найдена
    """
    if not category:
        return ""
    
    category_lower = str(category).lower().strip()
    
    # Проверяем каждую категорию в маппинге
    for normalized_name, mappings in CATEGORY_MAPPINGS.items():
        keywords = mappings.get("keywords", [])
        # Проверяем точное совпадение или вхождение ключевых слов
        if category_lower == normalized_name or category_lower in keywords:
            return normalized_name
        # Проверяем вхождение ключевых слов в категорию
        if any(kw in category_lower for kw in keywords):
            return normalized_name
    
    # Если не нашли совпадение, возвращаем исходную категорию в нижнем регистре
    return category_lower


def check_category_match(category: str, target_categories: list) -> bool:
    """
    Проверяет, соответствует ли категория одному из целевых типов категорий.
    
    :param category: Категория для проверки
    :param target_categories: Список целевых категорий (например, ["electronics", "tech"])
    :return: True если соответствует, False иначе
    """
    if not category:
        return False
    
    normalized = normalize_category(category)
    
    # Проверяем прямое совпадение
    if normalized in target_categories:
        return True
    
    # Проверяем по ключевым словам
    category_lower = str(category).lower()
    for target in target_categories:
        if target in CATEGORY_MAPPINGS:
            keywords = CATEGORY_MAPPINGS[target].get("keywords", [])
            if any(kw in category_lower for kw in keywords):
                return True
    
    return False


def get_category_keywords(category_type: str) -> list:
    """
    Возвращает список ключевых слов для категории.
    
    :param category_type: Тип категории (например, "electronics", "travel")
    :return: Список ключевых слов на разных языках
    """
    if category_type in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[category_type].get("keywords", [])
    return []


# Обратная совместимость: экспортируем маппинги для использования в nbo_model
CATEGORY_KEYWORDS = {
    "electronics": get_category_keywords("electronics"),
    "travel": get_category_keywords("travel"),
    "auto": get_category_keywords("auto"),
    "real_estate": get_category_keywords("real_estate"),
    "sport": get_category_keywords("sport"),
    "health": get_category_keywords("health"),
    "food": get_category_keywords("food"),
    "finance": get_category_keywords("finance"),
    "retail": get_category_keywords("retail"),
    "clothing": get_category_keywords("clothing"),
    "children": get_category_keywords("children"),
    "beauty": get_category_keywords("beauty"),
    "books": get_category_keywords("books"),
    "home_appliances": get_category_keywords("home_appliances"),
    "furniture": get_category_keywords("furniture"),
    "garden": get_category_keywords("garden"),
    "pets": get_category_keywords("pets"),
    "entertainment": get_category_keywords("entertainment"),
    "education": get_category_keywords("education"),
    "music": get_category_keywords("music"),
    "movies": get_category_keywords("movies"),
    "home_textiles": get_category_keywords("home_textiles"),
    "kitchen": get_category_keywords("kitchen"),
    "tools": get_category_keywords("tools"),
    "construction": get_category_keywords("construction"),
    "pharmacy": get_category_keywords("pharmacy"),
    "jewelry": get_category_keywords("jewelry"),
    "sports_goods": get_category_keywords("sports_goods"),
    # Алиасы для совместимости
    "tech": get_category_keywords("electronics"),
    "property": get_category_keywords("real_estate"),
    "fashion": get_category_keywords("clothing"),
    "kids": get_category_keywords("children"),
    "cosmetics": get_category_keywords("beauty"),
    "appliances": get_category_keywords("home_appliances"),
    "home": get_category_keywords("real_estate"),
    "toys": get_category_keywords("children"),
    "games": get_category_keywords("entertainment"),
    "medical": get_category_keywords("health"),
}

