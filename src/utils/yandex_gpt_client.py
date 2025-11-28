"""
Утилита для работы с YandexGPT Responses API.

Предоставляет единый интерфейс для работы с YandexGPT через OpenAI-compatible API.
"""

from typing import Optional, Dict, Any
from openai import OpenAI

from src.utils.yandex_cloud import get_cached_config


# Глобальный кэш клиента
_cached_client: Optional[OpenAI] = None
_cached_model: Optional[str] = None


def get_yandex_gpt_client() -> tuple[OpenAI, str]:
    """
    Получает клиент YandexGPT Responses API с кэшированием.
    
    :return: Кортеж (client, model)
    """
    global _cached_client, _cached_model
    
    if _cached_client is None or _cached_model is None:
        config = get_cached_config(auto_download=False)
        folder_id = config["folder_id"]
        api_key = config["api_key"]
        
        model = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"
        
        _cached_client = OpenAI(
            base_url="https://rest-assistant.api.cloud.yandex.net/v1",
            api_key=api_key,
            project=folder_id
        )
        _cached_model = model
    
    return _cached_client, _cached_model


def call_yandex_gpt(
    input_text: str,
    instructions: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    reasoning: Optional[Dict[str, str]] = None,
    store: bool = False,
    previous_response_id: Optional[str] = None
) -> str:
    """
    Вызывает YandexGPT Responses API.
    
    :param input_text: Входной текст для обработки
    :param instructions: Инструкции для модели
    :param temperature: Температура генерации (0.0-1.0)
    :param max_tokens: Максимальное количество токенов
    :param reasoning: Настройки рассуждения (например, {"effort": "low"})
    :param store: Сохранять ли ответ для последующего использования
    :param previous_response_id: ID предыдущего ответа для контекста
    :return: Текст ответа
    """
    client, model = get_yandex_gpt_client()
    
    params = {
        "model": model,
        "input": input_text,
        "store": store
    }
    
    if instructions:
        params["instructions"] = instructions
    
    if reasoning:
        params["reasoning"] = reasoning
    
    if previous_response_id:
        params["previous_response_id"] = previous_response_id
    
    res = client.responses.create(**params)
    
    return res.output_text


def call_yandex_gpt_with_tools(
    input_text: str,
    tools: list[Dict[str, Any]],
    instructions: Optional[str] = None,
    store: bool = True
) -> tuple[str, Optional[Any]]:
    """
    Вызывает YandexGPT с инструментами (function calling).
    
    :param input_text: Входной текст
    :param tools: Список инструментов (tools)
    :param instructions: Инструкции для модели
    :param store: Сохранять ли ответ
    :return: Кортеж (output_text, response_object)
    """
    client, model = get_yandex_gpt_client()
    
    params = {
        "model": model,
        "input": input_text,
        "tools": tools,
        "store": store
    }
    
    if instructions:
        params["instructions"] = instructions
    
    res = client.responses.create(**params)
    
    return res.output_text, res

