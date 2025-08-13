# llm_client.py
#Служит как функция для отправки запросов к локальной модели LM Studio

from typing import Optional
from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"   # для LM Studio обычно так
DEFAULT_MODEL = "AUTODETECT"

def chat_once(
    user_prompt: str,
    system_prompt: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 200,
) -> str:
    """
    Отправляет запрос в LM Studio (OpenAI-совместимый API) и возвращает текст ответа.

    :param user_prompt: Пользовательский запрос (что «спросить» у модели)
    :param system_prompt: Системный промт (инструкции ассистенту)
    :param base_url: База API (по умолчанию локальный LM Studio)
    :param api_key: API-ключ (для LM Studio можно оставить "lm-studio")
    :param model: Имя модели (в LM Studio можно "AUTODETECT")
    :param temperature: Креативность/вариативность ответа
    :param max_tokens: Ограничение на длину ответа модели
    :return: Строка — ответ модели
    """
    client = OpenAI(base_url=base_url, api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content or ""
