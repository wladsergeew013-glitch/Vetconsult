from openai import OpenAI
import json
# Скрипт для работы с локальной моделью LM Studio. 
# Чтобы скрипт запустился, необходимо запустить LM Studio 
# и убедиться, что он доступен по адресу http://localhost:1234/v1.
# Убедитесь, что у вас установлен пакет openai:
# pip install openai
# Такжэе необходимо, чтобы перейти в папку Developer в LM Studio и выбрать модель, 
# которую вы хотите использовать.

# Инициализация клиента для LM Studio
client = OpenAI(
    base_url="http://localhost:1234/v1",  # Адрес вашего локального сервера LM Studio
    api_key="lm-studio"  # API-ключ, для LM Studio обычно используется "lm-studio"
)

# Определение промта
messages = [
    {"role": "system", "content": "Ты полезный ИИ-ассистент, отвечай кратко и по делу."},
    {"role": "user", "content": "Напиши простой пример функции на Python."}
]

# Отправка запроса к модели
response = client.chat.completions.create(
    model="AUTODETECT",  # Используем AUTODETECT
    messages=messages,
    temperature=0.7,  # Настройка "креативности" ответа
    max_tokens=200    # Максимальная длина ответа
)

# Получение и вывод ответа
result = response.choices[0].message.content
print("Ответ модели:")
print(json.dumps(result, indent=2, ensure_ascii=False))