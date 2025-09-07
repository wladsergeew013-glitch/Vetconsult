# тестовый скрипт для проверки работы запроса к LLM studio через функцию, которая распиана в llm_client
from llm_client import chat_once
from pathlib import Path

if __name__ == "__main__":
    # Читаем системный промт из файла
    system_prompt_path = Path("LLM\system promt #1.txt")
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"Файл с системным промтом не найден: {system_prompt_path}")

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    # Пример запроса
    answer = chat_once(
        user_prompt="Привет, как дела?",
        system_prompt=system_prompt
    ) 

    print("Ответ модели:\n", answer)
