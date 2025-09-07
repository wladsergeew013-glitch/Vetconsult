# LLM/test_llm_rag.py
# -*- coding: utf-8 -*- 
# Для запуска python -m RAG.test_llm_rag 

"""
Тестовый скрипт: берём top-k релевантных чанков из RAG и передаём их в LLM.
— Системный промпт читается из LLM/system promt #1.txt
— Пользователь вводит запрос
— Чанки берутся через semantic_search() из RAG.embed_and_index
— Контекст аккуратно форматируется и ограничивается по длине
— Запрос уходит в llm_client.chat_once(system_prompt=..., user_prompt=...)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple
from timeit import default_timer as timer

# 1) Наш LLM-клиент, функция chat_once() в LLM/llm_client.py
from LLM.llm_client import chat_once

# 2) RAG-поиск, функция semantic_search() находится в RAG/embed_and_index.py
from RAG.embed_and_index import semantic_search  # quality-first версия


# ===== Настройки =====
SYSTEM_PROMPT_PATH = Path("LLM") / "system promt #1.txt"
DEFAULT_USER_QUERY = "Привет, как дела?"  # можно заменить на пустую строку
DEFAULT_TOP_K = 5

# Жёсткая отсечка по длине контекста (символы).
# Подбирай под твой LLM-лимит: 6000–12000 обычно ок.
CONTEXT_CHARS_LIMIT = 9000


def read_system_prompt(path: Path) -> str:
    """
    Читает системный промпт из файла. Падает с понятной ошибкой, если нет файла.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл с системным промптом не найден: {path}")
    return path.read_text(encoding="utf-8").strip()


def format_context(chunks: List[Tuple[float, dict]], *, limit_chars: int) -> str:
    """
    Превращает список [(sim, row_dict)] в текстовый контекст для LLM.
    Уважает лимит символов и аккуратно обрезает на границе чанка.
    """
    lines: List[str] = []
    used = 0

    lines.append("Выдержки из базы знаний (только для справки, проверяй факты):")
    lines.append("")

    for i, (sim, row) in enumerate(chunks, 1):
        filename = str(row.get("filename") or "—")
        idx = row.get("chunk_index", 0)
        sim_pct = f"{sim * 100:.1f}%"
        text = (row.get("text") or "").strip()

        block = (
            f"[{i}] Файл: {filename} | Чанк: #{idx} | Похожесть: {sim_pct}\n"
            f"{text}\n"
            f"{'-'*80}\n"
        )

        # Если добавление блока превышает лимит — прерываемся (ничего не режем внутри блока)
        if used + len(block) > limit_chars:
            lines.append("…(контекст обрезан по лимиту) …")
            break

        lines.append(block)
        used += len(block)

    return "\n".join(lines).rstrip()


def build_user_prompt(user_query: str, context: str) -> str:
    """
    Собирает единый пользовательский промпт:
    — сначала идёт «контекст из RAG»,
    — затем инструкция и вопрос пользователя.
    Модель может ссылаться на контекст, но не обязана копировать его.
    """
    return (
        f"{context}\n\n"
        "Инструкция: используя выдержки выше И/ИЛИ общие знания, кратко и по делу ответь на вопрос. "
        "Если контекст не содержит явного ответа, скажи это прямо и предложи, какие данные ещё нужны.\n\n"
        f"Вопрос пользователя: {user_query.strip() or '(пустой)'}"
    )


def main():
    
    # 1) Системный промпт
    system_prompt = read_system_prompt(SYSTEM_PROMPT_PATH)

    # 2) Параметры запроса
    try:
        top_k_str = input(f"Сколько релевантных чанков подставить? (по умолчанию {DEFAULT_TOP_K}): ").strip()
        top_k = int(top_k_str) if top_k_str else DEFAULT_TOP_K
        if top_k <= 0:
            top_k = DEFAULT_TOP_K
    except Exception:
        top_k = DEFAULT_TOP_K

    user_query = input(f"Введите пользовательский запрос (по умолчанию '{DEFAULT_USER_QUERY}'): ").strip() or DEFAULT_USER_QUERY
    start_time = timer()
    # 3) Поиск релевантных чанков через RAG
    hits = semantic_search(user_query, top_k=top_k)

    # 4) Формируем контекст для LLM (с лимитом)
    context = format_context(hits, limit_chars=CONTEXT_CHARS_LIMIT)

    # 5) Собираем user_prompt и вызываем LLM
    user_prompt = build_user_prompt(user_query, context)
    answer = chat_once(user_prompt=user_prompt, system_prompt=system_prompt)

    # 6) Вывод
    print("\n" + "=" * 100)
    print("ОТВЕТ МОДЕЛИ:\n")
    print(answer.strip())
    print("=" * 100)

    # 7) Для удобства — кратко покажем, какие куски подставлялись
    if hits:
        print("\nИСПОЛЬЗОВАННЫЕ КУСОЧКИ (ссылки):")
        for i, (sim, row) in enumerate(hits, 1):
            filename = str(row.get("filename") or "—")
            idx = row.get("chunk_index", 0)
            sim_pct = f"{sim * 100:.1f}%"
            print(f"  [{i}] {filename}  #{idx}  ({sim_pct})")
    elapsed = timer() - start_time
    print(f"\n⏱️ Время выполнения: {elapsed:.2f} сек.")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(130)
