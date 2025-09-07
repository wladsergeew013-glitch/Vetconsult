# RAG/embed_and_index.py
# -*- coding: utf-8 -*-

"""
RAG-пайплайн для локальной БД документов.

Что делает скрипт:
1) Чанкование текстов из vetconsult_documents.
2) Эмбеддинги SentenceTransformers.
3) Запись чанков + векторов (pgvector) в vetconsult_chunks.
4) Интерактив: если у документа уже есть чанки — спросить пересоздать/пропустить;
   при согласии на пересоздание спросить размеры чанков (max/overlap/min).
5) (Опционально) Семантический поиск: спросить запрос и количество ответов (top_k),
   красиво распечатать карточками.

Запуск из корня проекта:
    python -m RAG.embed_and_index
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# Стандартная библиотека
# ──────────────────────────────────────────────────────────────────────
import os
import re
import shutil
import textwrap
from typing import List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────
# Внешние зависимости
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# Локальная конфигурация подключения к БД
from PostgreSQL.local_postgres import load_db_config

print("[embed_and_index] v2025-09-07-quality-first")  # маячок версии при запуске

# ──────────────────────────────────────────────────────────────────────
# Константы по БД
# ──────────────────────────────────────────────────────────────────────
TABLE_DOCS = "vetconsult_documents"
TABLE_CHUNKS = "vetconsult_chunks"

# ──────────────────────────────────────────────────────────────────────
# Параметры чанкования (дефолты)
# ──────────────────────────────────────────────────────────────────────
MAX_CHARS = 1000   # целевой размер чанка, символов
MIN_CHARS = 400    # минимальная длина чанка (сливаем короткие хвосты)
OVERLAP   = 100    # перекрытие чанков, символов


# ╔════════════════════════════════════════════════════════════════════╗
# ║                 УТИЛИТЫ: консольный ввод / вывод                  ║
# ╚════════════════════════════════════════════════════════════════════╝

def _yes_no(prompt: str, default_yes: bool = True) -> bool:
    """Вопрос да/нет: [Y/n] или [y/N]. Enter -> значение по умолчанию."""
    suffix = "[Y/n]" if default_yes else "[y/N]"
    ans = input(f"{prompt} {suffix}: ").strip().lower()
    if not ans:
        return default_yes
    return ans in ("y", "yes", "д", "да")


def _ask_int_sequential(prompt: str, default: int, min_value: int | None = None) -> int:
    """Спрашивает целое число. Enter -> default. Проверяет min_value."""
    while True:
        s = input(f"{prompt} (по умолчанию {default}): ").strip()
        if not s:
            return default
        try:
            v = int(s)
            if min_value is not None and v < min_value:
                print(f"[!] Значение должно быть ≥ {min_value}.")
                continue
            return v
        except ValueError:
            print("[!] Введите целое число или просто Enter.")


def _ask_str_sequential(prompt: str, default: str) -> str:
    """Спрашивает строку. Enter -> default."""
    s = input(f"{prompt} (по умолчанию '{default}'): ").strip()
    return s or default


def ask_maybe_configure_settings() -> tuple[int, int, int]:
    """Опциональная преднастройка чанкования (до обхода документов)."""
    print("\n=== Настройки чанкования ===")
    if not _yes_no("Настроить параметры чанкования сейчас?", default_yes=False):
        print("→ Оставляю параметры по умолчанию.")
        return MAX_CHARS, OVERLAP, MIN_CHARS

    max_chars = _ask_int_sequential("Размер чанка", default=MAX_CHARS, min_value=50)
    overlap   = _ask_int_sequential("Перекрытие между чанками", default=OVERLAP, min_value=0)
    min_chars = _ask_int_sequential("Минимальный размер чанка", default=MIN_CHARS, min_value=50)

    if min_chars > max_chars:
        print(f"[!] MIN_CHARS ({min_chars}) > MAX_CHARS ({max_chars}). Ставлю MIN_CHARS = {max_chars // 2}.")
        min_chars = max(50, max_chars // 2)
    if overlap >= max_chars:
        print(f"[!] OVERLAP ({overlap}) ≥ MAX_CHARS ({max_chars}). Ставлю OVERLAP = {max_chars // 5}.")
        overlap = max(0, max_chars // 5)

    return max_chars, overlap, min_chars


def ask_chunk_sizes_for_rebuild(default_max: int, default_overlap: int, default_min: int) -> tuple[int, int, int]:
    """Спрашивает размеры чанков непосредственно перед пересозданием."""
    print("    ⚙ Размеры чанков для пересоздания")
    max_chars = _ask_int_sequential("    Размер чанка", default=default_max, min_value=50)
    overlap   = _ask_int_sequential("    Перекрытие между чанками", default=default_overlap, min_value=0)
    min_chars = _ask_int_sequential("    Минимальный размер чанка", default=default_min, min_value=50)

    if min_chars > max_chars:
        print(f"    [!] MIN_CHARS ({min_chars}) > MAX_CHARS ({max_chars}). Ставлю MIN_CHARS = {max_chars // 2}.")
        min_chars = max(50, max_chars // 2)
    if overlap >= max_chars:
        print(f"    [!] OVERLAP ({overlap}) ≥ MAX_CHARS ({max_chars}). Ставлю OVERLAP = {max_chars // 5}.")
        overlap = max(0, max_chars // 5)

    return max_chars, overlap, min_chars


def ask_maybe_search_params() -> tuple[bool, int, str]:
    """Диалог о запуске семантического поиска после индексации."""
    print("\n=== Семантический поиск ===")
    if not _yes_no("Запустить поиск после обработки документов?", default_yes=True):
        return False, 5, "Протокол лечения у кошек при гастроэнтерите"

    top_k = _ask_int_sequential("Сколько релевантных ответов показать?", default=5, min_value=1)
    query = _ask_str_sequential("Поисковый запрос", default="Протокол лечения у кошек при гастроэнтерерите")
    return True, top_k, query


# ╔════════════════════════════════════════════════════════════════════╗
# ║               КРАСИВЫЙ ВЫВОД РЕЗУЛЬТАТОВ ПОИСКА                   ║
# ╚════════════════════════════════════════════════════════════════════╝

def _term_width(default: int = 100) -> int:
    """Ширина терминала с запасом; fallback — default."""
    try:
        return max(60, shutil.get_terminal_size().columns)
    except Exception:
        return default


def _ellipsize(s: str, max_len: int) -> str:
    """Обрезает строку до max_len с многоточием, уважая границы слов."""
    s = " ".join(s.strip().split())
    if len(s) <= max_len:
        return s
    cut = max_len - 1
    space = s.rfind(" ", 0, cut)
    if space >= max(0, cut - 20):
        return s[:space] + "…"
    return s[:cut] + "…"


def _ansi_ok() -> bool:
    """Можно ли печатать ANSI-коды? (простая эвристика)."""
    return os.environ.get("TERM") not in (None, "dumb")


BOLD, DIM, RESET = ("\033[1m", "\033[2m", "\033[0m") if _ansi_ok() else ("", "", "")


def print_search_results(hits: List[Tuple[float, dict]], query: str, *, max_snippet_lines: int = 3) -> None:
    """Печатает результаты в виде компактных «карточек»."""
    width = _term_width()
    wrap_width = max(60, width - 4)
    sep = "─" * min(120, width)

    print("\n=== ТОП-результаты ===")
    if not hits:
        print("(нет результатов — возможно, таблица чанков пуста)")
        return

    for i, (sim, row) in enumerate(hits, 1):
        filename = str(row.get("filename", "")) or "—"
        chunk_idx = row.get("chunk_index", 0)
        sim_pct = f"{sim * 100:.1f}%"

        header_left = f"{i:>2}. [{sim_pct}] "
        fname_short = _ellipsize(filename, max_len=max(20, wrap_width - len(header_left) - 10))
        header = f"{header_left}{BOLD}{fname_short}{RESET}  {DIM}#{chunk_idx}{RESET}"
        print(header)

        raw = (row.get("text") or "").replace("\n", " ").strip()
        snippet = _ellipsize(raw, max_len=wrap_width * max_snippet_lines)
        wrapped = textwrap.fill(
            snippet,
            width=wrap_width,
            initial_indent="    ",
            subsequent_indent="    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        print(wrapped)
        print(sep)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                      ТЕКСТ → ЧАНКИ (chunking)                      ║
# ╚════════════════════════════════════════════════════════════════════╝

def normalize_text(s: str) -> str:
    """Небольшая чистка: NBSP/ZWSP → пробел, схлопываем множественные пробелы/переносы."""
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_into_sentences(text: str) -> List[str]:
    """Грубая сегментация: делим по .!? + пробел/конец, склеиваем коротыши."""
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    merged: List[str] = []
    buf: List[str] = []
    count = 0
    for p in parts:
        if not p:
            continue
        buf.append(p)
        count += len(p)
        if count >= 150:
            merged.append(" ".join(buf))
            buf, count = [], 0
    if buf:
        merged.append(" ".join(buf))
    return merged


def chunk_text(text: str, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS, overlap: int = OVERLAP) -> List[str]:
    """
    Перекрывающиеся чанки:
      — копим до max_chars,
      — следующий начинается с overlap-символов хвоста предыдущего,
      — короткие хвосты склеиваем с предыдущим чанком.
    """
    sents = split_into_sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for sent in sents:
        if cur_len + len(sent) + 1 <= max_chars:
            cur.append(sent)
            cur_len += len(sent) + 1
        else:
            if cur:
                chunks.append(" ".join(cur))
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                cur = ([tail, sent] if tail else [sent])
                cur_len = len(" ".join(cur))
            else:
                cur = [sent]
                cur_len = len(sent)

    if cur:
        chunks.append(" ".join(cur))

    out: List[str] = []
    for ch in chunks:
        if len(ch) < min_chars and out:
            out[-1] = (out[-1] + " " + ch).strip()
        else:
            out.append(ch.strip())

    return [c for c in out if c]


# ╔════════════════════════════════════════════════════════════════════╗
# ║          БАЗА ДАННЫХ: тип vector(dim), CRUD по чанкам             ║
# ╚════════════════════════════════════════════════════════════════════╝

def ensure_vector_dim(conn, dim: int, table: str = TABLE_CHUNKS, column: str = "embedding", schema: str = "public") -> bool:
    """
    Следит, чтобы {schema}.{table}.{column} имел тип vector(dim).
    Если уже так — False; иначе ALTER и True.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS coltype
            FROM pg_attribute a
            JOIN pg_class     c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s
            """,
            (schema, table, column),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Колонка {schema}.{table}.{column} не найдена")

        coltype = row["coltype"] if isinstance(row, dict) else row[0]
        m = re.search(r"vector\((\d+)\)", str(coltype or ""))
        current_dim = int(m.group(1)) if m else None

        if current_dim == dim:
            return False

        if dim <= 0:
            raise ValueError("dim должен быть положительным целым числом")

        cur.execute(
            f'ALTER TABLE "{schema}"."{table}" ALTER COLUMN "{column}" TYPE vector({dim});'
        )
        conn.commit()
        return True


def get_existing_chunk_count(conn, doc_id: int) -> int:
    """Возвращает количество чанков документа doc_id."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {TABLE_CHUNKS} WHERE doc_id = %s;", (doc_id,))
        row = cur.fetchone()
    return int(row["cnt"] if isinstance(row, dict) else row[0])


def delete_chunks_for_doc(conn, doc_id: int) -> None:
    """Удаляет все чанки документа doc_id."""
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {TABLE_CHUNKS} WHERE doc_id = %s;", (doc_id,))
    conn.commit()


def insert_chunks(conn, doc_id: int, chunks: List[str], embeddings: np.ndarray) -> None:
    """
    Вставляет строки (doc_id, chunk_index, text, embedding).
    Вектор передаём как list, чтобы адаптер pgvector корректно отработал.
    """
    with conn.cursor() as cur:
        for idx, (text, emb) in enumerate(zip(chunks, embeddings.tolist())):
            cur.execute(
                f"""
                INSERT INTO {TABLE_CHUNKS} (doc_id, chunk_index, text, embedding)
                VALUES (%s, %s, %s, %s);
                """,
                (doc_id, idx, text, emb)
            )
    conn.commit()


def get_all_docs(conn) -> List[dict]:
    """Загружает все документы (id, filename, content)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, filename, content FROM {TABLE_DOCS} ORDER BY id;")
        return cur.fetchall()


# ╔════════════════════════════════════════════════════════════════════╗
# ║                           ЭМБЕДДИНГИ                              ║
# ╚════════════════════════════════════════════════════════════════════╝

def load_embedding_model() -> Tuple[SentenceTransformer, int]:
    """Загружает модель эмбеддингов и определяет размерность."""
    print("[i] Загружаем модель эмбеддингов...")
    model = SentenceTransformer("google/embeddinggemma-300m")
    probe = model.encode(["dim_probe"], normalize_embeddings=True, convert_to_numpy=True)
    dim = int(probe.shape[1])
    print(f"[i] Размерность эмбеддинга: {dim}")
    return model, dim


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Считает эмбеддинги (используем normalize_embeddings=True; НИЧЕГО не правим)."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True
    )


def _encode_query(model: SentenceTransformer, query: str) -> List[float]:
    """Эмбеддинг запроса -> list[float] для pgvector (без ручной нормализации/чистки)."""
    q = model.encode([query or ""], normalize_embeddings=True, convert_to_numpy=True)[0]
    return q.astype(np.float32).tolist()


# ╔════════════════════════════════════════════════════════════════════╗
# ║       ОСНОВНОЙ ОБХОД ДОКУМЕНТОВ: skip/rebuild + автосоздание      ║
# ╚════════════════════════════════════════════════════════════════════╝

def rebuild_chunks_for_all_documents_interactive() -> None:
    """
    Обходит документы:
      — Если чанков НЕТ — создаёт их (по текущим глобальным параметрам).
      — Если чанки ЕСТЬ — спрашивает: [S]kip / [R]ebuild / s[k]ip all / re[b]uild all.
        При пересоздании спрашивает размеры (max/overlap/min).
    """
    cfg = load_db_config()
    conn = psycopg.connect(**cfg, row_factory=dict_row)
    register_vector(conn)

    model, dim = load_embedding_model()
    changed = ensure_vector_dim(conn, dim)
    if changed:
        print(f"[i] Привёл тип embedding к vector({dim}).")

    docs = get_all_docs(conn)
    print(f"[i] Документов: {len(docs)}")

    bulk_mode: Optional[str] = None
    bulk_rebuild_sizes: Optional[tuple[int, int, int]] = None

    stats = {"created_missing": 0, "rebuilt": 0, "skipped_existing": 0, "empty_content": 0, "processed": 0}

    for d in docs:
        stats["processed"] += 1
        doc_id = int(d["id"])
        filename = d["filename"]
        content = normalize_text(d["content"] or "")

        if not content:
            print(f"[!] doc_id={doc_id} '{filename}': пустой контент, пропуск.")
            stats["empty_content"] += 1
            continue

        existing = get_existing_chunk_count(conn, doc_id)

        # Нет чанков — создаём
        if existing == 0:
            print(f"[*] doc_id={doc_id} '{filename}': чанков нет -> создаём.")
            chunks = chunk_text(content)
            if not chunks:
                print(f"[!] doc_id={doc_id}: после чанкования пусто, пропуск.")
                stats["empty_content"] += 1
                continue
            embs = embed_texts(model, chunks)
            insert_chunks(conn, doc_id, chunks, embs)
            stats["created_missing"] += len(chunks)
            print(f"[+] создано {len(chunks)} чанков")
            continue

        # Чанки есть — спросим действие
        print(f"[?] doc_id={doc_id} '{filename}': уже есть {existing} чанков.")
        per_doc_sizes: Optional[tuple[int, int, int]] = None

        if bulk_mode is None:
            print("    Пересоздать? Варианты: [S]kip (по умолч.), [R]ebuild, s[k]ip all, re[b]uild all")
            choice = input("    Ваш выбор [S/R/k/b]: ").strip().lower() or "s"

            if choice == "b":
                bulk_mode = "rebuild_all"
                bulk_rebuild_sizes = ask_chunk_sizes_for_rebuild(MAX_CHARS, OVERLAP, MIN_CHARS)
            elif choice == "k":
                bulk_mode = "skip_all"
            elif choice == "r":
                per_doc_sizes = ask_chunk_sizes_for_rebuild(MAX_CHARS, OVERLAP, MIN_CHARS)
            else:
                print("    → Пропускаю этот документ (чанки уже есть).")
                stats["skipped_existing"] += 1
                continue

        if bulk_mode == "skip_all":
            print("    → Пропуск (режим skip all, т.к. чанки уже есть).")
            stats["skipped_existing"] += 1
            continue

        # Пересоздание
        print(f"    → Пересоздаю чанки (старых: {existing}).")
        delete_chunks_for_doc(conn, doc_id)

        if bulk_mode == "rebuild_all" and bulk_rebuild_sizes is not None:
            max_chars, overlap, min_chars = bulk_rebuild_sizes
        else:
            if per_doc_sizes is None:
                per_doc_sizes = ask_chunk_sizes_for_rebuild(MAX_CHARS, OVERLAP, MIN_CHARS)
            max_chars, overlap, min_chars = per_doc_sizes

        chunks = chunk_text(content, max_chars=max_chars, min_chars=min_chars, overlap=overlap)
        if not chunks:
            print(f"[!] doc_id={doc_id}: после чанкования пусто, пропуск.")
            stats["empty_content"] += 1
            continue

        embs = embed_texts(model, chunks)
        insert_chunks(conn, doc_id, chunks, embs)
        stats["rebuilt"] += len(chunks)
        print(f"[+] doc_id={doc_id}: пересоздано {len(chunks)} чанков (max={max_chars}, overlap={overlap}, min={min_chars})")

    print(
        "\n[✓] Готово."
        f"\n    Документов обработано: {stats['processed']}"
        f"\n    Создано (где не было чанков): {stats['created_missing']}"
        f"\n    Пересоздано (где чанки были): {stats['rebuilt']}"
        f"\n    Пропущено (чанки были, skip): {stats['skipped_existing']}"
        f"\n    Пропущено (пустой контент/после чанкования пусто): {stats['empty_content']}"
    )
    conn.close()


# ╔════════════════════════════════════════════════════════════════════╗
# ║                        СЕМАНТИЧЕСКИЙ ПОИСК                        ║
# ╚════════════════════════════════════════════════════════════════════╝

def _finite_predicate_sql(expr: str) -> str:
    """
    Возвращает SQL-предикат «expr — финитное число» без isfinite():
      — не NULL,
      — не NaN (x = x),
      — не +∞ и не −∞.
    """
    return (
        f"({expr}) IS NOT NULL "
        f"AND ({expr}) = ({expr}) "
        f"AND ({expr}) < 'Infinity'::float8 "
        f"AND ({expr}) > '-Infinity'::float8"
    )


def semantic_search(query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
    """
    Возвращает top_k ближайших чанков: [(similarity, row_dict)].
    НИКАКИХ модификаций эмбеддингов — только отбор по валидной метрике.
    """
    if top_k <= 0:
        return []

    cfg = load_db_config()
    conn = psycopg.connect(**cfg, row_factory=dict_row)
    register_vector(conn)

    # Эмбеддинг запроса (нормированный моделью)
    model = SentenceTransformer("google/embeddinggemma-300m")
    q_emb = _encode_query(model, query)

    dist_expr = "c.embedding <=> q.v"
    finite_pred = _finite_predicate_sql(dist_expr)

    # Диагностика (не обязательно, но полезно понять «почему меньше»)
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {TABLE_CHUNKS};")
        total_chunks = int(cur.fetchone()["cnt"])
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {TABLE_CHUNKS} WHERE embedding IS NOT NULL;")
        with_emb = int(cur.fetchone()["cnt"])
        cur.execute(
            f"""
            WITH q(v) AS (SELECT (%s)::vector)
            SELECT COUNT(*) AS cnt
            FROM {TABLE_CHUNKS} c
            CROSS JOIN q
            WHERE {finite_pred}
            """,
            (q_emb,)
        )
        rankable = int(cur.fetchone()["cnt"])

    # Основной запрос: ранжируем только финитные расстояния
    sql = f"""
        WITH q(v) AS (SELECT (%s)::vector)
        SELECT
            c.id, c.doc_id, c.chunk_index, c.text,
            COALESCE(d.filename, '') AS filename,
            ({dist_expr})::float AS cosine_distance
        FROM {TABLE_CHUNKS} c
        LEFT JOIN {TABLE_DOCS} d ON d.id = c.doc_id
        CROSS JOIN q
        WHERE {finite_pred}
        ORDER BY {dist_expr}
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (q_emb, int(top_k)))
        rows = cur.fetchall()

    # Подсказка, если вернулось меньше, чем просили
    if len(rows) < top_k:
        print(f"(доступно валидных результатов: {len(rows)} из запрошенных {top_k} — всего чанков: {total_chunks}, с embedding: {with_emb}, пригодных к ранжированию: {rankable})")

    conn.close()

    results: List[Tuple[float, dict]] = []
    for r in rows:
        dist = float(r["cosine_distance"])
        sim = 1.0 - dist
        results.append((sim, r))
    return results


# ╔════════════════════════════════════════════════════════════════════╗
# ║                             ТОЧКА ВХОДА                           ║
# ╚════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # 1) Преднастройка чанкования (по желанию)
    max_chars, overlap, min_chars = ask_maybe_configure_settings()
    MAX_CHARS, OVERLAP, MIN_CHARS = max_chars, overlap, min_chars

    # 2) Индексация
    rebuild_chunks_for_all_documents_interactive()

    # 3) Поиск
    run_search, top_k, query = ask_maybe_search_params()
    if run_search:
        hits = semantic_search(query, top_k=top_k)
        print_search_results(hits, query=query, max_snippet_lines=3)
    else:
        print("→ Поиск пропущен.")
