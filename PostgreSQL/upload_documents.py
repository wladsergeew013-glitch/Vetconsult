from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from striprtf.striprtf import rtf_to_text

from local_postgres import load_db_config, run_query

TABLE_NAME = "vetconsult_documents"
DOCS_FOLDER = Path("documents")

def ensure_table_exists():
    cfg = load_db_config()
    conn = psycopg.connect(**cfg, row_factory=dict_row)
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL
            );
        """)
        conn.commit()
    conn.close()

def get_existing_filenames():
    cfg = load_db_config()
    conn = psycopg.connect(**cfg, row_factory=dict_row)
    with conn.cursor() as cur:
        cur.execute(f"SELECT filename FROM {TABLE_NAME};")
        rows = cur.fetchall()
    conn.close()
    return {row["filename"] for row in rows}

def insert_document(filename: str, content: str):
    cfg = load_db_config()
    conn = psycopg.connect(**cfg, row_factory=dict_row)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {TABLE_NAME} (filename, content) VALUES (%s, %s) ON CONFLICT (filename) DO NOTHING;",
            (filename, content)
        )
        conn.commit()
    conn.close()

def update_document(filename: str, content: str):
    cfg = load_db_config()
    conn = psycopg.connect(**cfg, row_factory=dict_row)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {TABLE_NAME} SET content = %s WHERE filename = %s;",
            (content, filename)
        )
        conn.commit()
    conn.close()

def decode_best(raw_bytes: bytes) -> str:
    """Пробует разные кодировки и выбирает ту, где больше кириллицы"""
    encodings = ["utf-8", "cp1251", "latin1"]
    best_text = ""
    best_score = -1
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc, errors="ignore")
        except UnicodeDecodeError:
            continue
        cyr_count = sum(0x0400 <= ord(c) <= 0x04FF for c in text)
        score = cyr_count / max(len(text), 1)
        if score > best_score:
            best_text = text
            best_score = score
    return best_text

def read_rtf_file(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    decoded_text = decode_best(raw_bytes)
    return rtf_to_text(decoded_text)

def main():
    ensure_table_exists()
    existing_files = get_existing_filenames()

    if not DOCS_FOLDER.exists():
        print(f"[!] Папка {DOCS_FOLDER} не найдена")
        return

    # Спрашиваем один раз про перезапись
    overwrite = False
    if existing_files:
        choice = input(f"[?] Найдено {len(existing_files)} документов в базе. Перезаписать их текст? (y/n): ").strip().lower()
        overwrite = choice == "y"

    added_count, updated_count, skipped_count = 0, 0, 0

    for file_path in DOCS_FOLDER.glob("*.rtf"):
        text_content = read_rtf_file(file_path)

        if file_path.name in existing_files:
            if overwrite:
                update_document(file_path.name, text_content)
                updated_count += 1
                print(f"[~] Обновлён: {file_path.name}")
            else:
                skipped_count += 1
                print(f"[-] Пропущен: {file_path.name}")
        else:
            insert_document(file_path.name, text_content)
            added_count += 1
            print(f"[+] Добавлен: {file_path.name}")

    print(f"\nГотово. Добавлено {added_count}, обновлено {updated_count}, пропущено {skipped_count}.")
    print("\nСодержимое таблицы:")
    run_query(TABLE_NAME, limit=20)

if __name__ == "__main__":
    main()
