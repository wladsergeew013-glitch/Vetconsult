import psycopg
from psycopg.rows import dict_row
from rich.console import Console
from rich.table import Table
from pathlib import Path

# Читаем конфиг из файла
def load_db_config(path: str = "PostgreSQL/db_config.txt") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Файл настроек не найден: {cfg_path}")

    config = {}
    with cfg_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config

# Получить строки по имени таблицы или SQL-запросу
def fetch_rows(query_or_table: str, limit: int | None = None) -> list[dict]:
    sql = query_or_table.strip()

    # Если передано только имя таблицы — строим SELECT
    if not sql.lower().startswith("select"):
        sql = f"SELECT * FROM {sql}"

    # Добавляем LIMIT, если он передан и нет в SQL
    if limit is not None and "limit" not in sql.lower():
        sql = f"{sql.rstrip(';')} LIMIT {limit}"

    sql += ";" if not sql.endswith(";") else ""

    # Подключение
    config = load_db_config()
    conn = psycopg.connect(
        host=config["host"],
        port=int(config["port"]),
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        sslmode=config.get("sslmode", "prefer"),
        connect_timeout=int(config.get("connect_timeout", 10)),
        row_factory=dict_row
    )

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    conn.close()
    return rows

# Показать результат в красивой таблице
def run_query(query_or_table: str, limit: int | None = None):
    rows = fetch_rows(query_or_table, limit)
    if not rows:
        print("[!] Нет данных")
        return

    table_view = Table(show_header=True, header_style="bold magenta")
    for col in rows[0].keys():
        table_view.add_column(str(col))

    for row in rows:
        table_view.add_row(*[str(v) if v is not None else "" for v in row.values()])

    console = Console()
    console.print(table_view)
