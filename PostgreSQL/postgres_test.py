from local_postgres import run_query, fetch_rows

run_query("SELECT * FROM public.documents", limit=1)

# 3. Получить результат в переменную
data = fetch_rows("SELECT * FROM public.users", limit=10)
print(f"Получено {len(data)} строк")
print(data[0])
