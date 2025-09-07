LLM:
llm_client.py - файл для функции запроса к llm
local LLM - скрипт для работы с локальной моделью LM Studio
llm_client test - скрипт для тестирования через функцию запроса из llm_client.py
system promt #1 - первый системный промт


PostgreSQL:
local postgres.py - файл для функций с базой postgres
postgres_test.py - скрипт для тестирования функций local postgres.py
db_config.txt - конфиг для базы postgres
upload_documents.py - загрузка документов в базу postgres

RAG:
embed_and_index.py -делит на чанки уже готовую базу и возвращает несколько вариантов. Запускай через python -m RAG.embed_and_index
quaery.py - тестовый файл ,чтобы проверить работу моделей через hugging face
test_llm_rag - использует функцию из embed_and_index для семантического поиска и функцию из llm_client для связи с моделью. 




