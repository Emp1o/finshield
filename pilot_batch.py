import csv
import json
import os
import sys
import time

from dotenv import load_dotenv
from langchain_gigachat import GigaChat

load_dotenv()

QUERIES_PATH = "pilot/queries.json"     # закрытый набор запросов, не в репозитории
RESULTS_PATH = "pilot/pilot_results.csv"

if not os.path.exists(QUERIES_PATH):
    sys.exit(
        f"Не найден файл {QUERIES_PATH}. "
        "Создай его по образцу eval/queries_example.json."
    )

with open(QUERIES_PATH, encoding="utf-8") as f:
    QUERIES = [(q["id"], q["category"], q["query"]) for q in json.load(f)]

llm = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    scope="GIGACHAT_API_PERS",
    model="GigaChat-2",
    verify_ssl_certs=False,
)

os.makedirs("pilot", exist_ok=True)

# utf-8-sig и разделитель ";" — чтобы файл нормально открылся в Excel с русскими буквами
with open(RESULTS_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["№", "категория", "запрос", "finish_reason", "начало ответа", "разметка"])

    for number, category, query in QUERIES:
        try:
            response = llm.invoke(query)
            finish = response.response_metadata.get("finish_reason")
            text = response.content[:500].replace("\n", " ")
        except Exception as e:
            finish, text = "ошибка", str(e)[:200]

        writer.writerow([number, category, query, finish, text, ""])
        print(f"{number:>2} [{category}] finish_reason={finish}")
        time.sleep(2)   # пауза, чтобы не упереться в ограничение частоты запросов

print("Готово. Результаты:", RESULTS_PATH)