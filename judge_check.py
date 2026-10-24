"""Выгрузка ответов бота и вердиктов судьи для ручной проверки."""
import csv
import json
import sys

from components import get_judge
from evaluate import JUDGE_PROMPT
from rag import answer, format_docs, is_refusal

QUESTIONS_PATH = "eval/questions.json"
OUT_PATH = "eval/judge_check.csv"


def collect():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [q for q in json.load(f) if q["type"] == "answerable"]

    judge = get_judge()
    rows = []
    for q in questions:
        result = answer(q["question"], preshow_judge=False)
        if is_refusal(result):
            continue
        context = format_docs(result["sources"])
        prompt = JUDGE_PROMPT.format(context=context, answer=result["answer"])
        verdict = judge.invoke(prompt).content.strip()
        rows.append({
            "question": q["question"],
            "answer": result["answer"],
            "context": context,
            "judge": verdict,
            "my_verdict": "",
        })

    with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сохранено {len(rows)} ответов в {OUT_PATH}")


def is_yes(text):
    return text.strip().lower().startswith("да")


def score():
    with open(OUT_PATH, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if r["my_verdict"].strip()]

    agree = sum(is_yes(r["judge"]) == is_yes(r["my_verdict"]) for r in rows)
    print(f"Совпадений: {agree} из {len(rows)} ({agree / len(rows):.2f})")
    for r in rows:
        if is_yes(r["judge"]) != is_yes(r["my_verdict"]):
            print(f"- {r['question']}: судья «{r['judge']}», я «{r['my_verdict']}»")


if __name__ == "__main__":
    score() if "--score" in sys.argv else collect()