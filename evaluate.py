import json
import os
import sys
import time

import numpy as np
import pandas as pd

from components import get_judge
from rag import answer, format_docs, retrieve, is_refusal

QUESTIONS_PATH = next((a for a in sys.argv[1:] if a.endswith(".json")), "eval/questions.json")
RUN_NAME = os.path.splitext(os.path.basename(QUESTIONS_PATH))[0]

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    EVAL = json.load(f)

ANSWERABLE = [ex for ex in EVAL if ex["source"]]

JUDGE_PROMPT = (
    "Тебе даны фрагменты документов и ответ ассистента. Проверь, подтверждается ли каждое "
    "утверждение ответа этими фрагментами. Ответь одним словом: да или нет.\n\n"
    "Фрагменты:\n{context}\n\nОтвет ассистента:\n{answer}"
)


def rank_of_correct(docs, ex):
    """Return the 1-based rank of the first chunk from any expected source, or None."""
    expected = ex["source"] if isinstance(ex["source"], list) else [ex["source"]]
    for rank, d in enumerate(docs, 1):
        if os.path.basename(d.metadata["source"]) in expected:
            return rank
    return None


def bootstrap_ci(values, n_boot=5000, seed=0):
    """95% bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def evaluate_retrieval(use_reranker):
    hits, reciprocal_ranks = [], []
    for ex in ANSWERABLE:
        rank = rank_of_correct(retrieve(ex["question"], use_reranker=use_reranker), ex)
        hits.append(1 if rank else 0)
        reciprocal_ranks.append(1 / rank if rank else 0)
    low, high = bootstrap_ci(hits)
    return {
        "config": "reranker" if use_reranker else "no reranker",
        "hit_rate@4": np.mean(hits),
        "hit_rate 95% CI": f"[{low:.2f}; {high:.2f}]",
        "MRR": np.mean(reciprocal_ranks),
    }


def is_yes(verdict):
    return verdict.strip().lower().startswith("да")


def with_retry(fn, *args, attempts=4, base_delay=2, **kwargs):
    """Call fn, retrying with exponential backoff; re-raise after the last attempt."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            print(f"{type(e).__name__}: {e} — повтор {attempt}/{attempts - 1} через {delay} с",
                  file=sys.stderr)
            time.sleep(delay)


def evaluate_answers():
    """Faithfulness and false refusals on answerable questions, correct refusals on the rest.

    Per-question results are saved to eval/answers_<questions file>.csv.
    """
    judge = get_judge()
    faithful, false_refusals, refusals, rows = [], [], [], []
    for ex in EVAL:
        result = with_retry(answer, ex["question"], preshow_judge=False)
        refused = is_refusal(result)
        row = {"question": ex["question"], "type": ex.get("type", ""),
       "refused": refused, "top_score": result.get("top_score"),
       "answer": result["answer"], "judge": "", "context": ""}
        if ex["source"]:
            false_refusals.append(1 if refused else 0)
            if not refused:
                context = format_docs(result["sources"])
                verdict = with_retry(judge.invoke, JUDGE_PROMPT.format(
                    context=context, answer=result["answer"])).content.strip()
                faithful.append(1 if is_yes(verdict) else 0)
                row.update(judge=verdict, context=context)
        else:
            refusals.append(1 if refused else 0)
        rows.append(row)

    pd.DataFrame(rows).to_csv(f"eval/answers_{RUN_NAME}.csv", index=False,
                              sep=";", encoding="utf-8-sig")
    rejected = [r for r in rows if r["judge"] and not is_yes(r["judge"])]
    return {
        "faithfulness": np.mean(faithful) if faithful else None,
        "faithfulness 95% CI": bootstrap_ci(faithful) if faithful else None,
        "false_refusals": np.mean(false_refusals),
        "correct_refusals": np.mean(refusals) if refusals else None,
    }, rejected


if __name__ == "__main__":
    table = pd.DataFrame([evaluate_retrieval(False), evaluate_retrieval(True)])
    print(table.to_string(index=False))
    table.to_csv(f"eval/retrieval_{RUN_NAME}.csv", index=False)

    if "--retrieval-only" not in sys.argv:
        metrics, rejected = evaluate_answers()
        for name, value in metrics.items():
            if value is None:
                print(f"{name}: n/a")
            elif name.endswith("CI"):
                print(f"{name}: [{value[0]:.2f}; {value[1]:.2f}]")
            else:
                print(f"{name}: {value:.2f}")

        print(f"\nОтклонено судьёй: {len(rejected)}")
        for r in rejected:
            print(f"\n--- {r['question']}\nОтвет: {r['answer']}")