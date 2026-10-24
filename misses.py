import json
import os

from rag import answer, retrieve, source_name, is_refusal

with open("eval/questions.json", encoding="utf-8") as f:
    EVAL = json.load(f)

print("=== Retrieval misses ===")
for ex in EVAL:
    if not ex["source"]:
        continue
    docs = retrieve(ex["question"])
    found = [os.path.basename(d.metadata["source"]) for d in docs]
    expected = ex["source"] if isinstance(ex["source"], list) else [ex["source"]]
    if not set(expected) & set(found):
        print(f"\nQ: {ex['question']}")
        print(f"   expected: {', '.join(expected)}")
        print(f"   found:    {', '.join(source_name(d) for d in docs)}")

print("\n=== Refusal errors ===")
for ex in EVAL:
    result = answer(ex["question"], preshow_judge=False)
    refused = is_refusal(result)
    if ex["source"] and refused:
        print(f"\nFALSE REFUSAL (score {result['top_score']:.3f}): {ex['question']}")
        print(f"   answer: {result['answer']}")
    if not ex["source"] and not refused:
        print(f"\nMISSED REFUSAL: {ex['question']}")
        print(f"   answer: {result['answer'][:200]}")