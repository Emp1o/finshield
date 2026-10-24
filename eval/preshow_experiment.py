"""Прогон итоговой конфигурации с проверкой черновика судьёй перед показом.

Сравнивает один и тот же черновик до и после судьи: бот вызывается без фильтра,
затем тот же шаг, что в боте (`apply_preshow`), решает, показывать ли ответ.
Запуск из корня репозитория:

    python eval/preshow_experiment.py
    python eval/preshow_experiment.py --runs 3
    python eval/preshow_experiment.py eval/questions.json --strictness verbatim

Нужны GIGACHAT_CREDENTIALS, PDF в data/ и собранный chroma_db/.
Пишет eval/preshow_<имя набора>.csv (при --runs больше 1 — с суффиксом _runN)
и печатает доли прохода. В журнал числа сам не переносит.
"""
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preshow import (  # noqa: E402
    apply_preshow,
    check_draft,
    example_is_trap,
    in_near_band,
    is_case_9,
    strictness_text,
    summarize_rows,
)
from rag import NO_ANSWER, answer, format_docs, is_refusal  # noqa: E402

DEFAULT_PATHS = ["eval/questions.json", "eval/questions_hard.json"]
FIELDNAMES = [
    "id",
    "dataset",
    "question",
    "type",
    "category",
    "top_score",
    "near_threshold",
    "passed_before_judge",
    "passed_after_judge",
    "is_case_9",
    "is_trap",
    "judge_decision",
    "judge_reason",
    "latency_answer_s",
    "latency_judge_s",
    "latency_s",
    "strictness",
    "answer",
]


def with_retry(fn, *args, attempts=4, base_delay=2, **kwargs):
    """Вызвать fn с повторами и растущей паузой. После последней попытки ошибка летит дальше."""
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


def _positive_int(raw):
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit("--runs должен быть целым числом") from None
    if value < 1:
        raise SystemExit("--runs должен быть не меньше 1")
    return value


def parse_args(argv):
    strictness = None
    runs = 1
    paths = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--strictness":
            i += 1
            if i >= len(argv):
                raise SystemExit("нужно значение после --strictness: soft или verbatim")
            strictness = argv[i]
        elif arg.startswith("--strictness="):
            strictness = arg.split("=", 1)[1]
        elif arg == "--runs":
            i += 1
            if i >= len(argv):
                raise SystemExit("нужно целое число после --runs")
            runs = _positive_int(argv[i])
        elif arg.startswith("--runs="):
            runs = _positive_int(arg.split("=", 1)[1])
        elif arg.endswith(".json"):
            paths.append(arg)
        else:
            raise SystemExit(f"неизвестный аргумент: {arg}")
        i += 1
    return paths or DEFAULT_PATHS, strictness, runs


def run_question(example, *, dataset, index, strictness):
    """Одно сравнение до и после судьи. Черновик получается без фильтра.

    Решение «показать или нет» принимает apply_preshow — тот же шаг, что у бота.
    Судья вызывается только если у черновика есть ссылки и это ещё не отказ.
    """
    mode, _ = strictness_text(strictness)
    started = time.perf_counter()
    result = with_retry(answer, example["question"], preshow_judge=False)
    latency_answer = time.perf_counter() - started
    judge_time = {"seconds": 0.0}

    def checker(context, draft):
        judged_at = time.perf_counter()
        verdict = with_retry(check_draft, context, draft, strictness=strictness)
        judge_time["seconds"] = time.perf_counter() - judged_at
        return verdict

    _shown, _cited, _judge_pass, reason, decision = apply_preshow(
        result["answer"],
        result.get("cited") or [],
        format_docs(result.get("sources") or []),
        enabled=True,
        checker=checker,
        refusal_text=NO_ANSWER,
    )
    passed_before = not is_refusal(result)
    # Отказ до судьи так и остаётся отказом. «fail» — судья спрятал черновик.
    passed_after = passed_before and decision != "fail"
    latency_judge = judge_time["seconds"]
    score = result.get("top_score")
    return {
        "id": index,
        "dataset": dataset,
        "question": example["question"],
        "type": example.get("type", ""),
        "category": example.get("category", "") or "",
        "top_score": "" if score is None else score,
        "near_threshold": in_near_band(score),
        "passed_before_judge": passed_before,
        "passed_after_judge": passed_after,
        "is_case_9": is_case_9(example["question"]),
        "is_trap": example_is_trap(example),
        "judge_decision": decision,
        "judge_reason": reason,
        "latency_answer_s": round(latency_answer, 3),
        "latency_judge_s": round(latency_judge, 3),
        "latency_s": round(latency_answer + latency_judge, 3),
        "strictness": mode,
        "answer": result["answer"],
    }


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _share(hits, total, rate):
    if total == 0 or rate is None:
        return "n/a"
    return f"{rate:.2f} ({hits} из {total})"


def print_summary(dataset, rows):
    stats = summarize_rows(rows)
    print(f"\n{dataset}: n={stats['n']}, strictness={rows[0]['strictness'] if rows else ''}")
    print(f"  до судьи: {_share(stats['pass_before_n'], stats['n'], stats['pass_before'])}")
    print(f"  после судьи: {_share(stats['pass_after_n'], stats['n'], stats['pass_after'])}")
    print(
        "  полоса top_score 0,05–0,15: "
        f"n={stats['band_n']}, "
        f"до {_share(stats['band_pass_before_n'], stats['band_n'], stats['band_pass_before'])}, "
        f"после {_share(stats['band_pass_after_n'], stats['band_n'], stats['band_pass_after'])}"
    )
    print(
        "  полоса и type=answerable: "
        f"n={stats['band_answerable_n']}, "
        f"до {_share(stats['band_answerable_before_n'], stats['band_answerable_n'], stats['band_answerable_before'])}, "
        f"после {_share(stats['band_answerable_after_n'], stats['band_answerable_n'], stats['band_answerable_after'])}"
    )
    print(
        "  судья отклонил показанные черновики: "
        f"{stats['judge_rejected_n']} из {stats['shown_n']}, "
        f"из них type=answerable: {stats['answerable_rejected_n']}"
    )
    print(f"  случай 9: {stats['case_9']}")
    if stats["mean_latency_answer_s"] is not None:
        print(f"  средняя задержка ответа, с: {stats['mean_latency_answer_s']:.3f}")
    if stats["mean_latency_s"] is not None:
        print(f"  средняя задержка с судьёй, с: {stats['mean_latency_s']:.3f}")
    if stats["mean_judge_call_s"] is not None:
        print(f"  средняя задержка только вызова судьи, с: {stats['mean_judge_call_s']:.3f}")
    return stats


def main(argv=None):
    paths, strictness, runs = parse_args(sys.argv[1:] if argv is None else argv)
    strictness_text(strictness)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            examples = json.load(f)
        dataset = Path(path).stem
        for run in range(1, runs + 1):
            rows = []
            label = dataset if runs == 1 else f"{dataset} прогон {run}/{runs}"
            for index, example in enumerate(examples, 1):
                row = run_question(example, dataset=dataset, index=index, strictness=strictness)
                rows.append(row)
                print(
                    f"{label} {index}/{len(examples)} "
                    f"before={row['passed_before_judge']} after={row['passed_after_judge']} "
                    f"judge={row['judge_decision']} score={row['top_score']}",
                    flush=True,
                )
            suffix = "" if runs == 1 else f"_run{run}"
            out = ROOT / "eval" / f"preshow_{dataset}{suffix}.csv"
            write_csv(out, rows)
            print_summary(dataset, rows)
            print(f"  csv: {out}")


if __name__ == "__main__":
    main()
