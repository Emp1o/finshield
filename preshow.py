"""Проверка черновика ответа по найденным фрагментам перед показом.

Судья — GigaChat-2-Max с temperature=0 (`components.get_judge`). Вопрос судье
не передаётся: решается только опора на фрагменты, как в офлайн-оценке
faithfulness. Строгость сверки — отдельный абзац промпта. По умолчанию сверка
по смыслу; дословный режим включается тем же промптом со значением `verbatim`.
"""
import config

# Случай 9 в eval/error_analysis.md: ловушка обычного набора.
CASE_9_QUESTION = "Что делать, если у меня украли телефон с банковским приложением?"

# Полоса «едва прошли порог»: порог 0,05 включительно, верх 0,15 включительно.
NEAR_SCORE_LOW = 0.05
NEAR_SCORE_HIGH = 0.15

# Абзац строгость. Чтобы ужесточить проверку, поставьте PRESHOW_STRICTNESS=verbatim
# или замените текст STRICTNESS_SOFT.
STRICTNESS_SOFT = (
    "Сверяй по смыслу, не дословно. Перефраз и краткий пересказ засчитываются, "
    "если то же утверждение следует из фрагментов. "
    "Вердикт «нет», если в черновике есть факт, совет или вывод, которого во фрагментах нет даже по смыслу."
)

STRICTNESS_VERBATIM = (
    "Сверяй почти дословно. Каждое утверждение должно прямо читаться во фрагменте. "
    "Перефраз, обобщение и совет своими словами не засчитываются: для них вердикт «нет»."
)

STRICTNESS_RULES = {
    "soft": STRICTNESS_SOFT,
    "verbatim": STRICTNESS_VERBATIM,
}

PRESHOW_JUDGE_PROMPT = (
    "Тебе даны фрагменты документов и черновик ответа ассистента. "
    "Проверь, можно ли показать черновик пользователю: подтверждается ли каждое "
    "утверждение черновика этими фрагментами.\n"
    "Правило сверки: {strictness}\n"
    "Вопрос пользователя не дан. Полноту ответа не оценивай, только опору на фрагменты.\n"
    "Если черновик ничего не утверждает сверх фрагментов, это проход.\n"
    "Ответь ровно двумя строками и ничем больше.\n"
    "Первая строка начинается с «вердикт:» и дальше содержит только «да» (показать) "
    "или только «нет» (не показывать).\n"
    "Вторая строка начинается с «причина:» и содержит одно короткое предложение.\n\n"
    "Фрагменты:\n{context}\n\n"
    "Черновик:\n{answer}"
)


def strictness_text(name: str | None = None) -> tuple[str, str]:
    """Return (mode, rule paragraph). Unknown modes raise ValueError."""
    mode = (name if name is not None else config.PRESHOW_STRICTNESS).strip().lower() or "soft"
    if mode not in STRICTNESS_RULES:
        known = ", ".join(sorted(STRICTNESS_RULES))
        raise ValueError(f"PRESHOW_STRICTNESS must be one of: {known}")
    return mode, STRICTNESS_RULES[mode]


def build_preshow_prompt(context: str, answer: str, strictness: str | None = None) -> str:
    """Fill the pre-show prompt. `strictness` is `soft` or `verbatim`.

    Placeholders are substituted literally so braces inside a chunk are not format fields.
    """
    _, rule = strictness_text(strictness)
    return (
        PRESHOW_JUDGE_PROMPT
        .replace("{strictness}", rule)
        .replace("{context}", context)
        .replace("{answer}", answer)
    )


def _first_word(text: str) -> str:
    word = []
    for char in text.lower().replace("ё", "е"):
        if char.isalpha():
            word.append(char)
        elif word:
            break
    return "".join(word)


def parse_preshow_verdict(text: str) -> dict:
    """Parse a two-line verdict. Unrecognised text fails closed (do not show)."""
    raw = (text or "").strip()
    verdict_text = ""
    reason = ""
    for line in raw.splitlines():
        stripped = line.strip().strip("*").strip()
        lower = stripped.lower()
        if lower.startswith("вердикт"):
            verdict_text = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
        elif lower.startswith("причина"):
            reason = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
    if not verdict_text:
        verdict_text = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    word = _first_word(verdict_text)
    lowered = verdict_text.lower().replace("ё", "е")
    # «да или нет» — эхо инструкции, а не решение.
    if word == "да" and "нет" not in lowered:
        passed = True
    else:
        passed = False
        if word != "нет" and not reason:
            reason = "вердикт не распознан"
    return {
        "pass": passed,
        "decision": "pass" if passed else "fail",
        "reason": reason,
        "raw": raw,
    }


def check_draft(context: str, answer: str, *, judge=None, strictness: str | None = None) -> dict:
    """Ask the judge whether a draft may be shown. `judge` defaults to get_judge()."""
    mode, _ = strictness_text(strictness)
    prompt = build_preshow_prompt(context, answer, mode)
    if judge is None:
        from components import get_judge
        judge = get_judge()
    raw = judge.invoke(prompt).content
    parsed = parse_preshow_verdict(raw)
    parsed["strictness"] = mode
    return parsed


def apply_preshow(text: str, cited: list, context: str, *, enabled: bool, checker, refusal_text: str):
    """Gate a draft. Returns text, cited, judge_pass, judge_reason, judge_decision.

    `checker(context, draft)` is called only when the draft would be shown:
    the flag is on, there is at least one citation, and the text is not already
    the refusal. A failed check replaces the draft with `refusal_text`.
    """
    if not enabled or not cited or text == refusal_text:
        return text, cited, None, "", "skipped"
    verdict = checker(context, text)
    if verdict["pass"]:
        return text, cited, True, verdict.get("reason", ""), "pass"
    return refusal_text, [], False, verdict.get("reason", ""), "fail"


def is_case_9(question: str) -> bool:
    return question.strip() == CASE_9_QUESTION


def example_is_trap(example: dict) -> bool:
    """Close traps (`no_answer`), off-topic items, and hard-set `category=trap`."""
    if example.get("category") == "trap":
        return True
    return example.get("type") in {"no_answer", "off_topic"}


def in_near_band(top_score, low: float = NEAR_SCORE_LOW, high: float = NEAR_SCORE_HIGH) -> bool:
    if top_score is None or top_score == "":
        return False
    score = float(top_score)
    return low <= score <= high


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "да"}
    return bool(value)


def summarize_rows(rows: list[dict]) -> dict:
    """Pass rates before/after the judge, the near-threshold band, case 9, latency."""

    def count(items, key):
        return sum(1 for row in items if _as_bool(row.get(key)))

    band = [row for row in rows if _as_bool(row.get("near_threshold"))]
    case_rows = [row for row in rows if _as_bool(row.get("is_case_9"))]
    shown = [row for row in rows if _as_bool(row.get("passed_before_judge"))]
    rejected = [row for row in shown if not _as_bool(row.get("passed_after_judge"))]
    answerable_rejected = [row for row in rejected if row.get("type") == "answerable"]
    latencies = []
    answer_latencies = []
    judge_calls = []
    for row in rows:
        if row.get("latency_s") not in (None, ""):
            latencies.append(float(row["latency_s"]))
        if row.get("latency_answer_s") not in (None, ""):
            answer_latencies.append(float(row["latency_answer_s"]))
        called = row.get("judge_decision") not in (None, "", "skipped")
        if called and row.get("latency_judge_s") not in (None, ""):
            judge_calls.append(float(row["latency_judge_s"]))

    def share(hits, total):
        if not total:
            return None
        return hits / total

    case = case_rows[0] if case_rows else None
    if case is None:
        case_label = "нет в наборе"
    elif not _as_bool(case.get("passed_before_judge")):
        case_label = "до судьи не дошёл"
    else:
        case_label = case.get("judge_decision") or ""

    return {
        "n": len(rows),
        "pass_before": share(count(rows, "passed_before_judge"), len(rows)),
        "pass_after": share(count(rows, "passed_after_judge"), len(rows)),
        "pass_before_n": count(rows, "passed_before_judge"),
        "pass_after_n": count(rows, "passed_after_judge"),
        "band_n": len(band),
        "band_pass_before": share(count(band, "passed_before_judge"), len(band)),
        "band_pass_after": share(count(band, "passed_after_judge"), len(band)),
        "band_pass_before_n": count(band, "passed_before_judge"),
        "band_pass_after_n": count(band, "passed_after_judge"),
        "shown_n": len(shown),
        "judge_rejected_n": len(rejected),
        "answerable_rejected_n": len(answerable_rejected),
        "case_9": case_label,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "mean_latency_answer_s": (
            sum(answer_latencies) / len(answer_latencies) if answer_latencies else None
        ),
        "mean_judge_call_s": sum(judge_calls) / len(judge_calls) if judge_calls else None,
    }
