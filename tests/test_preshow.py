"""Проверка разбора вердикта и фильтра перед показом без вызова API."""
import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

import config
import preshow
import rag

ROOT = Path(__file__).resolve().parents[1]


def _experiment():
    path = ROOT / "eval" / "preshow_experiment.py"
    spec = importlib.util.spec_from_file_location("preshow_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Doc:
    def __init__(self, content, source="doc.pdf"):
        self.page_content = content
        self.metadata = {"source": source, "page": 0}


class _Judge:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self


def _draft_llm(text):
    return RunnableLambda(lambda _: text)


def test_soft_prompt_allows_paraphrase_and_verbatim_does_not():
    soft = preshow.build_preshow_prompt("фрагмент про код", "не сообщайте код", "soft")
    verbatim = preshow.build_preshow_prompt("фрагмент про код", "не сообщайте код", "verbatim")
    assert "Перефраз и краткий пересказ засчитываются" in soft
    assert "Перефраз, обобщение и совет своими словами не засчитываются" in verbatim
    assert "фрагмент про код" in soft
    assert "не сообщайте код" in verbatim
    assert "только «да»" in soft
    assert "только «нет»" in soft
    assert "причина:" in soft


def test_unknown_strictness_is_rejected():
    with pytest.raises(ValueError):
        preshow.build_preshow_prompt("контекст", "ответ", "strict")


def test_parse_two_line_verdicts():
    passed = preshow.parse_preshow_verdict("вердикт: да\nпричина: пересказ фрагмента [1]")
    assert passed["pass"] is True
    assert passed["decision"] == "pass"
    assert passed["reason"] == "пересказ фрагмента [1]"

    failed = preshow.parse_preshow_verdict(
        "вердикт: нет\nпричина: совета заблокировать приложение во фрагментах нет"
    )
    assert failed["pass"] is False
    assert failed["decision"] == "fail"
    assert "заблокировать" in failed["reason"]


def test_parse_bare_yes_no_like_the_offline_judge():
    assert preshow.parse_preshow_verdict("да")["pass"] is True
    assert preshow.parse_preshow_verdict("Да.")["pass"] is True
    assert preshow.parse_preshow_verdict("нет")["pass"] is False
    assert preshow.parse_preshow_verdict("Нет, во фрагментах этого нет")["pass"] is False


def test_unrecognised_verdict_fails_closed():
    parsed = preshow.parse_preshow_verdict("не могу решить")
    assert parsed["pass"] is False
    assert parsed["reason"] == "вердикт не распознан"
    echoed = preshow.parse_preshow_verdict("вердикт: да или нет\nпричина: не выбрал")
    assert echoed["pass"] is False


def test_check_draft_uses_injected_judge_and_soft_rule():
    judge = _Judge("вердикт: нет\nпричина: совета нет в фрагментах")
    verdict = preshow.check_draft("текст фрагмента", "заблокируйте телефон", judge=judge, strictness="soft")
    assert verdict["pass"] is False
    assert verdict["strictness"] == "soft"
    assert judge.prompts
    assert "текст фрагмента" in judge.prompts[0]
    assert "заблокируйте телефон" in judge.prompts[0]
    assert preshow.STRICTNESS_SOFT in judge.prompts[0]


def test_apply_preshow_skips_when_disabled_or_already_refused():
    def boom(context, draft):
        raise AssertionError("judge must not run")

    text, cited, judge_pass, reason, decision = preshow.apply_preshow(
        "ответ [1]", ["chunk"], "контекст", enabled=False, checker=boom, refusal_text=rag.NO_ANSWER,
    )
    assert text == "ответ [1]"
    assert cited == ["chunk"]
    assert judge_pass is None
    assert decision == "skipped"
    assert reason == ""

    text, cited, judge_pass, _, decision = preshow.apply_preshow(
        rag.NO_ANSWER, [], "контекст", enabled=True, checker=boom, refusal_text=rag.NO_ANSWER,
    )
    assert text == rag.NO_ANSWER
    assert decision == "skipped"
    assert judge_pass is None


def test_apply_preshow_hides_failed_draft_and_keeps_passed_one():
    def fail(context, draft):
        return {"pass": False, "reason": "нет опоры", "decision": "fail"}

    text, cited, judge_pass, reason, decision = preshow.apply_preshow(
        "заблокируйте приложение [1]", ["chunk"], "контекст",
        enabled=True, checker=fail, refusal_text=rag.NO_ANSWER,
    )
    assert text == rag.NO_ANSWER
    assert cited == []
    assert judge_pass is False
    assert decision == "fail"
    assert reason == "нет опоры"

    def ok(context, draft):
        return {"pass": True, "reason": "тот же смысл", "decision": "pass"}

    text, cited, judge_pass, reason, decision = preshow.apply_preshow(
        "не сообщайте код [1]", ["chunk"], "контекст",
        enabled=True, checker=ok, refusal_text=rag.NO_ANSWER,
    )
    assert text == "не сообщайте код [1]"
    assert cited == ["chunk"]
    assert judge_pass is True
    assert decision == "pass"


def test_case_9_band_and_trap_flags():
    assert preshow.is_case_9(preshow.CASE_9_QUESTION)
    assert not preshow.is_case_9("Как вернуть деньги, которые я уже перевёл мошенникам?")
    assert preshow.in_near_band(0.05)
    assert preshow.in_near_band(0.128)
    assert preshow.in_near_band(0.15)
    assert not preshow.in_near_band(0.049)
    assert not preshow.in_near_band(0.151)
    assert not preshow.in_near_band(None)
    assert preshow.example_is_trap({"type": "no_answer", "category": "trap"})
    assert preshow.example_is_trap({"type": "off_topic"})
    assert not preshow.example_is_trap({"type": "answerable", "category": "paraphrase"})


def test_summarize_before_after_band_and_case_9():
    rows = [
        {
            "type": "answerable",
            "near_threshold": False,
            "passed_before_judge": True,
            "passed_after_judge": True,
            "is_case_9": False,
            "latency_s": 1.0,
            "latency_answer_s": 0.8,
        },
        {
            "type": "answerable",
            "near_threshold": True,
            "passed_before_judge": True,
            "passed_after_judge": False,
            "is_case_9": False,
            "latency_s": 2.0,
            "latency_answer_s": 1.0,
        },
        {
            "type": "no_answer",
            "near_threshold": True,
            "passed_before_judge": True,
            "passed_after_judge": False,
            "is_case_9": True,
            "judge_decision": "fail",
            "latency_s": 3.0,
            "latency_answer_s": 1.2,
        },
        {
            "type": "off_topic",
            "near_threshold": False,
            "passed_before_judge": False,
            "passed_after_judge": False,
            "is_case_9": False,
            "judge_decision": "skipped",
            "latency_s": 0.2,
            "latency_answer_s": 0.2,
        },
    ]
    stats = preshow.summarize_rows(rows)
    assert stats["n"] == 4
    assert stats["pass_before_n"] == 3
    assert stats["pass_after_n"] == 1
    assert stats["band_n"] == 2
    assert stats["band_pass_before_n"] == 2
    assert stats["band_pass_after_n"] == 0
    assert stats["band_answerable_n"] == 1
    assert stats["band_answerable_before_n"] == 1
    assert stats["band_answerable_after_n"] == 0
    assert stats["answerable_rejected_n"] == 1
    assert stats["judge_rejected_n"] == 2
    assert stats["case_9"] == "fail"
    assert stats["mean_latency_s"] == pytest.approx(1.55)
    assert stats["mean_latency_answer_s"] == pytest.approx(0.8)


def test_summarize_case_9_absent_or_not_reached():
    assert preshow.summarize_rows([])["case_9"] == "нет в наборе"
    skipped = [{
        "passed_before_judge": False,
        "passed_after_judge": False,
        "is_case_9": True,
        "judge_decision": "skipped",
    }]
    assert preshow.summarize_rows(skipped)["case_9"] == "до судьи не дошёл"


def _patch_retrieval(monkeypatch, score=0.4, content="Не сообщайте код из СМС."):
    doc = _Doc(content)
    monkeypatch.setattr(rag, "retrieve_scored", lambda *args, **kwargs: [(doc, score)])
    return doc


def test_answer_does_not_call_judge_when_flag_off(monkeypatch):
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr(rag, "get_llm", lambda: _draft_llm("Не сообщайте код из СМС [1]"))

    def boom(*args, **kwargs):
        raise AssertionError("judge must stay off")

    monkeypatch.setattr(rag, "check_draft", boom)
    result = rag.answer("вопрос", preshow_judge=False)
    assert result["judge_decision"] == "skipped"
    assert result["judge_pass"] is None
    assert "Не сообщайте код" in result["answer"]
    assert result["cited"]
    assert not rag.is_refusal(result)


def test_answer_uses_config_flag(monkeypatch):
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr(rag, "get_llm", lambda: _draft_llm("Не сообщайте код [1]"))
    monkeypatch.setattr(config, "PRESHOW_JUDGE", False)

    def boom(*args, **kwargs):
        raise AssertionError("judge must stay off")

    monkeypatch.setattr(rag, "check_draft", boom)
    assert rag.answer("вопрос")["judge_decision"] == "skipped"

    monkeypatch.setattr(config, "PRESHOW_JUDGE", True)
    monkeypatch.setattr(
        rag, "check_draft",
        lambda *args, **kwargs: {"pass": True, "reason": "смысл тот же", "decision": "pass"},
    )
    result = rag.answer("вопрос")
    assert result["judge_pass"] is True
    assert "Не сообщайте код" in result["answer"]


def test_answer_replaces_draft_when_judge_fails(monkeypatch):
    _patch_retrieval(monkeypatch, score=0.128)
    monkeypatch.setattr(rag, "get_llm", lambda: _draft_llm("Заблокируйте банковское приложение [1]"))
    monkeypatch.setattr(
        rag, "check_draft",
        lambda context, draft, **kwargs: {
            "pass": False,
            "reason": "совета нет во фрагментах",
            "decision": "fail",
        },
    )
    result = rag.answer("Что делать, если украли телефон?", preshow_judge=True)
    assert result["answer"] == rag.NO_ANSWER
    assert result["cited"] == []
    assert result["judge_pass"] is False
    assert result["judge_decision"] == "fail"
    assert result["top_score"] == 0.128
    assert rag.is_refusal(result)
    assert result["filtered"] is False


def test_case_9_question_is_only_in_the_main_set():
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    hits = [item for item in questions if preshow.is_case_9(item["question"])]
    assert len(hits) == 1
    assert hits[0]["type"] == "no_answer"
    assert hits[0]["source"] is None
    hard = json.loads((ROOT / "eval" / "questions_hard.json").read_text(encoding="utf-8"))
    assert not any(preshow.is_case_9(item["question"]) for item in hard)


def test_experiment_row_records_case_9_reject_and_skips_refusals(monkeypatch):
    experiment = _experiment()
    calls = {}

    def fake_answer(question, preshow_judge=None):
        calls["preshow_judge"] = preshow_judge
        if preshow.is_case_9(question):
            return {
                "answer": "Заблокируйте банковское приложение [1]",
                "sources": [],
                "cited": ["chunk"],
                "top_score": 0.128,
            }
        return {"answer": rag.NO_ANSWER, "sources": [], "cited": [], "top_score": 0.01}

    def fake_judge(context, answer, strictness=None):
        calls["judged"] = answer
        return {
            "pass": False,
            "decision": "fail",
            "reason": "совета нет во фрагментах",
            "strictness": strictness or "soft",
        }

    monkeypatch.setattr(experiment, "answer", fake_answer)
    monkeypatch.setattr(experiment, "format_docs", lambda docs: "фрагменты")
    monkeypatch.setattr(experiment, "check_draft", fake_judge)

    trap = {"question": preshow.CASE_9_QUESTION, "type": "no_answer", "source": None}
    row = experiment.run_question(trap, dataset="questions", index=27, strictness="soft")
    assert calls["preshow_judge"] is False
    assert row["id"] == 27
    assert row["passed_before_judge"] is True
    assert row["passed_after_judge"] is False
    assert row["is_case_9"] is True
    assert row["is_trap"] is True
    assert row["near_threshold"] is True
    assert row["judge_decision"] == "fail"
    assert row["top_score"] == 0.128
    assert row["strictness"] == "soft"
    assert "Заблокируйте" in calls["judged"]

    def refuse_judge(*args, **kwargs):
        raise AssertionError("judge must not run on a refusal")

    monkeypatch.setattr(experiment, "check_draft", refuse_judge)
    off = {"question": "Как сварить борщ?", "type": "off_topic", "source": None}
    skipped = experiment.run_question(off, dataset="questions", index=40, strictness="soft")
    assert skipped["passed_before_judge"] is False
    assert skipped["passed_after_judge"] is False
    assert skipped["judge_decision"] == "skipped"
    assert skipped["near_threshold"] is False
    assert skipped["is_case_9"] is False


def test_experiment_parse_args():
    experiment = _experiment()
    paths, strictness, runs = experiment.parse_args([])
    assert paths == experiment.DEFAULT_PATHS
    assert strictness is None
    assert runs == 1
    paths, strictness, runs = experiment.parse_args(
        ["--strictness", "verbatim", "--runs", "2", "eval/questions.json"]
    )
    assert paths == ["eval/questions.json"]
    assert strictness == "verbatim"
    assert runs == 2
    with pytest.raises(SystemExit):
        experiment.parse_args(["--strictness"])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--runs", "0"])


def test_answer_skips_judge_below_threshold_and_without_citations(monkeypatch):
    _patch_retrieval(monkeypatch, score=0.01)

    def boom(*args, **kwargs):
        raise AssertionError("must not generate or judge")

    monkeypatch.setattr(rag, "get_llm", boom)
    monkeypatch.setattr(rag, "check_draft", boom)
    result = rag.answer("борщ", preshow_judge=True)
    assert rag.is_refusal(result)
    assert result["judge_decision"] == "skipped"

    _patch_retrieval(monkeypatch, score=0.4)
    monkeypatch.setattr(rag, "get_llm", lambda: _draft_llm("просто текст без ссылки"))
    monkeypatch.setattr(rag, "check_draft", boom)
    result = rag.answer("вопрос", preshow_judge=True)
    assert result["filtered"] is True
    assert result["judge_decision"] == "skipped"
    assert rag.is_refusal(result)
