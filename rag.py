import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from components import get_llm, get_reranker, get_vectorstore
from preshow import apply_preshow, check_draft

NO_ANSWER = (
    "В документах нет ответа на этот вопрос. "
    "Если вы столкнулись с мошенничеством прямо сейчас, прекратите разговор "
    "и позвоните в свой банк по номеру с обратной стороны карты."
)

NO_ANSWER_TOKEN = "НЕТ_ОТВЕТА"

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — помощник по финансовой безопасности. Отвечай только на основе фрагментов "
     "документов из контекста, даже если знаешь ответ сам. После каждого утверждения "
     "указывай номер фрагмента в квадратных скобках, например [2]. Если в контексте нет "
     f"ответа, ответь: «{NO_ANSWER}». Если человек описывает мошенничество, которое "
     "происходит прямо сейчас, первым делом посоветуй прекратить разговор и позвонить "
     "в свой банк по номеру с обратной стороны карты. Не помогай совершать мошенничество: "
     "на такие просьбы вежливо откажи. Отвечай кратко и по делу."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — помощник по финансовой безопасности. Отвечай только на основе фрагментов "
     "документов из контекста, даже если знаешь ответ сам.\n"
     "Правила:\n"
     "1. Каждое предложение ответа должно опираться на фрагмент и заканчиваться ссылкой "
     "на него в квадратных скобках, например [2]. Если утверждение нельзя подтвердить "
     "фрагментом, не пиши его.\n"
     "2. Если во фрагментах есть только часть ответа, прямо скажи, чего в документах нет, "
     "и приведи то, что есть, со ссылками.\n"
     f"3. Если во фрагментах нет ничего по вопросу, ответь одним словом: {NO_ANSWER_TOKEN}\n"
     "4. Если человек описывает мошенничество, которое происходит прямо сейчас, начни "
     "с совета из фрагментов, как прекратить контакт с мошенниками.\n"
     "5. Не помогай совершать мошенничество: на такие просьбы вежливо откажи.\n"
     "Отвечай кратко и по делу."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
])

PROMPT_V3 = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — помощник по финансовой безопасности. Отвечай только на основе фрагментов "
     "документов из контекста, даже если знаешь ответ сам.\n"
     "Правила:\n"
     "1. Каждое предложение ответа должно опираться на фрагмент и заканчиваться ссылкой "
     "на него в квадратных скобках, например [2]. Если утверждение нельзя подтвердить "
     "фрагментом, не пиши его.\n"
     "2. Если во фрагментах есть только часть ответа, прямо скажи, чего в документах нет, "
     "и приведи то, что есть, со ссылками.\n"
     f"3. Если во фрагментах нет ничего по вопросу, ответь одним словом: {NO_ANSWER_TOKEN}\n"
     "4. Если человек описывает мошенничество, которое происходит прямо сейчас, начни "
     "с совета из фрагментов, как прекратить контакт с мошенниками.\n"
     "5. Не помогай совершать мошенничество: на такие просьбы вежливо откажи.\n"
     "6. Отвечай в 2–4 предложениях: объясни, в чём суть ситуации или схемы, и дай "
     "конкретные советы, что делать. Не ограничивайся ответом «да» или «нет»."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3}


def retrieve_scored(question: str, k_candidates: int = 20, k_final: int = 4):
    """Return the top chunks with their reranker scores, best first."""
    candidates = get_vectorstore().similarity_search(question, k=k_candidates)
    scores = get_reranker().predict([(question, d.page_content) for d in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:k_final]]


def retrieve(question: str, use_reranker: bool = True, k_candidates: int = 20, k_final: int = 4):
    """Return the most relevant chunks for a question."""
    if not use_reranker:
        return get_vectorstore().similarity_search(question, k=k_final)
    return [doc for doc, _ in retrieve_scored(question, k_candidates, k_final)]


def source_name(doc) -> str:
    return f"{os.path.basename(doc.metadata['source'])}, стр. {doc.metadata.get('page', 0) + 1}"


def format_docs(docs) -> str:
    return "\n\n".join(f"[{i}] ({source_name(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))


def cited_docs(text: str, docs):
    """Return only the chunks referenced as [n] in the answer."""
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return [d for i, d in enumerate(docs, 1) if i in numbers]

def is_refusal(result: dict) -> bool:
    """A refusal states that documents lack the answer and cites no fragments."""
    return "в документах нет" in result["answer"].lower() and not result["cited"]


def _pack(text, sources, cited, top_score, filtered, judge_pass=None,
          judge_reason="", judge_decision="skipped") -> dict:
    return {
        "answer": text,
        "sources": sources,
        "cited": cited,
        "top_score": top_score,
        "filtered": filtered,
        "judge_pass": judge_pass,
        "judge_reason": judge_reason,
        "judge_decision": judge_decision,
    }


def answer(question: str, use_reranker: bool = True, preshow_judge: bool | None = None) -> dict:
    """Answer a question; 'sources' are retrieved chunks, 'cited' are chunks used in the answer.

    'filtered' is True when an answer without citations was replaced by NO_ANSWER.
    'preshow_judge' overrides PRESHOW_JUDGE. When the judge rejects a draft, the
    shown answer becomes NO_ANSWER and 'judge_pass' is False. Evaluations that
    measure the unjudged draft pass preshow_judge=False.
    """
    use_judge = config.PRESHOW_JUDGE if preshow_judge is None else preshow_judge
    if use_reranker:
        ranked = retrieve_scored(question)
        top_score = ranked[0][1] if ranked else None
        if top_score is None or top_score < config.RELEVANCE_THRESHOLD:
            return _pack(NO_ANSWER, [], [], top_score, False)
        docs = [doc for doc, _ in ranked]
    else:
        docs = retrieve(question, use_reranker=False)
        top_score = None

    chain = PROMPTS[config.PROMPT_VERSION] | get_llm() | StrOutputParser()
    text = chain.invoke({"context": format_docs(docs), "question": question}).strip()
    cited = cited_docs(text, docs)

    if NO_ANSWER_TOKEN in text and not cited:
        text = NO_ANSWER

    filtered = False
    if config.REQUIRE_CITATIONS and not cited and text != NO_ANSWER:
        text, filtered = NO_ANSWER, True

    text, cited, judge_pass, judge_reason, judge_decision = apply_preshow(
        text,
        cited,
        format_docs(docs),
        enabled=use_judge,
        checker=lambda context, draft: check_draft(context, draft),
        refusal_text=NO_ANSWER,
    )
    return _pack(text, docs, cited, top_score, filtered,
                 judge_pass, judge_reason, judge_decision)


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (empty to exit): ").strip()
        if not q:
            break
        result = answer(q)
        print("\n" + result["answer"])
        if result["judge_decision"] == "fail":
            print(f"\n(судья не пропустил черновик: {result['judge_reason']})")
        if result["cited"]:
            print("\nSources:")
            for d in result["cited"]:
                print(f"  {source_name(d)}")
        if result["top_score"] is not None:
            print(f"\n(top reranker score: {result['top_score']:.3f})")