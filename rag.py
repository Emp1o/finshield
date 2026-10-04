import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from components import get_llm, get_reranker, get_vectorstore

NO_ANSWER = "В документах нет ответа на этот вопрос."

PROMPT = ChatPromptTemplate.from_messages([
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


def answer(question: str, use_reranker: bool = True) -> dict:
    """Answer a question; 'sources' are retrieved chunks, 'cited' are chunks used in the answer."""
    if use_reranker:
        ranked = retrieve_scored(question)
        top_score = ranked[0][1] if ranked else None
        if top_score is None or top_score < config.RELEVANCE_THRESHOLD:
            return {"answer": NO_ANSWER, "sources": [], "cited": [], "top_score": top_score}
        docs = [doc for doc, _ in ranked]
    else:
        docs = retrieve(question, use_reranker=False)
        top_score = None

    chain = PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"context": format_docs(docs), "question": question})
    return {"answer": text, "sources": docs, "cited": cited_docs(text, docs), "top_score": top_score}


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (empty to exit): ").strip()
        if not q:
            break
        result = answer(q)
        print("\n" + result["answer"])
        if result["cited"]:
            print("\nSources:")
            for d in result["cited"]:
                print(f"  {source_name(d)}")
        if result["top_score"] is not None:
            print(f"\n(top reranker score: {result['top_score']:.3f})")