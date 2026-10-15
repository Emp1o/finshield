import os
import re
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from components import get_embeddings

BOILERPLATE_PATTERNS = [
    r"^107016, Москва.*$",
    r"^8 800 300-30-00$",
    r"^www\.cbr\.ru$",
    r"^Информационная безопасность Противодействие мошенническим практикам.*$",
    r"^Последнее обновление страницы:.*$",
    r"^Признаки мошенничества Что предпринять\?$",
    r"^Ответственное структурное подразделение:.*$",
    r"^\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2} .*$",
    r"^https?://\S+.*$",
]
URL_PATTERN = re.compile(r"https?://\S+")


def clean_document(doc):
    """Remove repeated headers and footers; store the page URL in metadata."""
    url = URL_PATTERN.search(doc.page_content)
    if url:
        doc.metadata["url"] = url.group(0)
    lines = [
        line for line in doc.page_content.splitlines()
        if not any(re.match(p, line.strip()) for p in BOILERPLATE_PATTERNS)
    ]
    doc.page_content = "\n".join(lines).strip()
    return doc


def main():
    docs = [clean_document(d) for d in PyPDFDirectoryLoader(config.DATA_DIR).load()]
    print(f"Pages loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks[:2]:
        print("-" * 60)
        print(chunk.metadata)
        print(chunk.page_content[:500])

    if os.path.exists(config.DB_DIR):
        shutil.rmtree(config.DB_DIR)
    Chroma.from_documents(chunks, get_embeddings(), persist_directory=config.DB_DIR)
    print(f"Index saved to {config.DB_DIR}")


if __name__ == "__main__":
    main()