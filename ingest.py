import os 

import shutil 

  

from langchain_community.document_loaders import PyPDFDirectoryLoader 

from langchain_chroma import Chroma 

from langchain_text_splitters import RecursiveCharacterTextSplitter 

  

import config 

from components import get_embeddings 

  

  

def main(): 

    # 1. Загрузка: каждая страница PDF становится объектом Document 

    #    с текстом (page_content) и метаданными (source — путь к файлу, page — номер страницы с нуля) 

    docs = PyPDFDirectoryLoader(config.DATA_DIR).load() 

    print(f"Загружено страниц: {len(docs)}") 

  

    # 2. Чанкинг: режем сначала по абзацам, если не влезает — по строкам, предложениям, словам 

    splitter = RecursiveCharacterTextSplitter( 

        chunk_size=config.CHUNK_SIZE, 

        chunk_overlap=config.CHUNK_OVERLAP, 

        separators=["\n\n", "\n", ". ", " ", ""], 

    ) 

    chunks = splitter.split_documents(docs) 

    print(f"Получилось чанков: {len(chunks)}") 

  

    # Посмотри глазами на пару чанков — это самая важная проверка качества данных 

    for chunk in chunks[:2]: 

        print("-" * 60) 

        print(chunk.metadata) 

        print(chunk.page_content[:500]) 

  

    # 3–4. Эмбеддинги и сохранение. Старую базу удаляем, чтобы не было дублей при повторном запуске 

    if os.path.exists(config.DB_DIR): 

        shutil.rmtree(config.DB_DIR) 

    Chroma.from_documents(chunks, get_embeddings(), persist_directory=config.DB_DIR) 

    print("Индекс сохранён в", config.DB_DIR) 

  

  

if __name__ == "__main__": 

    main() 