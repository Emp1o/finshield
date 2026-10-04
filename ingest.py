import os 

import shutil 

  

from langchain_chroma import Chroma 

from langchain_community.document_loaders import PyPDFDirectoryLoader 

from langchain_text_splitters import RecursiveCharacterTextSplitter 

  

import config 

from components import get_embeddings 

  

  

def main(): 

    docs = PyPDFDirectoryLoader(config.DATA_DIR).load() 

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