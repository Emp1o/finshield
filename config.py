import os 

from dotenv import load_dotenv 

  

load_dotenv() 

  

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS") 

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") 

GIGACHAT_MODEL = "GigaChat-2"   # модель GigaChat; список доступных — через get_models() 

  

DATA_DIR = "data" 

DB_DIR = "chroma_db" 

  

EMBEDDING_MODEL = "BAAI/bge-m3"            # мультиязычная модель эмбеддингов, хороша для русского 

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # мультиязычный cross-encoder 

  

CHUNK_SIZE = 800      # символов в чанке 

CHUNK_OVERLAP = 120   # перекрытие соседних чанков 