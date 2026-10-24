import os 

  

from dotenv import load_dotenv 

  

load_dotenv() 

  

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS") 

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") 

GIGACHAT_MODEL = "GigaChat-2" 

  

# LLM provider: "gigachat" or "local" (LM Studio) 

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat") 

LOCAL_BASE_URL = "http://localhost:1234/v1" 

LOCAL_MODEL = "qwen2.5-7b-instruct" 

  

DATA_DIR = "data" 

DB_DIR = "chroma_db" 

  

EMBEDDING_MODEL = "BAAI/bge-m3" 

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" 

  

CHUNK_SIZE = 800 

CHUNK_OVERLAP = 120 

RELEVANCE_THRESHOLD = 0.05

JUDGE_MODEL = "GigaChat-2-Max"

PROMPT_VERSION = "v3"
REQUIRE_CITATIONS = True

# Судья перед показом. Пустое значение оставляет текущий путь бота.
PRESHOW_JUDGE = os.getenv("PRESHOW_JUDGE", "").strip().lower() in {"1", "true", "yes", "on"}
# soft — тот же смысл, что во фрагментах. verbatim — почти дословно.
PRESHOW_STRICTNESS = os.getenv("PRESHOW_STRICTNESS", "soft").strip().lower() or "soft"