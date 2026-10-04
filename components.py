from functools import lru_cache 

  

from langchain_chroma import Chroma 

from langchain_gigachat import GigaChat 

from langchain_huggingface import HuggingFaceEmbeddings 

from langchain_openai import ChatOpenAI 

from sentence_transformers import CrossEncoder 

  

import config 

  

  

@lru_cache 

def get_embeddings(): 

    return HuggingFaceEmbeddings( 

        model_name=config.EMBEDDING_MODEL, 

        encode_kwargs={"normalize_embeddings": True}, 

    ) 

  

  

@lru_cache 

def get_vectorstore(): 

    return Chroma(persist_directory=config.DB_DIR, embedding_function=get_embeddings()) 

  

  

@lru_cache 

def get_reranker(): 

    return CrossEncoder(config.RERANKER_MODEL) 

  

  

@lru_cache 

def get_llm(): 

    if config.LLM_PROVIDER == "local": 

        return ChatOpenAI( 

            base_url=config.LOCAL_BASE_URL, 

            api_key="lm-studio", 

            model=config.LOCAL_MODEL, 

            temperature=0, 

        ) 

    return GigaChat( 

        credentials=config.GIGACHAT_CREDENTIALS, 

        scope="GIGACHAT_API_PERS", 

        model=config.GIGACHAT_MODEL, 

        verify_ssl_certs=False, 

        timeout=60, 

    ) 