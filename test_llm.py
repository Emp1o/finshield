import os 

from dotenv import load_dotenv 

from langchain_gigachat import GigaChat 

  

load_dotenv()

  

llm = GigaChat( 

    credentials=os.getenv("GIGACHAT_CREDENTIALS"), 

    scope="GIGACHAT_API_PERS",   # тариф для физических лиц 

    model="GigaChat-2",

    verify_ssl_certs=False,      # упрощение для учебного проекта 

) 

print(llm.invoke("Привет! Объясни в одном предложении, что такое RAG.").content) 