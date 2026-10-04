import os 

from dotenv import load_dotenv 

from langchain_gigachat import GigaChat 

  

load_dotenv()

  

llm = GigaChat( 

    credentials=os.getenv("GIGACHAT_CREDENTIALS"), 

    scope="GIGACHAT_API_PERS",   

    model="GigaChat-2",

    verify_ssl_certs=False,      

) 

print(llm.invoke("Привет! Что такое «безопасный счет» и почему это всегда обман?").content) 