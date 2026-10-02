import os
from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv()

with GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
) as giga:
    for model in giga.get_models().data:
        print(model)