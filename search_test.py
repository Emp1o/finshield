from components import get_vectorstore 

  

question = "Мне позвонили из банка и просят назвать код из СМС. Что делать?" 

results = get_vectorstore().similarity_search_with_score(question, k=4) 

  

for doc, distance in results: 

    print(round(distance, 3), doc.metadata["source"], "p.", doc.metadata["page"] + 1) 

    print(doc.page_content[:300]) 

    print("-" * 60) 