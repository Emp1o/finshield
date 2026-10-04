from components import get_vectorstore 

  

question = "Мне позвонили и говорят, что проводятся подозрительные операции на моем счету с цифровым рублем. Что делать?" 

results = get_vectorstore().similarity_search_with_score(question, k=4) 

  

for doc, score in results: 

    # в Chroma score — это расстояние: чем МЕНЬШЕ, тем ближе 

    print(round(score, 3), doc.metadata["source"], "стр.", doc.metadata["page"] + 1) 

    print(doc.page_content[:300]) 

    print("-" * 60) 