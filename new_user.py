from nano_graphrag import GraphRAG, QueryParam
from using_ollama_as_llm_and_embedding import ollama_model_if_cache

graph_func = GraphRAG(
    working_dir="./dickens",
    best_model_func=ollama_model_if_cache,
    cheap_model_func=ollama_model_if_cache,
)

with open("./book.txt") as f:
    graph_func.insert(f.read())

# Perform global graphrag search
print(graph_func.query("What are the top themes in this story?"))

# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))