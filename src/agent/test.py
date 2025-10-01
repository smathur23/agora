from src.agent.retriever import load_index, search

def retriever_test():
    question = "What AI regulation/funding is present in the One Big Beautiful Bill Act?"
    index = load_index("embeddings_output")
    results = search(question, index)
    print(results)

if __name__ == "__main__":
    retriever_test()