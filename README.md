# Fine tune retriever
```
python -m evaluation.finetune_colbert
```

# Generate vector embeddings
Modify src/data_processing/colbert_data_processing.py to use correct colbert model and index name (lines 155,156)
```
python -m src.data_processing.main
```
# Query RAG system
Updated the question variable in src/agent/main.py
```
python -m src.agent.main
```

# Run Streamlit App
```
streamlit run app.py
```
