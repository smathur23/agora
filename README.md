# Retriever Finetuning & RAG Pipeline

This repository provides tools for:

- Finetuning a **ColBERT retriever**
- Generating **vector embeddings**
- Running a **RAG system**
- Evaluating retriever performance
- Running a **Streamlit interface**

---

# 1. Finetune the Retriever

Finetuning the retriever is a multi-step process.

## Step 1: Configure LLM Connection

Modify the LLM connection in:

```
retriever_finetuning/llm.py
```

This file must be configured before any generation steps.

---

## Step 2: Generate Synthetic Query Prompts

Generate prompts used for synthetic query generation.

```bash
python -m retriever_finetuning.generate_prompts
```

The script contains parameters that control the **mix of prompt types** generated.

---

## Step 3: Generate Questions Using an LLM

```bash
python -m retriever_finetuning.generate_questions
```

Ensure the LLM connection in `llm.py` is configured before running this step.

---

## Step 4: Post-process Generated Questions (Recommended)

Clean up the generated question text:

```bash
python -m retriever_finetuning.post_process_question
```

---

## Step 5: Manually Label Questions

You must manually label the generated questions with **positive and negative examples**.

```bash
python -m retriever_finetuning.manual_triple_generation
```

This process is interactive and allows labeling **as many questions as desired**.

### Notes

The following files were part of a previous **automatic labeling approach**:

- `negative_examples.py`
- `tag_relations.csv`

This method produced **poor results**, so **manual labeling is required**.

---

## Step 6: Format Training Triples

After labeling, convert the data into training triples:

```bash
python -m retriever_finetuning.create_triples
```

---

## Step 7: Finetune ColBERT

Run the finetuning process:

```bash
python -m retriever_finetuning.finetune_colbert
```

Multiple finetuning configurations exist in the script, but **they cannot be run simultaneously**.

---

## Step 8: Locate the Finetuned Model

After training, the model will be stored in:

```
.ragatouille/colbert/none
```

Models are named using **timestamp-based directories**.

It is recommended to **rename them to something more descriptive**.

---

## Step 9: Configure the Retriever

Update the ColBERT model path in:

```
src/data_processing/colbert_data_processing.py
```

Replace the base model name with the **full directory path** to your finetuned model.

---

# 2. Generate Vector Embeddings

Ensure `src/data_processing/colbert_data_processing.py` contains the correct:

- ColBERT model path
- Index name

Then run:

```bash
python -m src.data_processing.main
```

This will build the **document index**.

---

# 3. Query the RAG System

Update the question variable in:

```
src/agent/main.py
```

Then run:

```bash
python -m src.agent.main
```

---

# 4. Run the Streamlit Application

Launch the UI:

```bash
streamlit run app.py
```

---

# 5. Retriever Evaluation

## Step 1: Build an Index

You must first build an index using the retriever being tested:

```bash
python -m src.data_processing.main
```

---

## Step 2: Label Evaluation Questions

Manually label evaluation questions:

```bash
python -m evaluation.retriever.manual_triple_generation
```

This is a **slightly modified version** of the labeling tool used during finetuning.

---

## Step 3: Configure Evaluation

Set the index name inside:

```
evaluation/retriever/retriever_evaluation.py
```

---

## Step 4: Run Evaluation

```bash
python -m evaluation.retriever.evaluate_retriever
```

Relevant metrics will be printed to the **console output**.

---

# Additional Notes

Some files in the evaluation directory are related to:

- **Offline triple generation**
- **Deprecated approaches to automatic question labeling**

These are retained for reference but are **not part of the current workflow**.