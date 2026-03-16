import pandas as pd

questions = pd.read_csv("retriever_finetuning/questions.csv")
documents = pd.read_csv("data/agora/documents.csv")
segments = pd.read_csv("data/agora/segments.csv")
for idx, row in questions.iterrows():
    q = row["question"]
    ids = row["relevant_ids"][:-1].split(",")
    ids = [i[2:-1] for i in ids]
    row["question"] = q.rstrip() # remove whitespace
    seg_ids = []
    doc_ids = []
    for i in ids:
        if "document" in i: doc_ids.append(i)
        else: seg_ids.append(i)
    for d in doc_ids:
        agora_id = d[9:]
        found = False
        for s in seg_ids:
            if f"segment_{agora_id}" in s: 
                found = True
                break
        if not found:
            doc_segs = segments[segments["Document ID"] == int(agora_id)]["Segment position"].tolist()
            doc_segs = [f"segment_{agora_id}_{s}" for s in doc_segs]
            seg_ids += doc_segs
    all_ids = doc_ids + seg_ids
    row["relevant_ids"] = ";".join(all_ids)

questions.to_csv("retriever_finetuning/questions_processed.csv", index=False)