import pandas as pd

docs = pd.read_csv("./data/agora/documents.csv")

def get_doc_ids_from_title(titles_str: str):
    titles = titles_str.split(",")
    ids = []
    bad = 0
    good = 0
    for t in titles:
        if t[0] == " ":
            t = t[1:]
        #matches = docs.loc[docs['Official name'] == t, 'AGORA ID'].tolist()
        matches = docs[docs["Official name"].str.contains(t, case=False, na=False,regex=False)]["AGORA ID"].tolist()
        #print(type(matches))
        if len(matches) != 1:
            bad += 1
            #print(f"{t} has {len(matches)} matches:")
            #print(matches)
        else:
            good += 1
            ids += matches
    out = ""
    for id in ids:
        out += "," + str(id)
    return(out[1:])       

def process_questions(file_path: str = "./evaluation/generated_questions_raw.csv", output_path: str = "./evaluation/generated_questions.csv"):
    questions = pd.read_csv(file_path)
    
    questions = questions[["question", "relevant_documents"]]
    #print(questions.loc[100, "relevant_documents"])
    questions["relevant_documents"] = questions["relevant_documents"].apply(get_doc_ids_from_title)
    questions.to_csv(output_path, index=False)

if __name__ == "__main__":
    process_questions()