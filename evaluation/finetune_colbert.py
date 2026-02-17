import os, torch
from ragatouille import RAGTrainer
import pandas as pd
import pickle
import json

segments_df = pd.read_csv("data/agora/segments.csv")
#ids2text = {f"segment_{row['Document ID']}_{row['Segment position']}": row["Text"] for idx, row in segments_df.iterrows()}
#chunk_texts = [row['Text'] for idx, row in segments_df.iterrows()]
with open("chunk_content/map.pkl", "rb") as f:
    chunk_content = pickle.load(f)


def finetune(name, input_file, output_dir, negs=True, add_negs=False):
    triplets = [("query", "positive_label", "negative_label")] if negs else [("query", "positive_label")]
    with open(f"evaluation/{input_file}", "r") as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            pos_id = item.get("positive_example", "")
            neg_id = item.get("negative_example", "")
            pos_text = chunk_content[pos_id]
            if negs:
                neg_text = chunk_content[neg_id]
                triplets.append((query, pos_text, neg_text))
            else:
                triplets.append((query, pos_text))
    print(triplets[1])
    trainer = RAGTrainer(
        model_name=f"colbert-{name}",
        pretrained_model_name="colbert-ir/colbertv2.0",  # or your model path
        n_usable_gpus=1
    )
    print(trainer.model_name)
    trainer.prepare_training_data(
        raw_data=triplets,
        mine_hard_negatives=add_negs,
        data_out_path=f".ragatouille/data/{name}"
    )
    print(trainer.train(
        learning_rate=2e-5,
        batch_size=16,
        maxsteps=600,  
    ))

if __name__ == "__main__":
    """
    print("Beginning finetuning with only labeled negatives")
    finetune("labeled", "train.jsonl", "labeled_only")
    """
    print("Beginning finetuning with only hard-mined negatives")
    finetune("hardmined", "train.jsonl", "hard_mined", negs=False)
    """
    print("Beginning finetuning with both mined and labeled negatives")
    finetune("both", "train.jsonl", "both", add_negs=True)



    print("Beginning finetuning naive")
    finetune("naive", "train_naive.jsonl", "naive_negatives")
    print("Beginning finetuning hard")
    finetune("hard", "train_true.jsonl", "hard_negatives")
    print("Beginning finetuning close")
    finetune("close", "train_close.jsonl", "close_negatives")
    print("Beginning finetuning combo")
    finetune("combo", "train_combo.jsonl", "combo_negatives")
    print("Beginning finetuning mined")
    finetune("mined", "train_combo.jsonl", "mined_negatives", negs=False)
    """
    print("done")
