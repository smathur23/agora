from ragatouille import RAGTrainer
import pandas as pd
import json

segments_df = pd.read_csv("data/agora/segments.csv")
ids2text = {f"segment_{row['Document ID']}_{row['Segment position']}": row["Text"] for idx, row in segments_df.iterrows()}
chunk_texts = [row['Text'] for idx, row in segments_df.iterrows()]


def finetune(name, input_file, output_dir, negs=True):
    triplets = [("query", "positive_label", "negative_label")] if negs else [("query", "positive_label")]
    with open(f"evaluation/{input_file}", "r") as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            pos_ids = item.get("positive_document_ids", [])
            neg_ids = item.get("negative_document_ids", [])
            for pos in pos_ids:
                pos_text = ids2text[pos]
                if negs:
                    for neg in neg_ids:
                        neg_text = ids2text[neg]
                        triplets.append((query, pos_text, neg_text))
                else:
                    triplets.append((query, pos_text))
    print(triplets[1])
    trainer = RAGTrainer(
        model_name=f"colbert-{name}",
        pretrained_model_name="colbert-ir/colbertv2.0",  # or your model path
        n_usable_gpus=2
    )
    print(trainer.model_name)
    if not negs:
        trainer.prepare_training_data(
            raw_data=triplets,
            mine_hard_negatives=False,
            data_out_path=f".ragatouille/data/{name}"
        )
    else:
        trainer.prepare_training_data(
            raw_data=triplets,
            mine_hard_negatives=True,
            data_out_path=f".ragatouille/data/{name}"
        )
    trainer.train(
        learning_rate=2e-5,
        batch_size=16,
        maxsteps=600,  
    )

if __name__ == "__main__":
    """print("Beginning finetuning naive")
    finetune("naive", "train_naive.jsonl", "naive_negatives")
    print("Beginning finetuning hard")
    finetune("hard", "train_true.jsonl", "hard_negatives")
    print("Beginning finetuning close")
    finetune("close", "train_close.jsonl", "close_negatives")
    print("Beginning finetuning combo")
    finetune("combo", "train_combo.jsonl", "combo_negatives")
    print("Beginning finetuning mined")
    """
    finetune("mined", "train_combo.jsonl", "mined_negatives", negs=False)
    print("done")