import os
from src.data_processing.embed_data import process_and_embed_data
from src.data_processing.preprocessing import process_policy_files


def main():
    # Run preprocessing and embedding scripts
    data_folder = "/".join(os.path.abspath(__file__).split("/")[:-3]) + "/data/agora"
    processed_data = process_policy_files(data_folder)
    embeddings_path = process_and_embed_data(processed_data)
    print("Vector embeddings stored at: " + embeddings_path)


if __name__ == "__main__":
    main()