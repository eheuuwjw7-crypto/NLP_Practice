import sys
from transformers import AutoTokenizer

from src.configuration.config import *
from datasets import load_dataset

def preprocess():
    # The raw files are tab-separated text classification data.
    # Keeping the split names aligned here makes the output DatasetDict
    # match the downstream train / test / valid loading code.
    dataset_dict = load_dataset(
        "csv",
        data_files={
            "train": str(TRAIN_PATH),
            "test": str(TEST_PATH),
            "valid": str(VALID_PATH),
        },
        delimiter="\t",

    )
    dataset_dict = dataset_dict.rename_column("text_a", "text")

    # Persist the label order so inference code can map predicted ids
    # back to human-readable category names consistently.
    all_labels = sorted(set(dataset_dict["train"]["label"]))

    # The label column starts as strings. class_encode_column converts it
    # into integer class ids and stores the mapping in the dataset schema.
    dataset_dict = dataset_dict.class_encode_column("label")

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(all_labels))

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    def batch_encode(batch):
        # Keep the target under "labels" because Hugging Face models expect
        # that key name during training.
        inputs = tokenizer(batch["text"], truncation=True)
        inputs['labels'] = batch["label"]
        return inputs

    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=dataset_dict["train"].column_names)
    dataset_dict.save_to_disk(PROCESSED_DIR)

if __name__ == "__main__":
    preprocess()
