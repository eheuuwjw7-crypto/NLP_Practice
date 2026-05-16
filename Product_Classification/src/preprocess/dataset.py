from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from src.configuration.config import *

def get_dataset(ds_type="train"):
    path = str(PROCESSED_DIR / ds_type)
    dataset = load_from_disk(path)
    return dataset

def get_dataloader(tokenizer, ds_type="train"):
    path = str(PROCESSED_DIR / ds_type)
    dataset = load_from_disk(path)
    dataset.set_format("torch")

    collate_fn = DataCollatorWithPadding(
        tokenizer,
        padding=True,
        return_tensors='pt'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    train_dataloader = get_dataloader(tokenizer, "train")
    for batch in train_dataloader:
        for k,v in batch.items():
            print(k, v)
        break