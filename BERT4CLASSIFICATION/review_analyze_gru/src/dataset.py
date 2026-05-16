import torch
from torch.utils.data import DataLoader
from config import *
from datasets import load_from_disk

def get_dataloader(train=True):
    path = TRAIN_PATH if train else TEST_PATH
    dataset = load_from_disk(path)
    dataset.set_format(type='torch')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    for batch in train_dataloader:
        for k, v in batch.items():
            print(k, '->', v.shape)
        break