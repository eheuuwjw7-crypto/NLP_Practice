import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import *

class SMDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True, orient='records').to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = torch.tensor(self.data[idx]['cn'], dtype=torch.long)
        target = torch.tensor(self.data[idx]['en'], dtype=torch.long)
        return input, target

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

def get_dataloader(train=True):
    path = TRAIN_PATH if train else TEST_PATH
    dataset = SMDataset(path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break