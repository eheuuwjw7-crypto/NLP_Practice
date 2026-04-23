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
        input = torch.tensor(self.data[idx]['review'], dtype=torch.long)
        target = torch.tensor(self.data[idx]['label'], dtype=torch.float)
        return input, target

def get_dataloader(train=True):
    path = TRAIN_PATH if train else TEST_PATH
    dataset = SMDataset(path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break