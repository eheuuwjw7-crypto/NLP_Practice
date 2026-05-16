import torch
from torch import nn, optim

from config import *
from dataset import get_dataloader
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        inputs= {k:v.to(device) for k,v in batch.items()}
        output = model(**inputs)
        loss = output.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader = get_dataloader(train=True)

    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        print(f'Epoch: {epoch+1}, Loss: {train_loss}')
        if train_loss < min_loss:
            min_loss = train_loss
            model.save_pretrained(MODELS_DIR)
            print('Saved best model')
    writer.close()

if __name__ == '__main__':
    train()