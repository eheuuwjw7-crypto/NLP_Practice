import torch
from torch import nn, optim

from config import *
from dataset import get_dataloader
from model import HFBERT
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        inputs= {k:v.to(device) for k,v in batch.items()}
        target = inputs.pop('labels').to(dtype=torch.float)
        output = model(**inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader = get_dataloader(train=True)

    model = HFBERT().to(device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, loss, train_loader, device)
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        print(f'Epoch: {epoch+1}, Loss: {train_loss}')
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), BEST_MODEL)
            print('Saved best model')
    writer.close()

if __name__ == '__main__':
    train()