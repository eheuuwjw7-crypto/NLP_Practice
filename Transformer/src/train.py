import torch
from torch import nn, optim

from config import *
from dataset import get_dataloader
from model import Transformer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from tokenizer import CNTokenizer, ENTokenizer

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)

        decoder_inputs = targets[:,:-1]
        decoder_targets = targets[:,1:]
        src_pad_mask = (inputs == model.cn_embedding.padding_idx)
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1]).to(device)

        decoder_outputs = model(inputs, decoder_inputs, src_pad_mask=src_pad_mask, tgt_mask=tgt_mask)

        loss = criterion(decoder_outputs.permute(0, 2, 1), decoder_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(train=True)

    cn_tokenizer = CNTokenizer.load_vocab(CN_PATH)
    en_tokenizer = ENTokenizer.load_vocab(EN_PATH)

    model = Transformer(cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_id, en_tokenizer.pad_id).to(device)
    loss = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)
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
