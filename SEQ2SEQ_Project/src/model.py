import torch
import torch.nn as nn
from config import *

class ENCODER(nn.Module):
    def __init__(self,vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)

    def forward(self,x):
        length = (x != self.embedding.padding_idx).sum(dim=1)
        x = self.embedding(x)
        out, _ = self.gru(x)
        indices = torch.arange(out.shape[0], device=out.device)
        out = out[indices, length-1]
        # print(out.shape)
        return out

class DECODER(nn.Module):
    def __init__(self,vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(self, x, h0=None):
        embed = self.embedding(x)
        output, hn = self.gru(embed, h0)
        out = self.fc(output)
        return out, hn

class Seq2Seq(nn.Module):
    def __init__(self, cn_vocab_size, en_vocab_size, cn_padding_idx, en_padding_idx):
        super().__init__()
        self.encoder = ENCODER(cn_vocab_size, cn_padding_idx)
        self.decoder = DECODER(en_vocab_size, en_padding_idx)

if __name__ == '__main__':
    model = Seq2Seq(1000, 1000, 0, 0)
    print(model.encoder)
    print(model.decoder)