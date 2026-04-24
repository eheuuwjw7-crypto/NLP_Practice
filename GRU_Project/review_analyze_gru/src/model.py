import torch
import torch.nn as nn
from config import *

class GRU(nn.Module):
    def __init__(self,vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.linear = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self,x):
        length = (x != self.embedding.padding_idx).sum(dim=1)
        x = self.embedding(x)
        out, _ = self.gru(x)
        indices = torch.arange(out.shape[0])
        out = out[indices, length-1]
        # print(out.shape)
        out = self.linear(out).squeeze(-1)
        return out

if __name__ == '__main__':
    vocab_size = 100
    model = GRU(vocab_size, padding_idx=0)
    x = torch.randint(vocab_size, (64, 5))
    out = model(x)
    print(out.shape)