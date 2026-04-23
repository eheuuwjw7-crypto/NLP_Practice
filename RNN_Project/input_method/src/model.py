import torch
import torch.nn as nn
from config import *

class RNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE)
        self.rnn = nn.RNN(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.linear = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out

if __name__ == '__main__':
    vocab_size = 100
    model = RNN(vocab_size)
    x = torch.randint(vocab_size, (64, 5))
    out = model(x)
    print(out.shape)