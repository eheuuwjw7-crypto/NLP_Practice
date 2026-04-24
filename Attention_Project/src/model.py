import torch
import torch.nn as nn
from torch.nn import attention

from config import *

class Attention(nn.Module):
    def forward(self, decoder_out, encoder_out):
        attention_score = torch.bmm(decoder_out, encoder_out.transpose(1,2))
        attention_weights = torch.softmax(attention_score, dim=-1)
        return torch.bmm(attention_weights, encoder_out)

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
        h0 = out[indices, length-1]
        # print(out.shape)
        return out, h0

class DECODER(nn.Module):
    def __init__(self,vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.attention = Attention()
        self.fc = nn.Linear(HIDDEN_SIZE * 2, vocab_size)

    def forward(self, x, h0, encoder_out):
        embed = self.embedding(x)
        output, hn = self.gru(embed, h0)
        context_vector = self.attention(output, encoder_out)
        combine = torch.cat([output, context_vector], dim=-1)
        out = self.fc(combine)
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