import torch
import torch.nn as nn
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]

class Transformer(nn.Module):
    def __init__(self, cn_vocab_size, en_vocab_size, cn_padding_idx, en_padding_idx):
        super().__init__()
        self.cn_embedding = nn.Embedding(cn_vocab_size, embedding_dim=D_MODEL, padding_idx=cn_padding_idx)
        self.en_embedding = nn.Embedding(en_vocab_size, embedding_dim=D_MODEL, padding_idx=en_padding_idx)
        self.position_encoding = PositionalEncoding(D_MODEL, max_seq_len=SEQ_LEN)
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            num_encoder_layers=N_LAYERS,
            num_decoder_layers=N_LAYERS,
            batch_first=True
        )
        self.fc = nn.Linear(D_MODEL, en_vocab_size)
    def forward(self, src, tgt, tgt_mask, src_pad_mask):
        memory = self.encode(src, src_pad_mask)
        return self.decode(tgt, memory, tgt_mask, src_pad_mask)

    def encode(self, src, src_pad_mask):
        embed = self.cn_embedding(src)
        input = self.position_encoding(embed)
        return self.transformer.encoder(input, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt, memory, tgt_mask, src_pad_mask):
        embed = self.en_embedding(tgt)
        input = self.position_encoding(embed)
        output =  self.transformer.decoder(input, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask)
        return self.fc(output)

if __name__ == '__main__':
    model = Transformer(1000, 1024, 0, 0)
    print(model)
