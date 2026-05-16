import torch
import torch.nn as nn
from config import *
from transformers import AutoModel

class HFBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_MODEL)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out= self.bert(input_ids, attention_mask, token_type_ids)
        cls_hidden_state = out.pooler_output
        out = self.linear(cls_hidden_state).squeeze(-1)
        return out

if __name__ == '__main__':
    model = HFBERT()
    print(model)