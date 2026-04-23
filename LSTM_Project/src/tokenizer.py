import jieba
from config import *

class Tokenizer():
    unk_token = UNK_TOKEN
    pad_token = PAD_TOKEN

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2id = {word:i for i, word in enumerate(vocab_list)}
        self.id2word = {i:word for i, word in enumerate(vocab_list)}
        self.unk_id = self.word2id[self.unk_token]
        self.pad_id = self.word2id[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        tokens = jieba.lcut(text)

        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens += [self.pad_token] * (seq_len - len(tokens))

        ids = [self.word2id.get(token, self.unk_id) for token in tokens]
        return ids

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        vocab_set = set()
        for sentence in sentences:
            vocab_set.update(jieba.lcut(sentence))
        vocab_list = [cls.pad_token, cls.unk_token] + list(vocab_set)
        print(len(vocab_list))
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def load_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [token.strip() for token in f.readlines()]
        tokenizer = cls(vocab_list)
        return tokenizer

if __name__ == '__main__':
    tokenizer = Tokenizer.load_vocab(MODEL_PATH)
    print(tokenizer.vocab_size)
    print(tokenizer.unk_token)
    print(tokenizer.pad_id)
    print(tokenizer.encode('今天', seq_len=SEQ_LEN))