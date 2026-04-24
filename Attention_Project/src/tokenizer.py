from config import *
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

class BaseTokenizer():
    unk_token = UNK_TOKEN
    pad_token = PAD_TOKEN
    start_token = START_TOKEN
    end_token = END_TOKEN

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2id = {word:i for i, word in enumerate(vocab_list)}
        self.id2word = {i:word for i, word in enumerate(vocab_list)}
        self.unk_id = self.word2id[self.unk_token]
        self.pad_id = self.word2id[self.pad_token]
        self.start_id = self.word2id[self.start_token]
        self.end_id = self.word2id[self.end_token]

    @classmethod
    def tokenize(cls, text) -> list[str]:
        pass

    def encode(self, text, mark=False):
        tokens = self.tokenize(text)
        if mark:
            tokens = [self.start_token] + tokens + [self.end_token]
        ids = [self.word2id.get(token, self.unk_id) for token in tokens]
        return ids

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        vocab_set = set()
        for sentence in sentences:
            vocab_set.update(cls.tokenize(sentence))
        vocab_list = [cls.pad_token, cls.unk_token, cls.start_token, cls.end_token] + list(vocab_set)
        print(len(vocab_list))
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def load_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [token.strip() for token in f.readlines()]
        tokenizer = cls(vocab_list)
        return tokenizer

class CNTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)

class ENTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    def detokenize(self, ids) -> str:
        tokens = [self.id2word[id] for id in ids]
        return self.detokenizer.detokenize(tokens)

if __name__ == '__main__':
    en_tokenizer = ENTokenizer.load_vocab(EN_PATH)
    cn_tokenizer = CNTokenizer.load_vocab(CN_PATH)

    print(en_tokenizer.vocab_size)
    print(cn_tokenizer.vocab_size)
    print(en_tokenizer.unk_token)
    print(en_tokenizer.pad_id)
    print((en_tokenizer.start_token))
    print(en_tokenizer.end_token)
    print(cn_tokenizer.encode('今天'))
    print(en_tokenizer.encode('Hello world', mark=True))