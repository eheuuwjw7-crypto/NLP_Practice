import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
from tokenizer import ENTokenizer, CNTokenizer

def build_dataset(sentences, tokenizer):
    dataset = []
    sentences_id = [tokenizer.encode(sentence, seq_len=SEQ_LEN) for sentence in sentences]
    for id in sentences_id:
        for i in range(len(id) - SEQ_LEN):
            input = id[i:i+SEQ_LEN]
            target = id[i+SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def preprocess():
    df = pd.read_csv(ORI_PATH, sep='\t', usecols=[0, 1], names=['en', 'cn'], encoding='utf-8').dropna()

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    CNTokenizer.build_vocab(train_df['cn'].tolist(), CN_PATH)
    ENTokenizer.build_vocab(train_df['en'].tolist(), EN_PATH)
    cn_tokenizer = CNTokenizer.load_vocab(CN_PATH)
    en_tokenizer = ENTokenizer.load_vocab(EN_PATH)

    train_df['cn'] = train_df['cn'].apply(lambda text: cn_tokenizer.encode(text, mark=False))
    test_df['cn'] = test_df['cn'].apply(lambda text: cn_tokenizer.encode(text, mark=False))
    train_df['en'] = train_df['en'].apply(lambda text: en_tokenizer.encode(text, mark=True))
    test_df['en'] = test_df['en'].apply(lambda text: en_tokenizer.encode(text, mark=True))

    train_df.to_json(TRAIN_PATH, orient='records', lines=True)
    test_df.to_json(TEST_PATH, orient='records', lines=True)

if __name__ == '__main__':
    preprocess()