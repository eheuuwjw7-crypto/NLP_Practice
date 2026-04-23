import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
from tokenizer import Tokenizer

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
    df = pd.read_csv(ORI_PATH, usecols=['label','review'], encoding='utf-8').dropna()

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

    Tokenizer.build_vocab(train_df['review'].tolist(), MODEL_PATH)
    tokenizer = Tokenizer.load_vocab(MODEL_PATH)

    train_df['review'] = train_df['review'].apply(lambda review: tokenizer.encode(review, SEQ_LEN))
    test_df['review'] = test_df['review'].apply(lambda review: tokenizer.encode(review, SEQ_LEN))
    train_df.to_json(TRAIN_PATH, orient='records', lines=True)
    test_df.to_json(TEST_PATH, orient='records', lines=True)

if __name__ == '__main__':
    preprocess()

