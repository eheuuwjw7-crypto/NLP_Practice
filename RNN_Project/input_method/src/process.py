import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
from input_method.src import tokenizer
from tokenizer import Tokenizer

def build_dataset(sentences, tokenizer):
    dataset = []
    sentences_id = [tokenizer.encode(sentence) for sentence in sentences]
    for id in sentences_id:
        for i in range(len(id) - SEQ_LEN):
            input = id[i:i+SEQ_LEN]
            target = id[i+SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def preprocess():
    df = pd.read_json(ORI_PATH, lines=True, orient='records').sample(frac=0.1, random_state=42)
    sentences = []
    # for dialog in df['dialog']:
    #     for item in dialog:
    #         if '：' in item:
    #             _, text = item.split('：', 1)
    #         elif ':' in item:
    #             _, text = item.split(':', 1)
    #         else:
    #             text = item
    #         sentences.append(text.strip())
    for dialog in df['dialog']:
        for item in dialog:
            sentences.append(item.split('：')[1])
    print(sentences[0])
    print(len(sentences))

    train_sentences, test_sentences = train_test_split(sentences, test_size=0.1, random_state=42)

    Tokenizer.build_vocab(train_sentences, MODEL_PATH)
    tokenizer = Tokenizer.load_vocab(MODEL_PATH)

    train_dataset = build_dataset(train_sentences, tokenizer)
    test_dataset = build_dataset(test_sentences, tokenizer)
    pd.DataFrame(train_dataset).to_json(TRAIN_PATH, orient='records', lines=True)
    pd.DataFrame(test_dataset).to_json(TEST_PATH, orient='records', lines=True)

if __name__ == '__main__':
    preprocess()
