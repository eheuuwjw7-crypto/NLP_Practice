from config import *
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

def preprocess():
    dataset = load_dataset('csv', data_files=str(ORI_PATH))['train']
    # print(dataset)
    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter(lambda x: x['review'] is not None)
    # print(dataset)
    dataset = dataset.cast_column('label', ClassLabel(names=['negative', 'positive']))
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    # print(dataset_dict)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    def batch_tokenize(batch):
        inputs = tokenizer(
            batch['review'],
            padding='max_length',
            truncation=True,
            max_length=SEQ_LEN,
            return_tensors='pt'
        )
        inputs['labels'] = batch['label']
        return inputs
    dataset_dict = dataset_dict.map(batch_tokenize, batched=True, remove_columns=['label', 'review'])
    dataset_dict.save_to_disk(PROC_DIR)

if __name__ == '__main__':
    preprocess()