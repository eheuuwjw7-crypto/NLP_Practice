from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
PROC_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'
MODELS_DIR = BASE_DIR / 'models'

TRAIN_PATH = PROC_DIR / 'train'
TEST_PATH = PROC_DIR / 'test'
ORI_PATH = RAW_DIR / 'online_shopping_10_cats.csv'
LOG_DIR = BASE_DIR / 'logs'
BERT_MODEL = 'google-bert/bert-base-chinese'

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

SEQ_LEN = 128
BATCH_SIZE = 12
EMBEDDING_SIZE = 128

LEARNING_RATE = 1e-5
EPOCHS = 20
