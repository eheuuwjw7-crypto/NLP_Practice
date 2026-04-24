from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
PROC_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'
MODELS_DIR = BASE_DIR / 'models'

TRAIN_PATH = PROC_DIR / 'train.jsonl'
TEST_PATH = PROC_DIR / 'test.jsonl'
ORI_PATH = RAW_DIR / 'cmn.txt'
EN_PATH = MODELS_DIR / 'en_vocab.txt'
CN_PATH = MODELS_DIR / 'cn_vocab.txt'
BEST_MODEL = MODELS_DIR / 'best_model.pt'
LOG_DIR = BASE_DIR / 'logs'

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
START_TOKEN = '<sos>'
END_TOKEN = '<eos>'

SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256

LEARNING_RATE = 1e-3
EPOCHS = 50