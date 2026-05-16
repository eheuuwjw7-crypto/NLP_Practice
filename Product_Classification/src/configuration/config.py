from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "checkpoint"
LOG_DIR = ROOT_DIR / "logs"

TRAIN_PATH = RAW_DIR / "train.txt"
TEST_PATH = RAW_DIR / "test.txt"
VALID_PATH = RAW_DIR / "valid.txt"
LABELS_PATH = MODELS_DIR / "labels.txt"

BERT_MODEL = str(ROOT_DIR / "hf_local" / "bert-base-chinese")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-5
SAVE_STEP = 500
