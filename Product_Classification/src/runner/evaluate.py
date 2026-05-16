import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from src.configuration.config import *
from src.preprocess.dataset import get_dataset
from src.runner.train import Trainer


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / 'best')

    test_dataset = get_dataset("test")
    collate_fn = DataCollatorWithPadding(
        tokenizer,
        padding=True,
        return_tensors='pt'
    )

    def compute_metrics(labels, preds) -> dict:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {
            'accuracy': acc,
            'f1': f1
        }

    trainer = Trainer(
        model=model,
        train_dataset=None,
        valid_dataset=test_dataset,
        collate_fn=collate_fn,
        compute_metrics=compute_metrics,
        device=device,
    )
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == '__main__':
    evaluate()
