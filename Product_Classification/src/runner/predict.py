import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.configuration.config import *


class Predictor:
    # 初始化，传入模型、分词器、设备
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, texts: str | list):
        is_str = isinstance(texts, str)
        if is_str:
            texts = [texts]
        # 分词编码，得到模型输入
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 前向传播，得到预测结果
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()
        labels = [self.model.config.id2label[pred_id] for pred_id in pred_ids]
        if is_str:
            return {
                "label": labels[0],
                "confidence": confidences[0],
            }
        return [
            {
                "label": label,
                "confidence": confidence,
            }
            for label, confidence in zip(labels, confidences)
        ]


def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR/'best')
    predictor = Predictor(model, tokenizer, device)
    while True:
        text = input("请输入要预测的文本（输入 q 或 quit 退出）：").strip()
        if text.lower() in {"q", "quit"}:
            print("已退出预测。")
            break
        if not text:
            print("输入不能为空，请重新输入。")
            continue
        result = predictor.predict(text)
        print(f"预测类别: {result['label']}")
        print(f"置信度: {result['confidence']:.2%}")

if __name__ == "__main__":
    predict()
