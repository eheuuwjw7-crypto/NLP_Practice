import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.configuration.config import BERT_MODEL, MODELS_DIR
from src.runner.predict import Predictor
from src.web.schemas import Category, Title
from src.web.service import TitleService

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR/'best')
predictor = Predictor(
    model=model,
    device=device,
    tokenizer=tokenizer,
)
service = TitleService(predictor)

@app.post("/predict")
def predict(title: Title) -> Category:
    label = service.predict(title.text)
    return Category(category=label)

def serve():
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000)
