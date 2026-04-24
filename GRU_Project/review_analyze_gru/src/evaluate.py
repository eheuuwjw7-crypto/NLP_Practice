import torch

from config import *
from dataset import get_dataloader
from model import GRU
from predict import predict_onebatch
from tokenizer import Tokenizer
from tqdm import tqdm

def evaluate(model, dataloader, device):
    acc_count, total_count = 0, 0
    with torch.no_grad():
        for input, targets in tqdm(dataloader, desc='Evaluating'):
            input, targets = input.to(device), targets.to(device)
            result_list = predict_onebatch(model, input)
            for target, result in zip(targets, result_list):
                total_count += 1
                result = 1 if result >= 0.5 else 0
                if target == result:
                    acc_count += 1
    acc_count = acc_count / total_count
    return acc_count

def evaluate_run():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = Tokenizer.load_vocab(MODEL_PATH)
    model = GRU(tokenizer.vocab_size, tokenizer.pad_id).to(device)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    print('Loaded best model')
    test_dataloader = get_dataloader(train=False)
    acc_count = evaluate(model, test_dataloader, device)
    print('Acc:', acc_count*100, '%')

if __name__ == '__main__':
    evaluate_run()