import torch

from config import *
from dataset import get_dataloader
from model import RNN
from predict import predict_topk
from tokenizer import Tokenizer
from tqdm import tqdm

def evaluate(model, dataloader, device):
    top1_acc_count, top5_acc_count, total_count = 0, 0, 0
    with torch.no_grad():
        for input, targets in tqdm(dataloader, desc='Evaluating'):
            input, targets = input.to(device), targets.to(device)
            top5_indices_list = predict_topk(model, input)
            for target, top5_indices in zip(targets, top5_indices_list):
                total_count += 1
                if target in top5_indices:
                    top5_acc_count += 1
                if target == top5_indices[0]:
                    top1_acc_count += 1
    top1_acc_count = top1_acc_count / total_count
    top5_acc_count = top5_acc_count / total_count
    return top1_acc_count, top5_acc_count

def evaluate_run():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = Tokenizer.load_vocab(MODEL_PATH)
    model = RNN(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    print('Loaded best model')
    test_dataloader = get_dataloader(train=False)
    top1_acc_count, top5_acc_count = evaluate(model, test_dataloader, device)
    print('Top1 Acc:', top1_acc_count*100, '%')
    print('Top5 Acc:', top5_acc_count*100, '%')

if __name__ == '__main__':
    evaluate_run()
