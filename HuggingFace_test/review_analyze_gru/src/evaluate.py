import torch

from config import *
from dataset import get_dataloader
from model import HFBERT
from predict import predict_onebatch

from tqdm import tqdm

def evaluate(model, dataloader, device):
    acc_count, total_count = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            labels = batch.pop('labels').tolist()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            result = predict_onebatch(model, inputs)
            for target, result in zip(labels, result):
                total_count += 1
                result = 1 if result >= 0.5 else 0
                if target == result:
                    acc_count += 1
    acc_count = acc_count / total_count
    return acc_count

def evaluate_run():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = HFBERT().to(device)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    print('Loaded best model')
    test_dataloader = get_dataloader(train=False)
    acc_count = evaluate(model, test_dataloader, device)
    print('Acc:', acc_count*100, '%')

if __name__ == '__main__':
    evaluate_run()