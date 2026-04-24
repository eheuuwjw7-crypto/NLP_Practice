import torch
from config import *
from dataset import get_dataloader
from model import Seq2Seq
from predict import predict_onebatch
from tokenizer import CNTokenizer, ENTokenizer
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.tolist()
            batch_result = predict_onebatch(model, inputs, tokenizer, device)
            hypotheses.extend(batch_result)
            for target in targets:
                if tokenizer.end_id in target:
                    target = target[1:target.index(tokenizer.end_id)]
                else:
                    target = target[1:]
                references.append([target])
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

def evaluate_run():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cn_tokenizer = CNTokenizer.load_vocab(CN_PATH)
    en_tokenizer = ENTokenizer.load_vocab(EN_PATH)
    model = Seq2Seq(cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_id, en_tokenizer.pad_id).to(device)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    print('Loaded best model')
    test_dataloader = get_dataloader(train=False)
    acc_count = evaluate(model, test_dataloader, en_tokenizer, device)
    print('Bleu:', acc_count*100, '%')

if __name__ == '__main__':
    evaluate_run()
