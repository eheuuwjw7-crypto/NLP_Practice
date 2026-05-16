import torch
from config import *
from model import HFBERT
from transformers import AutoTokenizer

def predict_onebatch(model, input):
    model.eval()
    with torch.no_grad():
        output = model(**input)
    batch_result = torch.sigmoid(output)
    return batch_result.tolist()

def predict_batches(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=SEQ_LEN,
        return_tensors='pt'
    )
    inputs = {k:v.to(device) for k, v in inputs.items()}

    result = predict_onebatch(model, inputs)
    return result[0]

def predict_run():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = HFBERT().to(device)
    model.load_state_dict(torch.load(BEST_MODEL))
    print('Loaded best model')
    print('欢迎使用情感判断模型，输入q或quit退出')
    while True:
        user_input = input('请输入：')
        if user_input in ['q', 'quit']:
            print('Bye~')
            break
        if user_input == '':
            print('请输入内容')
            continue

        result = predict_batches(user_input, model, tokenizer, device)
        if result > 0.5:
            print('预测结果：positive')
        else:
            print('预测结果：negative')

if __name__ == '__main__':
    # text = '我今天'
    # top5_tokens = predict_run(text)
    # print(top5_tokens)
    predict_run()