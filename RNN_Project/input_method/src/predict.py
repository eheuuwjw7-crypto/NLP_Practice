import torch
from config import *
from model import RNN
from tokenizer import Tokenizer

def predict_topk(model, input, k=5):
    model.eval()
    with torch.no_grad():
        output = model(input)
    top_indices = torch.topk(output, k).indices
    return top_indices.tolist()

def predict_patch(text, model, tokenizer, k, device):
    ids = tokenizer.encode(text)
    input = torch.tensor([ids], dtype=torch.long).to(device)
    top_indices = predict_topk(model, input, k=k)
    top_tokens = [tokenizer.id2word[index] for index in top_indices[0]]
    return top_tokens

def predict_run():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = Tokenizer.load_vocab(MODEL_PATH)
    model = RNN(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(BEST_MODEL))
    print('Loaded best model')
    print('欢迎使用智能输入法模型，输入q或quit退出')
    input_history = ''
    while True:
        user_input = input('请输入：')
        if user_input in ['q', 'quit']:
            print('Bye~')
            break
        if user_input == '':
            print('请输入内容')
            continue
        input_history += user_input
        top5_tokens = predict_patch(input_history, model, tokenizer, k=5, device=device)
        print('预测结果：', top5_tokens)

if __name__ == '__main__':
    # text = '我今天'
    # top5_tokens = predict_run(text)
    # print(top5_tokens)
    predict_run()