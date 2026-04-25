import torch
from config import *
from model import Transformer
from tokenizer import CNTokenizer, ENTokenizer

def predict_onebatch(model, inputs, tokenizer, device):
    model.eval() 
    with torch.no_grad():
        batch_size = inputs.shape[0]
        src_pad_mask = (inputs == model.cn_embedding.padding_idx).to(device)
        memory = model.encode(inputs, src_pad_mask)

        decoder_input = torch.full(size=(batch_size, 1), fill_value=tokenizer.start_id).to(device)
        generated_ids = []
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i in range(SEQ_LEN):
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            decoder_output = model.decode(decoder_input, memory, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask)
            token_ids = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(token_ids.squeeze(1))
            decoder_input = torch.cat([decoder_input, token_ids], dim=-1)
            is_finished |= (token_ids.squeeze(1) == tokenizer.end_id)
            if is_finished.all():
                break

    generated_tensor = torch.stack(generated_ids, dim=1)
    gen_list = generated_tensor.tolist()
    for i, sentence_ids in enumerate(gen_list):
        if tokenizer.end_id in sentence_ids:
            eos_pos = sentence_ids.index(tokenizer.end_id)
            gen_list[i] = sentence_ids[:eos_pos]

    return gen_list

def predict_batches(text, model, cn_tokenizer, en_tokenizer, device):
    ids = cn_tokenizer.encode(text)
    input = torch.tensor([ids], dtype=torch.long).to(device)
    result = predict_onebatch(model, input, en_tokenizer, device)
    return en_tokenizer.detokenize(result[0])

def predict_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cn_tokenizer = CNTokenizer.load_vocab(CN_PATH)
    en_tokenizer = ENTokenizer.load_vocab(EN_PATH)

    model = Transformer(cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_id, en_tokenizer.pad_id).to(device)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    print('Loaded best model')
    print('欢迎使用中英翻译模型，输入q或quit退出')
    while True:
        user_input = input('请输入中文：')
        if user_input in ['q', 'quit']:
            print('Bye~')
            break
        if user_input == '':
            print('请输入内容')
            continue

        result = predict_batches(user_input, model, cn_tokenizer, en_tokenizer, device)
        print('英文译文为', result)

if __name__ == '__main__':
    # text = '我今天'
    # top5_tokens = predict_run(text)
    # print(top5_tokens)
    predict_run()
