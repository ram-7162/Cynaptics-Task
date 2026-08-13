import torch
from transformers import GPT2TokenizerFast
from model import GPTLanguageModel, block_size, d_model, vocab_size, num_block, num_heads, d_ffn, dropout



def generate(model, tokenizer, prompt, max_new_tokens, device):

    model.eval()
    idx = torch.tensor(encode(prompt), dtype=torch.long).to(device) ## (T, )
    idx = idx.unsqueeze(0)  ##(1, T)
    print(decode(idx[0]))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            T = idx.shape[-1]
            idx_cond = idx
            if(T >= block_size):
                idx_cond = idx[:, -block_size:]
            y_pred = model(idx_cond)   ###(1, T, vocab_size)
            logits = y_pred[:, -1, :]   ###(1, vocab_size)   ### select last token
            probab = torch.softmax(logits, dim = -1)  ###(1, vocab_size)
            output = torch.argmax(probab, dim = -1, keepdim = True)  ###(1,1) return argument with max prop
            idx = torch.cat((idx, output), dim = -1)
            print(decode(output[0]))

    print("=="*50)
    print(decode(idx[0]))

device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

vocab_size = tokenizer.vocab_size

checkpoint = torch.load("best_model.pt", map_location=device)

model = GPTLanguageModel(d_model, vocab_size, num_block, num_heads, d_ffn, dropout).to(device)
model.load_state_dict(checkpoint)

init_word = input("Enter the starting few words : ")

generate(model, tokenizer, init_word, max_new_tokens=100, device=device)


