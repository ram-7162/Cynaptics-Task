import torch
from task1.model import GPTLanguageModel
from transformers import GPT2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

vocab_size = tokenizer.vocab_size

checkpoint = torch.load("model.pth", map_location=device)

model = GPTLanguageModel(vocab_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "KING:\nMy lord, "

context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

output = model.generate(
    context,
    max_new_tokens=500,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)

print(decode(output[0].tolist()))
