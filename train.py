import torch
from task1.model import GPTLanguageModel, block_size
from transformers import GPT2TokenizerFast


batch_size = 16
max_iters = 8000
eval_interval = 100
learning_rate = 1e-4
eval_iters = 50

device = "cuda" if torch.cuda.is_available() else "cpu"


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    text = text * 5


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

vocab_size = tokenizer.vocab_size

data = torch.tensor(encode(text), dtype=torch.long)

print("Total tokens:", len(data))


n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_split = train_data if split == "train" else val_data

    if len(data_split) <= block_size:
        raise ValueError(f"Dataset too small: {len(data_split)} tokens")

    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


model = GPTLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)


best_val_loss = float("inf")
patience = 5
counter = 0


for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss(model)

        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

       
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

     
        context = torch.tensor(
            [encode("KING:\nMy lord, ")],
            dtype=torch.long,
            device=device
        )

        out = model.generate(
            context,
            max_new_tokens=100,
            temperature=0.6,
            top_k=40,
            top_p=0.85
        )

        # print(decode(out[0].tolist()))
        print("------------------------------------------------")

    xb, yb = get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()


torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size
}, "model.pth")
