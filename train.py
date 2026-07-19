import torch
import torch.nn as nn
import numpy as np
from task1.model import GPTLanguageModel, block_size
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math


d_model = 384
num_block = 6
num_heads = 6
d_ffn = 4 * d_model
dropout = 0.1
learning_rate = 3e-4
epochs = 5
batch_size = 5
device = "cuda" if torch.cuda.is_available() else "cpu"


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

encode = lambda s: tokenizer.encode(s)
decode = lambda l: tokenizer.decode(l)

vocab_size = tokenizer.vocab_size

data = torch.tensor(encode(text), dtype=torch.long)

print("Total tokens:", len(data))



model = GPTLanguageModel(d_model, vocab_size, num_block, num_heads, d_ffn, dropout).to(device)
crieterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)






class CustomDataset(Dataset):

    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self, ):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.block_size]
        y = self.tokens[idx+1:idx+self.block_size+1]
        return x, y



train_dataset = CustomDataset(train_data, block_size)
val_dataset = CustomDataset(val_data, block_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)





best_loss = float("inf")
for Epoch in range(epochs):
    model.train()
    step = 0
    losses = []
    for batch_features, batch_labels in tqdm(train_dataloader):
        step+=1
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        y_pred = model(batch_features)  ###(B, T, vocab_size)
        y_pred = y_pred.reshape(-1, vocab_size)   ###(B*T, vocab_size)
        batch_labels = batch_labels.reshape(-1)    ####(B*T, )
        loss_value = crieterion(y_pred, batch_labels)
        losses.append(loss_value.item())   
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


        # print(f"Epoch : {Epoch} | Step : {step} | train_loss : {loss_value.item()} ")

    new_arr = np.array(losses)
    print(f"Epoch : {Epoch} | train_mean_loss : {np.mean(new_arr)} ")

    model.eval()
    losses = []
    step = 0
    with torch.no_grad():
        for batch_features, batch_labels in tqdm(val_dataloader):
            step+=1
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            y_pred = model(batch_features)  ###(B, T, vocab_size)
            y_pred = y_pred.reshape(-1, vocab_size)   ###(B*T, vocab_size)
            batch_labels = batch_labels.reshape(-1)    ####(B*T, )
            loss_value = crieterion(y_pred, batch_labels)
            losses.append(loss_value.item())

            # print(f"Epoch : {epoch} | Step : {step} | val_loss : {loss_value.item():.4f}")

        avg_val_loss = np.mean(losses)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")

        print(f"Epoch : {Epoch} | val_mean_loss : {np.mean(losses):.4f}")
