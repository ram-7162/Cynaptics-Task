import torch
import torch.nn as nn
from torch.nn import functional as F
import math



block_size = 20

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.Embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.Embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div = 1 / (10000 ** (torch.arange(0, d_model, 2)/d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)  ##(1, max_se_length, d_model)
        self.register_buffer("pe", pe) ### it create pe as attribute as well make it untrainable

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class LayerNormalization(nn.Module):
    def __init__(self,d_model : int, eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim=-1, unbiased=False, keepdim = True)
        x = (x - mean)/(torch.sqrt(var + self.eps))
        return self.gamma*x + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ffn, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),  # or nn.GELU()
            nn.Dropout(p=dropout),
            nn.Linear(d_ffn, d_model)
        )
    def forward(self, x):
        return self.features(x)



class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ x -> (B, T, C)
            w_q -> (C, C)
            w_k -> (C, C)
            w_v -> (C, C)
            """
        query = self.w_q(x)  ### (B, T, C)
        key = self.w_k(x)  ### (B, T, C)
        value = self.w_v(x)  ### (B, T, C)
        out = query @ key.transpose(-2, -1)  ### ### (B, T, C) @ (B, C, T) -> (B, T, T)
        T = out.shape[-1]
        mask = torch.tril(torch.ones(T, T, device=out.device))    
        out = out.masked_fill(mask == 0, float("-inf"))   
        out = self.softmax(out/math.sqrt(key.size(-1)))
        out = self.dropout(out)
        out = out @ value  ## (B, T, T) @ (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        query = self.w_q(x)  ### (B, T, C)
        key   = self.w_k(x)  ### (B, T, C)
        value = self.w_v(x)   ### (B, T, C)
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  ### (B, H, T, d)
        key   = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  ### (B, H, T, d)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  ###(B, H, T, d)
        scores = query @ key.transpose(-2, -1)   #####(B, H, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.softmax(scores / math.sqrt(self.head_dim)) ###(B, H, T, T)
        attn = self.dropout(attn)
        out = attn @ value   ### (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)  # (B, T, C)
        out = self.w_o(out)
        return out
    


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ffn, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(self, num_block, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.list = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ffn, dropout) for _ in range(num_block)])

    def forward(self, x):
        for layer in self.list:
            x = layer(x)
        return x




class GPTLanguageModel(nn.Module):

    def __init__(self, d_model, vocab_size, num_block, num_heads, d_ffn, dropout):
        super().__init__()
        self.input_embedding = InputEmbedding(d_model, vocab_size)
        self.position_embedding = PositionalEmbedding(d_model, block_size)
        self.decoder_block = TransformerDecoder(num_block, d_model, num_heads, d_ffn, dropout)
        self.ln_norm = LayerNormalization(d_model)
        self.last_lyr = nn.Linear(d_model, vocab_size)

   

    def forward(self, idx):
        out = self.input_embedding(idx)   ##idx shape of (B, T)  ### out shape of (B, T, C)
        out = self.position_embedding(out)  ### out shape of (B, T, C)
        out = self.decoder_block(out)   ### out shape of (B, T, C)
        out = self.ln_norm(out)   ### out shape of (B, T, C)
        logits = self.last_lyr(out)  ##(B, T, vocab_size)
        return logits
