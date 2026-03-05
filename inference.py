import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import math


# -------- Paths --------
MODEL_PATH = r"model.bin"
TOKENIZER_PATH = r"tokenizer.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.6


# -------- Config --------
class Config:
    vocab_size  = 50304
    d_model     = 1024
    n_layers    = 24
    n_heads     = 16
    d_ff        = 4096
    max_seq_len = 512
    dropout     = 0.0


# -------- Model --------
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k     = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1,1,config.max_seq_len,config.max_seq_len)
        )

    def forward(self,x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(self.d_model,dim=2)
        q = q.view(B,T,self.n_heads,self.d_k).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.d_k).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.d_k).transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(self.d_k))
        attn = attn.masked_fill(self.mask[:,:,:T,:T]==0, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn,dim=-1)
        out = (attn @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model,config.d_ff)
        self.fc2 = nn.Linear(config.d_ff,config.d_model)

    def forward(self,x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff   = FeedForward(config)
        self.ln1  = nn.LayerNorm(config.d_model)
        self.ln2  = nn.LayerNorm(config.d_model)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self,config,pad_id):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size,config.d_model,padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(config.max_seq_len,config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model,config.vocab_size,bias=False)

    def forward(self,idx):

        B,T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T,device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# -------- Generate --------
@torch.no_grad()
def generate(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids],dtype=torch.long,device=DEVICE)
    for _ in range(MAX_NEW_TOKENS):
        logits = model(idx[:,-model.config.max_seq_len:])
        logits = logits[:,-1,:] / TEMPERATURE
        probs = F.softmax(logits,dim=-1)
        next_token = torch.multinomial(probs,1)
        idx = torch.cat([idx,next_token],dim=1)
        if idx.shape[1] >= model.config.max_seq_len:
            break
    output_ids = idx[0,len(ids):].tolist()
    return tokenizer.decode(output_ids)


# -------- Main --------
def main():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    pad_id = tokenizer.token_to_id("<|pad|>")
    config = Config()
    model = GPT(config,pad_id).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model Ready\n")
    while True:
        prompt = input(">> ")
        if prompt.lower() == "exit":
            break
        output = generate(model,tokenizer,prompt)
        print(prompt + output, "\n")


if __name__ == "__main__":
    main()
