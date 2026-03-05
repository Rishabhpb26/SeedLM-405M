import json
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

config = json.load(open("config.json"))
weights = torch.load("model.bin", map_location="cpu")
tokenizer = Tokenizer.from_file("tokenizer.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = {name: tensor.to(device) for name, tensor in weights.items()}
num_heads, hidden_size = config["n_heads"], config["d_model"]
temperature = 0.5

def next_token_logits(token_ids):
    x = torch.tensor([token_ids], device=device)
    seq_len = x.size(1)
    h = weights["token_emb.weight"][x] + weights["pos_emb.weight"][:seq_len]
    for layer in range(config["n_layers"]):
        z = F.layer_norm(h, (hidden_size,), weights[f"blocks.{layer}.ln1.weight"], weights[f"blocks.{layer}.ln1.bias"])
        qkv = F.linear(z, weights[f"blocks.{layer}.attn.qkv.weight"], weights[f"blocks.{layer}.attn.qkv.bias"])
        q, k, v = qkv.chunk(3, -1)
        q = q.view(1, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
        k = k.view(1, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
        v = v.view(1, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / ((hidden_size // num_heads) ** 0.5)
        mask = weights[f"blocks.{layer}.attn.mask"][:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, -1e9)
        h = h + F.linear((F.softmax(attn, -1) @ v).transpose(1, 2).reshape(1, seq_len, hidden_size), weights[f"blocks.{layer}.attn.out.weight"], weights[f"blocks.{layer}.attn.out.bias"])
        z = F.layer_norm(h, (hidden_size,), weights[f"blocks.{layer}.ln2.weight"], weights[f"blocks.{layer}.ln2.bias"])
        h = h + F.linear(F.gelu(F.linear(z, weights[f"blocks.{layer}.ff.fc1.weight"], weights[f"blocks.{layer}.ff.fc1.bias"])), weights[f"blocks.{layer}.ff.fc2.weight"], weights[f"blocks.{layer}.ff.fc2.bias"])
    z = F.layer_norm(h, (hidden_size,), weights["ln_f.weight"], weights["ln_f.bias"])
    return F.linear(z[:, -1], weights["head.weight"]) / temperature

def generate(prompt, max_new_tokens=128):
    tokens = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        probs = F.softmax(next_token_logits(tokens), -1)
        tokens.append(torch.multinomial(probs, 1).item())
    return tokenizer.decode(tokens)

while True:
    prompt = input("Prompt (or 'exit'/'quit'): ").strip()
    if prompt.lower() in {"exit", "quit"}:
        break
    if not prompt:
        continue
    print(generate(prompt))
