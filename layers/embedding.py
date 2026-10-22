from pathlib import Path
import json
import torch.nn as nn

ROOT_DIR=Path(__file__).resolve().parent.parent

CONFIG_FILE=ROOT_DIR/ "config.json"

with open (CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg=json.load(f)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], init_std=cfg["init_std"]):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std) # std= standard deviation of the initial values

    def forward(self, token_ids):
        return self.embedding(token_ids)