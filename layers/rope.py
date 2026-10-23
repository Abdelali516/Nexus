from pathlib import Path
import json
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT_DIR / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)


class RoPE(nn.Module):
    def __init__(self, head_dim=cfg["d_model"] // cfg["num_heads"],
                 max_seq_len=cfg["max_seq_len"], base=cfg["rope_theta"]):
        super().__init__()
        assert head_dim % 2 == 0

        freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        pos = torch.arange(max_seq_len).float()
        
        angles = torch.outer(pos, freq)  # (max_seq_len, head_dim / 2)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, start_pos=0):
        # x: (batch, heads, seq_len, head_dim)
        seq_len = x.shape[-2] # number of tokens per sequence. 
        assert start_pos + seq_len <= self.cos.shape[0] # sequence exceeds max_seq_len

        cos = self.cos[start_pos:start_pos + seq_len]  # (seq_len, head_dim / 2)
        sin = self.sin[start_pos:start_pos + seq_len]

        # first half and second half of the head form the pairs
        x1, x2 = x.float().chunk(2, dim=-1) 

        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        out = torch.cat((out1, out2), dim=-1)
        return out.to(x.dtype)